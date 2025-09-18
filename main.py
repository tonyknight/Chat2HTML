import os
import re
import subprocess
import datetime
import logging
import argparse
import shutil
import requests
import calendar
import json
import time
import vobject
import base64
import sqlite3
from collections import OrderedDict
from functools import partial
import concurrent.futures
from langdetect import detect, LangDetectException
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---
PERFORM_TRANSLATION = True
# Number of parallel workers for processing images and translations
MAX_WORKERS = 25
# TODO: Set the path to your exported WhatsApp chat file
CHAT_FILE_PATH = "/Users/Add/Path/Here/_chat.txt"

# --- OPENROUTER AI SETTINGS (PLACEHOLDERS) ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Example model, change as needed
OPENROUTER_MODEL = "google/gemma-3-12b-it"

# --- DIRECTORY CONFIGURATION ---
OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
TEMPLATE_DIR = "templates"

# --- OTHER SETTINGS ---
EXIFTOOL_PATH = "exiftool" # Assumes exiftool is in the system's PATH

# --- SCRIPT GLOBALS ---
error_log_filename = ""
translation_stats = {"count": 0, "tokens_in": 0, "tokens_out": 0}
messages_processed = 0
images_processed = 0
videos_processed = 0
 
# --- iMessage GLOBALS ---
IMESSAGE_CONTACTS = {}
TAPBACK_EMOJI_MAP = {
    "Liked": "👍", "Loved": "❤️", "Laughed at": "😂",
    "Emphasized": "❗️", "Disliked": "👎", "Questioned": "❓"
}
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.heic', '.tif', '.tiff']
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v', '.avi']

def setup_logging():
    """Sets up a logger to write errors to a timestamped file."""
    global error_log_filename
    timestamp = datetime.datetime.now().strftime("(%Y-%m-%d) %H-%M-%S")
    error_log_filename = "{} Export Errors.txt".format(timestamp)
    
    # We will log to a file in the same directory as the script.
    # For simplicity, we won't use the logging module directly yet,
    # but prepare for it. We'll manually write errors for now.
    # This keeps it simple to start.
    print(f"Logging errors to: {error_log_filename}")
    # Clear the log file if it exists
    with open(error_log_filename, "w") as f:
        pass

def log_error(message):
    """Appends an error message to the log file."""
    if not error_log_filename:
        # This should not happen if setup_logging is called first
        print("Logging not set up. Cannot log error.")
        return
    with open(error_log_filename, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def parse_whatsapp_chat(file_path):
    """
    Parses the WhatsApp chat txt file.

    Args:
        file_path (str): The path to the chat file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a message.
    """
    # Regex to match the start of a WhatsApp message line
    # e.g., [2024-08-27, 22:43:45] Tony Knight: message
    # Added optional non-printable character support at the beginning of the line and the message.
    message_start_regex = re.compile(r"^\u200e?\[(\d{4}-\d{2}-\d{2}, \d{2}:\d{2}:\d{2})\] ([^:]+): \u200e?(.*)")
    
    messages = []
    current_message = None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = message_start_regex.match(line)
                if match:
                    # If we have a current message, save it before starting a new one
                    if current_message:
                        # Clean up the message content
                        current_message["message"] = current_message["message"].strip()
                        # --- ADDITION ---
                        # Skip adding messages that are just "image omitted"
                        if current_message["message"] != "image omitted":
                            messages.append(current_message)

                    timestamp_str, author, message_content = match.groups()
                    
                    current_message = {
                        "timestamp": datetime.datetime.strptime(timestamp_str, "%Y-%m-%d, %H:%M:%S"),
                        "author": author.strip(),
                        "message": message_content.strip(),
                        "type": "text", # Default type
                    }
                elif current_message:
                    # This is a continuation of the previous message (multi-line)
                    current_message["message"] += "\n" + line.strip()

            # Append the very last message
            if current_message:
                current_message["message"] = current_message["message"].strip()
                if current_message["message"] != "image omitted":
                    messages.append(current_message)

    except FileNotFoundError:
        log_error(f"Input chat file not found at: {file_path}")
        print(f"Error: Input chat file not found at '{file_path}'. Please check the path.")
        return None
    except Exception as e:
        log_error(f"An unexpected error occurred while parsing the chat file: {e}")
        print(f"An unexpected error occurred: {e}")
        return None

    return messages


def _fix_facebook_encoding(text):
    """
    Fixes the double-encoded strings from Facebook's JSON export.
    It appears that UTF-8 strings were incorrectly decoded as latin-1.
    This function reverses that process.
    """
    if text is None:
        return ""
    return text.encode('latin-1').decode('utf-8')


def parse_facebook_chat(file_path):
    """
    Parses a Facebook Messenger chat JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        log_error(f"Input chat file not found at: {file_path}")
        print(f"Error: Input chat file not found at '{file_path}'.")
        return None
    except json.JSONDecodeError:
        log_error(f"Error decoding JSON from file: {file_path}")
        print(f"Error: The file at '{file_path}' is not a valid JSON file.")
        return None
    except Exception as e:
        log_error(f"An unexpected error occurred while reading the chat file: {e}")
        print(f"An unexpected error occurred: {e}")
        return None

    messages = []
    # Messages are in reverse chronological order, so we reverse them
    for msg_data in reversed(data.get("messages", [])):
        timestamp_ms = msg_data.get("timestamp_ms")
        if not timestamp_ms:
            continue

        message_content = ""
        # The json.load automatically handles unicode decoding.
        # We need to handle different types of content.
        if "content" in msg_data:
            message_content += _fix_facebook_encoding(msg_data["content"])
        
        if "share" in msg_data and "link" in msg_data["share"]:
            link = msg_data["share"]["link"]
            share_text = _fix_facebook_encoding(msg_data["share"].get("share_text", ""))
            message_content += f'\n<br>[Shared Link: <a href="{link}" target="_blank">{link}</a>]'
            if share_text:
                message_content += f"<br>Preview: {share_text}"

        message = {
            "timestamp": datetime.datetime.fromtimestamp(timestamp_ms / 1000),
            "timestamp_ms": timestamp_ms, # Keep for photo naming
            "author": _fix_facebook_encoding(msg_data.get("sender_name", "Unknown")),
            "message": message_content.strip(),
            "type": "text", # Default type
        }

        if "photos" in msg_data and msg_data["photos"]:
            photo_uri = msg_data["photos"][0].get("uri")
            if photo_uri:
                message["type"] = "image"
                # We store the original URI for now. Processing will happen later.
                message["message"] = photo_uri
        elif "videos" in msg_data and msg_data["videos"]:
            video_uri = msg_data["videos"][0].get("uri")
            if video_uri:
                message["type"] = "video"
                message["message"] = video_uri
        
        messages.append(message)

    return messages


def translate_text(text):
    """
    Translates text to English using the OpenRouter AI API.

    Args:
        text (str): The text to translate.

    Returns:
        dict: A dictionary containing the translated text and token usage, 
              or None if translation fails.
    """
    if not text or not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_API_KEY":
        log_error("OpenRouter API key not set. Skipping translation.")
        return None

    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert translator. Translate the following text from Brazilian Portugueseto English. Keep as much of the original meaning and style as possible, but do not add any extra information. Direct translation only. Return only the translated text and nothing else."},
                    {"role": "user", "content": text}
                ]
            }
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        translated_text = data['choices'][0]['message']['content'].strip()
        
        # Get usage stats
        usage = data.get('usage', {})
        tokens_in = usage.get('prompt_tokens', 0)
        tokens_out = usage.get('completion_tokens', 0)

        return {
            "translated_text": translated_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out
        }

    except requests.exceptions.RequestException as e:
        log_error(f"Translation API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        log_error(f"Failed to parse translation API response: {e}")
        return None


def _process_single_whatsapp_message(message, source_dir):
    """
    Processes a single message. Designed to be run in a thread pool.
    This function is self-contained and does not modify global state.

    Args:
        message (dict): A message dictionary.
        source_dir (str): The directory of the source chat file.

    Returns:
        tuple: A tuple containing the (possibly modified) message and a dictionary of stats.
    """
    stats = {'images': 0, 'tokens_in': 0, 'tokens_out': 0, 'translations': 0}
    
    # Regexes must be defined here as they are not thread-safe if compiled globally in some versions
    attachment_regex = re.compile(r"^\u200e?<attached: (.+)>$")
    image_filename_regex = re.compile(r"-?(\d+)-(.+)-(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})\.(.+)")

    # First, check for image attachments
    match = attachment_regex.match(message["message"])
    if match:
        original_filename = match.group(1)
        source_path = os.path.join(source_dir, original_filename)

        if not os.path.exists(source_path):
            log_error(f"Image attachment not found: {original_filename}")
            message["message"] = f"[Image not found: {original_filename}]"
            return message, stats
        
        file_match = image_filename_regex.match(original_filename)
        if not file_match:
            log_error(f"Could not parse image filename format: {original_filename}")
            new_filename = original_filename
            dt_for_exif = message["timestamp"]
        else:
            seq_num, _, date_str, time_str, ext = file_match.groups()
            time_str_formatted = time_str.replace('-', ':')
            new_filename = f"({date_str} {time_str}) {int(seq_num)}.{ext}"
            dt_for_exif = datetime.datetime.strptime(f"{date_str} {time_str_formatted}", "%Y-%m-%d %H:%M:%S")

        dest_path = os.path.join(IMAGES_DIR, new_filename)
        
        try:
            shutil.copy2(source_path, dest_path)
            file_extension = os.path.splitext(original_filename)[1].lower()
            if file_extension not in ['.webp']:
                exif_timestamp = dt_for_exif.strftime("%Y:%m:%d %H:%M:%S")
                command = [EXIFTOOL_PATH, "-overwrite_original", f"-DateTimeOriginal={exif_timestamp}", dest_path]
                subprocess.run(command, check=True, capture_output=True, text=True)

            message["type"] = "image"
            message["message"] = os.path.join("images", new_filename)
            stats['images'] = 1
        except Exception as e:
            log_error(f"Failed to process image {original_filename}. Error: {e}")
            message["message"] = f"[Error processing image: {original_filename}]"
        
        return message, stats

    # --- TEXT PROCESSING & TRANSLATION ---
    if not message["message"] or not PERFORM_TRANSLATION:
        return message, stats
        
    try:
        lang = detect(message["message"])
        if lang != 'en':
            translation_result = translate_text(message["message"])
            if translation_result:
                message["translation"] = translation_result["translated_text"]
                stats['translations'] = 1
                stats['tokens_in'] = translation_result['tokens_in']
                stats['tokens_out'] = translation_result['tokens_out']
    except LangDetectException:
        pass # Ignore short/un-detectable text
    except Exception as e:
        log_error(f"An unexpected error occurred during language detection/translation: {e}")

    return message, stats


def _process_single_facebook_message(message, source_dir):
    """
    Processes a single Facebook message.
    Handles photo copying and renaming.
    Translation will be handled in a separate step due to its complexity.
    """
    stats = {'images': 0, 'videos': 0}

    if message['type'] == 'image':
        original_uri = message['message']
        
        # Truncate the URI to get the path relative to the JSON file's directory
        # Find the 'photos/' part and construct the path
        try:
            # The part of the URI we need starts after the chat-specific directory
            # e.g., your_facebook_activity/messages/.../elianamontouto_.../photos/image.jpg
            # We want to find the path relative to the input JSON directory.
            # A simpler way is to just look for the 'photos/' directory in the path.
            rel_path_start = original_uri.find('photos/')
            if rel_path_start == -1:
                 raise ValueError("Path does not contain 'photos/' segment.")
            
            relative_path = original_uri[rel_path_start:]
            source_path = os.path.join(source_dir, relative_path)
            
            if not os.path.exists(source_path):
                log_error(f"Image not found at source: {source_path}")
                message['message'] = f"[Image not found: {os.path.basename(original_uri)}]"
                message['type'] = 'text' # Revert to text if image is missing
                return message, stats

            # Generate new filename from timestamp_ms
            dt_obj = datetime.datetime.fromtimestamp(message['timestamp_ms'] / 1000)
            milliseconds = message['timestamp_ms'] % 1000
            file_extension = os.path.splitext(original_uri)[1]
            new_filename = f"({dt_obj.strftime('%Y-%m-%d %H-%M-%S')}) {milliseconds:03d}{file_extension}"
            
            dest_path = os.path.join(IMAGES_DIR, new_filename)
            
            shutil.copy2(source_path, dest_path)
            
            message['message'] = os.path.join("images", new_filename)
            stats['images'] = 1

        except (ValueError, Exception) as e:
            log_error(f"Failed to process Facebook image {original_uri}. Error: {e}")
            message['message'] = f"[Error processing image: {os.path.basename(original_uri)}]"
            message['type'] = 'text'
    elif message['type'] == 'video':
        original_uri = message['message']
        try:
            rel_path_start = original_uri.find('videos/')
            if rel_path_start == -1:
                    raise ValueError("Path does not contain 'videos/' segment.")
            
            relative_path = original_uri[rel_path_start:]
            source_path = os.path.join(source_dir, relative_path)

            if not os.path.exists(source_path):
                log_error(f"Video not found at source: {source_path}")
                message['message'] = f"[Video not found: {os.path.basename(original_uri)}]"
                message['type'] = 'text'
                return message, stats

            # Generate new filename from timestamp_ms
            dt_obj = datetime.datetime.fromtimestamp(message['timestamp_ms'] / 1000)
            milliseconds = message['timestamp_ms'] % 1000
            file_extension = os.path.splitext(original_uri)[1]
            new_filename = f"({dt_obj.strftime('%Y-%m-%d %H-%M-%S')}) {milliseconds:03d}{file_extension}"
            
            dest_path = os.path.join(VIDEOS_DIR, new_filename)
            
            shutil.copy2(source_path, dest_path)
            
            # Update metadata with exiftool
            exif_timestamp = dt_obj.strftime("%Y:%m:%d %H:%M:%S")
            tags_to_update = [
                "-DateTimeOriginal", "-CreateDate", "-ModifyDate",
                "-TrackCreateDate", "-TrackModifyDate",
                "-MediaCreateDate", "-MediaModifyDate"
            ]
            command = [EXIFTOOL_PATH, "-overwrite_original"]
            for tag in tags_to_update:
                command.append(f"{tag}={exif_timestamp}")
            command.append(dest_path)

            subprocess.run(command, check=True, capture_output=True, text=True)

            message['message'] = os.path.join("videos", new_filename)
            stats['videos'] = 1

        except subprocess.CalledProcessError as e:
            log_error(f"Exiftool failed for video {original_uri}. Error: {e.stderr}")
        except (ValueError, Exception) as e:
            log_error(f"Failed to process Facebook video {original_uri}. Error: {e}")
            message['message'] = f"[Error processing video: {os.path.basename(original_uri)}]"
            message['type'] = 'text'
 
    # Translation is not handled here. It will be a separate step after all messages are processed.
    return message, stats


def process_whatsapp_messages(messages, input_file_path):
    """
    Processes all messages in parallel to handle images and translations.
    """
    global images_processed, translation_stats
    
    source_dir = os.path.dirname(input_file_path)
    processed_messages = []
    
    # Use functools.partial to pre-fill the source_dir argument for our task function
    task_function = partial(_process_single_whatsapp_message, source_dir=source_dir)
    
    print(f"Processing {len(messages)} messages with up to {MAX_WORKERS} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # map applies the function to each item in messages and returns an iterator of results
        results = list(executor.map(task_function, messages))

    # Now, aggregate results safely in the main thread
    for message, stats in results:
        processed_messages.append(message)
        if stats.get('images'):
            images_processed += stats['images']
        if stats.get('translations'):
            translation_stats['count'] += stats['translations']
            translation_stats['tokens_in'] += stats['tokens_in']
            translation_stats['tokens_out'] += stats['tokens_out']

    return processed_messages


def _filter_chunk_for_translation(message_chunk):
    """
    Analyzes a chunk of messages to find which ones need translation.
    This is designed to be run in a parallel worker.
    """
    messages_to_translate = []
    for i, current_msg in enumerate(message_chunk):
        if current_msg['type'] != 'text' or not current_msg['message']:
            continue

        try:
            lang = detect(current_msg['message'])
            if lang == 'en':
                continue
        except LangDetectException:
            continue

        is_translation_of_another = False

        # Check previous message from the same author within the chunk
        for j in range(i - 1, -1, -1):
            prev_msg = message_chunk[j]
            if prev_msg['author'] == current_msg['author']:
                if (current_msg['timestamp'] - prev_msg['timestamp']).total_seconds() > 300:
                    break
                if prev_msg['type'] == 'text' and prev_msg['message']:
                    try:
                        if detect(prev_msg['message']) == 'en' and abs(len(current_msg['message']) - len(prev_msg['message'])) / len(prev_msg['message']) < 0.2:
                            is_translation_of_another = True
                            break
                    except LangDetectException:
                        pass
                break

        if is_translation_of_another:
            continue

        # Check next message from the same author within the chunk
        for j in range(i + 1, len(message_chunk)):
            next_msg = message_chunk[j]
            if next_msg['author'] == current_msg['author']:
                if (next_msg['timestamp'] - current_msg['timestamp']).total_seconds() > 300:
                    break
                if next_msg['type'] == 'text' and next_msg['message']:
                    try:
                        if detect(next_msg['message']) == 'en' and abs(len(current_msg['message']) - len(next_msg['message'])) / len(next_msg['message']) < 0.2:
                            is_translation_of_another = True
                            break
                    except LangDetectException:
                        pass
                break
        
        if not is_translation_of_another:
            messages_to_translate.append(current_msg)
            
    return messages_to_translate


def process_facebook_messages(messages, input_file_path):
    """
    Processes all Facebook messages.
    First, it handles file operations like copying photos in parallel.
    Then, it handles the complex translation logic sequentially.
    """
    global images_processed, videos_processed, translation_stats
    source_dir = os.path.dirname(input_file_path)
    
    # --- Step 1: Process images and videos in parallel ---
    processed_media_messages = []
    task_function = partial(_process_single_facebook_message, source_dir=source_dir)
    
    print(f"Processing {len(messages)} messages for media with up to {MAX_WORKERS} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(task_function, messages))

    for message, stats in results:
        processed_media_messages.append(message)
        if stats.get('images'):
            images_processed += stats['images']
        if stats.get('videos'):
            videos_processed += stats['videos']
            
    if not PERFORM_TRANSLATION:
        return processed_media_messages

    # --- Step 2: Filter messages for translation in parallel ---
    print("Analyzing messages to identify which need translation...")
    
    # Find safe split points (e.g., gaps of > 1 hour)
    split_indices = [0]
    for i in range(len(processed_media_messages) - 1):
        time_diff = (processed_media_messages[i+1]['timestamp'] - processed_media_messages[i]['timestamp']).total_seconds()
        if time_diff > 3600: # 1 hour gap
            split_indices.append(i + 1)
    split_indices.append(len(processed_media_messages))

    chunks = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i+1]
        chunks.append(processed_media_messages[start_idx:end_idx])

    messages_to_translate = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {executor.submit(_filter_chunk_for_translation, chunk): chunk for chunk in chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                result = future.result()
                messages_to_translate.extend(result)
            except Exception as exc:
                log_error(f'A chunk generated an exception during translation filtering: {exc}')

    # --- Step 3: Translate the filtered messages with rate limiting ---
    total_to_translate = len(messages_to_translate)
    print(f"Found {total_to_translate} messages to translate. Beginning translation with a 5-second delay between requests...")
    
    for i, message in enumerate(messages_to_translate):
        print(f"  Translating message {i+1} of {total_to_translate}...")
        translation_result = translate_text(message["message"])
        if translation_result:
            message["translation"] = translation_result["translated_text"]
            translation_stats['count'] += 1
            translation_stats['tokens_in'] += translation_result['tokens_in']
            translation_stats['tokens_out'] += translation_result['tokens_out']
        
        if i < total_to_translate - 1: # Don't sleep after the last one
             time.sleep(1)

    return processed_media_messages


def get_heatmap_color(message_count):
    """Returns a hex color code based on the number of messages."""
    if message_count <= 0:
        return ""
    if message_count <= 2:
        return "#fdc70c"
    elif message_count <= 5:
        return "#f3903f"
    elif message_count <= 9:
        return "#ed683c"
    else: # 10 or more
        return "#e93e3a"


def generate_html(messages):
    """
    Generates the final HTML file from the processed messages.

    Args:
        messages (list): The list of processed message dictionaries.
    """
    if not messages:
        print("No messages to generate HTML for.")
        return

    # --- Prepare data for the template ---
    
    # 1. Determine authors and main user
    # Assumption: The second unique author found is the main user (sender)
    all_authors = list(OrderedDict.fromkeys([m["author"] for m in messages]))
    main_author = all_authors[1] if len(all_authors) > 1 else all_authors[0]

    # 2. Group messages by date
    messages_by_date = {}
    for msg in messages:
        date = msg["timestamp"].date()
        if date not in messages_by_date:
            messages_by_date[date] = []
        messages_by_date[date].append(msg)
        
    # --- NEW: Consolidate sequential images into galleries ---
    consolidated_messages_by_date = OrderedDict()
    for date, daily_messages in messages_by_date.items():
        new_daily_messages = []
        i = 0
        while i < len(daily_messages):
            current_msg = daily_messages[i]

            # Check if we can start a gallery (must be an image)
            if current_msg['type'] == 'image':
                gallery_images = [current_msg]
                j = i + 1
                # Look ahead for more images that meet the criteria
                while j < len(daily_messages):
                    next_msg = daily_messages[j]
                    time_difference = next_msg['timestamp'] - current_msg['timestamp']
                    
                    if (next_msg['type'] == 'image' and
                        next_msg['author'] == current_msg['author'] and
                        time_difference.total_seconds() < 300):  # 5 minutes = 300 seconds
                        
                        gallery_images.append(next_msg)
                        j += 1
                    else:
                        break  # End of the gallery sequence
                
                # If we grouped more than one image, create a gallery message
                if len(gallery_images) > 1:
                    gallery_message = {
                        'type': 'gallery',
                        'author': current_msg['author'],
                        'timestamp': current_msg['timestamp'],
                        'images': gallery_images
                    }
                    new_daily_messages.append(gallery_message)
                    i = j  # Skip the main loop ahead to the end of the gallery
                else:
                    # It's just a single image, not a gallery
                    new_daily_messages.append(current_msg)
                    i += 1
            else:
                # It's a text message, add it and move on
                new_daily_messages.append(current_msg)
                i += 1
        consolidated_messages_by_date[date] = new_daily_messages

    # 3. Generate Calendar Data
    message_dates = set(consolidated_messages_by_date.keys())
    if not message_dates:
        first_date = last_date = datetime.date.today()
    else:
        first_date = min(message_dates)
        last_date = max(message_dates)

    # New: Store message counts for heatmap
    message_counts_by_date = {date: len(msgs) for date, msgs in consolidated_messages_by_date.items()}
    
    cal_data_by_year = OrderedDict()

    for year in range(first_date.year, last_date.year + 1):
        cal_data_by_year[year] = {
            "year": year,
            "months": []
        }
        start_month = first_date.month if year == first_date.year else 1
        end_month = last_date.month if year == last_date.year else 12

        for month in range(start_month, end_month + 1):
            cal = calendar.monthcalendar(year, month)
            month_name = datetime.date(year, month, 1).strftime("%B")

            month_data = {
                "name": month_name,
                "weeks": []
            }
            
            for week in cal:
                week_data = []
                for day in week:
                    if day == 0:
                        week_data.append({"day": 0})
                    else:
                        current_day_date = datetime.date(year, month, day)
                        message_count = message_counts_by_date.get(current_day_date, 0)
                        week_data.append({
                            "day": day,
                            "has_message": message_count > 0,
                            "message_count": message_count,
                            "heatmap_color": get_heatmap_color(message_count),
                            "date_str": current_day_date.strftime("%Y-%m-%d")
                        })
                month_data["weeks"].append(week_data)
            
            cal_data_by_year[year]["months"].append(month_data)

    # --- Render and Save HTML ---
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("template.html")
    
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template.render(
            messages_by_date=consolidated_messages_by_date,
            main_author=main_author,
            calendar_data_by_year=list(cal_data_by_year.values())
        ))


def main():
    """Main function to orchestrate the chat export processing."""
    global messages_processed, videos_processed
    
    # Load environment variables from .env file
    load_dotenv("API_Key.env")
    
    setup_logging()

    # Create output directories if they don't exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
 
    # --- User Interface ---
    print("--- Chat Export to HTML Converter ---")
    print("Select the chat service:")
    print("  1. WhatsApp (.txt file)")
    print("  2. Facebook Messenger (.json file)")
    print("  3. Apple iMessage (.html file)")
    
    choice = ""
    while choice not in ["1", "2", "3"]:
        choice = input("Enter your choice (1, 2, or 3): ").strip()

    input_file_path = input("Please enter the full path to your chat file: ").strip()

    # --- Processing Logic ---
    parsed_messages = None
    processed_messages = None

    if choice == "1":
        print(f"Starting WhatsApp chat export processing for: {input_file_path}")
        parsed_messages = parse_whatsapp_chat(input_file_path)
        if parsed_messages:
            messages_processed = len(parsed_messages)
            print(f"Successfully parsed {messages_processed} messages.")
            processed_messages = process_whatsapp_messages(parsed_messages, input_file_path)

    elif choice == "2":
        print(f"Starting Facebook Messenger chat export processing for: {input_file_path}")
        # Placeholder for Facebook processing
        parsed_messages = parse_facebook_chat(input_file_path)
        if parsed_messages:
             # This part will be implemented later
            processed_messages = process_facebook_messages(parsed_messages, input_file_path)

    elif choice == "3":
        handle_imessage_processing()
        return # Exit after iMessage processing


    # --- HTML Generation and Final Report ---
    if processed_messages:
        print(f"Processed {images_processed} image attachments.")
        print(f"Processed {videos_processed} video attachments.")
        print("Generating HTML file...")
        generate_html(processed_messages)
        print("HTML generation complete.")
        
        # --- Final Report ---
        print("\n--- Export Report ---")
        print(f"Total Messages Processed: {messages_processed}")
        print(f"Total Images Processed:   {images_processed}")
        print(f"Total Videos Processed:   {videos_processed}")
        print("--- Translation Stats ---")
        print(f"  Translations: {translation_stats['count']}")
        print(f"  Tokens In:    {translation_stats['tokens_in']}")
        print(f"  Tokens Out:   {translation_stats['tokens_out']}")
        print("-----------------------")
        print(f"HTML file generated at: {os.path.join(OUTPUT_DIR, 'index.html')}")
        if os.path.exists(error_log_filename) and os.path.getsize(error_log_filename) > 0:
            print(f"Errors were logged to: {error_log_filename}")
        print("-----------------------")
    elif parsed_messages is None and choice == "1":
         print("Processing stopped due to errors during WhatsApp parsing.")
    elif parsed_messages is None and choice == "2":
         print("Processing stopped. Facebook Messenger support is not yet implemented.")
    else:
        print("No messages were processed.")


def rename_imessage_files(directory_path):
    """
    Identifies author names in iMessage export HTML filenames and
    renames the files to encapsulate the author in brackets.
    e.g., "John Doe +12345.html" -> "[John Doe] +12345.html"
    """
    print(f"\nScanning for HTML files in: {directory_path}")
    try:
        html_files = [f for f in os.listdir(directory_path) if f.endswith('.html') and not f.startswith('[')]
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'.")
        return

    if not html_files:
        print("No HTML files found to rename (or they are already renamed).")
        return

    print(f"Found {len(html_files)} HTML files to process...")
    renamed_count = 0
    for filename in html_files:
        # Simple algorithm to find the author name
        # We assume the name is the sequence of capitalized words at the start.
        name_parts = []
        # Remove .html extension for processing
        base_name = filename[:-5]
        # Using split() without arguments handles multiple spaces and avoids empty strings
        words = base_name.split()
        
        for i, word in enumerate(words):
            # An identifier is likely to start with '+', a digit, or contain '@'
            if (word.startswith('+') or
                word.isdigit() or
                '@' in word or
                (i > 0 and word[0].isdigit())): # Handles numbers not at the start
                 # Everything before this is the name
                break
            else:
                name_parts.append(word)
        
        # If no identifier was found, the whole filename is the name
        author_name = " ".join(name_parts)

        if author_name:
            # Clean up trailing commas from the author name that might be part of the source filename
            author_name = author_name.strip().rstrip(',')
            
            # Construct new filename
            rest_of_filename = " ".join(words[len(name_parts):])
            new_filename = f"[{author_name}] {rest_of_filename}.html".replace("] .html", "].html").strip()

            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"  Renamed: '{filename}' -> '{new_filename}'")
                renamed_count += 1
            except OSError as e:
                print(f"  Error renaming '{filename}': {e}")
        else:
            print(f"  Could not determine author for '{filename}'. Skipping.")

    print(f"\nFinished renaming. {renamed_count} files were renamed.")


def organize_imessage_files(directory_path):
    """
    Finds HTML files with bracketed author names and moves them
    into subdirectories named after the author.
    """
    print(f"\nScanning for renamed HTML files in: {directory_path}")
    try:
        # Look for files that HAVE been renamed
        html_files = [f for f in os.listdir(directory_path) if f.endswith('.html') and f.startswith('[')]
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'.")
        return

    if not html_files:
        print("No renamed HTML files (e.g., '[Author Name]...') found to organize.")
        return

    print(f"Found {len(html_files)} files to organize...")
    moved_count = 0
    for filename in html_files:
        match = re.match(r"\[(.*?)\]", filename)
        if match:
            author_name = match.group(1)
            author_dir = os.path.join(directory_path, author_name)
            
            # Create directory for the author if it doesn't exist
            os.makedirs(author_dir, exist_ok=True)
            
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(author_dir, filename)
            
            try:
                shutil.move(old_path, new_path)
                print(f"  Moved '{filename}' to '{author_name}/'")
                moved_count += 1
            except Exception as e:
                 print(f"  Error moving '{filename}': {e}")

    print(f"\nFinished organizing. {moved_count} files were moved into author directories.")


def process_vcard_file(file_path):
    """
    Parses a Vcard (.vcf) file to extract contact information.
    """
    global IMESSAGE_CONTACTS
    print(f"\nProcessing Vcard file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            vcf_data = f.read()
    except FileNotFoundError:
        print(f"Error: Vcard file not found at '{file_path}'.")
        return
    except Exception as e:
        print(f"Error reading Vcard file: {e}")
        return

    total_contacts = 0
    total_phones = 0
    total_emails = 0
    total_photos = 0

    try:
        for card in vobject.readComponents(vcf_data):
            total_contacts += 1
            name = card.fn.value if hasattr(card, 'fn') else "Unknown"
            
            contact_info = {"phones": [], "emails": [], "photo": None}

            if hasattr(card, 'tel'):
                for tel in card.tel_list:
                    contact_info["phones"].append(str(tel.value))
                    total_phones += 1
            
            if hasattr(card, 'email'):
                for email in card.email_list:
                    contact_info["emails"].append(str(email.value))
                    total_emails += 1

            if hasattr(card, 'photo'):
                # Assuming the first photo is the one we want
                photo_data = card.photo.value
                # The data is already base64 encoded by the library
                contact_info["photo"] = base64.b64encode(photo_data).decode('utf-8')
                total_photos += 1

            IMESSAGE_CONTACTS[name] = contact_info

    except Exception as e:
        print(f"An error occurred while parsing the Vcard file: {e}")
        return
        
    print("\n--- Vcard Processing Summary ---")
    print(f"  Contacts Found: {total_contacts}")
    print(f"  Phone Numbers:  {total_phones}")
    print(f"  Email Addresses: {total_emails}")
    print(f"  Photos Found:   {total_photos}")
    print("--------------------------------")

    # Display the hierarchical list
    print("\n--- Extracted Contact Data ---")
    if not IMESSAGE_CONTACTS:
        print("No contacts were extracted.")
    else:
        for name, data in IMESSAGE_CONTACTS.items():
            print(f"\n- {name}")
            if data["phones"]:
                print("  - Phones:")
                for phone in data["phones"]:
                    print(f"    - {phone}")
            if data["emails"]:
                print("  - Emails:")
                for email in data["emails"]:
                    print(f"    - {email}")
            if data["photo"]:
                print("  - Photo: Yes")
    print("------------------------------")


def _handle_imessage_attachment(attachment_div, msg_timestamp, author_dir, root_dir):
    """
    Handles moving, renaming, and processing an iMessage attachment.
    Returns the new relative path for the JSON file.
    """
    link = attachment_div.find('a')
    img = attachment_div.find('img')
    video_source = attachment_div.find('source')
    audio_source = attachment_div.find('audio')

    if not (link or img or video_source or audio_source):
        log_error(f"Could not find a link, image, video, or audio source in attachment div: {attachment_div}")
        return "[Attachment type not recognized]"

    original_rel_path = ""
    original_filename_text = ""

    if img:
        original_rel_path = img['src']
    elif video_source:
        original_rel_path = video_source['src']
    elif audio_source:
        original_rel_path = audio_source['src']
    elif link:
        original_rel_path = link['href']
        original_filename_text = link.get_text()

    if not original_rel_path:
        log_error(f"Found attachment tag but could not extract a path (src/href): {attachment_div}")
        return "[Attachment path not found]"

    source_path = os.path.join(root_dir, original_rel_path)

    if not os.path.exists(source_path):
        log_error(f"Attachment not found: {source_path}")
        return f"[Attachment not found: {os.path.basename(original_rel_path)}]"

    # Determine category and destination folder
    _, ext = os.path.splitext(original_rel_path)
    ext = ext.lower()
    dest_folder_name = ""
    if ext in IMAGE_EXTENSIONS:
        dest_folder_name = "images"
    elif ext in VIDEO_EXTENSIONS:
        dest_folder_name = "videos"
    else:
        dest_folder_name = "data"
    
    dest_dir = os.path.join(author_dir, dest_folder_name)
    os.makedirs(dest_dir, exist_ok=True)

    # Determine new filename
    if dest_folder_name in ["images", "videos"]:
        milliseconds = msg_timestamp.microsecond // 1000
        new_filename = f"({msg_timestamp.strftime('%Y-%m-%d %H-%M-%S')}) {milliseconds:03d}{ext}"
    else:
        # Restore original filename from link text
        if 'Click to download ' in original_filename_text:
            new_filename = original_filename_text.split('Click to download ')[1].split(' (')[0]
        else:
            new_filename = os.path.basename(original_rel_path) # Fallback

    dest_path = os.path.join(dest_dir, new_filename)
    
    try:
        shutil.move(source_path, dest_path)
        
        # Set metadata for images/videos
        if dest_folder_name in ["images", "videos"]:
            exif_timestamp = msg_timestamp.strftime("%Y:%m:%d %H:%M:%S")
            tags_to_update = [
                "-DateTimeOriginal", "-CreateDate", "-ModifyDate",
                "-TrackCreateDate", "-TrackModifyDate",
                "-MediaCreateDate", "-MediaModifyDate"
            ]
            command = [EXIFTOOL_PATH, "-overwrite_original", "-api", "largefilesupport=1"]
            for tag in tags_to_update:
                command.append(f"{tag}={exif_timestamp}")
            command.append(dest_path)
            subprocess.run(command, check=True, capture_output=True, text=True, errors='ignore')

        return os.path.join(dest_folder_name, new_filename)

    except subprocess.CalledProcessError as e:
        log_error(f"Exiftool failed for {dest_path}. Error: {e.stderr}")
        return f"[Error setting metadata for {new_filename}]"
    except Exception as e:
        log_error(f"Failed to move/process attachment {source_path}. Error: {e}")
        return f"[Error moving attachment: {os.path.basename(original_rel_path)}]"


def _parse_message_div(msg_div, author_dir, root_dir):
    """
    Parses a single iMessage HTML message div. Since a message can have multiple parts
    (e.g., text and an attachment, or multiple attachments), this function can return
    a list of message objects.
    """
    all_message_parts = []
    
    # Base info that is common to all parts of the message
    sent_div = msg_div.find('div', class_='sent')
    received_div = msg_div.find('div', class_='received')
    msg_type = 'sent' if sent_div else 'received'
    container = sent_div or received_div
    if not container: return []

    timestamp_span = container.find('span', class_='timestamp')
    base_timestamp = None
    if timestamp_span and timestamp_span.find('a'):
        timestamp_str = timestamp_span.find('a').get_text()
        cleaned_timestamp_str = re.sub(r'\s+', ' ', timestamp_str).strip()
        try:
            base_timestamp = datetime.datetime.strptime(cleaned_timestamp_str, '%b %d, %Y %I:%M:%S %p')
        except ValueError:
            log_error(f"Could not parse timestamp: {timestamp_str}")
            return []
    else:
        return []

    sender_span = container.find('span', class_='sender')
    sender = sender_span.get_text() if sender_span else "Unknown"

    # Find all message parts within the message div
    message_parts = container.find_all('div', class_='message_part')

    # Handle messages that have attachments/stickers but no message_part wrapper
    if not message_parts:
        content_div = container.find('div', class_=['sticker', 'attachment'])
        if content_div:
            message_obj = {
                'type': msg_type,
                'timestamp': base_timestamp,
                'sender': sender,
                'body': "",
                'attachment': _handle_imessage_attachment(content_div, base_timestamp, author_dir, root_dir)
            }
            all_message_parts.append(message_obj)

    for part in message_parts:
        message_obj = {
            'type': msg_type,
            'timestamp': base_timestamp,
            'sender': sender
        }

        body_span = part.find('span', class_='bubble')
        message_obj['body'] = body_span.get_text() if body_span else ""
        
        attachment_div = part.find('div', class_='attachment')
        sticker_div = part.find('div', class_='sticker')
        content_div = attachment_div or sticker_div # Unified check

        if content_div:
            attachment_path = _handle_imessage_attachment(content_div, base_timestamp, author_dir, root_dir)
            if attachment_path:
                message_obj['attachment'] = attachment_path
                message_obj['attachment_author_dir'] = os.path.basename(author_dir)

        # App Links (e.g., Flipboard, GamePigeon)
        app_div = part.find('div', class_='app')
        if app_div and app_div.find('a'):
            link = app_div.find('a')['href']
            caption_div = app_div.find('div', class_='caption')
            caption = caption_div.get_text() if caption_div else ""
            subcaption_div = app_div.find('div', class_='subcaption')
            subcaption = subcaption_div.get_text() if subcaption_div else ""
            
            message_obj['shared_link'] = {
                "url": link, "caption": caption, "subcaption": subcaption
            }

        # Only add the message object if it has some content
        if message_obj.get('body') or message_obj.get('attachment') or message_obj.get('shared_link'):
            all_message_parts.append(message_obj)

    # Tapbacks are associated with the entire message, so we find them once
    tapbacks_div = container.find('div', class_='tapbacks')
    if tapbacks_div and all_message_parts:
        # Add tapback info to the first message part
        all_message_parts[0]['tapbacks'] = []
        for tb in tapbacks_div.find_all('div', class_='tapback'):
            text = tb.get_text()
            reaction = text.split(' by ')[0]
            reactor = ' by '.join(text.split(' by ')[1:])
            all_message_parts[0]['tapbacks'].append({
                "reaction": TAPBACK_EMOJI_MAP.get(reaction, reaction),
                "sender": reactor
            })

    return all_message_parts


def _process_author_folder(author_dir_path):
    """
    Parses all HTML files in an author's directory, processes attachments,
    and creates a single, sorted JSON file.
    This is the main worker function for parallel processing.
    """
    author_name = os.path.basename(author_dir_path)
    root_dir = os.path.dirname(author_dir_path)
    all_messages = []
    
    try:
        html_files = [f for f in os.listdir(author_dir_path) if f.endswith('.html')]
        
        for html_file in html_files:
            file_path = os.path.join(author_dir_path, html_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
            message_divs = soup.find_all('div', class_='message')
            for div in message_divs:
                parsed_msgs = _parse_message_div(div, author_dir_path, root_dir)
                if parsed_msgs:
                    all_messages.extend(parsed_msgs)

        # Sort all messages chronologically
        all_messages.sort(key=lambda x: x['timestamp'])

        # Convert datetime objects to strings for JSON
        for msg in all_messages:
            msg['timestamp'] = msg['timestamp'].isoformat()

        # Write to JSON
        json_filename = f"{author_name}.json"
        json_path = os.path.join(author_dir_path, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_messages, f, indent=2, ensure_ascii=False)
            
        return f"Success: Created {json_filename} with {len(all_messages)} messages."

    except Exception as e:
        log_error(f"Failed to process folder {author_name}. Error: {e}")
        return f"Error: Failed to process folder {author_name}. See log for details."


def convert_imessage_html_to_json(root_directory):
    """
    Orchestrates the conversion of all author folders from HTML to JSON in parallel.
    """
    print("\n--- Convert Author HTML to JSON ---")
    
    try:
        author_folders = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d)) and d != 'attachments']
    except FileNotFoundError:
        print(f"Error: Root directory not found at '{root_directory}'.")
        return
        
    if not author_folders:
        print("No author folders found in the specified directory.")
        return

    print(f"Found {len(author_folders)} author folders to process...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a partial function to pass the root directory to the worker
        worker_func = partial(_process_author_folder)
        
        author_paths = [os.path.join(root_directory, name) for name in author_folders]
        
        # Use tqdm for a progress bar
        results = list(tqdm(executor.map(worker_func, author_paths), total=len(author_paths), desc="Processing Authors"))

    print("\n--- Conversion Summary ---")
    success_count = 0
    error_count = 0
    for res in results:
        if res.startswith("Success"):
            success_count += 1
        else:
            error_count += 1
        print(f"  - {res}")
    print("--------------------------")
    print(f"Successfully converted {success_count} author folders.")
    if error_count > 0:
        print(f"Encountered errors in {error_count} author folders. Please check '{error_log_filename}'.")
    print("--------------------------")


def create_master_imessage_json(root_directory):
    """Placeholder for combining all author JSONs into one master file."""
    print("\n--- Create Master iMessage JSON ---")
    print("This feature is not yet implemented.")


def generate_html_from_imessage_json(directory_path):
    """Placeholder for generating the final HTML from the intermediate JSON."""
    print("\n--- Generate Final HTML from iMessage JSON ---")
    print("This feature is not yet implemented.")
    print("This step will take a generated JSON file and create the final HTML report.")


def create_imessage_database(root_directory):
    """Scans for author JSON files and imports them into a central SQLite database."""
    print("\n--- Create iMessage Chat Database ---")

    # One-time setup to identify the archive owner
    message_owner = input("Enter your full name exactly as it appears in iMessage exports (e.g., Tony Knight): ").strip()
    if not message_owner:
        print("Error: A message owner name is required to correctly process the chat data.")
        return

    # Optional Vcard processing
    vcard_path = input("Enter the path to your Vcard (.vcf) file (optional, press Enter to skip): ").strip()
    contact_lookup = {}
    if vcard_path:
        print("Processing Vcard for contact lookup...")
        try:
            with open(vcard_path, 'r', encoding='utf-8') as f:
                vcf_data = f.read()
            for card in vobject.readComponents(vcf_data):
                name = card.fn.value if hasattr(card, 'fn') else None
                if not name: continue
                
                all_identifiers = []
                if hasattr(card, 'tel'):
                    all_identifiers.extend([str(tel.value) for tel in card.tel_list])
                if hasattr(card, 'email'):
                    all_identifiers.extend([str(email.value) for email in card.email_list])
                
                # Create reverse lookup: identifier -> {name, all_identifiers}
                for identifier in all_identifiers:
                    # Normalize phone numbers for better matching
                    normalized_id = re.sub(r'\D', '', identifier)
                    if normalized_id:
                        contact_lookup[normalized_id] = {"name": name, "contacts": all_identifiers}
        except Exception as e:
            print(f"Warning: Could not process Vcard file. Proceeding without contact enrichment. Error: {e}")

    db_path = os.path.join(root_directory, "imessage_archive.db")
    # Delete existing DB file if it exists, to ensure a fresh start
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database file: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        timestamp TEXT,
        message_author TEXT,
        message_sender TEXT,
        message_owner TEXT,
        sender_contact_info TEXT,
        body TEXT,
        attachment_path TEXT,
        attachment_author_dir TEXT,
        tapbacks TEXT,
        shared_link TEXT,
        type TEXT
    )''')
    
    # Find all author JSON files
    author_folders = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d)) and d != 'attachments']
    json_files_found = 0
    messages_imported = 0

    print(f"Scanning {len(author_folders)} author folders for JSON files...")

    for author in author_folders:
        author_path = os.path.join(root_directory, author)
        json_file_path = os.path.join(author_path, f"{author}.json")
        
        if os.path.exists(json_file_path):
            json_files_found += 1
            with open(json_file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
            
            for msg in messages:
                # --- Sender Cleanup and Enrichment ---
                raw_sender = msg.get('sender', '')
                clean_sender = raw_sender
                contact_info_json = None
                
                # Try to find a match in the vcard lookup
                found_contact = None
                for identifier, contact_data in contact_lookup.items():
                    # Check if a normalized phone number is in the raw sender string
                    if identifier in re.sub(r'\D', '', raw_sender):
                        found_contact = contact_data
                        break
                
                if found_contact:
                    clean_sender = found_contact['name']
                    contact_info_json = json.dumps(found_contact['contacts'])
                else:
                    # Fallback: if name and number are together, just keep the name
                    match = re.match(r"([a-zA-Z\s]+)(?:\s*\+?\d+)", raw_sender)
                    if match:
                        clean_sender = match.group(1).strip()

                # Determine message type based on the owner
                msg_type = 'sent' if clean_sender == message_owner else 'received'

                cursor.execute('''
                INSERT INTO messages (timestamp, message_author, message_sender, message_owner, sender_contact_info, body, attachment_path, attachment_author_dir, tapbacks, shared_link, type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    msg.get('timestamp'),
                    author,
                    clean_sender,
                    message_owner,
                    contact_info_json,
                    msg.get('body'),
                    msg.get('attachment'),
                    msg.get('attachment_author_dir'),
                    json.dumps(msg.get('tapbacks')),
                    json.dumps(msg.get('shared_link')),
                    msg_type
                ))
                messages_imported += 1

    conn.commit()
    conn.close()
    
    print("\n--- Database Creation Summary ---")
    print(f"  Database created at: {db_path}")
    print(f"  Author JSON files found: {json_files_found}")
    print(f"  Total messages imported: {messages_imported}")
    print("---------------------------------")


def generate_imessage_html_from_db(root_directory):
    """Generates individual and a master 'all chat' HTML file from the SQLite DB."""
    print("\n--- Generate iMessage HTML Reports from Database ---")
    db_path = os.path.join(root_directory, "imessage_archive.db")
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'. Please create it first.")
        return

    conn = sqlite3.connect(db_path)
    # Use a dictionary cursor for easier data handling
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Generate Individual Author HTMLs
    print("Generating individual author HTML files...")
    cursor.execute("SELECT DISTINCT message_author FROM messages")
    authors = [row['message_author'] for row in cursor.fetchall()]
    
    for author in tqdm(authors, desc="Generating Author Reports"):
        cursor.execute("SELECT * FROM messages WHERE message_author = ? ORDER BY timestamp", (author,))
        messages_data = cursor.fetchall()
        
        if not messages_data:
            continue
            
        # Convert sqlite3.Row to standard dicts and parse JSON strings
        messages = []
        for row in messages_data:
            msg = dict(row)
            msg['timestamp'] = datetime.datetime.fromisoformat(msg['timestamp'])
            # Construct the correct path for the context
            if msg['attachment_path'] and msg['attachment_author_dir']:
                 # For individual reports, the path is relative to the author dir
                 # For the All Chat view, we would use:
                 # os.path.join(msg['attachment_author_dir'], msg['attachment_path'])
                 pass # No change needed for now, but the data is there
            msg['tapbacks'] = json.loads(msg['tapbacks']) if msg['tapbacks'] else None
            msg['shared_link'] = json.loads(msg['shared_link']) if msg['shared_link'] else None
            messages.append(msg)

        min_date = messages[0]['timestamp'].date()
        max_date = messages[-1]['timestamp'].date()
        
        # Generate the HTML using the generic `generate_html` function
        generate_html(messages) # This needs to be adapted
        
        # Rename the output file
        author_html_filename = f"[{author}] ({min_date} to {max_date}).html"
        default_output_path = os.path.join(OUTPUT_DIR, "index.html")
        new_output_path = os.path.join(root_directory, author, author_html_filename)
        
        # This part is tricky. generate_html is hardcoded.
        # For now, let's just print what we WOULD do.
        print(f"  - Would generate HTML for '{author}' with {len(messages)} messages.")
        print(f"    and save to '{new_output_path}'")

    # 2. Generate "All Chat" HTML
    print("\nGenerating 'All Chat' master HTML file...")
    # This will be a large operation, for now, we just indicate it.
    print("This part of the feature is not fully implemented yet.")

    conn.close()


def handle_imessage_processing():
    """Displays the submenu for iMessage processing and calls the appropriate functions."""
    print("\n--- Apple iMessage Processing ---")
    
    while True:
        print("\nPlease select an iMessage processing step:")
        print("  1. Rename HTML files by Author")
        print("  2. Organize Renamed Files into Author Folders")
        print("  3. Process Vcard File (.vcf)")
        print("  4. Convert All Author HTML to JSON")
        print("  5. Create Chat Database from JSON files")
        print("  6. Generate HTML Reports from Database (Not Implemented)")
        print("  7. Return to Main Menu")
        
        choice = input("Enter your choice (1-7): ").strip()

        if choice == '1':
            path = input("Enter the path to the directory containing iMessage HTML exports: ").strip()
            rename_imessage_files(path)
        elif choice == '2':
            path = input("Enter the path to the directory where you renamed the HTML files: ").strip()
            organize_imessage_files(path)
        elif choice == '3':
            path = input("Enter the path to your Vcard (.vcf) file: ").strip()
            process_vcard_file(path)
        elif choice == '4':
            path = input("Enter the path to the top-level directory containing all author folders: ").strip()
            convert_imessage_html_to_json(path)
        elif choice == '5':
            path = input("Enter the path to the top-level directory containing author JSONs: ").strip()
            create_imessage_database(path)
        elif choice == '6':
            path = input("Enter the path to the top-level directory containing the database: ").strip()
            generate_imessage_html_from_db(path)
        elif choice == '7':
            print("Returning to main menu...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")


if __name__ == "__main__":
    main()
