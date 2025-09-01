import os
import re
import subprocess
import datetime
import logging
import argparse
import shutil
import requests
import calendar
from collections import OrderedDict
from functools import partial
import concurrent.futures
from langdetect import detect, LangDetectException
from jinja2 import Environment, FileSystemLoader

# --- CONFIGURATION ---
PERFORM_TRANSLATION = True
# Number of parallel workers for processing images and translations
MAX_WORKERS = 25
# TODO: Set the path to your exported WhatsApp chat file
CHAT_FILE_PATH = "/Users/Add/Path/Here/_chat.txt"

# --- OPENROUTER AI SETTINGS (PLACEHOLDERS) ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("ADD_API_KEY_HERE", "ADD_API_KEY_HERE")
# Example model, change as needed
OPENROUTER_MODEL = "google/gemma-3-12b-it"

# --- DIRECTORY CONFIGURATION ---
OUTPUT_DIR = "output"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
TEMPLATE_DIR = "templates"

# --- OTHER SETTINGS ---
EXIFTOOL_PATH = "exiftool" # Assumes exiftool is in the system's PATH

# --- SCRIPT GLOBALS ---
error_log_filename = ""
translation_stats = {"count": 0, "tokens_in": 0, "tokens_out": 0}
messages_processed = 0
images_processed = 0

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

def parse_chat_file(file_path):
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


def _process_single_message(message, source_dir):
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


def process_messages(messages, input_file_path):
    """
    Processes all messages in parallel to handle images and translations.
    """
    global images_processed, translation_stats
    
    source_dir = os.path.dirname(input_file_path)
    processed_messages = []
    
    # Use functools.partial to pre-fill the source_dir argument for our task function
    task_function = partial(_process_single_message, source_dir=source_dir)
    
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
    first_date = min(message_dates)
    last_date = max(message_dates)
    
    cal_data = OrderedDict()
    current_date = first_date.replace(day=1)

    while current_date <= last_date:
        month_key = current_date.strftime("%Y-%m")
        month_name = current_date.strftime("%B %Y")
        cal = calendar.monthcalendar(current_date.year, current_date.month)
        
        month_data = {
            "name": month_name,
            "weeks": []
        }
        
        for week in cal:
            week_data = []
            for day in week:
                if day == 0:
                    week_data.append({"day": 0, "has_message": False})
                else:
                    current_day_date = datetime.date(current_date.year, current_date.month, day)
                    week_data.append({
                        "day": day,
                        "has_message": current_day_date in message_dates,
                        "date_str": current_day_date.strftime("%Y-%m-%d")
                    })
            month_data["weeks"].append(week_data)
        
        cal_data[month_key] = month_data

        # Move to the next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
            
    # --- Render and Save HTML ---
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("template.html")
    
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template.render(
            messages_by_date=consolidated_messages_by_date,
            main_author=main_author,
            calendar_data=list(cal_data.values()) # Pass as a list for easier iteration
        ))


def main():
    """Main function to orchestrate the chat export processing."""
    global messages_processed
    
    parser = argparse.ArgumentParser(description="Convert a WhatsApp chat export to a self-contained HTML file.")
    parser.add_argument("input_file", nargs="?", default=CHAT_FILE_PATH,
                        help=f"Path to the WhatsApp chat text file. Defaults to '{CHAT_FILE_PATH}'.")
    args = parser.parse_args()
    
    setup_logging()
    
    print(f"Starting WhatsApp chat export processing for: {args.input_file}")
    
    # 1. Parse the chat file
    parsed_messages = parse_chat_file(args.input_file)
    if parsed_messages is None:
        print("Processing stopped due to errors during parsing.")
        return

    messages_processed = len(parsed_messages)
    print(f"Successfully parsed {messages_processed} messages.")

    # 2. Process images and messages
    processed_messages = process_messages(parsed_messages, args.input_file)
    print(f"Processed {images_processed} image attachments.")

    # 3. Translate messages (handled inside process_messages)
    # 4. Generate HTML
    print("Generating HTML file...")
    generate_html(processed_messages)
    print("HTML generation complete.")

    # --- Final Report ---
    print("\n--- Export Report ---")
    print(f"Total Messages Processed: {messages_processed}")
    print(f"Total Images Processed:   {images_processed}")
    print("--- Translation Stats ---")
    print(f"  Translations: {translation_stats['count']}")
    print(f"  Tokens In:    {translation_stats['tokens_in']}")
    print(f"  Tokens Out:   {translation_stats['tokens_out']}")
    print("-----------------------")
    print(f"HTML file generated at: {os.path.join(OUTPUT_DIR, 'index.html')}")
    # Check if the error log file was created and is not empty
    if os.path.exists(error_log_filename) and os.path.getsize(error_log_filename) > 0:
        print(f"Errors were logged to: {error_log_filename}")
    print("-----------------------")


if __name__ == "__main__":
    main()
