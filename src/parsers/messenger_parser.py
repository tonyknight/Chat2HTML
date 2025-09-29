import json
import datetime
from src.parsers.base_parser import BaseParser

def _fix_facebook_encoding(text):
    """Fixes the double-encoded strings from Facebook's JSON export."""
    if text is None:
        return ""
    return text.encode('latin-1').decode('utf-8')

class MessengerParser(BaseParser):
    """Parses exported Facebook Messenger chat .json files."""

    def parse(self, file_path):
        """
        Parses a Facebook Messenger chat JSON file.

        Args:
            file_path (str): The path to the chat file.

        Returns:
            list: A list of standardized message dictionaries.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Input chat file not found at '{file_path}'.")
            return []
        except json.JSONDecodeError:
            print(f"Error: The file at '{file_path}' is not a valid JSON file.")
            return []
        
        messages = []
        for msg_data in reversed(data.get("messages", [])):
            timestamp_ms = msg_data.get("timestamp_ms")
            if not timestamp_ms:
                continue

            message_content = ""
            if "content" in msg_data:
                message_content += _fix_facebook_encoding(msg_data["content"])
            
            if "share" in msg_data and "link" in msg_data["share"]:
                link = msg_data["share"]["link"]
                share_text = _fix_facebook_encoding(msg_data["share"].get("share_text", ""))
                message_content += f' [Shared Link: {link}]'
                if share_text:
                    message_content += f" Preview: {share_text}"

            message = {
                "timestamp": datetime.datetime.fromtimestamp(timestamp_ms / 1000),
                "author": _fix_facebook_encoding(msg_data.get("sender_name", "Unknown")),
                "message": message_content.strip(),
                "type": "text",
            }

            if "photos" in msg_data and msg_data["photos"]:
                message["type"] = "image"
                message["attachment_path"] = msg_data["photos"][0].get("uri")
            elif "videos" in msg_data and msg_data["videos"]:
                message["type"] = "video"
                message["attachment_path"] = msg_data["videos"][0].get("uri")
            
            messages.append(message)

        return messages
