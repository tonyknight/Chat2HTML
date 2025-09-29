import re
import datetime
from src.parsers.base_parser import BaseParser

class WhatsAppParser(BaseParser):
    """Parses exported WhatsApp chat .txt files."""

    def parse(self, file_path):
        """
        Parses the WhatsApp chat txt file.

        Args:
            file_path (str): The path to the chat file.

        Returns:
            list: A list of standardized message dictionaries.
        """
        message_start_regex = re.compile(r"^\u200e?\[(\d{4}-\d{2}-\d{2}, \d{2}:\d{2}:\d{2})\] ([^:]+): \u200e?(.*)")
        
        messages = []
        current_message = None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    match = message_start_regex.match(line)
                    if match:
                        if current_message:
                            current_message["message"] = current_message["message"].strip()
                            if current_message["message"] != "image omitted":
                                messages.append(current_message)

                        timestamp_str, author, message_content = match.groups()
                        
                        current_message = {
                            "timestamp": datetime.datetime.strptime(timestamp_str, "%Y-%m-%d, %H:%M:%S"),
                            "author": author.strip(),
                            "message": message_content.strip(),
                            "type": "text",
                        }
                    elif current_message:
                        current_message["message"] += "\n" + line.strip()

                if current_message:
                    current_message["message"] = current_message["message"].strip()
                    if current_message["message"] != "image omitted":
                        messages.append(current_message)
        except FileNotFoundError:
            print(f"Error: Input chat file not found at '{file_path}'.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while parsing the chat file: {e}")
            return []

        return messages
