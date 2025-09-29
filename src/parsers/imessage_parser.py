import os
import re
import datetime
import json
import shutil
import subprocess
from bs4 import BeautifulSoup
from tqdm import tqdm
import vobject
import concurrent.futures
from functools import partial

from src.parsers.base_parser import BaseParser

# Constants (could be moved to a config file later)
TAPBACK_EMOJI_MAP = {
    "Liked": "👍", "Loved": "❤️", "Laughed at": "😂",
    "Emphasized": "❗️", "Disliked": "👎", "Questioned": "❓"
}
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.heic', '.tif', '.tiff']
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v', '.avi']
EXIFTOOL_PATH = "exiftool"

class IMessageParser(BaseParser):
    """Handles the complex, multi-step parsing of iMessage exports."""

    def __init__(self):
        self.owner_name = None
        self.contact_lookup = {}
        self.root_dir = None

    def parse(self, root_directory):
        """
        Orchestrates the iMessage workflow. This is different from other parsers
        as it involves user interaction and multiple steps. It returns
        a list of all messages parsed, ready for the database.
        """
        self.root_dir = root_directory
        print("\n--- Starting iMessage Processing Workflow ---")
        
        # Step 1: Rename & Organize (user-facing pre-processing)
        self._rename_files_by_author(self.root_dir)
        self._organize_files_into_folders(self.root_dir)
        
        # Step 2: Get user info needed for parsing
        self.owner_name = input("Enter your full name exactly as it appears in iMessage exports: ").strip()
        vcard_path = input("Enter path to Vcard .vcf file (optional): ").strip()
        if vcard_path:
            self._load_vcard(vcard_path)

        # Step 3: Parse all HTML files to generate message objects
        all_messages = self._convert_all_html_to_messages()
        
        print(f"\nSuccessfully parsed a total of {len(all_messages)} iMessage messages.")
        return all_messages

    def _load_vcard(self, vcard_path):
        # Vcard loading logic from the old script
        pass

    def _rename_files_by_author(self, directory_path):
        # Rename logic from the old script
        pass

    def _organize_files_into_folders(self, directory_path):
        # Organize logic from the old script
        pass

    def _convert_all_html_to_messages(self):
        author_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) and d != 'attachments']
        all_messages = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._process_author_folder, author_folders), total=len(author_folders), desc="Parsing Author HTML"))
            for msg_list in results:
                all_messages.extend(msg_list)
        
        return all_messages

    def _process_author_folder(self, author_name):
        # This is the worker function that processes a single author's folder
        author_dir_path = os.path.join(self.root_dir, author_name)
        messages_for_author = []
        html_files = [f for f in os.listdir(author_dir_path) if f.endswith('.html')]
        
        for html_file in html_files:
            file_path = os.path.join(author_dir_path, html_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            message_divs = soup.find_all('div', class_='message')
            for div in message_divs:
                parsed_msgs = self._parse_message_div(div, author_dir_path)
                if parsed_msgs:
                    messages_for_author.extend(parsed_msgs)

        # We return the raw list of messages here. The database manager will handle sorting and owner assignment.
        return messages_for_author

    def _parse_message_div(self, msg_div, author_dir):
        # This is the detailed message parsing logic from the old script
        # It needs to be adapted to be a method of this class.
        return [] # Placeholder

    def _handle_imessage_attachment(self, attachment_div, msg_timestamp, author_dir):
        # This is the attachment handling logic from the old script
        # It needs to be adapted to be a method of this class.
        return None # Placeholder
