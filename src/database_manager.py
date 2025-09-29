import sqlite3
import os
import json

class DatabaseManager:
    """Handles all interactions with the SQLite database."""
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def create_tables(self):
        """Creates the necessary tables if they don't exist."""
        cursor = self.conn.cursor()
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
            type TEXT,
            service TEXT
        )''')

    def insert_messages(self, messages, service_name, author_name=None, owner_name=None):
        """Inserts a list of standardized message dictionaries into the database."""
        cursor = self.conn.cursor()
        for msg in messages:
            # Logic for iMessage where author/owner are passed in
            if service_name == "iMessage":
                 # Determine message type based on the owner
                msg_type = 'sent' if msg.get('sender') == owner_name else 'received'
                cursor.execute('''
                INSERT INTO messages (timestamp, message_author, message_sender, message_owner, sender_contact_info, body, attachment_path, attachment_author_dir, tapbacks, shared_link, type, service)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    msg.get('timestamp'), author_name, msg.get('sender'), owner_name,
                    msg.get('sender_contact_info'), msg.get('body'), msg.get('attachment'),
                    msg.get('attachment_author_dir'), json.dumps(msg.get('tapbacks')),
                    json.dumps(msg.get('shared_link')), msg_type, service_name
                ))
            else:
                # Simplified logic for WhatsApp/Messenger for now
                cursor.execute('''
                INSERT INTO messages (timestamp, message_author, message_sender, body, type, service)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    msg.get('timestamp').isoformat(), msg.get('author'), msg.get('author'),
                    msg.get('message'), msg.get('type'), service_name
                ))

    def get_all_messages_by_author(self, author):
        """Fetches all messages for a specific author, ordered by time."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM messages WHERE message_author = ? ORDER BY timestamp", (author,))
        return cursor.fetchall()

    def get_all_authors(self):
        """Returns a list of all unique authors in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT message_author FROM messages")
        return [row['message_author'] for row in cursor.fetchall()]

    def get_all_messages(self):
        """Fetches all messages from the database, ordered by time."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM messages ORDER BY timestamp")
        return cursor.fetchall()
