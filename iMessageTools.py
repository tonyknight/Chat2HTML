#!/usr/bin/env python3
"""
iMessage Database Analysis Tools
================================

A consolidated menu-based tool for analyzing iMessage database patterns, attachments, 
and message content. This tool combines multiple analysis scripts into a single 
interface for comprehensive database investigation.

Features:
- Database diagnostics with comprehensive health analysis
- Null message pattern analysis for spam detection
- Database-wide search functionality
- Comprehensive attachment reporting with SHA-256 hashing
- Null message extraction and categorization
- Remaining null message analysis for filter optimization

All tools are read-only and designed for analysis purposes only.

Author: iMessage Database Tool Suite
"""

import os
import sys
import json
import sqlite3
import csv
import hashlib
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

# Apple's timestamp epoch (January 1, 2001, 00:00:00 UTC)
APPLE_EPOCH = datetime(2001, 1, 1)

@dataclass
class NullMessage:
    """Represents a message with null text for analysis"""
    # Basic message info
    rowid: int
    guid: str
    text: Optional[str]
    service: str
    handle_id: Optional[int]
    handle_name: Optional[str]
    date: str
    is_from_me: bool
    is_read: bool
    
    # Message type and classification
    item_type: int
    associated_message_type: int
    associated_message_guid: Optional[str]
    group_action_type: int
    message_action_type: int
    
    # Content and metadata
    balloon_bundle_id: Optional[str]
    expressive_send_style_id: Optional[str]
    group_title: Optional[str]
    
    # Status flags
    is_empty: bool
    is_system_message: bool
    is_service_message: bool
    is_delivered: bool
    is_sent: bool
    is_finished: bool
    
    # Data fields
    has_attributed_body: bool
    attributed_body_length: int
    has_payload_data: bool
    payload_data_length: int
    has_message_summary_info: bool
    message_summary_info_length: int
    
    # Cache and attachment info
    cache_has_attachments: bool
    cache_roomnames: Optional[str]
    
    # Chat info
    chat_id: Optional[int]
    chat_identifier: Optional[str]
    chat_display_name: Optional[str]
    chat_style: Optional[int]

def convert_apple_timestamp(apple_timestamp: Optional[int]) -> Optional[datetime]:
    """Convert Apple timestamp (nanoseconds since 2001-01-01) to datetime"""
    if apple_timestamp is None or apple_timestamp == 0:
        return None
    
    try:
        # Convert nanoseconds to seconds and add to Apple epoch
        timestamp_seconds = apple_timestamp / 1_000_000_000
        return APPLE_EPOCH + timedelta(seconds=timestamp_seconds)
    except (ValueError, OverflowError):
        return None

def apple_timestamp_to_datetime(timestamp: int) -> str:
    """Convert Apple timestamp to ISO format string"""
    if timestamp == 0:
        return "1970-01-01T00:00:00"
    
    # Apple timestamps are nanoseconds since 2001-01-01
    apple_epoch = datetime(2001, 1, 1)
    dt = apple_epoch + timedelta(seconds=timestamp / 1_000_000_000)
    return dt.isoformat()

def calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculate SHA-256 hash of a file"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (FileNotFoundError, PermissionError, OSError) as e:
        return f"ERROR: {str(e)}"

class DatabaseDiagnostics:
    """Comprehensive database diagnostics for iMessage database health analysis"""

    def __init__(self, db_path: str, attachment_root: str = "~/Library/Messages/Attachments"):
        self.db_path = db_path
        self.attachment_root = attachment_root
        self.conn = None

    def run_full_diagnostic(self):
        """Run complete diagnostic suite matching original implementation"""
        try:
            self.conn = sqlite3.connect(self.db_path)

            print("Building cache...")
            self._build_cache()
            print("Cache built!\n")

            print("iMessage Database Diagnostics\n")
            self._run_handle_diagnostics()
            self._run_message_diagnostics()
            self._run_attachment_diagnostics()
            self._run_thread_diagnostics()
            self._run_global_diagnostics()

            print("\nEnvironment Diagnostics\n")
            self._run_environment_diagnostics()

        except Exception as e:
            print(f"❌ Error running diagnostics: {e}")
        finally:
            if self.conn:
                self.conn.close()

    def _build_cache(self):
        """Build diagnostic caches (mimics original cache building steps)"""
        print("  [1/4] Caching chats...")
        print("  [2/4] Caching chatrooms...")
        print("  [3/4] Caching participants...")
        print("  [4/4] Caching tapbacks...")

    def _run_handle_diagnostics(self):
        """Analyze handle/contact duplicates"""
        try:
            query = """
            SELECT COUNT(DISTINCT person_centric_id)
            FROM handle
            WHERE person_centric_id IS NOT NULL
            """
            cursor = self.conn.execute(query)
            duplicate_contacts = cursor.fetchone()[0]

            if duplicate_contacts > 0:
                print("Handle diagnostic data:")
                print(f"    Contacts with more than one ID: {duplicate_contacts}")
        except Exception as e:
            print(f"Handle diagnostics failed: {e}")

    def _run_message_diagnostics(self):
        """Analyze message table health"""
        try:
            # Total messages
            total_query = "SELECT COUNT(*) FROM message"
            total_messages = self.conn.execute(total_query).fetchone()[0]

            # Messages not associated with a chat (orphaned)
            orphaned_query = """
            SELECT COUNT(m.rowid)
            FROM message as m
            LEFT JOIN chat_message_join as c ON m.rowid = c.message_id
            WHERE c.chat_id IS NULL
            """
            orphaned_messages = self.conn.execute(orphaned_query).fetchone()[0]

            # Messages in multiple chats
            multi_chat_query = """
            SELECT COUNT(*)
            FROM (
                SELECT m.rowid
                FROM message as m
                JOIN chat_message_join as c ON m.rowid = c.message_id
                GROUP BY m.rowid
                HAVING COUNT(c.chat_id) > 1
            )
            """
            try:
                multi_chat_messages = self.conn.execute(multi_chat_query).fetchone()[0]
            except:
                multi_chat_messages = 0

            print("Message diagnostic data:")
            print(f"    Total messages: {total_messages}")
            if orphaned_messages > 0:
                print(f"    Messages not associated with a chat: {orphaned_messages}")
            if multi_chat_messages > 0:
                print(f"    Messages belonging to more than one chat: {multi_chat_messages}")

        except Exception as e:
            print(f"Message diagnostics failed: {e}")

    def _run_attachment_diagnostics(self):
        """Analyze attachment table and file system"""
        try:
            # Get attachment data
            query = """
            SELECT COUNT(*) as total,
                   SUM(COALESCE(total_bytes, 0)) as total_bytes,
                   SUM(CASE WHEN filename IS NULL THEN 1 ELSE 0 END) as null_paths
            FROM attachment
            """
            cursor = self.conn.execute(query)
            row = cursor.fetchone()
            total_attachments = row[0]
            total_bytes = row[1] or 0
            null_attachments = row[2]

            if total_attachments == 0:
                return

            # Check files on disk
            size_on_disk = 0
            missing_files = 0

            file_query = "SELECT filename FROM attachment WHERE filename IS NOT NULL"
            for (filename,) in self.conn.execute(file_query):
                file_path = self._resolve_attachment_path(filename)
                try:
                    if os.path.exists(file_path):
                        size_on_disk += os.path.getsize(file_path)
                    else:
                        missing_files += 1
                except (OSError, PermissionError):
                    missing_files += 1

            print("Attachment diagnostic data:")
            print(f"    Total attachments: {total_attachments}")
            print(f"        Data referenced in table: {self._format_file_size(total_bytes)}")
            print(f"        Data present on disk: {self._format_file_size(size_on_disk)}")

            total_missing = missing_files + null_attachments
            if total_missing > 0:
                percentage = (total_missing / total_attachments) * 100
                print(f"    Missing files: {total_missing} ({percentage:.0f}%)")
                print(f"        No path provided: {null_attachments}")
                print(f"        No file located: {missing_files}")

        except Exception as e:
            print(f"Attachment diagnostics failed: {e}")

    def _run_thread_diagnostics(self):
        """Analyze chat/thread health"""
        try:
            # Chats with no handles
            query = """
            SELECT COUNT(DISTINCT c.rowid)
            FROM chat c
            LEFT JOIN chat_handle_join ch ON c.rowid = ch.chat_id
            WHERE ch.handle_id IS NULL
            """
            chats_no_handles = self.conn.execute(query).fetchone()[0]

            if chats_no_handles > 0:
                print("Thread diagnostic data:")
                print(f"    Chats with no handles: {chats_no_handles}")

        except Exception as e:
            print(f"Thread diagnostics failed: {e}")

    def _run_global_diagnostics(self):
        """Analyze global database health"""
        try:
            print("Global diagnostic data:")

            # Database size
            db_size = os.path.getsize(self.db_path)
            print(f"    Total database size: {self._format_file_size(db_size)}")

            # Duplicated contacts (simplified version)
            try:
                duplicate_contacts_query = """
                SELECT COUNT(*) - COUNT(DISTINCT id)
                FROM handle
                """
                duplicated_contacts = self.conn.execute(duplicate_contacts_query).fetchone()[0]

                if duplicated_contacts > 0:
                    print(f"    Duplicated contacts: {duplicated_contacts}")
            except:
                pass

            # Duplicated chats (simplified version)
            try:
                duplicate_chats_query = """
                SELECT COUNT(*) - COUNT(DISTINCT chat_identifier)
                FROM chat
                WHERE chat_identifier IS NOT NULL
                """
                duplicated_chats = self.conn.execute(duplicate_chats_query).fetchone()[0]

                if duplicated_chats > 0:
                    print(f"    Duplicated chats: {duplicated_chats}")
            except:
                pass

        except Exception as e:
            print(f"Global diagnostics failed: {e}")

    def _run_environment_diagnostics(self):
        """Check for available conversion tools"""
        print("Detected converters:")

        # Image converters
        image_converter = None
        if shutil.which("sips"):
            image_converter = "sips"
        elif shutil.which("magick"):
            image_converter = "magick"

        print(f"    Image converter: {image_converter or 'None'}")

        # Audio converters
        audio_converter = None
        if shutil.which("afconvert"):
            audio_converter = "afconvert"
        elif shutil.which("ffmpeg"):
            audio_converter = "ffmpeg"

        print(f"    Audio converter: {audio_converter or 'None'}")

        # Video converters
        video_converter = None
        if shutil.which("ffmpeg"):
            video_converter = "ffmpeg"

        print(f"    Video converter: {video_converter or 'None'}")

    def _resolve_attachment_path(self, filename: str) -> str:
        """Resolve attachment path"""
        path_str = filename

        # Apply custom attachment root if provided
        default_root = "~/Library/Messages/Attachments"
        if self.attachment_root and default_root in path_str:
            path_str = path_str.replace(default_root, self.attachment_root)

        # Expand tilde to home directory
        if path_str.startswith('~'):
            return os.path.expanduser(path_str)

        return path_str

    def _format_file_size(self, bytes_size: int) -> str:
        """Format file size in human readable format"""
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(bytes_size)
        unit_index = 0

        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1

        return f"{size:.2f} {units[unit_index]}"

def show_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("🔍 iMessage Database Analysis Tools")
    print("="*60)
    print("1. Database Diagnostics (Comprehensive Health Analysis)")
    print("2. Analyze Null Message Patterns (Spam Detection)")
    print("3. Database Search Tool (Find Text in All Tables)")
    print("4. Generate Attachment Report (CSV with SHA-256)")
    print("5. Extract Null Messages (Full Database Export)")
    print("6. Analyze Remaining Null Messages (Filter Optimization)")
    print("7. Exit")
    print("="*60)

def get_user_input(prompt: str, required: bool = True) -> str:
    """Get user input with validation"""
    while True:
        value = input(prompt).strip()
        if value or not required:
            return value
        print("❌ This field is required. Please enter a value.")

def validate_file_exists(file_path: str) -> bool:
    """Validate that a file exists"""
    expanded_path = os.path.expanduser(file_path)
    if os.path.exists(expanded_path):
        return True
    print(f"❌ File not found: {expanded_path}")
    return False

def get_default_db_path() -> str:
    """Get the default database path"""
    return "~/Library/Messages/chat.db"

def get_default_attachment_path() -> str:
    """Get the default attachment path"""
    return "~/Library/Messages/Attachments"

# Tool 1: Database Diagnostics
def run_database_diagnostics():
    """Run comprehensive database diagnostics"""
    print("\n🔍 DATABASE DIAGNOSTICS")
    print("This tool runs comprehensive health analysis on the iMessage database.")

    db_path = get_user_input("Enter path to iMessage database [~/Library/Messages/chat.db]: ", required=False)
    if not db_path:
        db_path = "~/Library/Messages/chat.db"

    # Expand tilde
    db_path = os.path.expanduser(db_path)

    if not validate_file_exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return

    attachment_root = get_user_input("Enter attachment root directory [~/Library/Messages/Attachments]: ", required=False)
    if not attachment_root:
        attachment_root = "~/Library/Messages/Attachments"

    print(f"\n📊 Running diagnostics on: {db_path}")
    print(f"📎 Attachment root: {attachment_root}")

    try:
        diagnostics = DatabaseDiagnostics(db_path, attachment_root)
        diagnostics.run_full_diagnostic()
        print("\n✅ Database diagnostics completed successfully!")

    except Exception as e:
        print(f"❌ Error running diagnostics: {e}")

# Tool 2: Analyze Null Message Patterns
def analyze_null_patterns():
    """Analyze null text messages to identify spam patterns"""
    print("\n🔍 NULL MESSAGE PATTERN ANALYSIS")
    print("This tool analyzes null messages from a JSON export to identify spam patterns.")
    
    null_messages_file = get_user_input("Enter path to null messages JSON file: ")
    if not validate_file_exists(null_messages_file):
        return
    
    print(f"Loading null messages from: {null_messages_file}")
    
    try:
        with open(null_messages_file, 'r') as f:
            data = json.load(f)
        
        messages = data['messages']
        total_count = len(messages)
        
        print(f"Analyzing {total_count:,} null text messages...")
        print(f"=" * 60)
        
        # Basic categorization
        categories = {
            'system_messages': [],
            'unread_messages': [],
            'sms_messages': [],
            'rcs_messages': [],
            'item_type_4': [],  # Often spam/deleted
            'has_attributed_body': [],
            'potential_spam': []
        }
        
        # Pattern analysis
        handle_frequency = Counter()
        chat_frequency = Counter()
        service_stats = Counter()
        item_type_stats = Counter()
        read_status_by_service = defaultdict(lambda: {'read': 0, 'unread': 0})
        
        for msg in messages:
            # Basic stats
            service = msg['service']
            item_type = msg['item_type']
            is_read = msg['is_read']
            handle_name = msg['handle_name']
            chat_id = msg['chat_identifier']
            
            service_stats[service] += 1
            item_type_stats[item_type] += 1
            
            if is_read:
                read_status_by_service[service]['read'] += 1
            else:
                read_status_by_service[service]['unread'] += 1
            
            if handle_name:
                handle_frequency[handle_name] += 1
            if chat_id:
                chat_frequency[chat_id] += 1
            
            # Categorize messages
            if msg['is_system_message'] or msg['is_service_message']:
                categories['system_messages'].append(msg)
            elif not is_read:
                categories['unread_messages'].append(msg)
            elif service == 'SMS':
                categories['sms_messages'].append(msg)
            elif service == 'RCS':
                categories['rcs_messages'].append(msg)
            elif item_type == 4:
                categories['item_type_4'].append(msg)
            elif msg['has_attributed_body']:
                categories['has_attributed_body'].append(msg)
            else:
                categories['potential_spam'].append(msg)
        
        # Print analysis results
        print(f"SERVICE BREAKDOWN:")
        for service, count in service_stats.most_common():
            percentage = (count / total_count) * 100
            print(f"  {service}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nITEM TYPE BREAKDOWN:")
        for item_type, count in item_type_stats.most_common():
            percentage = (count / total_count) * 100
            print(f"  Type {item_type}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nREAD STATUS BY SERVICE:")
        for service in service_stats.keys():
            read_count = read_status_by_service[service]['read']
            unread_count = read_status_by_service[service]['unread']
            total_service = read_count + unread_count
            unread_pct = (unread_count / total_service) * 100 if total_service > 0 else 0
            print(f"  {service}: {unread_count:,} unread / {total_service:,} total ({unread_pct:.1f}% unread)")
        
        print(f"\nCATEGORY BREAKDOWN:")
        for category, msgs in categories.items():
            count = len(msgs)
            percentage = (count / total_count) * 100
            print(f"  {category.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
        
        # Export sample messages for manual review
        sample_output = {
            'unread_sample': categories['unread_messages'][:50],
            'item_type_4_sample': categories['item_type_4'][:50],
            'potential_spam_sample': categories['potential_spam'][:50],
            'high_frequency_senders': dict(handle_frequency.most_common(50))
        }
        
        sample_file = null_messages_file.replace('.json', '_samples.json')
        with open(sample_file, 'w') as f:
            json.dump(sample_output, f, indent=2)
        
        print(f"\n✅ Sample messages exported to: {sample_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Tool 2: Database Search
def database_search():
    """Generate search query for finding text in all database tables"""
    print("\n🔍 DATABASE SEARCH TOOL")
    print("This tool generates SQL queries to search for text across all database tables.")
    
    db_path = get_user_input(f"Enter database path (default: {get_default_db_path()}): ", required=False)
    if not db_path:
        db_path = get_default_db_path()
    
    db_path = os.path.expanduser(db_path)
    if not validate_file_exists(db_path):
        return
    
    search_string = get_user_input("Enter search string: ")
    
    print(f"\nGenerating search query for: '{search_string}'")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        union_all_queries = []

        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            
            # Skip internal sqlite tables
            if table_name.startswith('sqlite_'):
                continue

            # Get column names for the current table
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            columns = cursor.fetchall()

            for column_info in columns:
                column_name = column_info[1]
                
                # Construct the SELECT statement for each column
                query_part = f"SELECT '{table_name}' AS table_name, '{column_name}' AS column_name, \"{column_name}\" FROM \"{table_name}\" WHERE \"{column_name}\" LIKE '%{search_string}%'"
                union_all_queries.append(query_part)

        conn.close()

        if not union_all_queries:
            print("❌ No user tables found in the database.")
            return

        # Combine all individual queries with UNION ALL
        full_query = " UNION ALL ".join(union_all_queries)
        
        print("\n" + "="*80)
        print("📋 GENERATED SQL QUERY:")
        print("="*80)
        print("-- Copy and paste the following query into your SQLite tool:")
        print(full_query + ";")
        print("="*80)
        
        # Optionally save to file
        save_query = input("\nSave query to file? (y/N): ").strip().lower()
        if save_query == 'y':
            query_file = f"search_query_{search_string.replace(' ', '_')}.sql"
            with open(query_file, 'w') as f:
                f.write(f"-- Search query for: {search_string}\n")
                f.write(f"-- Generated: {datetime.now().isoformat()}\n\n")
                f.write(full_query + ";")
            print(f"✅ Query saved to: {query_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Tool 3: Generate Attachment Report
def generate_attachment_report():
    """Generate comprehensive attachment report with SHA-256 hashes"""
    print("\n📎 ATTACHMENT REPORT GENERATOR")
    print("This tool creates a comprehensive CSV report of all attachments with file analysis.")

    db_path = get_user_input(f"Enter database path (default: {get_default_db_path()}): ", required=False)
    if not db_path:
        db_path = get_default_db_path()

    db_path = os.path.expanduser(db_path)
    if not validate_file_exists(db_path):
        return

    attachment_root = get_user_input(f"Enter attachment root (default: {get_default_attachment_path()}): ", required=False)
    if not attachment_root:
        attachment_root = get_default_attachment_path()

    attachment_root = os.path.expanduser(attachment_root)
    if not os.path.exists(attachment_root):
        print(f"❌ Attachment directory not found: {attachment_root}")
        return

    output_csv = get_user_input("Enter output CSV filename (default: iMessage_attachment_report.csv): ", required=False)
    if not output_csv:
        output_csv = "iMessage_attachment_report.csv"

    print(f"🔍 ANALYZING iMessage ATTACHMENTS")
    print(f"Database: {db_path}")
    print(f"Attachment root: {attachment_root}")
    print(f"Output CSV: {output_csv}")
    print(f"Note: Skipping .pluginPayloadAttachment files (Apple proprietary blobs)")
    print("=" * 80)

    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Simplified query for the consolidated tool
        query = """
        SELECT
            a.ROWID as attachment_rowid,
            a.guid as attachment_guid,
            a.created_date as attachment_created_date,
            a.filename as attachment_filename,
            a.mime_type as attachment_mime_type,
            a.total_bytes as attachment_total_bytes,
            m.ROWID as message_rowid,
            m.date as message_date,
            m.service as message_service
        FROM attachment a
        LEFT JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
        LEFT JOIN message m ON maj.message_id = m.ROWID
        ORDER BY a.ROWID
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"📊 Found {len(rows)} attachment records")

        def find_attachment_file(filename: str, attachment_root: str) -> Dict[str, Any]:
            """Find attachment file and return basic info"""
            if not filename:
                return {'found': False, 'full_path': None, 'file_size': None, 'sha256_hash': 'NO_FILENAME'}

            # Try direct path first
            full_path = os.path.join(attachment_root, filename)
            if os.path.exists(full_path):
                try:
                    stat_info = os.stat(full_path)
                    return {
                        'found': True,
                        'full_path': full_path,
                        'file_size': stat_info.st_size,
                        'sha256_hash': calculate_file_hash(full_path)
                    }
                except OSError:
                    pass

            return {'found': False, 'full_path': None, 'file_size': None, 'sha256_hash': 'FILE_NOT_FOUND'}

        # Write simplified CSV
        csv_headers = ['attachment_rowid', 'attachment_guid', 'attachment_filename', 'attachment_mime_type',
                      'attachment_total_bytes', 'file_found', 'file_full_path', 'file_size_bytes', 'file_sha256_hash']

        print(f"📝 Writing CSV report...")

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)

            processed_count = 0
            skipped_count = 0
            found_files = 0
            total_size = 0

            for row in rows:
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count}/{len(rows)} attachments...")

                # Skip Apple's proprietary plugin payload attachments
                filename = row['attachment_filename']
                if filename and filename.lower().endswith('.pluginpayloadattachment'):
                    skipped_count += 1
                    continue

                # Analyze file on disk
                file_info = find_attachment_file(filename, attachment_root)

                if file_info['found']:
                    found_files += 1
                    if file_info['file_size']:
                        total_size += file_info['file_size']

                # Write CSV row
                csv_row = [
                    row['attachment_rowid'], row['attachment_guid'], row['attachment_filename'],
                    row['attachment_mime_type'], row['attachment_total_bytes'],
                    file_info['found'], file_info['full_path'], file_info['file_size'], file_info['sha256_hash']
                ]

                writer.writerow(csv_row)

        conn.close()

        # Print summary statistics
        print(f"\n📊 ATTACHMENT ANALYSIS COMPLETE")
        print(f"=" * 50)
        print(f"Total attachment records: {len(rows):,}")
        print(f"Plugin payload attachments skipped: {skipped_count:,}")
        print(f"Attachments analyzed: {len(rows) - skipped_count:,}")
        print(f"Files found on disk: {found_files:,}")
        print(f"Files missing: {(len(rows) - skipped_count) - found_files:,}")
        print(f"Success rate: {(found_files/(len(rows) - skipped_count)*100):.1f}%" if (len(rows) - skipped_count) > 0 else "N/A")
        print(f"Total size of found files: {total_size/1024/1024:.1f} MB")
        print(f"✅ CSV report saved to: {output_csv}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Tool 4: Extract Null Messages
def extract_null_messages():
    """Extract all messages with null text from the database"""
    print("\n📤 NULL MESSAGE EXTRACTOR")
    print("This tool extracts all messages with null text for comprehensive analysis.")

    db_path = get_user_input(f"Enter database path (default: {get_default_db_path()}): ", required=False)
    if not db_path:
        db_path = get_default_db_path()

    db_path = os.path.expanduser(db_path)
    if not validate_file_exists(db_path):
        return

    output_path = get_user_input("Enter output JSON filename (default: null_messages_export.json): ", required=False)
    if not output_path:
        output_path = "null_messages_export.json"

    print(f"Extracting null text messages from: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Comprehensive query to get all null text messages with full metadata
        query = """
        SELECT
            m.rowid, m.guid, m.text, m.service, m.handle_id, m.date, m.is_from_me, m.is_read,
            m.item_type, m.associated_message_type, m.associated_message_guid,
            m.group_action_type, m.message_action_type,
            m.balloon_bundle_id, m.expressive_send_style_id, m.group_title,
            m.is_empty, m.is_system_message, m.is_service_message,
            m.is_delivered, m.is_sent, m.is_finished,
            m.attributedBody, m.payload_data, m.message_summary_info,
            m.cache_has_attachments, m.cache_roomnames,
            h.id as handle_name,
            c.rowid as chat_id, c.chat_identifier, c.display_name as chat_display_name, c.style as chat_style
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.rowid
        LEFT JOIN chat_message_join cmj ON m.rowid = cmj.message_id
        LEFT JOIN chat c ON cmj.chat_id = c.rowid
        WHERE m.text IS NULL
        ORDER BY m.date DESC
        """

        cursor = conn.execute(query)
        null_messages = []

        for row in cursor:
            # Process data fields
            attributed_body = row['attributedBody']
            payload_data = row['payload_data']
            message_summary_info = row['message_summary_info']

            null_message = NullMessage(
                # Basic info
                rowid=row['rowid'],
                guid=row['guid'],
                text=row['text'],
                service=row['service'] or 'Unknown',
                handle_id=row['handle_id'],
                handle_name=row['handle_name'],
                date=apple_timestamp_to_datetime(row['date'] or 0),
                is_from_me=bool(row['is_from_me']),
                is_read=bool(row['is_read']),

                # Message type
                item_type=row['item_type'] or 0,
                associated_message_type=row['associated_message_type'] or 0,
                associated_message_guid=row['associated_message_guid'],
                group_action_type=row['group_action_type'] or 0,
                message_action_type=row['message_action_type'] or 0,

                # Content
                balloon_bundle_id=row['balloon_bundle_id'],
                expressive_send_style_id=row['expressive_send_style_id'],
                group_title=row['group_title'],

                # Status
                is_empty=bool(row['is_empty']),
                is_system_message=bool(row['is_system_message']),
                is_service_message=bool(row['is_service_message']),
                is_delivered=bool(row['is_delivered']),
                is_sent=bool(row['is_sent']),
                is_finished=bool(row['is_finished']),

                # Data fields
                has_attributed_body=attributed_body is not None,
                attributed_body_length=len(attributed_body) if attributed_body else 0,
                has_payload_data=payload_data is not None,
                payload_data_length=len(payload_data) if payload_data else 0,
                has_message_summary_info=message_summary_info is not None,
                message_summary_info_length=len(message_summary_info) if message_summary_info else 0,

                # Cache
                cache_has_attachments=bool(row['cache_has_attachments']),
                cache_roomnames=row['cache_roomnames'],

                # Chat
                chat_id=row['chat_id'],
                chat_identifier=row['chat_identifier'],
                chat_display_name=row['chat_display_name'],
                chat_style=row['chat_style']
            )

            null_messages.append(null_message)

        conn.close()

        print(f"Found {len(null_messages)} messages with null text")

        # Convert to JSON-serializable format
        messages_data = [asdict(msg) for msg in null_messages]

        # Add summary statistics
        output_data = {
            "summary": {
                "total_null_messages": len(null_messages),
                "extraction_date": datetime.now().isoformat(),
                "database_path": db_path
            },
            "messages": messages_data
        }

        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Exported to: {output_path}")

        # Print some basic statistics
        print(f"\n=== BASIC STATISTICS ===")

        # Group by item_type
        item_types = {}
        services = {}
        is_read_stats = {"read": 0, "unread": 0}
        is_from_me_stats = {"from_me": 0, "from_others": 0}

        for msg in null_messages:
            # Item type stats
            item_type = msg.item_type
            item_types[item_type] = item_types.get(item_type, 0) + 1

            # Service stats
            service = msg.service
            services[service] = services.get(service, 0) + 1

            # Read status
            if msg.is_read:
                is_read_stats["read"] += 1
            else:
                is_read_stats["unread"] += 1

            # Sender stats
            if msg.is_from_me:
                is_from_me_stats["from_me"] += 1
            else:
                is_from_me_stats["from_others"] += 1

        print(f"Item types: {dict(sorted(item_types.items()))}")
        print(f"Services: {dict(sorted(services.items()))}")
        print(f"Read status: {is_read_stats}")
        print(f"Sender: {is_from_me_stats}")

    except Exception as e:
        print(f"❌ Error: {e}")

# Tool 5: Analyze Remaining Null Messages
def analyze_remaining_null_messages():
    """Extract and analyze the remaining null text messages from an export"""
    print("\n🔍 REMAINING NULL MESSAGE ANALYZER")
    print("This tool analyzes null messages from a JSON export to optimize filtering.")

    export_file = get_user_input("Enter path to messages JSON export file: ")
    if not validate_file_exists(export_file):
        return

    output_file = get_user_input("Enter output analysis filename (default: remaining_null_analysis.json): ", required=False)
    if not output_file:
        output_file = "remaining_null_analysis.json"

    print(f"Loading messages from: {export_file}")

    try:
        with open(export_file, 'r') as f:
            messages = json.load(f)

        # Find messages with null text
        null_messages = [msg for msg in messages if msg.get('text') is None]

        print(f"Found {len(null_messages)} messages with null text out of {len(messages)} total")

        if not null_messages:
            print("No null text messages found!")
            return

        # Detailed pattern analysis
        patterns = {
            'service_patterns': Counter(),
            'sender_patterns': Counter(),
            'read_status': Counter(),
            'service_read_combinations': Counter(),
        }

        # Categorize messages for suppression analysis
        suppression_categories = {
            'rcs_unread_system': [],
            'rcs_group_system': [],
            'unknown_sender_unread': [],
            'attachment_only': [],
            'other_patterns': []
        }

        for msg in null_messages:
            service = msg.get('service', 'Unknown')
            sender = msg.get('sender', 'Unknown')
            chat = msg.get('chat', 'Unknown')
            is_read = msg.get('is_read', False)
            is_from_me = msg.get('is_from_me', False)
            has_attachments = len(msg.get('attachments', [])) > 0
            has_tapbacks = len(msg.get('tapbacks', [])) > 0

            # Pattern analysis
            patterns['service_patterns'][service] += 1
            patterns['sender_patterns'][sender] += 1
            patterns['read_status'][is_read] += 1
            patterns['service_read_combinations'][f"{service}_{is_read}"] += 1

            # Categorize for suppression analysis
            if service == 'RCS' and not is_read:
                if sender == 'Unknown':
                    suppression_categories['rcs_unread_system'].append(msg)
                else:
                    suppression_categories['rcs_group_system'].append(msg)
            elif sender == 'Unknown' and not is_read:
                suppression_categories['unknown_sender_unread'].append(msg)
            elif has_attachments and not has_tapbacks:
                suppression_categories['attachment_only'].append(msg)
            else:
                suppression_categories['other_patterns'].append(msg)

        # Print analysis
        print(f"\n{'='*60}")
        print(f"PATTERN ANALYSIS")
        print(f"{'='*60}")

        print(f"\nSERVICE BREAKDOWN:")
        for service, count in patterns['service_patterns'].most_common():
            percentage = (count / len(null_messages)) * 100
            print(f"  {service}: {count} ({percentage:.1f}%)")

        print(f"\nREAD STATUS:")
        for status, count in patterns['read_status'].most_common():
            percentage = (count / len(null_messages)) * 100
            print(f"  {'Read' if status else 'Unread'}: {count} ({percentage:.1f}%)")

        print(f"\nTOP SENDERS:")
        for sender, count in patterns['sender_patterns'].most_common(10):
            percentage = (count / len(null_messages)) * 100
            print(f"  {sender}: {count} ({percentage:.1f}%)")

        print(f"\n{'='*60}")
        print(f"SUPPRESSION ANALYSIS")
        print(f"{'='*60}")

        total_suppressible = 0
        for category, msgs in suppression_categories.items():
            count = len(msgs)
            percentage = (count / len(null_messages)) * 100
            total_suppressible += count
            print(f"{category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

        print(f"\nTotal potentially suppressible: {total_suppressible} ({(total_suppressible/len(null_messages))*100:.1f}%)")
        print(f"Remaining for manual review: {len(null_messages) - total_suppressible}")

        # Create output data
        output_data = {
            "summary": {
                "total_null_messages": len(null_messages),
                "analysis_date": datetime.now().isoformat(),
                "source_file": export_file,
                "patterns": {
                    "services": dict(patterns['service_patterns']),
                    "read_status": dict(patterns['read_status']),
                    "top_senders": dict(patterns['sender_patterns'].most_common(20)),
                },
                "suppression_categories": {
                    category: len(msgs) for category, msgs in suppression_categories.items()
                }
            },
            "messages_by_category": {
                category: msgs for category, msgs in suppression_categories.items()
            },
            "all_null_messages": null_messages
        }

        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Detailed analysis exported to: {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main menu loop"""
    print("🔍 iMessage Database Analysis Tools")
    print("Consolidated analysis suite for iMessage database investigation")

    while True:
        show_menu()

        try:
            choice = input("\nSelect an option (1-7): ").strip()

            if choice == '1':
                run_database_diagnostics()
            elif choice == '2':
                analyze_null_patterns()
            elif choice == '3':
                database_search()
            elif choice == '4':
                generate_attachment_report()
            elif choice == '5':
                extract_null_messages()
            elif choice == '6':
                analyze_remaining_null_messages()
            elif choice == '7':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-7.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        # Pause before returning to menu
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
