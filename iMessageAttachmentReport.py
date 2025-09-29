#!/usr/bin/env python3
"""
iMessage Attachment Report Generator

This script analyzes all attachments in the iMessage database and creates a comprehensive
CSV report containing attachment metadata, file system information, and SHA-256 hashes.

Usage:
    python3 iMessageAttachmentReport.py [--db-path PATH] [--attachment-root PATH] [--output CSV_FILE]

The script examines:
- All records from the 'attachment' table
- All records from the 'message_attachment_join' table  
- File system presence and SHA-256 hashes for each attachment
- Comprehensive metadata for analysis and planning

Author: iMessage Database Tool Suite
"""

import sqlite3
import os
import csv
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import sys

# Apple's timestamp epoch (January 1, 2001, 00:00:00 UTC)
APPLE_EPOCH = datetime(2001, 1, 1)

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

def find_attachment_file(filename: str, attachment_root: str) -> Optional[Dict[str, Any]]:
    """
    Find attachment file in the filesystem and return file information
    
    Args:
        filename: The filename from the database
        attachment_root: Root directory to search for attachments
        
    Returns:
        Dictionary with file information or None if not found
    """
    if not filename:
        return None
    
    # Try direct path first
    full_path = os.path.join(attachment_root, filename)
    if os.path.exists(full_path):
        try:
            stat_info = os.stat(full_path)
            return {
                'found': True,
                'full_path': full_path,
                'file_size': stat_info.st_size,
                'modified_time': datetime.fromtimestamp(stat_info.st_mtime),
                'sha256_hash': calculate_file_hash(full_path)
            }
        except OSError:
            pass
    
    # If direct path doesn't work, search recursively (for complex folder structures)
    try:
        basename = os.path.basename(filename)
        for root, dirs, files in os.walk(attachment_root):
            if basename in files:
                found_path = os.path.join(root, basename)
                try:
                    stat_info = os.stat(found_path)
                    return {
                        'found': True,
                        'full_path': found_path,
                        'file_size': stat_info.st_size,
                        'modified_time': datetime.fromtimestamp(stat_info.st_mtime),
                        'sha256_hash': calculate_file_hash(found_path)
                    }
                except OSError:
                    continue
    except OSError:
        pass
    
    return {
        'found': False,
        'full_path': None,
        'file_size': None,
        'modified_time': None,
        'sha256_hash': 'FILE_NOT_FOUND'
    }

def analyze_attachments(db_path: str, attachment_root: str, output_csv: str):
    """
    Analyze all attachments in the iMessage database and generate comprehensive report

    Args:
        db_path: Path to the chat.db SQLite database
        attachment_root: Root directory where attachments are stored
        output_csv: Output CSV file path
    """
    print(f"🔍 ANALYZING iMessage ATTACHMENTS")
    print(f"Database: {db_path}")
    print(f"Attachment root: {attachment_root}")
    print(f"Output CSV: {output_csv}")
    print(f"Note: Skipping .pluginPayloadAttachment files (Apple proprietary blobs)")
    print("=" * 80)
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"❌ Error connecting to database: {e}")
        return
    
    # Get all attachment records with join information
    query = """
    SELECT
        a.ROWID as attachment_rowid,
        a.guid as attachment_guid,
        a.created_date as attachment_created_date,
        a.start_date as attachment_start_date,
        a.filename as attachment_filename,
        a.uti as attachment_uti,
        a.mime_type as attachment_mime_type,
        a.transfer_state as attachment_transfer_state,
        a.is_outgoing as attachment_is_outgoing,
        a.user_info as attachment_user_info,
        a.transfer_name as attachment_transfer_name,
        a.total_bytes as attachment_total_bytes,
        a.is_sticker as attachment_is_sticker,
        a.sticker_user_info as attachment_sticker_user_info,
        a.attribution_info as attachment_attribution_info,
        a.hide_attachment as attachment_hide_attachment,
        a.ck_sync_state as attachment_ck_sync_state,
        a.ck_server_change_token_blob as attachment_ck_server_change_token_blob,
        a.ck_record_id as attachment_ck_record_id,
        a.original_guid as attachment_original_guid,
        a.is_commsafety_sensitive as attachment_is_commsafety_sensitive,
        a.emoji_image_content_identifier as attachment_emoji_image_content_identifier,
        a.emoji_image_short_description as attachment_emoji_image_short_description,
        a.preview_generation_state as attachment_preview_generation_state,

        maj.message_id as join_message_id,
        maj.attachment_id as join_attachment_id,

        m.ROWID as message_rowid,
        m.guid as message_guid,
        m.text as message_text,
        m.date as message_date,
        m.date_read as message_date_read,
        m.date_delivered as message_date_delivered,
        m.is_from_me as message_is_from_me,
        m.is_read as message_is_read,
        m.service as message_service

    FROM attachment a
    LEFT JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
    LEFT JOIN message m ON maj.message_id = m.ROWID
    ORDER BY a.ROWID
    """
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"📊 Found {len(rows)} attachment records")
    except sqlite3.Error as e:
        print(f"❌ Error querying database: {e}")
        conn.close()
        return
    
    # Prepare CSV output
    csv_headers = [
        # Attachment table fields
        'attachment_rowid', 'attachment_guid', 'attachment_created_date', 'attachment_start_date',
        'attachment_filename', 'attachment_uti', 'attachment_mime_type', 'attachment_transfer_state',
        'attachment_is_outgoing', 'attachment_user_info', 'attachment_transfer_name', 'attachment_total_bytes',
        'attachment_is_sticker', 'attachment_sticker_user_info', 'attachment_attribution_info',
        'attachment_hide_attachment', 'attachment_ck_sync_state', 'attachment_ck_server_change_token_blob',
        'attachment_ck_record_id', 'attachment_original_guid', 'attachment_is_commsafety_sensitive',
        'attachment_emoji_image_content_identifier', 'attachment_emoji_image_short_description',
        'attachment_preview_generation_state',

        # Message attachment join fields
        'join_message_id', 'join_attachment_id',

        # Message fields (for context)
        'message_rowid', 'message_guid', 'message_text_preview', 'message_date', 'message_date_read',
        'message_date_delivered', 'message_is_from_me', 'message_is_read', 'message_service',

        # File system analysis
        'file_found', 'file_full_path', 'file_size_bytes', 'file_modified_time', 'file_sha256_hash'
    ]
    
    print(f"📝 Writing CSV report...")
    
    try:
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

                # Convert timestamps
                created_date = convert_apple_timestamp(row['attachment_created_date'])
                start_date = convert_apple_timestamp(row['attachment_start_date'])
                message_date = convert_apple_timestamp(row['message_date'])
                message_date_read = convert_apple_timestamp(row['message_date_read'])
                message_date_delivered = convert_apple_timestamp(row['message_date_delivered'])
                
                # Analyze file on disk (filename already extracted above for filtering)
                file_info = find_attachment_file(filename, attachment_root) if filename else {
                    'found': False, 'full_path': None, 'file_size': None,
                    'modified_time': None, 'sha256_hash': 'NO_FILENAME'
                }
                
                if file_info['found']:
                    found_files += 1
                    if file_info['file_size']:
                        total_size += file_info['file_size']
                
                # Truncate message text for preview
                message_text = row['message_text']
                message_text_preview = (message_text[:100] + '...') if message_text and len(message_text) > 100 else message_text
                
                # Write CSV row
                csv_row = [
                    # Attachment fields
                    row['attachment_rowid'], row['attachment_guid'],
                    created_date.isoformat() if created_date else None,
                    start_date.isoformat() if start_date else None,
                    row['attachment_filename'], row['attachment_uti'], row['attachment_mime_type'],
                    row['attachment_transfer_state'], row['attachment_is_outgoing'], row['attachment_user_info'],
                    row['attachment_transfer_name'], row['attachment_total_bytes'], row['attachment_is_sticker'],
                    row['attachment_sticker_user_info'], row['attachment_attribution_info'],
                    row['attachment_hide_attachment'], row['attachment_ck_sync_state'],
                    row['attachment_ck_server_change_token_blob'], row['attachment_ck_record_id'],
                    row['attachment_original_guid'], row['attachment_is_commsafety_sensitive'],
                    row['attachment_emoji_image_content_identifier'], row['attachment_emoji_image_short_description'],
                    row['attachment_preview_generation_state'],

                    # Join fields
                    row['join_message_id'], row['join_attachment_id'],

                    # Message fields
                    row['message_rowid'], row['message_guid'], message_text_preview,
                    message_date.isoformat() if message_date else None,
                    message_date_read.isoformat() if message_date_read else None,
                    message_date_delivered.isoformat() if message_date_delivered else None,
                    row['message_is_from_me'], row['message_is_read'], row['message_service'],

                    # File system fields
                    file_info['found'], file_info['full_path'], file_info['file_size'],
                    file_info['modified_time'].isoformat() if file_info['modified_time'] else None,
                    file_info['sha256_hash']
                ]
                
                writer.writerow(csv_row)
    
    except IOError as e:
        print(f"❌ Error writing CSV file: {e}")
        conn.close()
        return
    
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
    print(f"CSV report saved to: {output_csv}")
    print(f"\n🎯 Use this data to plan your attachment processing strategy!")
    print(f"💡 Plugin payload attachments ({skipped_count:,}) are automatically excluded from processing")

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive iMessage attachment report (excludes .pluginPayloadAttachment files)")
    parser.add_argument("--db-path", "-d",
                       default="~/Library/Messages/chat.db",
                       help="Path to chat.db file (default: ~/Library/Messages/chat.db)")
    parser.add_argument("--attachment-root", "-a",
                       default="~/Library/Messages/Attachments",
                       help="Root directory for attachments (default: ~/Library/Messages/Attachments)")
    parser.add_argument("--output", "-o",
                       default="iMessage_attachment_report.csv",
                       help="Output CSV file (default: iMessage_attachment_report.csv)")
    
    args = parser.parse_args()
    
    # Expand user paths
    db_path = os.path.expanduser(args.db_path)
    attachment_root = os.path.expanduser(args.attachment_root)
    output_csv = args.output
    
    # Validate inputs
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        sys.exit(1)
    
    if not os.path.exists(attachment_root):
        print(f"❌ Attachment directory not found: {attachment_root}")
        sys.exit(1)
    
    # Run analysis
    analyze_attachments(db_path, attachment_root, output_csv)

if __name__ == "__main__":
    main()
