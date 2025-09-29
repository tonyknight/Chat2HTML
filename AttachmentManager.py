#!/usr/bin/env python3
"""
AttachmentManager - Universal Attachment Processing System

A comprehensive attachment processing system designed for archival quality storage
across multiple messaging platforms. Handles file copying, format conversion,
metadata preservation, and standardized naming conventions.

🎯 KEY FEATURES:
- Universal attachment processing for multiple messaging services
- Archival quality format conversion (HEIC/PNG → JPG at 85% quality)
- EXIF metadata preservation and enhancement
- Standardized date-based folder structure and naming
- Duplicate detection with hash-based deduplication
- Parallel processing for large datasets
- Swift-compatible architecture for future porting

📁 ARCHIVAL STRUCTURE:
/Attachments/{service}/{year}/{month_number} - {month_name}/{filename}
Example: /Attachments/iMessage/2024/3 - March/(2024-03-15) 14-30-25 [123].jpg

🔧 PROCESSING PIPELINE:
1. Source file validation and hash calculation
2. EXIF metadata extraction and date determination
3. Format conversion (if needed) with quality optimization
4. Metadata embedding using ExifTool
5. Archival naming and folder organization
6. Duplicate detection and symlink creation
7. Database record updates with new paths

🎛️ SUPPORTED FORMATS:
- Images: HEIC, PNG, WebP → JPG (85% quality), JPEG (preserved), GIF (preserved)
- Videos: MP4, MOV (metadata enhancement only)
- Documents: PDF, DOC, XLS, TXT (date-prefixed naming)
- Audio: CAF → AAC (lossless), M4A, MP3 (date-prefixed naming)
- Other: VCF, ZIP, etc. (date-prefixed naming)

⚡ PERFORMANCE:
- Multi-threaded processing with configurable workers
- Progress tracking with detailed statistics
- Efficient duplicate detection using SHA-256 hashing
- Batch processing for optimal memory usage
"""

import os
import shutil
import hashlib
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tempfile

try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Image conversion will be disabled.")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()  # Register HEIC/HEIF support for Pillow
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    print("Warning: pillow-heif not available. HEIC conversion will be limited.")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("Warning: ffmpeg-python not available. Audio conversion will be disabled.")


@dataclass
class AttachmentInfo:
    """
    Data structure representing an attachment with all necessary metadata
    
    This class is designed to be easily portable to Swift as a struct
    """
    # Original attachment data
    original_path: str
    original_filename: str
    mime_type: Optional[str]
    file_size: int
    created_date: Optional[datetime]
    
    # Processing results
    archival_path: Optional[str] = None
    archival_filename: Optional[str] = None
    file_hash: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    conversion_applied: bool = False
    metadata_enhanced: bool = False
    
    # Processing status
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None


@dataclass
class ProcessingStats:
    """Statistics tracking for attachment processing operations"""
    total_attachments: int = 0
    processed_successfully: int = 0
    failed_processing: int = 0
    skipped_processing: int = 0  # Files skipped (e.g., plugin payload attachments)
    duplicates_found: int = 0
    conversions_applied: int = 0
    metadata_enhanced: int = 0
    total_size_bytes: int = 0
    processing_time_seconds: float = 0.0


class AttachmentProcessor:
    """
    Core attachment processing engine
    
    Handles individual file operations including conversion, metadata enhancement,
    and archival naming. Designed with MVC pattern for Swift compatibility.
    """
    
    def __init__(self, service_name: str = "iMessage"):
        self.service_name = service_name
        self.supported_image_formats = {'.heic', '.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif', '.bmp', '.webp'}
        self.supported_video_formats = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.3gp', '.webm'}
        self.supported_audio_formats = {'.m4a', '.caf', '.mp3', '.wav', '.aac', '.flac'}
        self.image_conversion_formats = {'.heic', '.png', '.webp'}  # Formats to convert to JPG
        self.audio_conversion_formats = {'.caf'}  # Formats to convert to AAC
        self.conversion_formats = self.image_conversion_formats | self.audio_conversion_formats
        
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for duplicate detection"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            raise Exception(f"Failed to calculate hash for {file_path}: {e}")
    
    def extract_exif_date(self, file_path: str) -> Optional[datetime]:
        """Extract DateTimeOriginal from EXIF data using ExifTool with robust parsing"""
        try:
            # Use ExifTool without format specification to get raw data
            result = subprocess.run([
                'exiftool', '-DateTimeOriginal', '-s3', file_path
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout.strip():
                date_output = result.stdout.strip()
                if date_output and date_output != '-':
                    # Try to parse various date formats
                    return self._parse_exif_date_string(date_output)
        except Exception as e:
            print(f"Warning: Failed to extract EXIF date from {file_path}: {e}")

        return None

    def _parse_exif_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse EXIF date string with multiple format support"""
        # Clean up the string - take only the first line if multiple lines exist
        date_str = date_str.split('\n')[0].strip()

        # Common EXIF date formats to try
        date_formats = [
            '%Y:%m:%d %H:%M:%S',      # Standard EXIF format: 2018:02:23 22:28:12
            '%Y-%m-%d %H:%M:%S',      # ISO format: 2018-02-23 22:28:12
            '%d.%m.%Y %H:%M:%S',      # European format: 04.03.2011 10:03:50
            '%m/%d/%Y %H:%M:%S',      # US format: 03/04/2011 10:03:50
            '%Y:%m:%d %H:%M:%S%z',    # With timezone
            '%Y-%m-%d %H:%M:%S%z',    # ISO with timezone
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If no format matches, try to extract just the date part
        import re

        # Try to find YYYY:MM:DD or YYYY-MM-DD pattern
        date_match = re.search(r'(\d{4})[:-](\d{2})[:-](\d{2})', date_str)
        if date_match:
            year, month, day = date_match.groups()

            # Try to find time part
            time_match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', date_str)
            if time_match:
                hour, minute, second = time_match.groups()
                try:
                    return datetime(int(year), int(month), int(day),
                                  int(hour), int(minute), int(second))
                except ValueError:
                    pass

            # Just date without time
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                pass

        print(f"Warning: Could not parse EXIF date format: '{date_str}'")
        return None
    
    def embed_exif_date(self, file_path: str, date_time: datetime) -> bool:
        """Embed DateTimeOriginal into file using ExifTool"""
        try:
            date_str = date_time.strftime('%Y:%m:%d %H:%M:%S')
            result = subprocess.run([
                'exiftool', '-overwrite_original', f'-DateTimeOriginal={date_str}', file_path
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0
        except Exception as e:
            print(f"Warning: Failed to embed EXIF date in {file_path}: {e}")
            return False
    
    def _validate_image_file(self, file_path: str) -> bool:
        """Validate that an image file can be opened and read"""
        if not PIL_AVAILABLE:
            return False

        # Check if it's a HEIC file and we don't have HEIF support
        file_ext = Path(file_path).suffix.lower()
        if file_ext in ['.heic', '.heif'] and not HEIF_AVAILABLE:
            print(f"Warning: HEIC file {file_path} requires pillow-heif library")
            return False

        try:
            with Image.open(file_path) as img:
                # Try to load the image data to verify it's not corrupted
                img.load()
                # Check if image has reasonable dimensions
                if img.size[0] > 0 and img.size[1] > 0:
                    return True
        except Exception as e:
            # Provide more specific error information for HEIC files
            if file_ext in ['.heic', '.heif']:
                print(f"Warning: HEIC file validation failed for {file_path}: {e}")
                if not HEIF_AVAILABLE:
                    print("  → Install pillow-heif: pip install pillow-heif")
            pass

        return False

    def convert_image_to_jpg(self, input_path: str, output_path: str, quality: int = 85) -> bool:
        """Convert HEIC/PNG/WebP to JPG with specified quality and validation"""
        if not PIL_AVAILABLE:
            return False

        # Check for HEIC files specifically
        file_ext = Path(input_path).suffix.lower()
        if file_ext in ['.heic', '.heif'] and not HEIF_AVAILABLE:
            print(f"Warning: Cannot convert HEIC file {input_path} - pillow-heif not available")
            print("  → Install with: pip install pillow-heif")
            return False

        # Pre-validate the image file
        if not self._validate_image_file(input_path):
            print(f"Warning: Image file validation failed for {input_path} (corrupted or unreadable)")
            return False

        try:
            with Image.open(input_path) as img:
                # For HEIC files, ensure we handle them properly
                if file_ext in ['.heic', '.heif']:
                    print(f"Converting HEIC file: {Path(input_path).name}")

                # Convert to RGB if necessary (for HEIC/PNG/WebP with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save as JPG with specified quality
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                return True
        except Exception as e:
            # Provide specific guidance for HEIC conversion failures
            if file_ext in ['.heic', '.heif']:
                print(f"Warning: HEIC conversion failed for {input_path}: {e}")
                if not HEIF_AVAILABLE:
                    print("  → This likely requires: pip install pillow-heif")
            else:
                print(f"Warning: Failed to convert {input_path} to JPG: {e}")
            return False

    def convert_caf_to_aac(self, input_path: str, output_path: str) -> bool:
        """Convert CAF audio to AAC format (lossless when possible)"""
        if not FFMPEG_AVAILABLE:
            return False

        try:
            # Use FFmpeg to convert CAF to AAC
            # -c:a aac uses the built-in AAC encoder
            # -b:a 128k sets a reasonable bitrate for good quality
            # -movflags +faststart optimizes for streaming
            (
                ffmpeg
                .input(input_path)
                .output(output_path, acodec='aac', audio_bitrate='128k', movflags='+faststart')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to convert {input_path} to AAC: {e}")
            return False

    def generate_archival_filename(self, attachment: AttachmentInfo, target_date: datetime) -> str:
        """Generate standardized archival filename based on date and type"""
        # Extract file extension
        original_ext = Path(attachment.original_filename).suffix.lower()
        
        # Determine target extension based on conversion rules
        if original_ext in self.image_conversion_formats:
            target_ext = '.jpg'
        elif original_ext in self.audio_conversion_formats:
            target_ext = '.aac'
        else:
            target_ext = original_ext
        
        # Check if we have milliseconds in the date (from EXIF or message timestamp)
        if target_date.microsecond > 0:
            # Format with milliseconds
            milliseconds = target_date.microsecond // 1000
            date_part = target_date.strftime('(%Y-%m-%d) %H-%M-%S')
            filename = f"{date_part} [{milliseconds:03d}]{target_ext}"
        else:
            # Format without milliseconds
            filename = target_date.strftime('(%Y-%m-%d) %H-%M-%S') + target_ext
        
        # For non-image/video files, prepend date to original filename
        if (original_ext not in self.supported_image_formats and 
            original_ext not in self.supported_video_formats):
            date_prefix = target_date.strftime('(%Y-%m-%d) %H-%M-%S ')
            original_name = Path(attachment.original_filename).stem
            filename = f"{date_prefix}{original_name}{target_ext}"
        
        return filename
    
    def generate_archival_path(self, attachment: AttachmentInfo, target_date: datetime, 
                             base_attachments_dir: str) -> Tuple[str, str]:
        """Generate full archival path and filename"""
        # Create year and month folder structure
        year = target_date.year
        month_num = target_date.month
        month_name = target_date.strftime('%B')
        
        # Create folder structure: /Attachments/{service}/{year}/{month_number} - {month_name}/
        folder_path = os.path.join(
            base_attachments_dir,
            self.service_name,
            str(year),
            f"{month_num} - {month_name}"
        )
        
        # Generate filename
        filename = self.generate_archival_filename(attachment, target_date)
        
        # Full path
        full_path = os.path.join(folder_path, filename)
        
        return folder_path, full_path


class AttachmentManager:
    """
    Main attachment management controller
    
    Orchestrates the entire attachment processing pipeline with support for
    multiple messaging services, parallel processing, and comprehensive statistics.
    
    Designed with MVC architecture for easy Swift porting.
    """
    
    def __init__(self, base_attachments_dir: str, service_name: str = "iMessage", 
                 move_files: bool = False, max_workers: int = 4):
        """
        Initialize AttachmentManager
        
        Args:
            base_attachments_dir: Root directory for archival attachments
            service_name: Name of messaging service (iMessage, WhatsApp, etc.)
            move_files: If True, move files instead of copying (for non-iMessage services)
            max_workers: Number of parallel processing workers
        """
        self.base_attachments_dir = Path(base_attachments_dir)
        self.service_name = service_name
        self.move_files = move_files
        self.max_workers = max_workers
        
        # Initialize processor
        self.processor = AttachmentProcessor(service_name)
        
        # Duplicate tracking
        self.hash_to_path: Dict[str, str] = {}
        self.processed_hashes: Dict[str, AttachmentInfo] = {}
        
        # Statistics
        self.stats = ProcessingStats()
        
        # Ensure base directory exists
        self.base_attachments_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"AttachmentManager initialized for {service_name}")
        print(f"Base directory: {base_attachments_dir}")
        print(f"Mode: {'MOVE' if move_files else 'COPY'} files")
        print(f"Workers: {max_workers}")

    def process_single_attachment(self, attachment: AttachmentInfo,
                                message_date: datetime) -> AttachmentInfo:
        """
        Process a single attachment through the complete pipeline

        Args:
            attachment: AttachmentInfo object with source details
            message_date: Date when message was sent (fallback for EXIF date)

        Returns:
            Updated AttachmentInfo with processing results
        """
        attachment.processing_status = "processing"

        try:
            # Step 0: Skip Apple's proprietary plugin payload attachments
            if attachment.original_filename.lower().endswith('.pluginpayloadattachment'):
                attachment.processing_status = "skipped"
                attachment.error_message = "Apple proprietary blob format - not suitable for archival"
                self.stats.skipped_processing += 1
                return attachment

            # Step 1: Validate source file exists
            if not os.path.exists(attachment.original_path):
                raise Exception(f"Source file not found: {attachment.original_path}")

            # Step 2: Calculate file hash for duplicate detection
            attachment.file_hash = self.processor.calculate_file_hash(attachment.original_path)

            # Step 3: Check for duplicates
            if attachment.file_hash in self.processed_hashes:
                duplicate_attachment = self.processed_hashes[attachment.file_hash]
                attachment.is_duplicate = True
                attachment.duplicate_of = duplicate_attachment.archival_path
                attachment.processing_status = "completed"
                self.stats.duplicates_found += 1
                return attachment

            # Step 4: Determine target date (EXIF first, then message date)
            exif_date = self.processor.extract_exif_date(attachment.original_path)
            target_date = exif_date if exif_date else message_date

            # Step 5: Generate archival path and filename
            folder_path, full_archival_path = self.processor.generate_archival_path(
                attachment, target_date, str(self.base_attachments_dir)
            )

            # Step 6: Create target directory
            os.makedirs(folder_path, exist_ok=True)

            # Step 7: Determine if conversion is needed
            original_ext = Path(attachment.original_filename).suffix.lower()
            needs_conversion = original_ext in self.processor.conversion_formats

            if needs_conversion:
                # Step 8a: Convert format based on type
                temp_path = full_archival_path
                conversion_success = False

                if original_ext in self.processor.image_conversion_formats:
                    # Convert image to JPG
                    conversion_success = self.processor.convert_image_to_jpg(attachment.original_path, temp_path)
                elif original_ext in self.processor.audio_conversion_formats:
                    # Convert audio to AAC
                    conversion_success = self.processor.convert_caf_to_aac(attachment.original_path, temp_path)

                if conversion_success:
                    attachment.conversion_applied = True
                    self.stats.conversions_applied += 1
                else:
                    # Fallback to copy if conversion fails
                    if self.move_files:
                        shutil.move(attachment.original_path, full_archival_path)
                    else:
                        shutil.copy2(attachment.original_path, full_archival_path)
            else:
                # Step 8b: Copy/move file without conversion
                if self.move_files:
                    shutil.move(attachment.original_path, full_archival_path)
                else:
                    shutil.copy2(attachment.original_path, full_archival_path)

            # Step 9: Embed EXIF metadata if needed
            if not exif_date and target_date:
                if self.processor.embed_exif_date(full_archival_path, target_date):
                    attachment.metadata_enhanced = True
                    self.stats.metadata_enhanced += 1

            # Step 10: Update attachment info
            attachment.archival_path = os.path.relpath(full_archival_path, self.base_attachments_dir)
            attachment.archival_filename = os.path.basename(full_archival_path)
            attachment.processing_status = "completed"

            # Step 11: Track for duplicate detection
            self.processed_hashes[attachment.file_hash] = attachment
            self.hash_to_path[attachment.file_hash] = full_archival_path

            self.stats.processed_successfully += 1

        except Exception as e:
            attachment.processing_status = "failed"
            attachment.error_message = str(e)
            self.stats.failed_processing += 1
            print(f"Error processing {attachment.original_filename}: {e}")

        return attachment

    def create_symlink_for_duplicate(self, duplicate_attachment: AttachmentInfo,
                                   target_date: datetime) -> bool:
        """Create symlink for duplicate file in appropriate date folder with conflict resolution"""
        try:
            if not duplicate_attachment.duplicate_of:
                return False

            # Generate path where symlink should be created
            folder_path, symlink_path = self.processor.generate_archival_path(
                duplicate_attachment, target_date, str(self.base_attachments_dir)
            )

            # Create target directory
            os.makedirs(folder_path, exist_ok=True)

            # Handle naming conflicts by adding suffix
            original_symlink_path = symlink_path
            counter = 1
            while os.path.exists(symlink_path):
                # Add counter suffix to filename
                base_name = os.path.splitext(os.path.basename(original_symlink_path))[0]
                extension = os.path.splitext(os.path.basename(original_symlink_path))[1]
                new_filename = f"{base_name}_dup{counter}{extension}"
                symlink_path = os.path.join(folder_path, new_filename)
                counter += 1

                # Prevent infinite loop
                if counter > 1000:
                    print(f"Warning: Too many duplicate conflicts for {duplicate_attachment.original_filename}")
                    return False

            # Create relative symlink to original file
            original_path = os.path.join(str(self.base_attachments_dir), duplicate_attachment.duplicate_of)
            relative_path = os.path.relpath(original_path, folder_path)

            os.symlink(relative_path, symlink_path)

            # Update duplicate attachment info
            duplicate_attachment.archival_path = os.path.relpath(symlink_path, self.base_attachments_dir)
            duplicate_attachment.archival_filename = os.path.basename(symlink_path)

            return True

        except Exception as e:
            print(f"Warning: Failed to create symlink for duplicate: {e}")
            return False

    def process_attachments_parallel(self, attachments_with_dates: List[Tuple[AttachmentInfo, datetime]]) -> List[AttachmentInfo]:
        """
        Process multiple attachments in parallel with progress tracking

        Args:
            attachments_with_dates: List of (AttachmentInfo, message_date) tuples

        Returns:
            List of processed AttachmentInfo objects
        """
        if not attachments_with_dates:
            return []

        print(f"Processing {len(attachments_with_dates):,} attachments with {self.max_workers} workers...")

        # Initialize statistics
        self.stats.total_attachments = len(attachments_with_dates)
        self.stats.total_size_bytes = sum(att.file_size for att, _ in attachments_with_dates)

        start_time = datetime.now()
        processed_attachments = []

        # Process attachments in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_attachment = {
                executor.submit(self.process_single_attachment, attachment, message_date): (attachment, message_date)
                for attachment, message_date in attachments_with_dates
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(attachments_with_dates), desc="Processing attachments") as pbar:
                for future in as_completed(future_to_attachment):
                    try:
                        processed_attachment = future.result()
                        processed_attachments.append(processed_attachment)

                        # Create symlinks for duplicates if needed
                        if processed_attachment.is_duplicate:
                            _, message_date = future_to_attachment[future]
                            self.create_symlink_for_duplicate(processed_attachment, message_date)

                        pbar.update(1)

                    except Exception as e:
                        attachment, _ = future_to_attachment[future]
                        print(f"Error processing attachment {attachment.original_filename}: {e}")
                        attachment.processing_status = "failed"
                        attachment.error_message = str(e)
                        processed_attachments.append(attachment)
                        self.stats.failed_processing += 1
                        pbar.update(1)

        # Calculate processing time
        self.stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()

        # Print statistics
        self.print_processing_statistics()

        return processed_attachments

    def print_processing_statistics(self):
        """Print comprehensive processing statistics"""
        print(f"\n📊 ATTACHMENT PROCESSING STATISTICS:")
        print(f"Total attachments: {self.stats.total_attachments:,}")
        print(f"Successfully processed: {self.stats.processed_successfully:,}")
        print(f"Failed processing: {self.stats.failed_processing:,}")
        print(f"Skipped processing: {self.stats.skipped_processing:,} (plugin payload attachments)")
        print(f"Duplicates found: {self.stats.duplicates_found:,}")
        print(f"Format conversions: {self.stats.conversions_applied:,}")
        print(f"Metadata enhanced: {self.stats.metadata_enhanced:,}")
        print(f"Total size processed: {self.stats.total_size_bytes / (1024*1024):.1f} MB")
        print(f"Processing time: {self.stats.processing_time_seconds:.1f} seconds")

        if self.stats.total_attachments > 0:
            success_rate = (self.stats.processed_successfully / self.stats.total_attachments) * 100
            print(f"Success rate: {success_rate:.1f}%")

        if self.stats.processing_time_seconds > 0:
            throughput = self.stats.total_attachments / self.stats.processing_time_seconds
            print(f"Throughput: {throughput:.1f} attachments/second")


def create_attachment_info_from_message_attachment(attachment_dict: Dict[str, Any],
                                                 source_attachments_dir: str) -> Optional[AttachmentInfo]:
    """
    Create AttachmentInfo from message attachment dictionary

    This function bridges the gap between the message system and attachment manager.
    Designed to be easily adaptable for different messaging platforms.

    Args:
        attachment_dict: Attachment data from message export
        source_attachments_dir: Base directory where original attachments are stored

    Returns:
        AttachmentInfo object or None if attachment cannot be processed
    """
    try:
        filename = attachment_dict.get('filename', '')
        if not filename:
            return None

        # Handle tilde expansion for iMessage paths
        if filename.startswith('~/'):
            original_path = os.path.expanduser(filename)
        else:
            original_path = os.path.join(source_attachments_dir, filename)

        # Verify file exists
        if not os.path.exists(original_path):
            print(f"Warning: Attachment file not found: {original_path}")
            return None

        # Extract file info
        file_size = attachment_dict.get('file_size', 0)
        if file_size == 0:
            file_size = os.path.getsize(original_path)

        # Create AttachmentInfo
        return AttachmentInfo(
            original_path=original_path,
            original_filename=os.path.basename(filename),
            mime_type=attachment_dict.get('mime_type'),
            file_size=file_size,
            created_date=None  # Will be determined from EXIF or message date
        )

    except Exception as e:
        print(f"Error creating AttachmentInfo: {e}")
        return None


# Example usage and testing functions
if __name__ == "__main__":
    # This section can be used for testing the AttachmentManager
    print("AttachmentManager module loaded successfully")
    print("Key classes: AttachmentManager, AttachmentProcessor, AttachmentInfo")
    print("Use create_attachment_info_from_message_attachment() to bridge with message system")
