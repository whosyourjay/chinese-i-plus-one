#!/usr/bin/env python3
"""
Split audio file into segments based on VTT subtitle timestamps.
Uses ffmpeg to extract audio segments with optional padding.
"""

import re
import subprocess
import os
from pathlib import Path
from typing import List, Tuple


def parse_timestamp(timestamp: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_vtt_file(vtt_path: str) -> List[Tuple[float, float, str]]:
    """
    Parse VTT file and extract timing and text information.

    Returns:
        List of tuples: (start_time, end_time, text)
    """
    segments = []

    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp lines (format: 00:00:00.000 --> 00:00:00.000)
        if '-->' in line:
            # Parse timestamps
            match = re.match(r'(\S+)\s+-->\s+(\S+)', line)
            if match:
                start_time = parse_timestamp(match.group(1))
                end_time = parse_timestamp(match.group(2))

                # Get the text content (next non-empty line)
                i += 1
                if i < len(lines):
                    text = lines[i].strip()
                    if text:
                        segments.append((start_time, end_time, text))

        i += 1

    return segments


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text to create a valid filename.

    Args:
        text: Original text
        max_length: Maximum length for the filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
    # Replace spaces and other whitespace with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def split_audio(
    audio_file: str,
    vtt_file: str,
    output_dir: str,
    padding: float = 0.0
):
    """
    Split audio file based on VTT timestamps.

    Args:
        audio_file: Path to input audio file
        vtt_file: Path to VTT subtitle file
        output_dir: Directory to save output files
        padding: Extra time (in seconds) to add before and after each segment
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Parse VTT file
    print(f"Parsing VTT file: {vtt_file}")
    segments = parse_vtt_file(vtt_file)
    print(f"Found {len(segments)} segments")

    # Process each segment
    for idx, (start_time, end_time, text) in enumerate(segments, start=1):
        # Apply padding
        start_with_padding = max(0, start_time - padding)
        end_with_padding = end_time + padding
        duration = end_with_padding - start_with_padding

        # Create filename
        safe_text = sanitize_filename(text)
        output_filename = f"{idx:04d}_{safe_text}.mp3"
        output_file = output_path / output_filename

        # Build ffmpeg command
        # -ss: start time, -t: duration, -i: input file
        # Using -ss before -i for faster seeking
        cmd = [
            'ffmpeg',
            '-ss', str(start_with_padding),
            '-t', str(duration),
            '-i', audio_file,
            '-acodec', 'copy',  # Copy audio codec for faster processing
            '-y',  # Overwrite output file if exists
            str(output_file)
        ]

        print(f"Processing segment {idx}/{len(segments)}: {text[:30]}...")

        try:
            # Run ffmpeg (suppress output)
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"  ✓ Created: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error processing segment {idx}: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")

    print(f"\nDone! All segments saved to: {output_dir}")


if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "Learn Intermediate⧸Advanced Chinese：  Do You Really Know Yourself？ 你真的了解你自己吗？ [nw7c9zWduRo].mp3"
    VTT_FILE = "Learn Intermediate⧸Advanced Chinese：  Do You Really Know Yourself？ 你真的了解你自己吗？ [nw7c9zWduRo].zh.vtt"
    OUTPUT_DIR = "audio_segments"
    PADDING_SECONDS = 0.2  # Add 0.2 seconds before and after each segment

    # Check if files exist
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found: {AUDIO_FILE}")
        exit(1)

    if not os.path.exists(VTT_FILE):
        print(f"Error: VTT file not found: {VTT_FILE}")
        exit(1)

    # Split audio
    split_audio(AUDIO_FILE, VTT_FILE, OUTPUT_DIR, PADDING_SECONDS)
