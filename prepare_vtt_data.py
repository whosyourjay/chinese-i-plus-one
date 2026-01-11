#!/usr/bin/env python3
"""
Parse VTT file, split audio into segments, and create a basic CSV.
This CSV will be enhanced later with segmentation and translation.
"""

import csv
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
    segments: List[Tuple[float, float, str]],
    output_dir: str,
    padding: float = 0.0
) -> List[str]:
    """
    Split audio file based on VTT segments.

    Args:
        audio_file: Path to input audio file
        segments: List of (start_time, end_time, text) tuples
        output_dir: Directory to save output files
        padding: Extra time (in seconds) to add before and after each segment

    Returns:
        List of output filenames created
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_files = []

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
        output_files.append(output_filename)

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-ss', str(start_with_padding),
            '-t', str(duration),
            '-i', audio_file,
            '-acodec', 'copy',
            '-y',
            str(output_file)
        ]

        print(f"Processing segment {idx}/{len(segments)}: {text[:30]}...")

        try:
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

    return output_files


def create_basic_csv(
    segments: List[Tuple[float, float, str]],
    audio_filenames: List[str],
    output_csv_path: str
):
    """
    Create a basic CSV with Sentence and audio columns.

    Args:
        segments: List of (start_time, end_time, text) tuples
        audio_filenames: List of audio filenames
        output_csv_path: Path for the output CSV file
    """
    print("\nCreating basic CSV...")

    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Sentence', 'audio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for (_, _, text), audio_filename in zip(segments, audio_filenames):
            audio_ref = f"[sound:{audio_filename}]"
            writer.writerow({
                'Sentence': text,
                'audio': audio_ref
            })

    print(f"CSV file created: {output_csv_path}")


def process_vtt(
    audio_file: str,
    vtt_file: str,
    output_csv: str,
    audio_output_dir: str = "audio_segments",
    padding: float = 0.2
):
    """
    Main function to process VTT: split audio and create basic CSV.

    Args:
        audio_file: Path to input audio file (in data_files)
        vtt_file: Path to VTT subtitle file (in data_files)
        output_csv: Path for output CSV (will be saved in data_files)
        audio_output_dir: Directory to save audio segments
        padding: Extra time (in seconds) to add before and after each segment
    """
    # Check if files exist
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return

    if not os.path.exists(vtt_file):
        print(f"Error: VTT file not found: {vtt_file}")
        return

    # Parse VTT file
    print(f"Parsing VTT file: {vtt_file}")
    segments = parse_vtt_file(vtt_file)
    print(f"Found {len(segments)} segments")

    # Split audio
    print(f"\nSplitting audio into segments...")
    audio_filenames = split_audio(audio_file, segments, audio_output_dir, padding)

    # Create basic CSV
    create_basic_csv(segments, audio_filenames, output_csv)

    print(f"\nDone! Audio segments saved to: {audio_output_dir}")
    print(f"Basic CSV saved to: {output_csv}")


if __name__ == "__main__":
    # Example usage
    # You can modify these paths as needed
    AUDIO_FILE = "data_files/The Best Chinese.mp3"
    VTT_FILE = "data_files/The Best Chinese.vtt"
    OUTPUT_CSV = "data_files/sentences_basic.csv"
    AUDIO_OUTPUT_DIR = "audio_segments"
    PADDING_SECONDS = 0.2

    process_vtt(AUDIO_FILE, VTT_FILE, OUTPUT_CSV, AUDIO_OUTPUT_DIR, PADDING_SECONDS)
