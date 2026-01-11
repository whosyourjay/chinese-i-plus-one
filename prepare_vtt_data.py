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
from typing import List, Tuple, Optional


def download_youtube_video(video_url: str, output_base: str = "data_files/video"):
    """Download YouTube video with audio and Chinese subtitles."""
    cmd = [
        'yt-dlp',
        '-x',
        '--audio-format', 'mp3',
        '--write-subs',
        '--sub-langs', 'zh,zh-CN,zh-Hans',
        '-o', output_base,
        video_url
    ]
    subprocess.run(cmd, check=True)


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

        if idx % 100 == 0:
            print(f"Processing segment {idx}/{len(segments)}: {text[:30]}...")

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            #print(f"  ✓ Created: {output_filename}")
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


if __name__ == "__main__":
    import sys

    # Get video URL from command line argument
    if len(sys.argv) < 2:
        print("Usage: python prepare_vtt_data.py <youtube_url>")
        sys.exit(1)

    video_url = sys.argv[1]

    # Configuration
    OUTPUT_BASE = "data_files/video"
    AUDIO_FILE = f"{OUTPUT_BASE}.mp3"
    OUTPUT_CSV = "data_files/sentences_basic.csv"
    AUDIO_OUTPUT_DIR = "audio_segments"
    PADDING_SECONDS = 0.2
    SUBTITLE_LANGS = ['zh', 'zh-CN', 'zh-Hans']

    # Remove old VTT files to avoid using stale data
    for lang in SUBTITLE_LANGS:
        old_vtt = f"{OUTPUT_BASE}.{lang}.vtt"
        if os.path.exists(old_vtt):
            os.remove(old_vtt)
            print(f"Removed old subtitle file: {old_vtt}")

    # Download video and subtitles
    download_youtube_video(video_url, OUTPUT_BASE)

    # Find which VTT file was created
    VTT_FILE = None
    for lang in SUBTITLE_LANGS:
        candidate = f"{OUTPUT_BASE}.{lang}.vtt"
        if os.path.exists(candidate):
            VTT_FILE = candidate
            break

    if not VTT_FILE:
        print(f"Error: No Chinese subtitle file found (tried {', '.join(f'.{lang}.vtt' for lang in SUBTITLE_LANGS)})")
        sys.exit(1)

    # Parse VTT file
    print(f"Parsing VTT file: {VTT_FILE}")
    segments = parse_vtt_file(VTT_FILE)
    print(f"Found {len(segments)} segments")

    # Split audio
    print(f"\nSplitting audio into segments...")
    audio_filenames = split_audio(AUDIO_FILE, segments, AUDIO_OUTPUT_DIR, PADDING_SECONDS)

    # Create basic CSV
    create_basic_csv(segments, audio_filenames, OUTPUT_CSV)

    print(f"\nDone! Audio segments saved to: {AUDIO_OUTPUT_DIR}")
    print(f"Basic CSV saved to: {OUTPUT_CSV}")
