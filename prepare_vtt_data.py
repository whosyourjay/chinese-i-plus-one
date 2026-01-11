#!/usr/bin/env python3
"""
Parse VTT file and create a basic CSV with timing information.
This CSV will be enhanced later with segmentation and translation.
Audio files will be created after sentence selection.
"""

import csv
import re
import subprocess
import os
from typing import List, Tuple


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


def create_basic_csv(
    segments: List[Tuple[float, float, str]],
    output_csv_path: str
):
    """
    Create a basic CSV with Sentence and timing columns.

    Args:
        segments: List of (start_time, end_time, text) tuples
        output_csv_path: Path for the output CSV file
    """
    print("\nCreating basic CSV with timing info...")

    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Sentence', 'start_time', 'end_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for start_time, end_time, text in segments:
            writer.writerow({
                'Sentence': text,
                'start_time': start_time,
                'end_time': end_time
            })

    print(f"CSV file created: {output_csv_path}")


def process_video(video_url, output_base="data_files/video",
                  output_csv="data_files/sentences_basic.csv"):
    """Process a YouTube video: download, parse subtitles, create CSV with timing info."""
    audio_file = f"{output_base}.mp3"
    subtitle_langs = ['zh', 'zh-CN', 'zh-Hans']

    # Remove old files to ensure fresh download
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"Removed old audio file: {audio_file}")

    for lang in subtitle_langs:
        old_vtt = f"{output_base}.{lang}.vtt"
        if os.path.exists(old_vtt):
            os.remove(old_vtt)

    # Download video and subtitles
    download_youtube_video(video_url, output_base)

    # Find VTT file
    vtt_file = None
    for lang in subtitle_langs:
        candidate = f"{output_base}.{lang}.vtt"
        if os.path.exists(candidate):
            vtt_file = candidate
            break

    if not vtt_file:
        raise FileNotFoundError(f"No Chinese subtitle file found")

    # Parse VTT file
    print(f"Parsing VTT file: {vtt_file}")
    segments = parse_vtt_file(vtt_file)
    print(f"Found {len(segments)} segments")

    # Create CSV with timing info (audio will be created after selection)
    create_basic_csv(segments, output_csv)
    print(f"Basic CSV with timing info saved to: {output_csv}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prepare_vtt_data.py <youtube_url>")
        sys.exit(1)

    process_video(sys.argv[1])
