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
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def process_audio_segment(args):
    """Process a single audio segment with ffmpeg."""
    idx, start_time, end_time, text, audio_file, output_dir, padding = args

    output_path = Path(output_dir)

    # Apply padding
    start_with_padding = max(0, start_time - padding)
    end_with_padding = end_time + padding
    duration = end_with_padding - start_with_padding

    # Create filename
    safe_text = sanitize_filename(text)
    output_filename = f"{idx:04d}_{safe_text}.mp3"
    output_file = output_path / output_filename

    # Remove old file if exists
    if output_file.exists():
        output_file.unlink()

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

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return {'idx': idx, 'filename': output_filename, 'success': True}
    except subprocess.CalledProcessError as e:
        return {'idx': idx, 'filename': output_filename, 'success': False, 'error': str(e)}
    except Exception as e:
        return {'idx': idx, 'filename': output_filename, 'success': False, 'error': str(e)}


def split_audio(
    audio_file: str,
    segments: List[Tuple[float, float, str]],
    output_dir: str,
    padding: float = 0.0,
    max_workers: int = 8
) -> List[str]:
    """
    Split audio file based on VTT segments using parallel processing.

    Args:
        audio_file: Path to input audio file
        segments: List of (start_time, end_time, text) tuples
        output_dir: Directory to save output files
        padding: Extra time (in seconds) to add before and after each segment
        max_workers: Number of parallel ffmpeg processes to run

    Returns:
        List of output filenames created
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Prepare arguments for parallel processing
    args_list = [
        (idx, start_time, end_time, text, audio_file, output_dir, padding)
        for idx, (start_time, end_time, text) in enumerate(segments, start=1)
    ]

    output_files = [''] * len(segments)  # Pre-allocate list
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_audio_segment, args): args[0]
            for args in args_list
        }

        for future in as_completed(future_to_idx):
            result = future.result()
            output_files[result['idx'] - 1] = result['filename']
            completed += 1

            if completed % 100 == 0:
                status = "✓" if result['success'] else "✗"
                print(f"[{completed}/{len(segments)}] {status}")

            if not result['success']:
                print(f"  ✗ Error processing segment {result['idx']}: {result.get('error', 'Unknown error')}")

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


def process_video(video_url, output_base="data_files/video",
                  output_csv="data_files/sentences_basic.csv",
                  audio_output_dir="audio_segments",
                  padding_seconds=0.2, max_workers=8):
    """Process a YouTube video: download, parse subtitles, split audio, create CSV."""
    import shutil
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

    # Clear audio segments directory
    if os.path.exists(audio_output_dir):
        shutil.rmtree(audio_output_dir)
        print(f"Cleared old audio segments: {audio_output_dir}")

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

    # Split audio
    print(f"\nSplitting audio with {max_workers} workers...")
    audio_filenames = split_audio(audio_file, segments, audio_output_dir, padding_seconds, max_workers)

    # Create CSV
    create_basic_csv(segments, audio_filenames, output_csv)
    print(f"Basic CSV saved to: {output_csv}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prepare_vtt_data.py <youtube_url>")
        sys.exit(1)

    process_video(sys.argv[1])
