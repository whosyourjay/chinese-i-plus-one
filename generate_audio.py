#!/usr/bin/env python3
"""
Generate audio files for selected sentences after the selection step.
Reads timing info from basic CSV and creates audio files only for selected sentences.
Also generates TTS audio for target words using Edge TTS.
"""

import re
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
import edge_tts


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text to create a valid filename."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized


async def generate_word_tts_async(word: str, output_file: Path) -> str:
    """Generate TTS audio for a word using Edge TTS."""
    communicate = edge_tts.Communicate(word, "zh-CN-XiaoxiaoNeural", rate="-20%")
    await communicate.save(str(output_file))
    return output_file.name


def generate_word_audio(sequence_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Add word_audio column with TTS for the target word."""
    print("\nGenerating Edge TTS for target words...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    word_audio_list = []
    for idx, row in sequence_df.iterrows():
        word = str(row.get('New_Words', '')).strip()

        if not word:
            word_audio_list.append('')
            continue

        filename = f"word_{sanitize_filename(word, 30)}.mp3"
        filepath = output_path / filename

        if not filepath.exists():
            try:
                asyncio.run(generate_word_tts_async(word, filepath))
            except Exception as e:
                print(f"  ✗ Error for '{word}': {e}")
                word_audio_list.append('')
                continue

        word_audio_list.append(f"[sound:{filename}]")

    sequence_df['word_audio'] = word_audio_list
    print(f"  ✓ Generated audio for {len(sequence_df)} sentences")
    return sequence_df


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


def generate_audio_for_selected_sentences(
    sequence_csv: str,
    audio_file: str,
    output_dir: str = "audio_segments",
    padding_seconds: float = 0.2,
    max_workers: int = 8
):
    """Generate audio files for sentences and TTS for target words."""
    print(f"Loading selected sentences from: {sequence_csv}")
    sequence_df = pd.read_csv(sequence_csv)

    # Generate sentence audio clips
    segments = [
        (row['start_time'], row['end_time'], row['Sentence'])
        for _, row in sequence_df.iterrows()
    ]

    print(f"\nCreating audio files for {len(segments)} selected sentences...")
    audio_filenames = split_audio(audio_file, segments, output_dir, padding_seconds, max_workers)

    sentence_to_audio = {segments[i][2]: f"[sound:{audio_filenames[i]}]" for i in range(len(segments))}
    sequence_df['audio'] = sequence_df['Sentence'].map(lambda s: sentence_to_audio.get(s, ''))

    # Generate word TTS audio and add pinyin
    from pinyin_jyutping_sentence import pinyin
    sequence_df = generate_word_audio(sequence_df, output_dir)
    sequence_df['sentence_pinyin'] = sequence_df['Sentence'].apply(pinyin)
    sequence_df['word_pinyin'] = sequence_df['New_Words'].apply(
        lambda x: pinyin(str(x).strip()) if str(x).strip() else ''
    )

    # Save updated sequence CSV
    sequence_df.to_csv(sequence_csv, index=False)
    print(f"\nUpdated {sequence_csv} with audio and pinyin")
    print(f"Audio files saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    # Default paths
    SEQUENCE_CSV = "data_files/sentence_sequence.csv"
    AUDIO_FILE = "data_files/video.mp3"

    if len(sys.argv) > 1:
        AUDIO_FILE = sys.argv[1]

    generate_audio_for_selected_sentences(
        SEQUENCE_CSV,
        AUDIO_FILE
    )
