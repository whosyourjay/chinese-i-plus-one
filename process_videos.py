#!/usr/bin/env python3
import os
import sys
import time
import pandas as pd
from prepare_vtt_data import process_video as prepare_vtt_data
from enhance_csv import enhance_csv_with_segmentation
from selection import run_i_plus_1_selection
from generate_audio import generate_audio_for_selected_sentences


def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


def append_new_words_to_known(csv_path, known_file='known'):
    df = pd.read_csv(csv_path)
    if 'New_Words' not in df.columns:
        return

    words = []
    for words_str in df['New_Words'].dropna():
        words.extend([w.strip() for w in str(words_str).split(',') if w.strip()])

    if words:
        with open(known_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(words) + '\n')
        print(f"Appended {len(words)} new words to known file")


def process_video(video_url, num, total):
    print(f"\n{'='*50}\nVideo {num}/{total}: {video_url}\n{'='*50}")

    print("Step 1: Preparing VTT data...")
    _, elapsed = time_function(prepare_vtt_data, video_url)
    print(f"Time: {elapsed:.1f}s")

    print("Step 2: Removing old enhanced CSV...")
    os.remove("data_files/sentences_enhanced.csv") if os.path.exists("data_files/sentences_enhanced.csv") else None

    print("Step 3: Enhancing CSV...")
    _, elapsed = time_function(enhance_csv_with_segmentation,
                               "data_files/sentences_basic.csv",
                               "data_files/sentences_enhanced.csv", 5)
    print(f"Time: {elapsed:.1f}s")

    print("Step 4: Running selection...")
    _, elapsed = time_function(run_i_plus_1_selection,
                               "data_files/sentences_enhanced.csv",
                               "data_files/sentence_sequence.csv", 6, True)
    print(f"Time: {elapsed:.1f}s")

    print("Step 5: Generating audio files for selected sentences...")
    _, elapsed = time_function(generate_audio_for_selected_sentences,
                               "data_files/sentence_sequence.csv",
                               "data_files/video.mp3")
    print(f"Time: {elapsed:.1f}s")

    print("Step 6: Appending results...")
    seq_csv = "data_files/sentence_sequence.csv"
    all_csv = "data_files/all_sentences.csv"

    if not os.path.exists(seq_csv):
        return False

    df = pd.read_csv(seq_csv)
    df.to_csv(all_csv, mode='a', header=not os.path.exists(all_csv), index=False)
    print(f"Appended {len(df)} sentences")

    print("Step 7: Updating known words...")
    append_new_words_to_known(seq_csv)

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 process_videos.py <video_list_file>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not urls:
        print("No URLs found")
        sys.exit(1)

    print(f"Processing {len(urls)} videos")
    os.makedirs("data_files", exist_ok=True)

    successful = 0
    for i, url in enumerate(urls, 1):
        try:
            if process_video(url, i, len(urls)):
                successful += 1
        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'='*50}\nComplete: {successful}/{len(urls)} successful\n{'='*50}")


if __name__ == "__main__":
    main()
