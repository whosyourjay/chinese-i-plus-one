import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from chinese_translator import segment_and_translate

def parse_vtt_file(vtt_path):
    """
    Parse a VTT file and extract subtitle entries.

    Args:
        vtt_path: Path to the .zh.vtt file

    Returns:
        List of dictionaries with 'Sentence' for each subtitle
    """
    subtitles = []

    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp lines (format: HH:MM:SS.mmm --> HH:MM:SS.mmm)
        if '-->' in line:
            # The next line should contain the subtitle text
            if i + 1 < len(lines):
                text = lines[i + 1].strip()
                if text:  # Only add non-empty subtitles
                    subtitles.append({'Sentence': text})
            i += 2
        else:
            i += 1

    return subtitles


def process_subtitle(idx, text):
    """
    Process a single subtitle: translate and segment.

    Args:
        idx: Subtitle index (1-based)
        text: Chinese subtitle text

    Returns:
        Dictionary with processed data
    """
    audio_filename = f"{idx:04d}_{text}.mp3"
    audio_ref = f"[sound:{audio_filename}]"

    try:
        result = segment_and_translate(text)
        translation = result['translation']
        segmented_words = result['words']

        # Convert segmented words list to string
        if isinstance(segmented_words, list):
            segmented_words_str = ', '.join(segmented_words)
        else:
            segmented_words_str = str(segmented_words)

        return {
            'idx': idx,
            'Sentence': text,
            'audio': audio_ref,
            'translation': translation,
            'segmented_words': segmented_words_str,
            'success': True
        }
    except Exception as e:
        print(f"  Error processing subtitle {idx}: {e}")
        return {
            'idx': idx,
            'Sentence': text,
            'audio': audio_ref,
            'translation': "",
            'segmented_words': "",
            'success': False
        }


def create_csv_from_vtt(vtt_path, output_csv_path, max_workers=5):
    """
    Create a CSV file from VTT subtitles with translations and segmentations.
    Uses concurrent processing to speed up API calls.

    Args:
        vtt_path: Path to the .zh.vtt file
        output_csv_path: Path for the output CSV file
        max_workers: Maximum number of concurrent workers (default: 10)
    """
    print("Parsing VTT file...")
    subtitles = parse_vtt_file(vtt_path)
    print(f"Found {len(subtitles)} subtitles")
    print(f"Processing with {max_workers} concurrent workers...\n")

    # Process all subtitles concurrently
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_subtitle, idx, subtitle['Sentence']): idx
            for idx, subtitle in enumerate(subtitles, start=1)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            result = future.result()
            results[result['idx']] = result
            completed += 1
            status = "✓" if result['success'] else "✗"
            print(f"[{completed}/{len(subtitles)}] {status} {result['Sentence']}")

    print("\nWriting CSV file...")
    # Write results to CSV in order
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Sentence', 'audio', 'translation', 'segmented_words']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for idx in sorted(results.keys()):
            result = results[idx]
            writer.writerow({
                'Sentence': result['Sentence'],
                'audio': result['audio'],
                'translation': result['translation'],
                'segmented_words': result['segmented_words']
            })

    print(f"CSV file created: {output_csv_path}")


if __name__ == "__main__":
    vtt_file = "Learn Intermediate⧸Advanced Chinese.zh.vtt"
    output_csv = "subtitles_with_translations.csv"

    create_csv_from_vtt(vtt_file, output_csv)
