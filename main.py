import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from organizer import SentenceOrganizer
from segmenters.openai import segment_and_translate


def load_frequency_data():
    """Load word frequency data from TSV"""
    df = pd.read_csv('words/100k', sep='\t')
    # Convert count to rank (higher count = lower rank)
    sorted_df = df.sort_values('Count', ascending=False)
    sorted_df['Rank'] = range(1, len(sorted_df) + 1)
    return {row['Vocab']: row['Rank'] for _, row in sorted_df.iterrows()}


def enhance_csv_with_segmentation(
    input_csv: str,
    output_csv: str,
    max_workers: int = 5
):
    """
    Read basic CSV and add segmentation and translation columns.
    Appends new sentences to existing output CSV if it exists.

    Args:
        input_csv: Path to basic CSV with Sentence and audio columns
        output_csv: Path to save enhanced CSV
        max_workers: Number of concurrent workers for API calls
    """
    import os

    print(f"Loading basic CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    # Check if output CSV already exists and filter out already processed sentences
    already_processed = set()
    if os.path.exists(output_csv):
        print(f"Output CSV exists, loading to check for already processed sentences...")
        existing_df = pd.read_csv(output_csv)
        already_processed = set(existing_df['Sentence'].tolist())
        print(f"Found {len(already_processed)} already processed sentences")

        # Filter to only new sentences
        df = df[~df['Sentence'].isin(already_processed)]
        print(f"Filtered to {len(df)} new sentences to process")

    print(f"Found {len(df)} sentences to process")

    if len(df) == 0:
        print("No new sentences to process!")
        return pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()

    print(f"Processing with {max_workers} concurrent workers...\n")

    # Process all sentences concurrently
    results = {}

    def process_sentence(idx, sentence):
        """Process a single sentence: translate and segment."""
        try:
            result = segment_and_translate(sentence)
            translation = result['translation']
            segmented_words = result['words']

            # Convert segmented words list to string
            if isinstance(segmented_words, list):
                segmented_words_str = ', '.join(segmented_words)
            else:
                segmented_words_str = str(segmented_words)

            return {
                'idx': idx,
                'translation': translation,
                'segmented_words': segmented_words_str,
                'success': True
            }
        except Exception as e:
            print(f"  Error processing sentence {idx}: {e}")
            return {
                'idx': idx,
                'translation': "",
                'segmented_words': "",
                'success': False
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_sentence, idx, row['Sentence']): idx
            for idx, row in df.iterrows()
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            result = future.result()
            results[result['idx']] = result
            completed += 1
            status = "✓" if result['success'] else "✗"
            sentence = df.loc[result['idx'], 'Sentence']
            print(f"[{completed}/{len(df)}] {status} {sentence}")

    # Add new columns to dataframe
    df['translation'] = df.index.map(lambda idx: results[idx]['translation'])
    df['segmented_words'] = df.index.map(lambda idx: results[idx]['segmented_words'])

    # Append to existing CSV or create new one
    print(f"\nSaving enhanced CSV to: {output_csv}")
    if os.path.exists(output_csv) and len(already_processed) > 0:
        # Append to existing file
        df.to_csv(output_csv, mode='a', header=False, index=False)
        print(f"Appended {len(df)} new sentences to existing CSV")
        # Return combined dataframe
        return pd.concat([existing_df, df], ignore_index=True)
    else:
        # Create new file
        df.to_csv(output_csv, index=False)
        print(f"Created new CSV with {len(df)} sentences")
        return df

    print("Done!")


def run_i_plus_1_selection(
    enhanced_csv: str,
    output_csv: str,
    initial_words_count: int = 6,
    use_known_file: bool = True
):
    """
    Run i+1 sentence selection algorithm on enhanced CSV.

    Args:
        enhanced_csv: Path to CSV with Sentence, audio, translation, segmented_words
        output_csv: Path to save the ordered sentence sequence
        initial_words_count: Number of most frequent words to start with
        use_known_file: Whether to load additional known words from 'known' file
    """
    # Load and process frequency data
    start_time = time.time()
    word_ranks = load_frequency_data()
    sorted_words = sorted(word_ranks.items(), key=lambda x: x[1])
    freq_time = time.time() - start_time
    print(f"Loading frequency data: {freq_time:.2f} seconds")

    # Get top N most frequent words
    initial_words = {word for word, _ in sorted_words[:initial_words_count]}
    print(f"Initial words: {initial_words}")

    # Load sentences
    t1 = time.time()
    df = pd.read_csv(enhanced_csv)
    # Strip <b> and </b> tags from all sentences if present
    df['Sentence'] = df['Sentence'].str.replace('<b>', '', regex=False).str.replace('</b>', '', regex=False)
    load_time = time.time() - t1

    # Create sentence lookup index and pre-segmented data
    t2 = time.time()
    sentence_to_row = {row['Sentence']: row for _, row in df.iterrows()}

    # Parse segmented_words column (comma-separated) into lists
    pre_segmented_data = {}
    for _, row in df.iterrows():
        sentence = row['Sentence']
        segmented_str = row['segmented_words']
        # Split by comma and strip whitespace
        words = [w.strip() for w in segmented_str.split(',') if w.strip()]
        pre_segmented_data[sentence] = words

    index_time = time.time() - t2

    print(f"Loading sentences: {load_time:.2f} seconds")
    print(f"Creating index and segmented data: {index_time:.2f} seconds")

    # Initialize organizer with pre-segmented data
    t3 = time.time()
    organizer = SentenceOrganizer(
        df['Sentence'].tolist(),
        word_ranks,
        pre_segmented_data,
        initial_words,
        use_known_file=use_known_file
    )
    print(f"Initial sentence processing took {time.time() - t3:.2f} seconds")

    print(f"Total unique words in corpus: {len(organizer.all_words)}")
    print(f"Initially known words: {len(initial_words)}")

    # Create sequence data
    sequence_data = []
    sequence_num = 1
    get_next_time = 0

    print(f"Skipped {organizer.skipped_sentences} sentences")

    # Process sentences until we run out
    while True:
        start_time = time.time()
        sentence = organizer.get_next_sentence()
        get_next_time += time.time() - start_time

        if not sentence:
            break

        new_words, segmented = organizer.learn_sentence(sentence)

        # Use index to get row data
        row = sentence_to_row[sentence]
        sequence_data.append({
            'Sequence': sequence_num,
            'Sentence': sentence,
            'New_Words': ', '.join(new_words),
            'Word_Rank': min(organizer.word_ranks.get(w, float('inf')) for w in new_words),
            **row.to_dict()
        })
        sequence_num += 1

    # Create and save new dataframe
    sequence_df = pd.DataFrame(sequence_data)
    sequence_df.to_csv(output_csv, index=False)

    # Count remaining sentences
    remaining = sum(len(bucket) for bucket in organizer.sentence_buckets.values())

    print(f"\nProcessed {len(sequence_data)} sentences")
    print(f"Skipped {organizer.skipped_sentences} sentences")
    print(f"Used {organizer.n2_sentences_used} n+2 sentences")
    print(f"Time spent in get_next_sentence: {get_next_time:.2f} seconds")
    print(f"Time spent in update_buckets: {organizer.update_buckets_time:.2f} seconds")
    print(f"  Collecting sentences: {organizer.collect_sentences_time:.2f} seconds")
    print(f"  Processing sentences: {organizer.process_sentences_time:.2f} seconds")
    print(f"    Bucket removal: {organizer.bucket_remove_time:.2f} seconds")
    print(f"    Unknown word updates: {organizer.unknown_update_time:.2f} seconds")
    print(f"    Bucket addition: {organizer.bucket_add_time:.2f} seconds")
    print(f"Remaining unprocessed: {remaining} sentences")

    if remaining > 0:
        print("\nSentences per bucket:")
        for bucket_size, sentences in sorted(organizer.sentence_buckets.items())[:5]:
            if sentences:
                print(f"{bucket_size} unknown words: {len(sentences)} sentences")

    print(f"\nSentence sequence saved to: {output_csv}")


def main():
    """Main entry point for processing pipeline."""
    import os

    # Configuration
    BASIC_CSV = "data_files/sentences_basic.csv"
    ENHANCED_CSV = "data_files/sentences_enhanced.csv"
    SEQUENCE_CSV = "data_files/sentence_sequence.csv"

    # Step 1: Check if we need to enhance the CSV
    if not os.path.exists(ENHANCED_CSV) or os.path.getmtime(BASIC_CSV) > os.path.getmtime(ENHANCED_CSV):
        print("=" * 60)
        print("STEP 1: Enhancing CSV with segmentation and translation")
        print("=" * 60)
        enhance_csv_with_segmentation(BASIC_CSV, ENHANCED_CSV, max_workers=5)
        print()
    else:
        print(f"Enhanced CSV already exists: {ENHANCED_CSV}")
        print()

    # Step 2: Run i+1 selection
    print("=" * 60)
    print("STEP 2: Running i+1 sentence selection")
    print("=" * 60)
    run_i_plus_1_selection(ENHANCED_CSV, SEQUENCE_CSV, initial_words_count=6, use_known_file=True)


if __name__ == "__main__":
    main()
