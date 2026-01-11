import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pkuseg

# Initialize pkuseg once at module level for efficiency
seg = pkuseg.pkuseg()


def process_sentence(idx, sentence):
    """Process a single sentence: segment with pkuseg."""
    try:
        segmented_words = seg.cut(sentence)
        if isinstance(segmented_words, list):
            segmented_words = ', '.join(segmented_words)
        return {
            'idx': idx,
            'translation': '',
            'segmented_words': segmented_words,
            'success': True
        }
    except Exception as e:
        print(f"  Error processing sentence {idx}: {e}")
        return {'idx': idx, 'translation': "", 'segmented_words': "", 'success': False}


def load_and_filter_sentences(input_csv, output_csv):
    """Load sentences and filter out already processed ones."""
    df = pd.read_csv(input_csv)
    existing_df = None

    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        already_processed = set(existing_df['Sentence'].tolist())
        df = df[~df['Sentence'].isin(already_processed)]
        print(f"Found {len(already_processed)} already processed, {len(df)} new sentences")

    return df, existing_df


def process_sentences_concurrent(df, max_workers):
    """Process sentences concurrently and return results."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_sentence, idx, row['Sentence']): idx
            for idx, row in df.iterrows()
        }

        completed = 0
        for future in as_completed(future_to_idx):
            result = future.result()
            results[result['idx']] = result
            completed += 1
            status = "✓" if result['success'] else "✗"
            if completed % 100 == 0:
                print(f"[{completed}/{len(df)}] {status} {df.loc[result['idx'], 'Sentence']}")

    return results


def save_enhanced_csv(df, existing_df, output_csv):
    """Save enhanced CSV, appending if necessary."""
    if existing_df is not None:
        df.to_csv(output_csv, mode='a', header=False, index=False)
        print(f"Appended {len(df)} sentences")
        return pd.concat([existing_df, df], ignore_index=True)
    else:
        df.to_csv(output_csv, index=False)
        print(f"Created CSV with {len(df)} sentences")
        return df


def enhance_csv_with_segmentation(input_csv, output_csv, max_workers=5):
    """Add segmentation and translation columns. Appends to existing CSV."""
    print(f"Loading: {input_csv}")
    df, existing_df = load_and_filter_sentences(input_csv, output_csv)

    if len(df) == 0:
        print("No new sentences to process!")
        return pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()

    print(f"Processing {len(df)} sentences with {max_workers} workers...\n")
    results = process_sentences_concurrent(df, max_workers)

    # Add new columns
    df['translation'] = df.index.map(lambda idx: results[idx]['translation'])
    df['segmented_words'] = df.index.map(lambda idx: results[idx]['segmented_words'])

    return save_enhanced_csv(df, existing_df, output_csv)

if __name__ == "__main__":
    """Run i+1 sentence selection and add pinyin."""
    BASIC_CSV = "data_files/sentences_basic.csv"
    ENHANCED_CSV = "data_files/sentences_enhanced.csv"

    enhance_csv_with_segmentation(BASIC_CSV, ENHANCED_CSV, max_workers=5)
