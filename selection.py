import os
import time

import pandas as pd
from pinyin_jyutping_sentence import pinyin

from organizer import SentenceOrganizer


def load_frequency_data():
    """Load word frequency data from CSV"""
    df = pd.read_csv('words/100k', sep='\t')
    sorted_df = df.sort_values('Count', ascending=False)
    sorted_df['Rank'] = range(1, len(sorted_df) + 1)
    return {row['Vocab']: row['Rank'] for _, row in sorted_df.iterrows()}


def load_and_prepare_data(enhanced_csv, initial_words_count):
    """Load frequency data, sentences, and prepare segmented data."""
    word_ranks = load_frequency_data()
    initial_words = {word for word, _ in sorted(word_ranks.items(), key=lambda x: x[1])[:initial_words_count]}
    print(f"Initial words: {initial_words}")

    df = pd.read_csv(enhanced_csv)
    df['Sentence'] = df['Sentence'].str.replace('<b>', '', regex=False).str.replace('</b>', '', regex=False)

    sentence_to_row = {row['Sentence']: row for _, row in df.iterrows()}
    pre_segmented_data = {
        row['Sentence']: [w.strip() for w in row['segmented_words'].split(',') if w.strip()]
        for _, row in df.iterrows()
        if isinstance(row['segmented_words'], str)
    }

    return word_ranks, initial_words, df, sentence_to_row, pre_segmented_data


def generate_sequence(organizer, sentence_to_row):
    """Generate sentence sequence using i+1 algorithm."""
    sequence_data = []
    sequence_num = 1

    while True:
        sentence = organizer.get_next_sentence()
        if not sentence:
            break

        new_words, _ = organizer.learn_sentence(sentence)
        row = sentence_to_row[sentence]
        sequence_data.append({
            'Sequence': sequence_num,
            'Sentence': sentence,
            'New_Words': ', '.join(new_words),
            'Word_Rank': min(organizer.word_ranks.get(w, float('inf')) for w in new_words),
            **row.to_dict()
        })
        sequence_num += 1

    return sequence_data


def add_pinyin_columns(df):
    """Add pinyin columns for sentences and new words."""
    print("\nAdding pinyin...")
    df['sentence_pinyin'] = df['Sentence'].apply(pinyin)
    df['new_words_pinyin'] = df['New_Words'].apply(
        lambda x: ', '.join([pinyin(w.strip()) for w in x.split(',') if w.strip()])
    )
    return df


def print_summary(organizer, sequence_data):
    """Print summary statistics."""
    remaining = sum(len(bucket) for bucket in organizer.sentence_buckets.values())
    print(f"\nProcessed {len(sequence_data)} sentences")
    print(f"Skipped {organizer.skipped_sentences} sentences")
    print(f"Remaining unprocessed: {remaining} sentences")


def run_i_plus_1_selection(enhanced_csv, output_csv, initial_words_count=6, use_known_file=True):
    """Run i+1 sentence selection algorithm and add pinyin."""
    word_ranks, initial_words, df, sentence_to_row, pre_segmented_data = load_and_prepare_data(
        enhanced_csv, initial_words_count
    )

    organizer = SentenceOrganizer(
        df['Sentence'].tolist(),
        word_ranks,
        pre_segmented_data,
        initial_words,
        use_known_file=use_known_file
    )

    sequence_data = generate_sequence(organizer, sentence_to_row)
    sequence_df = pd.DataFrame(sequence_data)
    sequence_df = add_pinyin_columns(sequence_df)
    sequence_df.to_csv(output_csv, index=False)

    print_summary(organizer, sequence_data)
    print(f"Sentence sequence saved to: {output_csv}")


if __name__ == "__main__":
    """Run i+1 sentence selection and add pinyin."""
    ENHANCED_CSV = "data_files/sentences_enhanced.csv"
    SEQUENCE_CSV = "data_files/sentence_sequence.csv"

    run_i_plus_1_selection(ENHANCED_CSV, SEQUENCE_CSV, initial_words_count=6, use_known_file=True)
