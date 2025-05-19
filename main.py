import pandas as pd
from organizer import SentenceOrganizer
import time
from line_profiler import profile

def load_frequency_data():
    """Load word frequency data from CSV"""
    df = pd.read_csv('words/frequency.csv')
    return {row['Vocab']: row['Rank'] for _, row in df.iterrows()}

@profile
def main():
    # Load and process frequency data
    start_time = time.time()
    word_ranks = load_frequency_data()
    sorted_words = sorted(word_ranks.items(), key=lambda x: x[1])
    freq_time = time.time() - start_time
    print(f"Loading frequency data: {freq_time:.2f} seconds")

    # Get top 5 most frequent words
    initial_words = {word for word, _ in sorted_words[:6]}
    print(f"Initial words: {initial_words}")

    # Load sentences
    t1 = time.time()
    df = pd.read_csv('iknow_table.csv', sep=',', index_col=False)
    # Strip <b> and </b> tags from all sentences and sentence pinyin
    df['Sentence'] = df['Sentence'].str.replace('<b>', '', regex=False).str.replace('</b>', '', regex=False)
    df['Sentence pinyin'] = df['Sentence pinyin'].str.replace('<b>', '', regex=False).str.replace('</b>', '', regex=False)
    load_time = time.time() - t1

    # Create sentence lookup index
    t2 = time.time()
    sentence_to_row = {row['Sentence']: row for _, row in df.iterrows()}
    index_time = time.time() - t2

    print(f"Loading sentences: {load_time:.2f} seconds")
    print(f"Creating index: {index_time:.2f} seconds")

    # Initialize organizer with known words flag
    t3 = time.time()
    organizer = SentenceOrganizer(df['Sentence'].tolist(), word_ranks, initial_words, use_known_file=True)
    print(f"Initial sentence processing took {time.time() - t3:.2f} seconds")
    
    print(f"Total unique words in corpus: {len(organizer.all_words)}")
    print(f"Initially known words: {len(initial_words)}")
    
    # Create sequence data
    sequence_data = []
    sequence_num = 1
    get_next_time = 0
    
    print(f"Skipped {organizer.skipped_sentences} sentences")
    for _ in range(11092 - 3111):
        start_time = time.time()
        sentence = organizer.get_next_sentence()
        get_next_time += time.time() - start_time
        
        if not sentence:
            break
            
        new_words, segmented = organizer.learn_sentence(sentence)
        
        # Use index instead of searching
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
    sequence_df.to_csv('sentence_sequence.csv', index=False, sep='\t')
    
    # Count remaining sentences
    remaining = sum(len(bucket) for bucket in organizer.sentence_buckets.values())
    
    print(f"Processed {len(sequence_data)} sentences")
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

if __name__ == "__main__":
    main()
