import pandas as pd
from collections import defaultdict
import sys

def load_word_list(filename):
    """Load HSK3 words from CSV file"""
    df = pd.read_csv(filename)
    # Assuming first column contains the Chinese words
    return set(df.iloc[:, 0].astype(str).tolist())

def load_sentences(filename):
    """Load sentences from iknow CSV file"""
    df = pd.read_csv(filename)
    return df['Sentence'].tolist()

def segment_sentence(sentence, word_list, max_word_len):
    """Segment a sentence using greedy algorithm"""
    words = []
    i = 0
    while i < len(sentence):
        # Try to find longest possible word starting at current position
        found_word = False
        for j in range(min(i + max_word_len, len(sentence)), i, -1):
            candidate = sentence[i:j]
            if candidate in word_list:
                words.append(candidate)
                i = j
                found_word = True
                break
        
        # If no word found, take single character
        if not found_word:
            # Add debug info to see what words we're missing
            print(f"No word found for: {sentence[i:i+max_word_len]}", file=sys.stderr)
            words.append(sentence[i])
            i += 1
    
    return words

def main():
    # Load data
    word_list = load_word_list('words/hsk3_words')
    sentences = load_sentences('iknow_table.csv')
    
    # Calculate maximum word length from word list
    max_word_len = max(len(word) for word in word_list)
    
    # Segment all sentences and bucket by word count
    buckets = defaultdict(list)
    
    for sentence in sentences:
        segmented = segment_sentence(sentence, word_list, max_word_len)
        buckets[len(segmented)].append((sentence, segmented))
    
    # Print statistics and examples
    print("Segmentation Statistics:")
    print("-----------------------")
    for word_count in sorted(buckets.keys()):
        bucket = buckets[word_count]
        print(f"\nBucket with {word_count} words: {len(bucket)} sentences")
        # Print first example from each bucket
        example = bucket[0]
        print(f"Example sentence: {example[0]}")
        print(f"Segmentation: {' | '.join(example[1])}")

if __name__ == "__main__":
    main() 