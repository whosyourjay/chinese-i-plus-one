import pandas as pd
from collections import defaultdict
import sys

def load_words(filename):
    """Load words from file into a set"""
    words = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word)
    return words

def load_sentences(filename):
    """Load sentences from iknow CSV file"""
    df = pd.read_csv(filename)
    return df['Sentence'].tolist()

def segment_sentence(sentence, word_list, max_word_len):
    """Segment a sentence using greedy algorithm"""
    words = []
    unknown_words = []
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
        
        # If no word found, take single character and learn the bigram
        if not found_word:
            if i + 1 < len(sentence):
                bigram = sentence[i:i+2]
                unknown_words.append(bigram)
                word_list.add(sentence[i:i+1])  # Add to dictionary for future use
            words.append(sentence[i])
            i += 1
    
    return words, unknown_words

def segment_text(text, words):
    # Calculate maximum word length from word list
    max_word_len = max(len(word) for word in words)
    
    # Segment all sentences and bucket by word count
    buckets = defaultdict(list)
    all_unknown_words = set()
    
    for sentence in text:
        segmented, unknown = segment_sentence(sentence, words, max_word_len)
        buckets[len(segmented)].append((sentence, segmented))
        all_unknown_words.update(unknown)
    
    # Print statistics and examples
    print(f"\nInitial words in dictionary: {len(words) - len(all_unknown_words)}")
    print(f"New words learned: {len(all_unknown_words)}")
    print(f"Final dictionary size: {len(words)}")
    print("\nNew words learned:")
    for word in sorted(all_unknown_words):
        print(word)

def main():
    # Load words from file
    words = load_words('words/all_words')
    text = load_sentences('iknow_table.csv')

    segments = segment_text(text, words)
    print(segments)

if __name__ == "__main__":
    main() 