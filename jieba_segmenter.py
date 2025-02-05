import pandas as pd
from collections import defaultdict
import jieba
import sys

def load_words(filename):
    """Load words from file into a set and add them to jieba dictionary"""
    words = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word)
                jieba.add_word(word)  # Add word to jieba dictionary
    return words

def load_sentences(filename):
    """Load sentences from iknow CSV file"""
    df = pd.read_csv(filename)
    return df['Sentence'].tolist()

def segment_text(text, words):
    """Segment text using jieba"""
    all_unknown_words = set()
    buckets = defaultdict(list)
    
    for sentence in text:
        # Use jieba to segment the sentence
        segmented = list(jieba.cut(sentence))
        
        # Find unknown words (words not in our dictionary)
        unknown = {word for word in segmented if word not in words and len(word) > 1}
        all_unknown_words.update(unknown)
        
        buckets[len(segmented)].append((sentence, segmented))
    
    # Print statistics
    print(f"\nInitial words in dictionary: {len(words) - len(all_unknown_words)}")
    print(f"New words found: {len(all_unknown_words)}")
    print(f"Final dictionary size: {len(words)}")
    # print("\nNew words found:")
    # for word in sorted(all_unknown_words):
    #     print(word)
    
    return buckets

def main():
    # Load words from file
    words = load_words('words/all_words')
    text = load_sentences('iknow_table.csv')

    segments = segment_text(text, words)
    
    # Print some examples
    print("\nExample segmentations:")
    for bucket in sorted(segments.keys()):
        print(f"\n{bucket} segments, found {len(segments[bucket])} sentences:")
        for orig, seg in segments[bucket][:1]:  # Show 2 examples per bucket
            print(f"Original: {orig}")
            print(f"Segmented: {' '.join(seg)}")

if __name__ == "__main__":
    main() 