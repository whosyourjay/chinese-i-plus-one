import pandas as pd
from collections import defaultdict
import jieba
from chinese_segmenter import ChineseSegmenter

PUNCTUATION = {'。', '，', '？', '！', '、', '：', '；', '"', '"', ''', ''', '（', '）', '【', '】', '《', '》'}

def load_sentences(filename):
    """Load sentences and their metadata from iknow CSV file"""
    df = pd.read_csv(filename)
    return df

def load_frequency_data():
    """Load word frequency data from CSV"""
    df = pd.read_csv('words/frequency.csv')
    return {row['Vocab']: row['Rank'] for _, row in df.iterrows()}

class SentenceOrganizer:
    def __init__(self, sentences, word_ranks, initial_words=None):
        self.known_words = PUNCTUATION.copy()
        if initial_words:
            self.known_words.update(initial_words)
        self.word_ranks = word_ranks
        self.sentence_buckets = defaultdict(list)
        self.sentence_data = {}
        self.skipped_sentences = 0
        self.n2_sentences_used = 0
        
        # Initialize segmenter
        self.segmenter = ChineseSegmenter(word_ranks, PUNCTUATION)
        
        # Count total unique words while initializing buckets
        self.all_words = set()
        for sentence in sentences:
            words = self.segmenter.segment(sentence)
            self.all_words.update(words)
            self._process_sentence(sentence, words)  # Pass already segmented words
    
    def _process_sentence(self, sentence, words=None):
        unknown = {w for w in words if w not in self.known_words}
        
        if not unknown:
            self.skipped_sentences += 1
            return
        
        self.sentence_data[sentence] = {
            'words': words,
            'unknown': unknown,
            'max_rank': min((self.word_ranks.get(w, float('inf')) for w in unknown), default=float('inf'))
        }
        self.sentence_buckets[len(unknown)].append(sentence)
    
    def _update_buckets(self, new_words):
        """Update buckets after learning new words"""
        new_buckets = defaultdict(list)
        
        for sentence in list(self.sentence_data.keys()):
            # Update unknown words
            self.sentence_data[sentence]['unknown'] -= new_words
            unknown = self.sentence_data[sentence]['unknown']
            
            if not unknown:
                # Remove sentences with no unknown words
                del self.sentence_data[sentence]
                self.skipped_sentences += 1
                continue
                
            new_buckets[len(unknown)].append(sentence)
        
        self.sentence_buckets = new_buckets
    
    def get_next_sentence(self):
        """Get the next best sentence to learn"""
        # First try to find a sentence with exactly one unknown word
        if self.sentence_buckets[1]:
            # Sort bucket by rank to get most frequent unknown word
            return min(
                self.sentence_buckets[1],
                key=lambda s: self.sentence_data[s]['max_rank']
            )
        
        # if self.sentence_buckets[2]:
        #     return min(
        #         self.sentence_buckets[2],
        #         key=lambda s: self.sentence_data[s]['max_rank']
        #     )
        
        return None
    
    def learn_sentence(self, sentence):
        """Learn all unknown words from a sentence"""
        # Get data before removing the sentence
        sentence_info = self.sentence_data[sentence]
        new_words = sentence_info['unknown']
        segmented_words = sentence_info['words']
        
        # Track if this was an n+2 sentence
        if len(new_words) == 2:
            self.n2_sentences_used += 1
        
        # Update known words
        self.known_words.update(new_words)
        
        # Remove the learned sentence
        count = len(new_words)
        self.sentence_buckets[count].remove(sentence)
        del self.sentence_data[sentence]
        
        # Update all other sentences
        self._update_buckets(new_words)
        
        return new_words, segmented_words

def main():
    # Load frequency data
    word_ranks = load_frequency_data()
    sorted_words = sorted(word_ranks.items(), key=lambda x: x[1])
    
    # Get top 10 most frequent words
    initial_words = {word for word, _ in sorted_words[:10]}

    # Process sentences, starting with top 10 words as known
    df = load_sentences('iknow_table.csv')
    organizer = SentenceOrganizer(df['Sentence'].tolist(), word_ranks, initial_words)
    
    print(f"Total unique words in corpus: {len(organizer.all_words)}")
    print(f"Initially known words: {len(initial_words)}")
    
    # Create sequence data
    sequence_data = []
    sequence_num = 1
    
    print(f"Skipped {organizer.skipped_sentences} sentences")
    while True:
        sentence = organizer.get_next_sentence()
        if not sentence:
            break
            
        new_words, segmented = organizer.learn_sentence(sentence)
        
        # Find matching row in original dataframe
        row = df[df['Sentence'] == sentence].iloc[0]
        sequence_data.append({
            'Sequence': sequence_num,
            'Sentence': sentence,
            'New_Words': ', '.join(new_words),
            'Word_Rank': min(organizer.word_ranks.get(w, float('inf')) for w in new_words),
            **row.to_dict()  # Include all original columns
        })
        sequence_num += 1
    
    # Create and save new dataframe
    sequence_df = pd.DataFrame(sequence_data)
    sequence_df.to_csv('iknow_sequence.csv', index=False)
    
    # Count remaining sentences
    remaining = sum(len(bucket) for bucket in organizer.sentence_buckets.values())
    
    print(f"Processed {len(sequence_data)} sentences")
    print(f"Skipped {organizer.skipped_sentences} sentences")
    print(f"Used {organizer.n2_sentences_used} n+2 sentences")
    print(f"Remaining unprocessed: {remaining} sentences")
    if remaining > 0:
        print("\nSentences per bucket:")
        for bucket_size, sentences in sorted(organizer.sentence_buckets.items()):
            if sentences:
                print(f"{bucket_size} unknown words: {len(sentences)} sentences")

if __name__ == "__main__":
    main() 