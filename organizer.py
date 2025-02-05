import pandas as pd
from collections import defaultdict
import jieba
from jieba_segmenter import ChineseSegmenter
import time

PUNCTUATION = {'。', '，', '？', '！', '、', '：', '；', '"', '"', ''', ''', '（', '）', '【', '】', '《', '》'}

def load_frequency_data():
    """Load word frequency data from CSV"""
    df = pd.read_csv('words/frequency.csv')
    return {row['Vocab']: row['Rank'] for _, row in df.iterrows()}

class SentenceOrganizer:
    def __init__(self, sentences, word_ranks, initial_words=None):
        self.update_buckets_time = 0
        self.known_words_update_time = 0
        self.remove_sentence_time = 0
        self.known_words = PUNCTUATION.copy()
        if initial_words:
            self.known_words.update(initial_words)
        self.word_ranks = word_ranks
        self.sentence_buckets = defaultdict(set)  # Most buckets are sets
        self.sentence_buckets[1] = []  # Special n+1 bucket as list for sorting
        self.sentence_data = {}
        self.word_to_sentences = defaultdict(set)
        self.skipped_sentences = 0
        self.n2_sentences_used = 0
        
        # Initialize segmenter
        self.segmenter = ChineseSegmenter(word_ranks, PUNCTUATION)
        
        # Count total unique words while initializing buckets
        self.all_words = set()
        for sentence in sentences:
            words = self.segmenter.segment(sentence)
            self.all_words.update(words)
            self._process_sentence(sentence, words)
    
    def _process_sentence(self, sentence, words=None):
        unknown = {w for w in words if w not in self.known_words}
        
        if not unknown:
            self.skipped_sentences += 1
            return
        
        self.sentence_data[sentence] = {
            'words': words,
            'unknown': unknown,
            'max_rank': max((self.word_ranks.get(w, float('inf')) for w in unknown), default=float('inf'))
        }
        
        # Special handling for n+1 bucket
        if len(unknown) == 1:
            self.sentence_buckets[1].append(sentence)
        else:
            self.sentence_buckets[len(unknown)].add(sentence)
        
        # Add sentence to word mapping
        for word in words:
            self.word_to_sentences[word].add(sentence)
    
    def _update_buckets(self, new_words):
        start_time = time.time()
        
        # Only check sentences that contain the new words
        t1 = time.time()
        sentences_to_check = set()
        for word in new_words:
            sentences_to_check.update(self.word_to_sentences[word])
            del self.word_to_sentences[word]  # Clean up word mapping for learned words
        self.collect_sentences_time = getattr(self, 'collect_sentences_time', 0) + (time.time() - t1)
        
        t2 = time.time()
        for sentence in sentences_to_check:
            if sentence not in self.sentence_data:
                continue
                
            # Remove from old bucket
            old_count = len(self.sentence_data[sentence]['unknown'])
            if old_count == 1:
                try:
                    self.sentence_buckets[1].remove(sentence)
                except ValueError:
                    pass  # Sentence might have been already removed
            else:
                self.sentence_buckets[old_count].discard(sentence)
            
            # Update unknown words
            self.sentence_data[sentence]['unknown'] -= new_words
            unknown = self.sentence_data[sentence]['unknown']
            
            if not unknown:
                # Remove sentences with no unknown words
                del self.sentence_data[sentence]
                self.skipped_sentences += 1
                continue
            
            # Add to new bucket
            if len(unknown) == 1:
                self.sentence_buckets[1].append(sentence)
            else:
                self.sentence_buckets[len(unknown)].add(sentence)
        self.process_sentences_time = getattr(self, 'process_sentences_time', 0) + (time.time() - t2)
                
        self.update_buckets_time += time.time() - start_time
    
    def get_next_sentence(self):
        """Get the next best sentence to learn"""
        if self.sentence_buckets[1]:
            bucket = self.sentence_buckets[1]
            
            # If we don't have a counter or have reached it, clean and resort
            if not hasattr(self, '_n1_counter') or self._n1_counter <= 0:
                # Clean the bucket of any fully known sentences
                new_bucket = []
                for sentence in bucket:
                    if sentence in self.sentence_data:
                        new_bucket.append(sentence)
                bucket = new_bucket
                self.sentence_buckets[1] = bucket
                
                if bucket:
                    bucket.sort(key=lambda s: self.sentence_data[s]['max_rank'])
                    self._n1_counter = (len(bucket) + 1) // 2
                else:
                    return None
            
            # Skip any sentences that are now fully known
            while self._n1_counter > 0 and bucket:
                self._n1_counter -= 1
                sentence = bucket[0]
                if sentence in self.sentence_data:
                    return sentence
                bucket.pop(0)
            
            return self.get_next_sentence()  # Recursively try again if we didn't find a valid sentence
        
        return None
    
    def learn_sentence(self, sentence):
        """Learn all unknown words from a sentence"""
        start_time = time.time()
        
        # Get data before removing the sentence
        sentence_info = self.sentence_data[sentence]
        new_words = sentence_info['unknown']
        segmented_words = sentence_info['words']
        
        # Track if this was an n+2 sentence
        if len(new_words) == 2:
            self.n2_sentences_used += 1
        
        t1 = time.time()
        # Update known words
        self.known_words.update(new_words)
        self.known_words_update_time += time.time() - t1
        
        # Remove the learned sentence from front of bucket
        t2 = time.time()
        count = len(new_words)
        if count == 1:
            self.sentence_buckets[1].pop(0)  # Remove from front since we know it's the one we just processed
        else:
            self.sentence_buckets[count].discard(sentence)
        del self.sentence_data[sentence]
        self.remove_sentence_time += time.time() - t2
        
        # Update all other sentences
        t3 = time.time()
        self._update_buckets(new_words)
        self.update_buckets_time += time.time() - t3
        
        return new_words, segmented_words

def main():
    # Load frequency data
    word_ranks = load_frequency_data()
    sorted_words = sorted(word_ranks.items(), key=lambda x: x[1])
    
    # Get top 10 most frequent words
    initial_words = {word for word, _ in sorted_words[:5]}
    print(initial_words)

    # Process sentences
    start_time = time.time()
    
    df = pd.read_csv('rezero_v1-8.tsv', sep='\t', index_col=False)
    # Create sentence lookup index
    sentence_to_row = {row['Sentence']: row for _, row in df.iterrows()}
    
    organizer = SentenceOrganizer(df['Sentence'].tolist(), word_ranks, initial_words)
    
    processing_time = time.time() - start_time
    print(f"\nInitial sentence processing took {processing_time:.2f} seconds")
    
    print(f"Total unique words in corpus: {len(organizer.all_words)}")
    print(f"Initially known words: {len(initial_words)}")
    
    # Create sequence data
    sequence_data = []
    sequence_num = 1
    get_next_time = 0
    
    print(f"Skipped {organizer.skipped_sentences} sentences")
    for _ in range(11092):
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
    sequence_df.to_csv('sentence_sequence.csv', index=False)
    
    # Count remaining sentences
    remaining = sum(len(bucket) for bucket in organizer.sentence_buckets.values())
    
    print(f"Processed {len(sequence_data)} sentences")
    print(f"Skipped {organizer.skipped_sentences} sentences")
    print(f"Used {organizer.n2_sentences_used} n+2 sentences")
    print(f"Time spent in get_next_sentence: {get_next_time:.2f} seconds")
    print(f"Time spent in update_buckets: {organizer.update_buckets_time:.2f} seconds")
    print(f"  Collecting sentences: {organizer.collect_sentences_time:.2f} seconds")
    print(f"  Processing sentences: {organizer.process_sentences_time:.2f} seconds")
    print(f"Remaining unprocessed: {remaining} sentences")
    if remaining > 0:
        print("\nSentences per bucket:")
        for bucket_size, sentences in sorted(organizer.sentence_buckets.items())[:5]:
            if sentences:
                print(f"{bucket_size} unknown words: {len(sentences)} sentences")

if __name__ == "__main__":
    main() 