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
        self.collect_sentences_time = 0
        self.process_sentences_time = 0
        self.bucket_remove_time = 0
        self.bucket_add_time = 0
        self.unknown_update_time = 0
        
        start_time = time.time()
        
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
        self.all_words = set()
        
        # Initialize segmenter
        t1 = time.time()
        self.segmenter = ChineseSegmenter(word_ranks, PUNCTUATION)
        segmenter_time = time.time() - t1
        
        # Process all sentences
        t2 = time.time()
        for sentence in sentences:
            words = self.segmenter.segment(sentence)
            self.all_words.update(words)
            self._process_sentence(sentence, words)
        process_time = time.time() - t2
        
        total_time = time.time() - start_time
        print(f"\nInitialization timing:")
        print(f"  Segmenter setup: {segmenter_time:.2f} seconds")
        print(f"  Process sentences: {process_time:.2f} seconds")
        print(f"  Total: {total_time:.2f} seconds")
    
    def _process_sentence(self, sentence, words):
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
            
            # Take from front of sorted list and decrement counter
            while self._n1_counter > 0 and bucket:
                self._n1_counter -= 1
                sentence = bucket[0]
                if sentence in self.sentence_data:
                    return sentence
                bucket.pop(0)  # Remove invalid sentence
            
            # If we get here, try again with a fresh sort
            self._n1_counter = 0
            return self.get_next_sentence()
        
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
        
        # Remove the learned sentence from front of bucket
        count = len(new_words)
        if count == 1:
            self.sentence_buckets[1].pop(0)
        else:
            self.sentence_buckets[count].discard(sentence)
        del self.sentence_data[sentence]
        
        # Update all other sentences
        self._update_buckets(new_words)
        
        return new_words, segmented_words
    
    def _update_buckets(self, new_words):
        start_time = time.time()
        
        # Only check sentences that contain the new words
        t1 = time.time()
        sentences_to_check = set()
        for word in new_words:
            sentences_to_check.update(self.word_to_sentences[word])
            del self.word_to_sentences[word]  # Clean up word mapping for learned words
        self.collect_sentences_time += time.time() - t1
        
        t2 = time.time()
        for sentence in sentences_to_check:
            if sentence not in self.sentence_data:
                continue
                
            # Remove from old bucket
            t3 = time.time()
            old_count = len(self.sentence_data[sentence]['unknown'])
            # if old_count == 1:
            #     try:
            #         self.sentence_buckets[1].remove(sentence)
            #     except ValueError:
            #         pass  # Sentence might have been already removed
            if old_count > 1:
                self.sentence_buckets[old_count].discard(sentence)
            self.bucket_remove_time += time.time() - t3
            
            # Update unknown words
            t4 = time.time()
            self.sentence_data[sentence]['unknown'] -= new_words
            unknown = self.sentence_data[sentence]['unknown']
            self.unknown_update_time += time.time() - t4
            
            if not unknown:
                # Remove sentences with no unknown words
                del self.sentence_data[sentence]
                self.skipped_sentences += 1
                continue
            
            # Add to new bucket
            t5 = time.time()
            if len(unknown) == 1:
                self.sentence_buckets[1].append(sentence)
            else:
                self.sentence_buckets[len(unknown)].add(sentence)
            self.bucket_add_time += time.time() - t5
        
        self.process_sentences_time += time.time() - t2
        self.update_buckets_time += time.time() - start_time