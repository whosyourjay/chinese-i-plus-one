import jieba
from typing import List, Set

class ChineseSegmenter:
    def __init__(self, word_ranks: dict, punctuation: Set[str]):
        """Initialize segmenter with known words and punctuation"""
        self.word_ranks = word_ranks
        self.punctuation = punctuation
        
        # Configure jieba for aggressive splitting
        jieba.initialize()
        
        # Add our known words to jieba's dictionary
        for word in word_ranks:
            jieba.add_word(word)
    
    def segment(self, text: str) -> List[str]:
        """
        Segment text, aggressively splitting anything not in our word list.
        Returns list of words with punctuation removed.
        """
        # First pass with HMM=False for aggressive splitting
        initial_segments = [w for w in jieba.cut(text, HMM=False) 
                          if w not in self.punctuation]
        
        #Further split any segments not in our word list
        final_segments = []
        for segment in initial_segments:
            if segment in self.word_ranks or len(segment) == 1:
                final_segments.append(segment)
            else:
                # Split into individual characters
                final_segments.extend(list(segment))
        
        return final_segments 
        # return initial_segments
