from snownlp import SnowNLP
from typing import List, Set

class ChineseSegmenter:
    def __init__(self, word_ranks: dict, punctuation: Set[str]):
        """Initialize segmenter with known words and punctuation"""
        self.word_ranks = word_ranks
        self.punctuation = punctuation

    def segment(self, text: str) -> List[str]:
        """
        Segment text using SnowNLP.
        Returns list of words with punctuation removed.
        """
        s = SnowNLP(text)
        seg_result = [word for word in s.words if word not in self.punctuation]

        return seg_result
