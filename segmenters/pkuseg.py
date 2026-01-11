import pkuseg
from typing import List, Set

class ChineseSegmenter:
    def __init__(self, word_ranks: dict, punctuation: Set[str]):
        """Initialize segmenter with known words and punctuation"""
        self.word_ranks = word_ranks
        self.punctuation = punctuation

        # Initialize pkuseg with default model
        # Can also use domain-specific models: 'news', 'web', 'medicine', 'tourism'
        self.seg = pkuseg.pkuseg()

    def segment(self, text: str) -> List[str]:
        """
        Segment text using pkuseg.
        Returns list of words with punctuation removed.
        """
        result = self.seg.cut(text)
        seg_result = [word for word in result if word not in self.punctuation]

        return seg_result
