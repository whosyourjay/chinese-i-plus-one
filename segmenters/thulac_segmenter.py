import thulac
from typing import List, Set

class ChineseSegmenter:
    def __init__(self, word_ranks: dict, punctuation: Set[str]):
        """Initialize segmenter with known words and punctuation"""
        self.word_ranks = word_ranks
        self.punctuation = punctuation

        # Initialize THULAC with seg_only=True for segmentation only
        # text=False returns list instead of string
        self.thu = thulac.thulac(seg_only=True)

    def segment(self, text: str) -> List[str]:
        """
        Segment text using THULAC.
        Returns list of words with punctuation removed.
        """
        # THULAC returns list of [word, pos_tag] pairs when text=False
        # We only need the words, not the POS tags
        result = self.thu.cut(text, text=False)

        # Extract words from [word, pos_tag] pairs and filter punctuation
        seg_result = [word[0] for word in result if word[0] not in self.punctuation]

        return seg_result
