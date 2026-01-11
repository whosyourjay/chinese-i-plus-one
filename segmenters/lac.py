from LAC import LAC
from typing import List, Set

class ChineseSegmenter:
    def __init__(self, word_ranks: dict, punctuation: Set[str]):
        self.lac = LAC(mode="seg")
        """Initialize segmenter with known words and punctuation"""
    
    def segment(self, text: str) -> List[str]:
        seg_result = lac.run(text)
        return seg_result
