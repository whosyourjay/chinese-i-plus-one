import stanza
from typing import List, Set
import os

class ChineseSegmenter:
    def __init__(self, word_ranks: dict, punctuation: Set[str]):
        """Initialize segmenter with known words and punctuation"""
        self.word_ranks = word_ranks
        self.punctuation = punctuation

        # Initialize Stanza pipeline for tokenization
        # Suppress download messages and use CPU
        try:
            # Check if model is downloaded, download if not
            stanza_dir = os.path.expanduser('~/stanza_resources')
            if not os.path.exists(os.path.join(stanza_dir, 'zh')):
                stanza.download('zh', processors='tokenize', verbose=False)

            # Initialize with use_gpu=False to avoid torch issues
            self.nlp = stanza.Pipeline(
                'zh',
                processors='tokenize',
                verbose=False,
                use_gpu=False,
                download_method=None  # Don't try to download again
            )
        except Exception as e:
            raise ImportError(f"Failed to initialize Stanza: {e}")

    def segment(self, text: str) -> List[str]:
        """
        Segment text using Stanza tokenizer.
        Returns list of words with punctuation removed.
        """
        doc = self.nlp(text)
        seg_result = []

        for sentence in doc.sentences:
            for word in sentence.words:
                if word.text not in self.punctuation:
                    seg_result.append(word.text)

        return seg_result
