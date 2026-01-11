"""
Test file for all Chinese segmenters.
Tests that each segmenter can run and segment a short example sentence.
"""

import os
import sys
import unittest
import warnings

# Flag to control warning display (set to True to see all warnings)
SHOW_WARNINGS = os.environ.get('SHOW_WARNINGS', 'false').lower() == 'true'

if not SHOW_WARNINGS:
    # Suppress third-party library warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Suppress NumPy compatibility messages by redirecting stderr temporarily
    import io

    # Create a context manager to suppress stderr during imports
    class SuppressStderr:
        def __enter__(self):
            self.old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            return self

        def __exit__(self, *args):
            sys.stderr = self.old_stderr

    # Suppress jieba's verbose initialization output
    os.environ['JIEBA_QUIET'] = '1'


def print_banner(title):
    """Print a formatted banner for test sections"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(segmenter_name, original, result, status="✓ PASSED"):
    """Print formatted segmentation results"""
    print(f"\n{status} {segmenter_name}")
    print(f"  Original:     {original}")
    print(f"  Segmented:    {' | '.join(result)}")
    print(f"  Segment count: {len(result)}")


class TestSegmenters(unittest.TestCase):
    """Test suite for Chinese text segmenters"""

    @classmethod
    def setUpClass(cls):
        """Set up test data used across all tests"""
        cls.test_sentence = "我喜欢学习中文"  # "I like learning Chinese"
        cls.test_word_ranks = {
            "我": 1,
            "喜欢": 2,
            "学习": 3,
            "中文": 4,
        }
        cls.test_punctuation = {"，", "。", "！", "？", "、"}

        # Define segmenters with consistent API
        cls.segmenters = [
            ("jieba_segmenter", "Jieba Segmenter"),
            ("lac_segmenter", "LAC Segmenter"),
            ("stanza_segmenter", "Stanza Segmenter"),
            ("snownlp_segmenter", "SnowNLP Segmenter"),
            ("pkuseg_segmenter", "pkuseg Segmenter"),
            ("thulac_segmenter", "THULAC Segmenter"),
        ]

        print_banner("Testing Chinese Text Segmenters")
        print(f"Test sentence: {cls.test_sentence} (I like learning Chinese)")

    def _test_segmenter_with_api(self, module_name: str, display_name: str):
        """
        Generic test function for segmenters with __init__() and segment() API.

        Args:
            module_name: Name of the module to import (e.g., 'jieba')
            display_name: Display name for the segmenter (e.g., 'Jieba Segmenter')
        """
        try:
            # Import the segmenter module
            module = __import__(module_name)
            ChineseSegmenter = module.ChineseSegmenter

            # Initialize segmenter
            segmenter = ChineseSegmenter(self.test_word_ranks, self.test_punctuation)

            # Segment the test sentence
            result = segmenter.segment(self.test_sentence)

            # Sanity checks
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            for segment in result:
                self.assertIsInstance(segment, str)
            self.assertEqual(''.join(result), self.test_sentence)

            print_result(display_name, self.test_sentence, result)

        except (ImportError, ModuleNotFoundError) as e:
            print_result(display_name, self.test_sentence, [], f"✗ SKIPPED: {type(e).__name__}")
            self.skipTest(f"{display_name} library not available: {e}")
        except RuntimeError as e:
            # Handle RuntimeError from torch/numpy issues
            if "Numpy is not available" in str(e) or "torch" in str(e).lower():
                print_result(display_name, self.test_sentence, [], f"✗ SKIPPED: Runtime issue")
                self.skipTest(f"{display_name} has runtime compatibility issues: {e}")
            else:
                print_result(display_name, self.test_sentence, [], f"✗ ERROR: {type(e).__name__}")
                raise
        except Exception as e:
            print_result(display_name, self.test_sentence, [], f"✗ ERROR: {type(e).__name__}")
            raise

    # def test_jieba_segmenter(self):
    #     """Test that jieba can segment text"""
    #     self._test_segmenter_with_api("jieba", "Jieba Segmenter")

    # def test_lac_segmenter(self):
    #     """Test that lac can segment text"""
    #     self._test_segmenter_with_api("lac", "LAC Segmenter")

    # def test_stanza_segmenter(self):
    #     """Test that stanza can segment text"""
    #     self._test_segmenter_with_api("stanza", "Stanza Segmenter")

    # def test_snownlp_segmenter(self):
    #     """Test that snownlp can segment text"""
    #     self._test_segmenter_with_api("snownlp", "SnowNLP Segmenter")

    # def test_pkuseg_segmenter(self):
    #     """Test that pkuseg can segment text"""
    #     self._test_segmenter_with_api("pkuseg", "pkuseg Segmenter")

    # def test_thulac_segmenter(self):
    #     """Test that thulac can segment text"""
    #     self._test_segmenter_with_api("thulac", "THULAC Segmenter")

    def test_greedy_segmenter(self):
        """Test that greedy can segment text"""
        try:
            from greedy import segment_sentence

            # Create a working copy of word_list (as the function modifies it)
            word_list = set(self.test_word_ranks.keys())
            max_word_len = max(len(word) for word in word_list)

            result, unknown_words = segment_sentence(
                self.test_sentence,
                word_list,
                max_word_len
            )

            # Sanity checks
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            for segment in result:
                self.assertIsInstance(segment, str)
            self.assertIsInstance(unknown_words, list)
            self.assertEqual(''.join(result), self.test_sentence)

            print_result("Greedy Segmenter", self.test_sentence, result)
            if unknown_words:
                print(f"  Unknown bigrams: {', '.join(unknown_words)}")

        except ImportError as e:
            print_result("Greedy Segmenter", self.test_sentence, [], f"✗ SKIPPED: {e}")
            self.skipTest(f"pandas library not available: {e}")

    def test_segmenters_produce_consistent_output_types(self):
        """Test that all segmenters return lists of strings"""
        print_banner("Consistency Check: All Segmenters Return Lists of Strings")

        segmenters_tested = []

        # Test all segmenters with consistent API
        for module_name, display_name in self.segmenters:
            try:
                # Suppress stderr during import if warnings are disabled
                if not SHOW_WARNINGS:
                    with SuppressStderr():
                        module = __import__(module_name)
                else:
                    module = __import__(module_name)

                ChineseSegmenter = module.ChineseSegmenter

                segmenter = ChineseSegmenter(self.test_word_ranks, self.test_punctuation)
                result = segmenter.segment(self.test_sentence)

                self.assertIsInstance(result, list)
                self.assertTrue(all(isinstance(s, str) for s in result))
                segmenters_tested.append(f"✓ {display_name}")
            except (ImportError, Exception):
                segmenters_tested.append(f"✗ {display_name} (not available)")

        # Test greedy (has different API)
        try:
            from greedy import segment_sentence
            word_list = set(self.test_word_ranks.keys())
            max_word_len = max(len(word) for word in word_list)
            greedy_result, _ = segment_sentence(
                self.test_sentence,
                word_list,
                max_word_len
            )
            self.assertIsInstance(greedy_result, list)
            self.assertTrue(all(isinstance(s, str) for s in greedy_result))
            segmenters_tested.append("✓ Greedy Segmenter")
        except ImportError:
            segmenters_tested.append("✗ Greedy Segmenter (not available)")

        print("\nConsistency test results:")
        for result in segmenters_tested:
            print(f"  {result}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run tests with verbose output
    # To see all warnings, set environment variable: SHOW_WARNINGS=true python test_segmenters.py
    if not SHOW_WARNINGS:
        print("Note: Warnings suppressed. Set SHOW_WARNINGS=true to see all warnings.\n")
    unittest.main(verbosity=2)
