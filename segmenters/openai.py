from openai import OpenAI
import json
from typing import List, Set, Union, Dict

endpoint = "https://segmentor-resource.cognitiveservices.azure.com/openai/v1/"
model_name = "gpt-5-nano"
deployment_name = "gpt-5-nano-segmentor"

with open('api.key', 'r') as f:
    api_key = f.read().strip()

client = OpenAI(
    base_url=f"{endpoint}",
    api_key=api_key
)


def segment_and_translate(sentence: str) -> dict:
    """
    Segments a Chinese sentence into words and translates the overall sentence into English.

    Args:
        sentence: The Chinese sentence to segment and translate

    Returns:
        A dictionary with keys:
        - 'words': list of segmented Chinese words
        - 'translation': English translation of the sentence
    """
    messages = [
        {
            "role": "user",
            "content": f"Separate the following Chinese sentence into individual words. Output as a json array. {sentence}"
        },
        {
            "role": "user",
            "content": f"Translate the following Chinese sentence to English. Return ONLY one translation - pick the best option. Do not provide alternatives or multiple options. Just output the single best English translation: {sentence}"
        }
    ]

    # Get word segmentation
    segmentation_response = client.chat.completions.create(
        model=deployment_name,
        messages=[messages[0]],
    )
    words_result = segmentation_response.choices[0].message.content

    # Parse the JSON array response
    try:
        words = json.loads(words_result)
    except json.JSONDecodeError:
        words = words_result  # Fallback if not valid JSON

    # Get translation
    translation_response = client.chat.completions.create(
        model=deployment_name,
        messages=[messages[1]],
    )
    translation = translation_response.choices[0].message.content

    # Clean up translation - force single sentence
    # Remove any lines after the first line (alternatives)
    translation = translation.strip().split('\n')[0]
    # Remove parenthetical alternatives like "(Alternative: ...)"
    translation = translation.split('(Optional')[0].strip()
    translation = translation.strip('"').strip()

    return {
        'words': words,
        'translation': translation
    }


def translate_word_in_context(word: str, sentence: str) -> str:
    """
    Translates a word in the context of a given sentence.

    Args:
        word: The Chinese word to translate
        sentence: The Chinese sentence containing the word

    Returns:
        A single word or short phrase translation that matches the word's usage in the sentence
    """
    message = {
        "role": "user",
        "content": f"In the sentence {sentence}，define the word {word}. Give a single word or short phrase as appropriate. Try to make the definition match the part of speech of the word in this sentence."
    }

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[message],
    )

    return response.choices[0].message.content


class ChineseSegmenter:
    """
    Segmenter that uses OpenAI API for word segmentation and optional translation.
    Compatible with the standard segmenter API used by organizer.py.
    """
    def __init__(self, word_ranks: dict = None, punctuation: Set[str] = None):
        """Initialize segmenter (word_ranks and punctuation kept for API compatibility)"""
        self.word_ranks = word_ranks
        self.punctuation = punctuation or set()

    def segment(self, text: str, include_translation: bool = False) -> Union[List[str], Dict[str, any]]:
        """
        Segment text using OpenAI API, optionally including translation.

        Args:
            text: The Chinese sentence to segment
            include_translation: If True, return dict with 'words' and 'translation'.
                               If False, return just the list of words.

        Returns:
            If include_translation is False: list of segmented Chinese words
            If include_translation is True: dict with keys 'words' and 'translation'
        """
        if include_translation:
            return segment_and_translate(text)
        else:
            # Just get segmentation without translation
            result = segment_and_translate(text)
            return result['words']


# Example usage
if __name__ == "__main__":
    test_sentence = "少女的话席卷周围，"
    test_word = "席卷"

    # Test segment and translate
    print("=== Segment and Translate ===")
    result = segment_and_translate(test_sentence)
    print(f"Words: {result['words']}")
    print(f"Translation: {result['translation']}")
    print()

    # Test word translation in context
    print("=== Word in Context ===")
    word_translation = translate_word_in_context(test_word, test_sentence)
    print(f"Word '{test_word}' in context: {word_translation}")
