import csv
import re
from collections import Counter

def clean_word(word):
    # Remove spaces and dots
    word = word.strip()
    word = word.replace('…', '')
    word = word.replace('...', '')
    return word

def load_hsk3(filename):
    """
    HSK 3.0 CSV format (example header):
      HSK Level,No,Simplified,Pinyin,English
    Performs the following cleaning:
      - Remove part-of-speech or usage info (e.g. "白（形）" becomes "白")
      - If the Simplified field contains multiple usages (separated by "｜"),
        split them into separate entries.
      - Remove ellipsis (...) and (…) from words
      - Loads the English translation into a separate column (if needed later)
    """
    words = set()
    with open(filename, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Remove part-of-speech info in full-width parentheses.
            chinese = re.sub(r'（.*?）', '', row['Simplified']).strip()
            if not chinese:
                continue
            # Split into multiple words if there is a separator (full-width vertical bar)
            for part in chinese.split('｜'):
                part = clean_word(part)
                if part:
                    words.add(part)
    return words

def parse_line(line):
    # Split by tab or multiple spaces
    parts = re.split(r'\t|\s{2,}', line.strip())
    if len(parts) < 2:
        return None
        
    word = clean_word(parts[0])
    pinyin = parts[1] if len(parts) > 1 else ''
    english = parts[2] if len(parts) > 2 else ''
    
    return word, pinyin, english

def main():
    hsk3_file = 'words/hsk3_words'
    hsk3_words = load_hsk3(hsk3_file)
    
    # Write cleaned words to output file
    with open('words/all_words', 'w', encoding='utf-8') as f:
        for word in sorted(hsk3_words):
            f.write(word + '\n')

if __name__ == "__main__":
    main()
