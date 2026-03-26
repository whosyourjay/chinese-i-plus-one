#!/usr/bin/env python3
"""Extract Simplified column from Anki export to known words file."""

import sys

ANKI_EXPORT = 'Selected Notes.txt'
KNOWN_FILE = 'known'


def parse_anki_export(path):
    """Parse tab-separated Anki export, handling embedded newlines."""
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2 and parts[0].strip().isdigit():
                word = parts[1].strip()
                if word:
                    words.append(word)
    return words


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ANKI_EXPORT
    words = parse_anki_export(path)
    with open(KNOWN_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')
    print(f"Wrote {len(words)} words to {KNOWN_FILE}")


if __name__ == '__main__':
    main()
