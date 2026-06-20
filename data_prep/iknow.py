import re
import csv

# Change this to the path of your input file
input_file = 'iknow2'
output_file = 'iknow_table.csv'

# Regular expression to capture the headword and its pinyin (e.g. "一旦 [yídàn]")
headword_re = re.compile(r'^(.*?)\s*\[(.*?)\]$')

entries = []

# Read the file and strip blank lines
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines):
    # Parse headword line
    header_match = headword_re.match(lines[i])
    if not header_match:
        print(f"Warning: Line does not match headword format: {lines[i]}")
        i += 1
        continue
    word = header_match.group(1)
    word_pinyin = header_match.group(2)
    i += 1

    # Next line is the headword's translation
    if i >= len(lines):
        break
    word_translation = lines[i]
    i += 1

    # Collect one or more sentence blocks (each block has 3 lines)
    # Stop if we reach a line that looks like a new headword (matches "text [pinyin]")
    while i < len(lines) and not headword_re.match(lines[i]):
        # Ensure there are 3 lines left for a complete sentence block
        if i + 2 < len(lines):
            sentence = lines[i]         # Example sentence in Chinese
            sentence_pinyin = lines[i+1]  # Sentence pinyin
            sentence_translation = lines[i+2]  # Sentence translation
            entries.append([word, word_pinyin, word_translation,
                          sentence, sentence_pinyin, sentence_translation])
            i += 3
        else:
            # If not enough lines remain for a complete block, break out.
            break

# Write the entries to a CSV file with 6 columns.
with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Word', 'Pinyin', 'Translation',
                    'Sentence', 'Sentence Pinyin', 'Sentence Translation'])
    writer.writerows(entries)

print(f"Extraction complete. {len(entries)} rows written to '{output_file}'.")