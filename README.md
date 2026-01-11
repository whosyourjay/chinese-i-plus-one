# Chinese Incremental Learning System

A system for organizing Chinese learning materials using i+1 sentence selection.

## Directory Structure

- `data_files/` - Contains all input data and generated CSVs
  - Audio files (.mp3)
  - VTT subtitle files (.vtt)
  - TSV data files (.tsv)
  - Generated CSV files
- `audio_segments/` - Individual audio clips for each sentence
- `segmenters/` - Different Chinese segmentation implementations
- `words/` - Word frequency data

## Workflow

### Step 1: Prepare VTT Data
Use `prepare_vtt_data.py` to process VTT files:
- Splits audio into individual sentence segments
- Creates basic CSV with `Sentence` and `audio` columns
- Saves CSV to `data_files/`

```bash
python prepare_vtt_data.py
```

Edit the file to configure:
- `AUDIO_FILE` - Path to audio file in data_files
- `VTT_FILE` - Path to VTT subtitle file in data_files
- `OUTPUT_CSV` - Output path for basic CSV
- `AUDIO_OUTPUT_DIR` - Where to save audio segments (default: audio_segments)

### Step 2: Run Main Pipeline
Use `main.py` to enhance data and run i+1 selection:

```bash
python main.py
```

This will:
1. Read basic CSV from `data_files/sentences_basic.csv`
2. Add segmentation and translation using `segmenters/openai.py`
3. Save enhanced CSV to `data_files/sentences_enhanced.csv`
4. Run i+1 sentence selection algorithm
5. Save ordered sequence to `data_files/sentence_sequence.csv`

The enhanced CSV will include:
- `Sentence` - Chinese sentence
- `audio` - Anki-style audio reference
- `translation` - English translation
- `segmented_words` - Comma-separated Chinese words

The sequence CSV additionally includes:
- `Sequence` - Learning order
- `New_Words` - Words to learn in this sentence
- `Word_Rank` - Frequency rank of new words

## Configuration

### API Key
The system requires an OpenAI-compatible API key in `api.key` file.

### Word Frequency
Word frequency data should be in `words/100k` (TSV format with `Vocab` and `Count` columns).

### Known Words
Optionally create a `known` file with one word per line to mark words as already known.

## Components

- `prepare_vtt_data.py` - VTT processing and audio splitting
- `main.py` - Main pipeline orchestration
- `organizer.py` - i+1 sentence selection algorithm
- `segmenters/openai.py` - Segmentation and translation using AI
- `segmenters/` - Alternative segmentation implementations
  - `pkuseg.py`
  - `jieba.py`
  - `greedy.py`
  - And more...

## Old Files (for reference)

- `process_vtt_to_csv.py` - Old combined approach (now split into prepare + main)
- `split_audio_by_vtt.py` - Old audio splitting only (functionality now in prepare_vtt_data.py)
