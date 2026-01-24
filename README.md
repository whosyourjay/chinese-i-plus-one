# Chinese Incremental Learning System

A system for organizing Chinese learning materials using i+1 sentence selection across multiple videos.

## Overview

The system processes YouTube videos with Chinese subtitles to create an optimal learning sequence. It selects sentences that introduce exactly the right amount of new vocabulary based on what you already know, then incrementally builds your vocabulary across multiple videos.

## Quick Start

```bash
# Process a single video
python prepare_vtt_data.py  # Configure video URL in the file first
python main.py

# Process multiple videos in sequence
python process_videos.py video_urls.txt
```

## Processing Multiple Videos

`process_videos.py` is the main entry point for building a learning path across multiple videos:

1. **Input**: A text file with one YouTube URL per line (lines starting with `#` are ignored)
2. **Loop**: For each video, it:
   - Downloads subtitles and splits audio into sentence segments
   - Adds word segmentation and English translations
   - Runs i+1 selection to pick the best sentences (typically 6 sentences per video)
   - Generates audio files for the selected sentences
   - Appends results to `data_files/all_sentences.csv`
   - **Adds new words to the `known` file** so the next video builds on this vocabulary
3. **Output**: A cumulative CSV with the optimal sentence learning sequence across all videos

This creates an incremental learning path where each video introduces new vocabulary while reinforcing what you've already learned.

## Pipeline Steps (per video)

1. **VTT Processing** (`prepare_vtt_data.py`): Downloads subtitles, splits audio by sentence
2. **Enhancement** (`enhance_csv.py`): Adds word segmentation and translations via OpenAI API
3. **Selection** (`selection.py`): Picks sentences using i+1 algorithm (new words + known words)
4. **Audio Generation** (`generate_audio.py`): Creates audio clips for sentences, generates TTS for target words, adds pinyin

## Directory Structure

- `data_files/` - All CSVs, audio files, and VTT subtitles
- `audio_segments/` - Individual audio clips for each sentence
- `words/100k` - Word frequency data (TSV with `Vocab` and `Count` columns)
- `known` - List of known words (one per line), automatically updated after each video

## Configuration

- **API Key**: Create `api.key` file with OpenAI-compatible API key
- **Known Words**: Optional `known` file with pre-existing vocabulary
- **Video URLs**: Create a text file listing YouTube URLs (see `video_urls.txt`)

## Key Components

- `process_videos.py` - Main loop for processing multiple videos
- `prepare_vtt_data.py` - Downloads and processes subtitles
- `enhance_csv.py` - Adds segmentation and translations
- `selection.py` - i+1 sentence selection algorithm (formerly `organizer.py`)
- `generate_audio.py` - Creates audio clips for selected sentences
- `segmenters/` - Various Chinese segmentation implementations (OpenAI, jieba, pkuseg, etc.)

## Output Files

- `data_files/all_sentences.csv` - Cumulative learning sequence from all videos
- `data_files/sentence_sequence.csv` - Results from the most recent video
- Columns: `Sentence`, `audio`, `translation`, `segmented_words`, `Sequence`, `New_Words`, `Word_Rank`, `word_audio`, `word_pinyin`
