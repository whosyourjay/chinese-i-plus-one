#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_list_file>"
    echo "Example: $0 video_urls.txt"
    exit 1
fi

VIDEO_LIST_FILE=$1
OUTPUT_FILE="all_sentences.csv"

# Check if the input file exists
if [ ! -f "$VIDEO_LIST_FILE" ]; then
    echo "Error: File '$VIDEO_LIST_FILE' not found"
    exit 1
fi

# Initialize the output file with header (if starting fresh)
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Initializing $OUTPUT_FILE..."
    touch "$OUTPUT_FILE"
fi

# Process each video URL
line_num=0
while IFS= read -r video_url || [ -n "$video_url" ]; do
    # Skip empty lines and comments
    if [ -z "$video_url" ] || [[ "$video_url" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    line_num=$((line_num + 1))
    echo ""
    echo "=========================================="
    echo "Processing video $line_num: $video_url"
    echo "=========================================="

    # Step 1: Run prepare_vtt_data.py
    echo "Step 1: Preparing VTT data..."
    time python3 prepare_vtt_data.py "$video_url"
    if [ $? -ne 0 ]; then
        echo "Error in prepare_vtt_data.py for $video_url"
        continue
    fi

    # Step 2: Remove sentences_enhanced.csv
    echo "Step 2: Removing old enhanced CSV..."
    rm -f data_files/sentences_enhanced.csv

    # Step 3: Run enhance_csv.py
    echo "Step 3: Enhancing CSV..."
    time python3 enhance_csv.py
    if [ $? -ne 0 ]; then
        echo "Error in enhance_csv.py for $video_url"
        continue
    fi

    # Step 4: Run selection.py
    echo "Step 4: Running selection..."
    time python3 selection.py
    if [ $? -ne 0 ]; then
        echo "Error in selection.py for $video_url"
        continue
    fi

    # Step 5: Append sentence_sequence.csv to the growing list
    echo "Step 5: Appending to master list..."
    if [ -f "sentence_sequence.csv" ]; then
        cat sentence_sequence.csv >> "$OUTPUT_FILE"
        echo "Successfully appended sentences from video $line_num"
    else
        echo "Warning: sentence_sequence.csv not found for $video_url"
    fi

    echo "Completed processing video $line_num"
done < "$VIDEO_LIST_FILE"

echo ""
echo "=========================================="
echo "All videos processed!"
echo "Combined output saved to: $OUTPUT_FILE"
echo "Total sentences collected: $(wc -l < "$OUTPUT_FILE")"
echo "=========================================="
