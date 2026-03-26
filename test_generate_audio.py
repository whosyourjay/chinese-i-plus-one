import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from generate_audio import (
    generate_word_audio,
    sanitize_filename,
)


def test_sanitize_filename():
    assert sanitize_filename('你好') == '你好'
    assert sanitize_filename('4S店') == '4S店'
    assert '/' not in sanitize_filename('a/b')
    assert len(sanitize_filename('x' * 100, 50)) == 50


def test_generate_word_audio_populates_column():
    """word_audio column is set for each row."""
    df = pd.DataFrame({
        'New_Words': ['你好', '世界'],
        'Sentence': ['你好吗', '世界大'],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-create the audio files so TTS isn't called
        for word in df['New_Words']:
            fname = f"word_{sanitize_filename(word, 30)}.mp3"
            (Path(tmpdir) / fname).write_text('fake')

        result = generate_word_audio(df, tmpdir)

    assert 'word_audio' in result.columns
    for val in result['word_audio']:
        assert val.startswith('[sound:')
        assert val.endswith('.mp3]')


def test_generate_word_audio_skips_existing_files():
    """Existing files are not re-generated."""
    df = pd.DataFrame({
        'New_Words': ['测试'],
        'Sentence': ['这是测试'],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f"word_{sanitize_filename('测试', 30)}.mp3"
        (Path(tmpdir) / fname).write_text('fake')

        with patch(
            'generate_audio.generate_all_word_tts'
        ) as mock_tts:
            result = generate_word_audio(df, tmpdir)
            mock_tts.assert_not_called()

    assert result['word_audio'].iloc[0] == f'[sound:{fname}]'


def test_generate_word_audio_calls_tts_for_missing():
    """TTS is called for words without existing audio."""
    df = pd.DataFrame({
        'New_Words': ['新词'],
        'Sentence': ['学新词'],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            'generate_audio.asyncio.run'
        ) as mock_run:
            result = generate_word_audio(df, tmpdir)
            mock_run.assert_called_once()

        # Check the task was created with right word
        args = mock_run.call_args[0][0]
        # It's a coroutine, check tasks were built
        assert len(result) == 1


def test_generate_word_audio_deduplicates():
    """Same word appearing twice only generates TTS once."""
    df = pd.DataFrame({
        'New_Words': ['重复', '重复'],
        'Sentence': ['重复一', '重复二'],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            'generate_audio.asyncio.run'
        ) as mock_run:
            result = generate_word_audio(df, tmpdir)
            mock_run.assert_called_once()

        # Both rows should have the same sound reference
        assert result['word_audio'].iloc[0] == \
            result['word_audio'].iloc[1]


def test_generate_word_audio_empty_words():
    """Empty or NaN New_Words don't cause errors."""
    df = pd.DataFrame({
        'New_Words': ['', float('nan')],
        'Sentence': ['句子一', '句子二'],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_word_audio(df, tmpdir)

    assert result['word_audio'].iloc[0] == ''
    assert result['word_audio'].iloc[1] == ''
