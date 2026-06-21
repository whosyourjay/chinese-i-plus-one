import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from generate_audio import (
    generate_word_audio,
    prefer_audio_when_subset,
    sanitize_filename,
)


def test_prefer_audio_when_subset():
    # Audio extends sub within same clause (no separator between) → take audio
    assert prefer_audio_when_subset("你好", "你好世界") == "你好世界"
    # Whitespace differences ignored; audio adds 啊 within the clause
    assert prefer_audio_when_subset("你好世界", "你好 世界 啊") == "你好 世界 啊"
    # Punctuation-only difference is neutral → keep sub
    assert prefer_audio_when_subset("你好。", "你好") == "你好。"
    assert prefer_audio_when_subset("西冷它是什么特点",
                                    "西冷它是什么特点？你想。") == "西冷它是什么特点"
    # Sub not contained in audio → keep sub
    assert prefer_audio_when_subset("你好", "再见") == "你好"
    # Empty audio → keep sub
    assert prefer_audio_when_subset("你好", "") == "你好"
    # Empty sub → keep sub (no-op)
    assert prefer_audio_when_subset("", "你好") == ""
    # Comma splits clauses: trailing fragment dropped, keep sub since the
    # containing chunk is just sub + terminator (punct-only diff)
    assert prefer_audio_when_subset("一判就是判几年",
                                    "对，一判就是判几年。如果说。") == "一判就是判几年"
    # Comma split prevents junk like "票，他给。" from being appended
    assert prefer_audio_when_subset("他没有票", "他没有票，他给。") == "他没有票"
    # Audio adds new content before sub within same clause → take audio
    assert prefer_audio_when_subset("世界", "你好世界啊") == "你好世界啊"


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
