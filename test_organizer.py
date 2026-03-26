import pytest
from organizer import SentenceOrganizer


def test_cedict_filter_excludes_non_cedict_words():
    """Target words must appear in cedict vocab."""
    # 咋 and 整 are individually in cedict so resegmentation
    # splits 咋整 into them. Use a word with chars NOT in cedict
    # to test pure filtering.
    cedict_vocab = {'你好', '世界', '今天', '是'}
    word_ranks = {'你好': 1, '世界': 2, '今天': 3}
    sentences = ['你好世界', '今天是你好的']
    pre_segmented = {
        '你好世界': ['你好', '世界'],
        '今天是你好的': ['今天', '是', '你好', '的'],
    }

    org = SentenceOrganizer(
        sentences, word_ranks, pre_segmented,
        initial_words={'你好', '是', '的'},
        cedict_vocab=cedict_vocab,
    )

    sent = org.get_next_sentence()
    assert sent == '你好世界'
    new_words, _ = org.learn_sentence(sent)
    assert new_words == {'世界'}

    sent = org.get_next_sentence()
    assert sent == '今天是你好的'
    new_words, _ = org.learn_sentence(sent)
    assert new_words == {'今天'}


def test_resegment_non_cedict_chinese():
    """Non-cedict segments are split via greedy longest-match."""
    cedict_vocab = {'你', '好', '世界', '今天'}
    word_ranks = {'你': 1, '好': 2, '世界': 3, '今天': 4}
    # Segmenter outputs 你好 as one word, but it's not in cedict
    # Should be resegmented into 你 + 好
    sentences = ['你好世界今天']
    pre_segmented = {
        '你好世界今天': ['你好', '世界', '今天'],
    }

    org = SentenceOrganizer(
        sentences, word_ranks, pre_segmented,
        initial_words={'你'},
        cedict_vocab=cedict_vocab,
    )

    # After resegmentation: ['你', '好', '世界', '今天']
    # 你 is known, so unknowns are {好, 世界, 今天} -> bucket 3
    data = org.sentence_data['你好世界今天']
    assert '好' in data['unknown']
    assert '世界' in data['unknown']
    assert '今天' in data['unknown']


def test_resegment_mixed_english_chinese():
    """Mixed english+chinese segments handled correctly."""
    cedict_vocab = {'4S店', '好', '今天', '去'}
    word_ranks = {'4S店': 1, '好': 2, '今天': 3, '去': 4}

    # '4S店' is in cedict, kept as-is
    # 'XY好' is NOT in cedict, resegmented to X, Y, 好
    sentences = ['今天去4S店', '今天XY好去']
    pre_segmented = {
        '今天去4S店': ['今天', '去', '4S店'],
        '今天XY好去': ['今天', 'XY好', '去'],
    }

    org = SentenceOrganizer(
        sentences, word_ranks, pre_segmented,
        initial_words={'今天', '去'},
        cedict_vocab=cedict_vocab,
    )

    # 4S店 is in cedict and not resegmented -> valid target
    data = org.sentence_data['今天去4S店']
    assert data['unknown'] == {'4S店'}

    # XY好 resegmented to ['X', 'Y', '好']
    # X and Y are non-Chinese -> ignored by unknown filter
    # 好 is Chinese + in cedict -> unknown
    data = org.sentence_data['今天XY好去']
    assert data['unknown'] == {'好'}


def test_english_words_ignored_for_i_plus_1():
    """English segments don't count as unknown; sentence still selected."""
    cedict_vocab = {'今天', '用', '很', '好'}
    word_ranks = {'今天': 1, '用': 2, '很': 3, '好': 4}
    sentences = ['今天用iPhone很好']
    pre_segmented = {
        '今天用iPhone很好': ['今天', '用', 'iPhone', '很', '好'],
    }

    org = SentenceOrganizer(
        sentences, word_ranks, pre_segmented,
        initial_words={'今天', '用', '很'},
        cedict_vocab=cedict_vocab,
    )

    # iPhone has no Chinese chars -> not unknown
    # 好 is the only unknown -> bucket 1 (i+1 sentence)
    sent = org.get_next_sentence()
    assert sent == '今天用iPhone很好'
    new_words, _ = org.learn_sentence(sent)
    assert new_words == {'好'}


def test_no_cedict_vocab_allows_all_words():
    """Without cedict_vocab, all Chinese words are candidates."""
    word_ranks = {'你好': 1, '咋整': 2}
    sentences = ['你好咋整']
    pre_segmented = {'你好咋整': ['你好', '咋整']}

    org = SentenceOrganizer(
        sentences, word_ranks, pre_segmented,
        cedict_vocab=None,
    )

    # Both words are unknown, so this is in bucket 2
    assert '你好咋整' not in org.sentence_buckets[1]
    assert '你好咋整' in org.sentence_buckets[2]


def test_cedict_loading():
    """Test parse_cedict returns vocab and definitions."""
    import tempfile, os
    from cedict import parse_cedict

    content = (
        "傳統 传统 [chuan2 tong3] /tradition/conventional/\n"
        "你好 你好 [ni3 hao3] /hello/hi/\n"
    )
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.u8', delete=False, encoding='utf-8'
    ) as f:
        f.write(content)
        path = f.name

    try:
        vocab, defs = parse_cedict(path)
        assert '传统' in vocab
        assert '你好' in vocab
        assert '傳統' not in vocab  # traditional excluded
        assert defs['传统'] == 'tradition\nconventional'
        assert defs['你好'] == 'hello\nhi'
    finally:
        os.unlink(path)
