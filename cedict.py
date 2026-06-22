"""Load CC-CEDICT dictionary data."""

CEDICT_PATH = 'words/cedict_ts.u8'


def parse_cedict(path=CEDICT_PATH):
    """Parse cedict, returning (vocab set, definitions dict).

    Definitions dict maps simplified word to its definition string
    with multiple senses separated by newlines.
    """
    vocab = set()
    definitions = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 2)
            if len(parts) < 3:
                continue
            simplified = parts[1]
            vocab.add(simplified)
            # Extract definition between /.../ markers
            rest = parts[2]
            slash_start = rest.find('/')
            if slash_start == -1:
                continue
            def_text = rest[slash_start + 1:].rstrip('/')
            definition = def_text.replace('/', '\n')
            definitions[simplified] = definition
    return vocab, definitions


def load_cedict_vocab(path=CEDICT_PATH):
    """Load just the simplified Chinese vocabulary set."""
    vocab, _ = parse_cedict(path)
    return vocab


def load_cedict_definitions(path=CEDICT_PATH):
    """Load simplified word -> definition mapping."""
    _, definitions = parse_cedict(path)
    return definitions


def has_chinese(word):
    return any("\u4e00" <= ch <= "\u9fff" for ch in word)


def resegment_word(word, cedict_vocab):
    """Split a token not in cedict via greedy longest-match against cedict.
    Single chars are kept as-is when no longer prefix matches."""
    result = []
    i = 0
    while i < len(word):
        best = None
        for end in range(len(word), i, -1):
            candidate = word[i:end]
            if candidate in cedict_vocab:
                best = candidate
                break
        if best:
            result.append(best)
            i += len(best)
        else:
            result.append(word[i])
            i += 1
    return result


def expand_segmented_words(words, cedict_vocab):
    """Greedy-resegment any Chinese token that isn't in cedict; keep the rest."""
    expanded = []
    for word in words:
        if has_chinese(word) and word not in cedict_vocab:
            expanded.extend(resegment_word(word, cedict_vocab))
        else:
            expanded.append(word)
    return expanded
