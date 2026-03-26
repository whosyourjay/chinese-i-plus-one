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
