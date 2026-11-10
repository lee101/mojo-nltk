import random

import pytest

import mojonltk as m

nltk_stem = pytest.importorskip("nltk.stem")


PORTER_WORDS = [
    "caresses",
    "ponies",
    "ties",
    "cats",
    "feed",
    "agreed",
    "plastered",
    "motoring",
    "conflated",
    "troubled",
    "hopping",
    "falling",
    "filing",
    "happy",
    "sky",
    "relational",
    "conditional",
    "digitizer",
    "vietnamization",
    "decisiveness",
    "sensibiliti",
    "triplicate",
    "formalize",
    "electrical",
    "goodness",
    "replacement",
    "adoption",
    "communism",
    "effective",
    "bowdlerize",
    "probate",
    "cease",
    "controll",
    "skies",
    "dying",
    "lying",
    "tying",
    "news",
    "innings",
    "proceed",
    "spied",
    "died",
    "flies",
    "enjoy",
    "spy",
    "by",
    "cafés",
]


@pytest.mark.parametrize(
    "mode",
    [
        m.PorterStemmer.ORIGINAL_ALGORITHM,
        m.PorterStemmer.MARTIN_EXTENSIONS,
        m.PorterStemmer.NLTK_EXTENSIONS,
    ],
)
def test_porter_published_and_extension_vectors(mode):
    ours = m.PorterStemmer(mode)
    reference = nltk_stem.PorterStemmer(mode)
    assert [ours.stem(word) for word in PORTER_WORDS] == [
        reference.stem(word) for word in PORTER_WORDS
    ]


def test_porter_generated_suffix_parity():
    rng = random.Random(11)
    pieces = ["a", "be", "con", "de", "fli", "hop", "log", "sens", "tion", "y"]
    suffixes = [
        "s",
        "ies",
        "eed",
        "ing",
        "ational",
        "izer",
        "alli",
        "ization",
        "fulness",
        "biliti",
        "ative",
        "ical",
        "ement",
        "ion",
        "ize",
        "ll",
    ]
    ours = m.PorterStemmer()
    reference = nltk_stem.PorterStemmer()
    for _ in range(500):
        word = "".join(rng.choices(pieces, k=rng.randrange(1, 5)))
        word += rng.choice(suffixes)
        assert ours.stem(word) == reference.stem(word)


def test_porter_lowercase_flag_and_repr():
    ours = m.PorterStemmer()
    reference = nltk_stem.PorterStemmer()
    for word in ("RELATIONAL", "Caresses", "SKIES"):
        assert ours.stem(word, False) == reference.stem(word, False)
        assert ours.stem(word, True) == reference.stem(word, True)
    assert repr(ours) == repr(reference)


@pytest.mark.parametrize(
    "word",
    [
        "",
        "y",
        "by",
        "é",
        "éy",
        "caféed",
        "naïvely",
        "東京ing",
        "sensßïïïing",
    ],
)
def test_porter_ascii_fast_path_and_unicode_edges(word):
    for mode in (
        m.PorterStemmer.ORIGINAL_ALGORITHM,
        m.PorterStemmer.MARTIN_EXTENSIONS,
        m.PorterStemmer.NLTK_EXTENSIONS,
    ):
        ours = m.PorterStemmer(mode)
        reference = nltk_stem.PorterStemmer(mode)
        assert ours.stem(word) == reference.stem(word)


def test_porter_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Mode must be one of"):
        m.PorterStemmer("fast")


def test_regexp_stemmer_matches_nltk():
    ours = m.RegexpStemmer(r"ing$|s$|e$|able$", min=4)
    reference = nltk_stem.RegexpStemmer(r"ing$|s$|e$|able$", min=4)
    words = ["cars", "mass", "was", "bee", "compute", "advisable"]
    assert [ours.stem(word) for word in words] == [
        reference.stem(word) for word in words
    ]
    assert repr(ours) == repr(reference)


@pytest.mark.parametrize(
    "word",
    [
        "",
        "a",
        "ab",
        "y",
        "yy",
        "yyy",
        "yyyy",
        "by",
        "s",
        "ss",
        "ies",
        "eed",
        "ing",
        "ed",
        "-" * 8,
        "y" * 40,
        "a" * 300,
        "y" * 300,
        "é" * 300,
        "e" * 2000,
        "東京",
        "e\u0301",
        "İstanbul",
        "ǅungla",
        "eee",
        "EEE",
    ],
)
def test_porter_degenerate_words_match_nltk(word):
    # Covers the kernel's short-buffer edge (empty input writes nothing),
    # is_consonant's y-run walk, and non-ASCII code-point buffers.
    for mode in (
        m.PorterStemmer.ORIGINAL_ALGORITHM,
        m.PorterStemmer.MARTIN_EXTENSIONS,
        m.PorterStemmer.NLTK_EXTENSIONS,
    ):
        ours = m.PorterStemmer(mode)
        reference = nltk_stem.PorterStemmer(mode)
        for to_lowercase in (True, False):
            assert ours.stem(word, to_lowercase) == reference.stem(
                word, to_lowercase
            ), (word[:20], len(word), mode, to_lowercase)


def test_porter_short_word_bypass_matches_nltk():
    # nltk.stem.porter.PorterStemmer.stem returns lowercased input unchanged
    # when mode != ORIGINAL_ALGORITHM and len(word) <= 2.
    for mode in (
        m.PorterStemmer.MARTIN_EXTENSIONS,
        m.PorterStemmer.NLTK_EXTENSIONS,
    ):
        ours = m.PorterStemmer(mode)
        reference = nltk_stem.PorterStemmer(mode)
        for word in ("", "A", "aS", "ý", "Ys"):
            assert ours.stem(word) == reference.stem(word)


def test_porter_irregular_table_matches_nltk():
    # The NLTK_EXTENSIONS irregular map in nltk/stem/porter.py.
    ours = m.PorterStemmer(m.PorterStemmer.NLTK_EXTENSIONS)
    reference = nltk_stem.PorterStemmer(nltk_stem.PorterStemmer.NLTK_EXTENSIONS)
    for word in reference.pool:
        assert ours.stem(word) == reference.stem(word), word
        assert ours.stem(word.upper(), False) == reference.stem(word.upper(), False)


def test_porter_non_string_raises_same_error():
    # Upstream calls word.lower() first, so non-strings raise AttributeError.
    with pytest.raises(AttributeError):
        nltk_stem.PorterStemmer().stem(5)
    with pytest.raises(AttributeError):
        m.PorterStemmer().stem(5)


def test_regexp_stemmer_degenerate_matches_nltk():
    cases = [
        ("", 0),
        ("s", 0),
        ("ss", 2),
        ("", 4),
        ("abc", 4),
        ("abc", 3),
        ("aaa", 99),
    ]
    for word, minimum in cases:
        ours = m.RegexpStemmer(r"a", min=minimum)
        reference = nltk_stem.RegexpStemmer(r"a", min=minimum)
        assert ours.stem(word) == reference.stem(word), (word, minimum)
    assert m.RegexpStemmer(r"s$", 4).stem("") == nltk_stem.RegexpStemmer(
        r"s$", 4
    ).stem("")
