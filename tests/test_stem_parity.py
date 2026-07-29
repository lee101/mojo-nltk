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

