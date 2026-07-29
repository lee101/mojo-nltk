import pytest

import mojonltk as m

nltk_tokenize = pytest.importorskip("nltk.tokenize")

TEXTS = [
    "",
    "Good muffins cost $3.88\nin New York.",
    "wait... what?!  yes_no",
    "café déjà-vu 你好！",
    "\t one\v two\fthree\r\n",
]


@pytest.mark.parametrize("text", TEXTS)
def test_wordpunct_tokens_and_spans_match_nltk(text):
    ours = m.WordPunctTokenizer()
    reference = nltk_tokenize.WordPunctTokenizer()
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))
    assert m.wordpunct_tokenize(text) == nltk_tokenize.wordpunct_tokenize(text)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65])
def test_wordpunct_ascii_boundaries_match_nltk(length):
    text = ("a" * length) + " " + ("!" * length) + "_z"
    ours = m.WordPunctTokenizer()
    reference = nltk_tokenize.WordPunctTokenizer()
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "WhitespaceTokenizer",
        "BlanklineTokenizer",
        "SpaceTokenizer",
        "TabTokenizer",
        "CharTokenizer",
    ],
)
def test_simple_tokenizers_match_nltk(tokenizer_name):
    text = "alpha beta\tgamma\n\n delta "
    ours = getattr(m, tokenizer_name)()
    if tokenizer_name == "CharTokenizer":
        from nltk.tokenize.simple import CharTokenizer

        reference = CharTokenizer()
    else:
        reference = getattr(nltk_tokenize, tokenizer_name)()
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))
    assert ours.tokenize_sents([text, "x y"]) == reference.tokenize_sents(
        [text, "x y"]
    )
    assert list(ours.span_tokenize_sents([text, "x y"])) == list(
        reference.span_tokenize_sents([text, "x y"])
    )


@pytest.mark.parametrize(
    ("tokenizer_name", "text"),
    [
        ("SpaceTokenizer", "  alpha  beta  "),
        ("TabTokenizer", "\t\talpha\t\tbeta\t"),
    ],
)
def test_repeated_string_delimiters_match_nltk(tokenizer_name, text):
    ours = getattr(m, tokenizer_name)()
    reference = getattr(nltk_tokenize, tokenizer_name)()
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))


@pytest.mark.parametrize("blanklines", ["discard", "keep", "discard-eof"])
def test_line_tokenizer_matches_nltk(blanklines):
    text = "first\n \nsecond\n\n"
    ours = m.LineTokenizer(blanklines)
    reference = nltk_tokenize.LineTokenizer(blanklines)
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))


@pytest.mark.parametrize("gaps", [False, True])
@pytest.mark.parametrize("discard_empty", [False, True])
def test_regexp_tokenizer_matches_nltk(gaps, discard_empty):
    text = "a1, b22;;c333"
    pattern = r"\W+" if gaps else r"\w+"
    ours = m.RegexpTokenizer(pattern, gaps, discard_empty)
    reference = nltk_tokenize.RegexpTokenizer(pattern, gaps, discard_empty)
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))
    assert m.regexp_tokenize(text, pattern, gaps, discard_empty) == (
        nltk_tokenize.regexp_tokenize(text, pattern, gaps, discard_empty)
    )


def test_blankline_and_line_functions_match_nltk():
    text = "first\n\n second\n \nthird"
    assert m.blankline_tokenize(text) == nltk_tokenize.blankline_tokenize(text)
    assert m.line_tokenize(text) == nltk_tokenize.line_tokenize(text)


def test_invalid_line_mode_matches_nltk():
    with pytest.raises(ValueError, match="Blank lines"):
        m.LineTokenizer("unknown")
