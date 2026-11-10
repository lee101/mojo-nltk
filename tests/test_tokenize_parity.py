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


# Every ASCII byte in word, punctuation, separator, and mixed contexts.
ASCII_CONTEXTS = [
    "a{c}b",
    "{c}a",
    "a{c}",
    "{c}{c}{c}",
    "a{c}{c}b{c}",
    "{c}",
]


@pytest.mark.parametrize("code", range(128))
def test_wordpunct_ascii_classes_match_nltk(code):
    char = chr(code)
    for template in ASCII_CONTEXTS:
        text = template.format(c=char)
        ours = m.WordPunctTokenizer()
        reference = nltk_tokenize.WordPunctTokenizer()
        assert ours.tokenize(text) == reference.tokenize(text), repr(text)
        assert list(ours.span_tokenize(text)) == list(
            reference.span_tokenize(text)
        ), repr(text)


@pytest.mark.parametrize("code", [0x1C, 0x1D, 0x1E, 0x1F])
def test_wordpunct_ascii_information_separators_are_whitespace(code):
    # Python's re classifies U+001C..U+001F as \s, so nltk's pattern
    # r"\w+|[^\w\s]+" drops them like other separators.
    text = "a" + chr(code) + "b"
    reference = nltk_tokenize.WordPunctTokenizer()
    assert reference.tokenize(text) == ["a", "b"]
    assert m.WordPunctTokenizer().tokenize(text) == reference.tokenize(text)
    assert list(m.WordPunctTokenizer().span_tokenize(text)) == list(
        reference.span_tokenize(text)
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\n",
        "\t",
        "a",
        "  ",
        "\u00a0",
        "a\u00a0b",
        "é",
        "aé",
        "你 好",
        "a\u200bb",
        "\u2028",
    ],
)
def test_wordpunct_degenerate_texts_match_nltk(text):
    ours = m.WordPunctTokenizer()
    reference = nltk_tokenize.WordPunctTokenizer()
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))
    assert m.wordpunct_tokenize(text) == nltk_tokenize.wordpunct_tokenize(text)


@pytest.mark.parametrize(
    "text", ["", " ", "\n", "\n\n", "\n \n\n\t\n", "a\n\nb", "a\nb\n", "\r\n", "\r\n\r\n"]
)
@pytest.mark.parametrize("tokenizer_name", ["BlanklineTokenizer", "WhitespaceTokenizer"])
def test_whitespace_tokenizers_degenerate_match_nltk(tokenizer_name, text):
    ours = getattr(m, tokenizer_name)()
    reference = getattr(nltk_tokenize, tokenizer_name)()
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))


@pytest.mark.parametrize(
    "text", ["", " ", "\n", "\n\n", "a\n\n", "a\n \n\nb\n", "\r\n\r\n", "a\nb"]
)
@pytest.mark.parametrize("blanklines", ["discard", "keep", "discard-eof"])
def test_line_tokenizer_degenerate_match_nltk(blanklines, text):
    ours = m.LineTokenizer(blanklines)
    reference = nltk_tokenize.LineTokenizer(blanklines)
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))


def test_line_tokenizer_discard_eof_matches_nltk():
    # Docstring of nltk.tokenize.simple.LineTokenizer: "discard-eof: if the
    # string ends with a newline, then do not generate a corresponding token
    # '' after that newline."
    text = "a\nb\n"
    assert m.LineTokenizer("keep").tokenize(text) == nltk_tokenize.LineTokenizer(
        "keep"
    ).tokenize(text)
    assert m.LineTokenizer("discard-eof").tokenize(text) == nltk_tokenize.LineTokenizer(
        "discard-eof"
    ).tokenize(text)


def test_line_tokenizer_default_mode_matches_nltk():
    text = "a\n\nb"
    assert nltk_tokenize.LineTokenizer().tokenize(text) == ["a", "b"]
    assert m.LineTokenizer().tokenize(text) == nltk_tokenize.LineTokenizer().tokenize(
        text
    )


@pytest.mark.parametrize(
    ("tokenizer_name", "text"),
    [
        ("SpaceTokenizer", ""),
        ("SpaceTokenizer", " "),
        ("SpaceTokenizer", "  "),
        ("SpaceTokenizer", " a "),
        ("TabTokenizer", ""),
        ("TabTokenizer", "\t"),
        ("TabTokenizer", "\t\t"),
        ("TabTokenizer", "\ta\t"),
    ],
)
def test_string_tokenizers_degenerate_match_nltk(tokenizer_name, text):
    ours = getattr(m, tokenizer_name)()
    reference = getattr(nltk_tokenize, tokenizer_name)()
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))


@pytest.mark.parametrize("text", ["", "a", "ab", "a b", "\n", "é"])
def test_char_tokenizer_degenerate_matches_nltk(text):
    from nltk.tokenize.simple import CharTokenizer

    ours = m.CharTokenizer()
    reference = CharTokenizer()
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))


@pytest.mark.parametrize(
    ("pattern", "gaps", "discard_empty", "text"),
    [
        (r"\w+", False, True, ""),
        (r"\w+", False, True, " ,;"),
        (r"\W+", True, True, ""),
        (r"\W+", True, False, ""),
        (r"\W+", True, False, ",,,"),
        (r"\s+", True, False, " "),
        (r"x", False, True, "abc"),
        (r"a?", False, True, "aba"),
    ],
)
def test_regexp_tokenizer_degenerate_match_nltk(
    pattern, gaps, discard_empty, text
):
    ours = m.RegexpTokenizer(pattern, gaps, discard_empty)
    reference = nltk_tokenize.RegexpTokenizer(pattern, gaps, discard_empty)
    assert ours.tokenize(text) == reference.tokenize(text)
    assert list(ours.span_tokenize(text)) == list(reference.span_tokenize(text))
    assert m.regexp_tokenize(text, pattern, gaps, discard_empty) == (
        nltk_tokenize.regexp_tokenize(text, pattern, gaps, discard_empty)
    )


def test_empty_span_tokenize_sents_match_nltk():
    for tokenizer_name in ("SpaceTokenizer", "WhitespaceTokenizer", "WordPunctTokenizer"):
        ours = getattr(m, tokenizer_name)()
        reference = getattr(nltk_tokenize, tokenizer_name)()
        assert ours.tokenize_sents([]) == reference.tokenize_sents([])
        assert list(ours.span_tokenize_sents([])) == list(
            reference.span_tokenize_sents([])
        )
        assert list(ours.span_tokenize_sents([""])) == list(
            reference.span_tokenize_sents([""])
        )


def test_regexp_tokenizer_repr_matches_nltk():
    assert repr(m.RegexpTokenizer(r"\w+")) == repr(
        nltk_tokenize.RegexpTokenizer(r"\w+")
    )
    assert repr(m.WordPunctTokenizer()) == repr(nltk_tokenize.WordPunctTokenizer())
