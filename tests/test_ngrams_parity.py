import pytest

import mojonltk as m

nltk_util = pytest.importorskip("nltk.util")


def test_pad_sequence_matches_nltk():
    kwargs = {
        "pad_left": True,
        "pad_right": True,
        "left_pad_symbol": "<s>",
        "right_pad_symbol": "</s>",
    }
    assert list(m.pad_sequence(iter("abc"), 3, **kwargs)) == list(
        nltk_util.pad_sequence(iter("abc"), 3, **kwargs)
    )


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
def test_ngrams_match_nltk(n):
    sequence = [1, 2, 3, 4, 5]
    assert list(m.ngrams(sequence, n)) == list(nltk_util.ngrams(sequence, n))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_padded_ngrams_match_nltk(n):
    kwargs = {
        "pad_left": True,
        "pad_right": True,
        "left_pad_symbol": "<s>",
        "right_pad_symbol": "</s>",
    }
    assert list(m.ngrams(iter("abcd"), n, **kwargs)) == list(
        nltk_util.ngrams(iter("abcd"), n, **kwargs)
    )


def test_bigrams_and_trigrams_match_nltk():
    sequence = "one two three four".split()
    assert list(m.bigrams(sequence)) == list(nltk_util.bigrams(sequence))
    assert list(m.trigrams(sequence)) == list(nltk_util.trigrams(sequence))


@pytest.mark.parametrize(("minimum", "maximum"), [(1, -1), (2, 3), (3, 5)])
def test_everygrams_match_nltk(minimum, maximum):
    sequence = "a b c d".split()
    assert list(m.everygrams(sequence, minimum, maximum)) == list(
        nltk_util.everygrams(sequence, minimum, maximum)
    )


def test_everygrams_padding_matches_nltk():
    kwargs = {
        "min_len": 1,
        "max_len": 3,
        "pad_left": True,
        "pad_right": True,
        "left_pad_symbol": "<s>",
        "right_pad_symbol": "</s>",
    }
    assert list(m.everygrams("abc", **kwargs)) == list(
        nltk_util.everygrams("abc", **kwargs)
    )


@pytest.mark.parametrize(("n", "k"), [(2, 1), (2, 2), (3, 2)])
def test_skipgrams_match_nltk(n, k):
    sequence = "Insurgents killed in ongoing fighting".split()
    assert list(m.skipgrams(sequence, n, k)) == list(
        nltk_util.skipgrams(sequence, n, k)
    )


def test_generators_accept_one_shot_iterators():
    ours = list(m.ngrams((value for value in range(8)), 4))
    reference = list(nltk_util.ngrams((value for value in range(8)), 4))
    assert ours == reference


@pytest.mark.parametrize("sequence", [list(range(12)), tuple(range(12)), range(12)])
@pytest.mark.parametrize("n", [1, 3, 7])
def test_sequence_fast_path_matches_nltk(sequence, n):
    assert list(m.ngrams(sequence, n)) == list(nltk_util.ngrams(sequence, n))


def test_everygrams_default_safety_limit_matches_nltk():
    with pytest.raises(ValueError, match="default max_len"):
        list(m.everygrams(range(257)))


def test_everygrams_default_limit_boundary_matches_nltk():
    # nltk.util.everygrams raises only when the defaulted max_len exceeds
    # nltk.util.MAX_EVERYGRAMS_DEFAULT_LEN (256); 256 itself is allowed.
    with pytest.raises(ValueError, match="default max_len"):
        list(nltk_util.everygrams(range(257)))
    with pytest.raises(ValueError, match="default max_len"):
        list(m.everygrams(range(257)))
    assert len(list(m.everygrams(range(256), max_len=1))) == 256


def test_everygrams_reads_default_limit_dynamically():
    import mojonltk.util as m_util

    saved = [
        module.MAX_EVERYGRAMS_DEFAULT_LEN
        for module in (m_util, nltk_util)
    ]
    try:
        m_util.MAX_EVERYGRAMS_DEFAULT_LEN = 3
        nltk_util.MAX_EVERYGRAMS_DEFAULT_LEN = 3
        assert list(m.everygrams(range(3))) == list(nltk_util.everygrams(range(3)))
        with pytest.raises(ValueError):
            list(nltk_util.everygrams(range(4)))
        with pytest.raises(ValueError):
            list(m.everygrams(range(4)))
    finally:
        m_util.MAX_EVERYGRAMS_DEFAULT_LEN, nltk_util.MAX_EVERYGRAMS_DEFAULT_LEN = saved


@pytest.mark.parametrize(
    "sequence",
    [[], [1], [1, 1, 1], (), (1,), "a", "", list(range(10))],
)
@pytest.mark.parametrize("n", [1, 2, 3, 10])
@pytest.mark.parametrize("pad_left", [False, True])
@pytest.mark.parametrize("pad_right", [False, True])
def test_ngrams_degenerate_inputs_match_nltk(sequence, n, pad_left, pad_right):
    kwargs = {
        "pad_left": pad_left,
        "pad_right": pad_right,
        "left_pad_symbol": "<s>",
        "right_pad_symbol": "</s>",
    }
    assert list(m.ngrams(sequence, n, **kwargs)) == list(
        nltk_util.ngrams(sequence, n, **kwargs)
    )


def test_ngrams_zero_degree_matches_nltk():
    # nltk.util.ngrams([1, 2], 0) emits one empty tuple per item.
    for sequence in ([1, 2], "ab", []):
        assert list(m.ngrams(sequence, 0)) == list(nltk_util.ngrams(sequence, 0))


@pytest.mark.parametrize("n", [-1, -3])
def test_ngrams_negative_degree_raises_same_error(n):
    with pytest.raises(ValueError):
        list(nltk_util.ngrams([1, 2], n))
    with pytest.raises(ValueError):
        list(m.ngrams([1, 2], n))


def test_pad_sequence_degenerate_matches_nltk():
    cases = [
        ([], 2, False, False),
        ([], 2, True, True),
        ([1], 3, True, True),
        ([1], 1, True, True),
        ([1, 2], 0, True, True),
    ]
    for sequence, n, pad_left, pad_right in cases:
        kwargs = {
            "pad_left": pad_left,
            "pad_right": pad_right,
            "left_pad_symbol": "<s>",
            "right_pad_symbol": "</s>",
        }
        assert list(m.pad_sequence(sequence, n, **kwargs)) == list(
            nltk_util.pad_sequence(sequence, n, **kwargs)
        )


def test_pad_sequence_is_lazy_like_nltk():
    # nltk.util.pad_sequence returns an iterator, not a list.
    assert not isinstance(m.pad_sequence([1, 2], 2), list)
    assert not isinstance(nltk_util.pad_sequence([1, 2], 2), list)


@pytest.mark.parametrize(
    ("min_len", "max_len", "sequence"),
    [
        (1, -1, []),
        (1, 1, [1]),
        (2, 1, [1, 2, 3]),
        (3, 5, [1, 2]),
        (1, 2, [1, 1, 1]),
        (0, 2, [1, 2]),
        (1, 4, "a"),
    ],
)
def test_everygrams_degenerate_match_nltk(min_len, max_len, sequence):
    assert list(m.everygrams(sequence, min_len, max_len)) == list(
        nltk_util.everygrams(sequence, min_len, max_len)
    )


def test_everygrams_padding_degenerate_matches_nltk():
    kwargs = {
        "pad_left": True,
        "pad_right": True,
        "left_pad_symbol": "<s>",
        "right_pad_symbol": "</s>",
    }
    for sequence, min_len, max_len in [([], 1, 2), ([1], 1, 3), ([1, 2], 2, 3)]:
        assert list(m.everygrams(sequence, min_len, max_len, **kwargs)) == list(
            nltk_util.everygrams(sequence, min_len, max_len, **kwargs)
        )


@pytest.mark.parametrize(
    ("sequence", "n", "k"),
    [
        ([], 2, 1),
        ([1], 2, 1),
        ([1, 2], 3, 2),
        ([1, 1, 1, 1], 2, 2),
        ([1, 2, 3, 4, 5], 2, 0),
    ],
)
def test_skipgrams_degenerate_match_nltk(sequence, n, k):
    assert list(m.skipgrams(sequence, n, k)) == list(
        nltk_util.skipgrams(sequence, n, k)
    )


def test_skipgrams_unary_degree_raises_same_error():
    # nltk.util.skipgrams indexes skip_tail[-1], so n=1 raises IndexError.
    with pytest.raises(IndexError):
        list(nltk_util.skipgrams([1, 2], 1, 1))
    with pytest.raises(IndexError):
        list(m.skipgrams([1, 2], 1, 1))


def test_skipgrams_padding_matches_nltk():
    kwargs = {
        "pad_left": True,
        "pad_right": True,
        "left_pad_symbol": "<s>",
        "right_pad_symbol": "</s>",
    }
    assert list(m.skipgrams("abc", 2, 1, **kwargs)) == list(
        nltk_util.skipgrams("abc", 2, 1, **kwargs)
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bigrams_trigrams_degenerate_match_nltk(n):
    sequence = [1]
    assert list(m.ngrams(sequence, n)) == list(nltk_util.ngrams(sequence, n))
    assert list(m.bigrams([])) == list(nltk_util.bigrams([]))
    assert list(m.trigrams([])) == list(nltk_util.trigrams([]))
