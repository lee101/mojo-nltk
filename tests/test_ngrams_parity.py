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
