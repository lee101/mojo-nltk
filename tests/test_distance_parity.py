import random

import pytest

import mojonltk as m

nltk_distance = pytest.importorskip("nltk.metrics.distance")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("", ""),
        ("", "empty"),
        ("kitten", "sitting"),
        ("rain", "shine"),
        ("ab", "ba"),
        ("ca", "abc"),
        ("café", "coffee"),
        ("𐍈a", "a𐍈"),
        ([1, 2, 3], [1, 3]),
        (("x", None, 4), ("x", 4)),
    ],
)
@pytest.mark.parametrize("substitution_cost", [1, 2, 3])
@pytest.mark.parametrize("transpositions", [False, True])
def test_edit_distance_matches_nltk(
    first, second, substitution_cost, transpositions
):
    assert m.edit_distance(
        first, second, substitution_cost, transpositions
    ) == nltk_distance.edit_distance(
        first, second, substitution_cost, transpositions
    )


def test_edit_distance_random_parity():
    rng = random.Random(7)
    for _ in range(200):
        first = "".join(rng.choices("abcde", k=rng.randrange(15)))
        second = "".join(rng.choices("abcde", k=rng.randrange(15)))
        assert m.edit_distance(first, second, 2, True) == (
            nltk_distance.edit_distance(first, second, 2, True)
        )


def test_edit_distance_rejects_silent_integer_narrowing():
    with pytest.raises(OverflowError, match="signed 64-bit"):
        m.edit_distance("a", "b", 1 << 64)
    with pytest.raises(OverflowError, match="distance table"):
        m.edit_distance("aa", "bb", -(1 << 62) - 1)


def test_large_positive_substitution_cost_preserves_python_integer_semantics():
    assert m.edit_distance("a", "b", (1 << 63) - 1) == 2


@pytest.mark.parametrize(
    ("first", "second", "cost"),
    [
        ("", "", 1),
        ("rain", "shine", 1),
        ("kitten", "sitting", 1),
        ("ab", "ba", 3),
        ("café", "cafe", 2),
    ],
)
def test_edit_distance_align_matches_nltk(first, second, cost):
    assert m.edit_distance_align(first, second, cost) == (
        nltk_distance.edit_distance_align(first, second, cost)
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("", ""),
        ("", "nonempty"),
        ("a", "a"),
        ("a", "b"),
        ("billy", "blily"),
        ("MARTHA", "MARHTA"),
        ("DWAYNE", "DUANE"),
        ("你好世界", "你号世界"),
    ],
)
def test_jaro_matches_nltk(first, second):
    assert m.jaro_similarity(first, second) == pytest.approx(
        nltk_distance.jaro_similarity(first, second), abs=1e-15
    )


@pytest.mark.parametrize("p", [0.1, 0.125, 0.2])
def test_jaro_winkler_matches_nltk(p):
    pairs = [
        ("billy", "bill"),
        ("massie", "massey"),
        ("dixon", "dickson"),
        ("SHACKLEFORD", "SHACKELFORD"),
    ]
    for first, second in pairs:
        assert m.jaro_winkler_similarity(first, second, p=p) == pytest.approx(
            nltk_distance.jaro_winkler_similarity(first, second, p=p)
        )


def test_jaro_winkler_warning_matches_contract():
    with pytest.warns(UserWarning):
        m.jaro_winkler_similarity("abc", "abd", p=0.3, max_l=4)
