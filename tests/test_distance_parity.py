import random

import numpy as np
import pytest

import mojonltk as m
from conftest import assert_close

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


def test_substitution_cost_beyond_int64_matches_nltk():
    # nltk.metrics.distance.edit_distance stores the table as Python numbers and
    # never bounds substitution_cost, so neither of these costs overflows there.
    for cost in (1 << 64, -(1 << 62) - 1, (1 << 63) - 1):
        assert m.edit_distance("a", "b", cost) == nltk_distance.edit_distance(
            "a", "b", cost
        )


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


# Papers cited in nltk.metrics.distance.jaro_winkler_similarity's docstring.
# The reference score is the published table value; nltk itself is compared
# with the same tolerance so neither side is trusted on its own.
CENSUS_WINKLER = [
    # (s1, s2, p, published jaro, published winkler), rr93-8 Table 5
    ("billy", "billy", 0.1, 1.000, 1.000),
    ("billy", "bill", 0.125, 0.933, 0.967),
    ("billy", "blily", 0.2, 0.933, 0.947),
    ("massie", "massey", 0.125, 0.889, 0.944),
    ("yvette", "yevett", 0.2, 0.889, 0.911),
    ("billy", "bolly", 0.2, 0.867, 0.893),
    ("dwayne", "duane", 0.2, 0.822, 0.858),
    ("dixon", "dickson", 0.15, 0.790, 0.853),
    ("billy", "susan", 0.1, 0.000, 0.000),
    # rr94-5 Table 2.1
    ("SHACKLEFORD", "SHACKELFORD", 0.1, 0.970, 0.982),
    ("NICHLESON", "NICHULSON", 0.1, 0.926, 0.956),
    ("JONES", "JOHNSON", 0.1, 0.790, 0.832),
    ("MARHTA", "MARTHA", 0.1, 0.944, 0.961),
    ("DWAYNE", "DUANE", 0.2, 0.822, 0.858),
    ("BROOKHAVEN", "BRROKHAVEN", 0.1, 0.933, 0.947),
    ("BROOK HALLOW", "BROOK HLLW", 0.1, 0.944, 0.967),
    ("DECATUR", "DECATIR", 0.1, 0.905, 0.943),
    ("FITZRUREITER", "FITZENREITER", 0.1, 0.856, 0.913),
    ("IOWA", "IONA", 0.1, 0.833, 0.867),
]


@pytest.mark.parametrize(
    ("first", "second", "p", "jaro", "winkler"), CENSUS_WINKLER
)
def test_jaro_and_winkler_against_published_census_tables(
    first, second, p, jaro, winkler
):
    assert round(m.jaro_similarity(first, second), 3) == jaro
    assert round(m.jaro_winkler_similarity(first, second, p=p), 3) == winkler
    assert_close(
        m.jaro_similarity(first, second),
        nltk_distance.jaro_similarity(first, second),
    )
    assert_close(
        m.jaro_winkler_similarity(first, second, p=p),
        nltk_distance.jaro_winkler_similarity(first, second, p=p),
    )


def test_jaro_winkler_max_l_only_bounds_prefix():
    # Docstring of nltk.metrics.distance.jaro_winkler_similarity pins this value,
    # showing the output depends on l * p and not on max_l * p.
    ours = m.jaro_winkler_similarity("TANYA", "TONYA", p=0.1, max_l=100)
    assert round(ours, 3) == 0.88
    assert ours == pytest.approx(
        nltk_distance.jaro_winkler_similarity("TANYA", "TONYA", p=0.1, max_l=100),
        abs=1e-15,
    )


def test_jaro_empty_cases_match_nltk_docstring():
    # Exact values asserted in nltk.metrics.distance.jaro_similarity's docstring.
    assert m.jaro_similarity("", "") == 1.0
    assert m.jaro_similarity("", "nonempty") == 0.0
    assert m.jaro_similarity("a", "a") == 1.0
    assert m.jaro_similarity("a", "b") == 0.0
    assert m.jaro_winkler_similarity("", "") == 1.0
    assert m.jaro_winkler_similarity("a", "a") == 1.0


def test_jaro_mismatched_shapes_matches_nltk():
    # Upstream zips the prefix scan but loops over s1 by index, so different
    # lengths must still agree term by term.
    for first, second in [("abc", "ab"), ("ab", "abc"), ("ab", ""), ("", "ab")]:
        assert_close(
            m.jaro_similarity(first, second),
            nltk_distance.jaro_similarity(first, second),
        )
        assert_close(
            m.jaro_winkler_similarity(first, second),
            nltk_distance.jaro_winkler_similarity(first, second),
        )


def test_jaro_repeated_characters_matches_nltk():
    pairs = [("aaaa", "aa"), ("aa", "aaaa"), ("a" * 9, "a" * 4), ("abab", "baba")]
    for first, second in pairs:
        assert_close(
            m.jaro_similarity(first, second),
            nltk_distance.jaro_similarity(first, second),
        )


def test_jaro_nan_elements_match_nltk():
    # NaN never equals NaN, so upstream reports no matches for these pairs.
    for first, second in [
        ([float("nan"), 1.0], [float("nan"), 2.0]),
        ([float("inf")], [float("inf")]),
        ([float("nan")], [1.0]),
    ]:
        assert_close(
            m.jaro_similarity(first, second),
            nltk_distance.jaro_similarity(first, second),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 3])
def test_scalar_edit_distance_raises_same_error(value):
    # nltk.metrics.distance.edit_distance/edit_distance_align call len(s1)
    # before anything else, so a bare float has no length.
    for first, second in [(value, value), (value, "ab"), ("ab", value)]:
        with pytest.raises(TypeError):
            nltk_distance.edit_distance(first, second)
        with pytest.raises(TypeError):
            m.edit_distance(first, second)
        with pytest.raises(TypeError):
            m.edit_distance_align(first, second)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 3])
def test_scalar_jaro_inputs_match_nltk(value):
    # nltk.metrics.distance.jaro_similarity returns 1.0 for equal arguments via
    # its s1 == s2 shortcut, and otherwise indexes s1[i] (TypeError for floats).
    if value == value:
        assert nltk_distance.jaro_similarity(value, value) == 1.0
        assert m.jaro_similarity(value, value) == 1.0
    else:
        with pytest.raises(TypeError):
            nltk_distance.jaro_similarity(value, value)
        with pytest.raises(TypeError):
            m.jaro_similarity(value, value)
    for first, second in [(value, "ab"), ("ab", value)]:
        with pytest.raises(TypeError):
            nltk_distance.jaro_similarity(first, second)
        with pytest.raises(TypeError):
            m.jaro_similarity(first, second)


def test_jaro_unicode_and_code_points_match_nltk():
    for first, second in [
        ("café", "cafe"),
        ("𐍈a", "a𐍈"),
        ("東京", "東亰"),
        ("e\u0301", "\u00e9"),
    ]:
        assert_close(
            m.jaro_similarity(first, second),
            nltk_distance.jaro_similarity(first, second),
        )
        assert_close(
            m.jaro_winkler_similarity(first, second),
            nltk_distance.jaro_winkler_similarity(first, second),
        )


@pytest.mark.parametrize("offset", [-1, -2, -(1 << 62)])
def test_edit_distance_negative_costs_match_nltk(offset):
    pairs = [("", ""), ("a", ""), ("abc", "abd"), ("ab", "ba"), ("kitten", "sitting")]
    for first, second in pairs:
        assert m.edit_distance(first, second, offset) == nltk_distance.edit_distance(
            first, second, offset
        )


@pytest.mark.parametrize(
    "cost",
    [0.5, 1.0, 2.5, -0.5, float("inf"), float("nan"), np.int64(3), np.float64(2.0), True],
)
def test_edit_distance_numeric_cost_types_match_nltk(cost):
    # nltk.metrics.distance.edit_distance never coerces substitution_cost, so any
    # real number is legal there; the port must agree, including float costs.
    for first, second in [("", ""), ("a", "b"), ("abc", "abd"), ("kitten", "sitting")]:
        assert_close(
            m.edit_distance(first, second, cost),
            nltk_distance.edit_distance(first, second, cost),
        )
        assert m.edit_distance_align(first, second, cost) == (
            nltk_distance.edit_distance_align(first, second, cost)
        )


def test_edit_distance_non_numeric_cost_raises_type_error():
    # Upstream fails in `lev[i - 1][j - 1] + substitution_cost`.
    for cost in ["x", None, [1]]:
        with pytest.raises(TypeError):
            nltk_distance.edit_distance("a", "b", cost)
        with pytest.raises(TypeError):
            m.edit_distance("a", "b", cost)
        with pytest.raises(TypeError):
            m.edit_distance_align("a", "b", cost)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("", ""),
        ("", "a"),
        ("a", ""),
        ("ab", "ba"),
        ("abc", "abd"),
        ("abcdef", "azced"),
        ("aaaa", "aa"),
        ("kitten", "sitting"),
        ("café", "cafe"),
        ([1, 2, 3], [3, 2]),
    ],
)
@pytest.mark.parametrize("cost", [1, 2, 3, 2.5])
def test_edit_distance_align_cases_match_nltk(first, second, cost):
    assert m.edit_distance_align(first, second, cost) == (
        nltk_distance.edit_distance_align(first, second, cost)
    )


def test_edit_distance_align_random_parity():
    rng = random.Random(5)
    for _ in range(150):
        first = "".join(rng.choices("abcd", k=rng.randrange(8)))
        second = "".join(rng.choices("abcd", k=rng.randrange(8)))
        cost = rng.choice([1, 2, 3, 2.5])
        assert m.edit_distance_align(first, second, cost) == (
            nltk_distance.edit_distance_align(first, second, cost)
        )


def test_edit_distance_duplicates_and_repeats_match_nltk():
    for transpositions in (False, True):
        for first, second in [
            ("aaaa", "aa"),
            ("aa", "aaaa"),
            ("abab", "baba"),
            ("ab", "ba"),
            ("ca", "abc"),
        ]:
            assert m.edit_distance(first, second, 1, transpositions) == (
                nltk_distance.edit_distance(first, second, 1, transpositions)
            )
