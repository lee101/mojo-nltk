from __future__ import annotations

import math
import numbers
import warnings

import numpy as np

from .._lib import address, lib

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _pair_ids(first, second) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(first, str) and isinstance(second, str):
        a = np.fromiter(map(ord, first), dtype=np.int64, count=len(first))
        b = np.fromiter(map(ord, second), dtype=np.int64, count=len(second))
        sigma = max(int(a.max(initial=-1)), int(b.max(initial=-1))) + 1
        return a, b, max(1, sigma)

    # nltk.metrics.distance.edit_distance/edit_distance_align both call len()
    # on their arguments, so an unsized input raises TypeError there too.
    len(first)
    len(second)
    values = []

    def encode(sequence):
        encoded = []
        for item in sequence:
            try:
                index = values.index(item)
            except ValueError:
                values.append(item)
                index = len(values) - 1
            encoded.append(index)
        return np.asarray(encoded, dtype=np.int64)

    return encode(first), encode(second), max(1, len(values))


def _numeric_cost(substitution_cost):
    # Matches nltk.metrics.distance.edit_distance: substitution_cost is only
    # added to table numbers, so any real number is accepted and anything else
    # raises TypeError when upstream would try to add it.
    if isinstance(substitution_cost, numbers.Real):
        return substitution_cost
    raise TypeError(
        "substitution_cost must be a real number, got "
        f"{type(substitution_cost).__name__}"
    )


def _as_int(cost):
    """Return cost as an exact Python int when it is integral, else None.

    Integers always convert exactly; a non-integral real (0.5, inf, nan) must
    take the float table because upstream stores the raw float in it.
    """
    if isinstance(cost, numbers.Integral):
        return int(cost)
    if not isinstance(cost, numbers.Real):
        return None
    value = float(cost)
    if not math.isfinite(value) or not value.is_integer():
        return None
    try:
        return int(cost)
    except (TypeError, ValueError, OverflowError):
        return int(value)


def _step_cost(cost, c1, c2):
    # The substitution term of nltk.metrics.distance._edit_dist_step, kept
    # separate so a mismatch adds the cost exactly once, as upstream does.
    return 0 if c1 == c2 else cost


def _fills_int64(len1, len2, cost):
    """Prove that every cell of upstream's recurrence stays inside int64.

    Every cell is at least the border distance and at most the plain path
    len1 + len2; a transposition reads such a cell plus i + j + 1, i.e. at most
    3 * (len1 + len2). A substitution adds at most min(len1, len2) * cost. A
    negative integer cost subtracts, so only use the exact int path otherwise.
    """
    bound = abs(cost) * min(len1, len2) + 3 * (len1 + len2) + 2
    return cost >= 0 and bound <= INT64_MAX


def _native_table(first, second, cost, transpositions):
    """Fill lev[][] natively; only called when _fills_int64 proved it safe."""
    len1, len2 = first.size, second.size
    first_buf = first if first.size else np.zeros(1, dtype=np.int64)
    second_buf = second if second.size else np.zeros(1, dtype=np.int64)
    table = np.empty((len1 + 1, len2 + 1), dtype=np.int64)
    if transpositions is None:
        lib().mnltk_edit_distance_align(
            address(first_buf, np.int64),
            address(second_buf, np.int64),
            len1,
            len2,
            cost,
            address(table, np.int64, writable=True),
        )
        return table
    sigma = max(int(first.max(initial=-1)), int(second.max(initial=-1))) + 1
    last = np.zeros(max(1, sigma), dtype=np.int64)
    lib().mnltk_edit_distance(
        address(first_buf, np.int64),
        address(second_buf, np.int64),
        len1,
        len2,
        cost,
        int(transpositions),
        address(table, np.int64, writable=True),
        address(last, np.int64, writable=True),
    )
    return table


def _python_table(first, second, cost, transpositions):
    """The same table in Python ints, for costs the int64 kernel cannot hold."""
    len1, len2 = first.size, second.size
    table = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        table[i][0] = i
    for j in range(len2 + 1):
        table[0][j] = j
    a_ids = first.tolist()
    b_ids = second.tolist()
    last_left = {}
    for i in range(1, len1 + 1):
        last_right_buf = 0
        for j in range(1, len2 + 1):
            c1 = a_ids[i - 1]
            c2 = b_ids[j - 1]
            last_left_j = last_left.get(c2, 0)
            last_right = last_right_buf
            if c1 == c2:
                last_right_buf = j
            a = table[i - 1][j] + 1
            b = table[i][j - 1] + 1
            c = table[i - 1][j - 1] + _step_cost(cost, c1, c2)
            d = c + 1
            if transpositions and last_left_j > 0 and last_right > 0:
                d = (
                    table[last_left_j - 1][last_right - 1]
                    + i
                    - last_left_j
                    + j
                    - last_right
                    - 1
                )
            table[i][j] = min(a, b, c, d)
        last_left[a_ids[i - 1]] = i
    return table


def _table(first, second, cost, transpositions):
    native_cost = _as_int(cost)
    if (
        native_cost is not None
        and INT64_MIN <= native_cost <= INT64_MAX
        and _fills_int64(first.size, second.size, native_cost)
    ):
        return _native_table(first, second, native_cost, transpositions)
    return _python_table(first, second, cost, transpositions)


def _cell(table, i, j):
    if isinstance(table, np.ndarray):
        return int(table[i, j])
    return table[i][j]


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    # Matches nltk.metrics.distance.edit_distance.
    cost = _numeric_cost(substitution_cost)
    first, second, _ = _pair_ids(s1, s2)
    table = _table(first, second, cost, transpositions)
    return _cell(table, first.size, second.size)


def edit_distance_align(s1, s2, substitution_cost=1):
    # Matches nltk.metrics.distance.edit_distance_align, including the
    # _edit_dist_backtrace precedence: substitute, skip s1, skip s2.
    cost = _numeric_cost(substitution_cost)
    first, second, _ = _pair_ids(s1, s2)
    table = _table(first, second, cost, False)
    i, j = first.size, second.size
    alignment = [(i, j)]
    while (i, j) != (0, 0):
        if i > 0 and j > 0:
            diagonal = _cell(table, i - 1, j - 1) + _step_cost(
                cost, int(first[i - 1]), int(second[j - 1])
            )
        else:
            diagonal = float("inf")
        skip_s1 = _cell(table, i - 1, j) + 1 if i > 0 else float("inf")
        skip_s2 = _cell(table, i, j - 1) + 1 if j > 0 else float("inf")
        # Upstream takes min(costs, key=itemgetter(0)): the first candidate wins
        # unless a later one is strictly smaller. Because NaN fails every "<"
        # comparison, a NaN diagonal (substitution_cost=nan) is kept as-is,
        # which makes upstream step to (i - 1, j - 1).
        candidates = (
            (diagonal, (i - 1, j - 1)),
            (skip_s1, (i - 1, j)),
            (skip_s2, (i, j - 1)),
        )
        best_cost, best = candidates[0]
        for candidate in candidates[1:]:
            if candidate[0] < best_cost:
                best_cost, best = candidate
        i, j = best
        alignment.append((i, j))
    alignment.reverse()
    return alignment


def jaro_similarity(s1, s2):
    # Upstream returns 1.0 before indexing anything, so equal scalars qualify.
    if s1 == s2:
        return 1.0
    first, second, _ = _pair_ids(s1, s2)
    first_buf = first if first.size else np.zeros(1, dtype=np.int64)
    second_buf = second if second.size else np.zeros(1, dtype=np.int64)
    first_flags = np.zeros(max(1, first.size), dtype=np.int64)
    second_flags = np.zeros(max(1, second.size), dtype=np.int64)
    return lib().mnltk_jaro_similarity(
        address(first_buf, np.int64),
        address(second_buf, np.int64),
        first.size,
        second.size,
        address(first_flags, np.int64, writable=True),
        address(second_flags, np.int64, writable=True),
    )


def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
    if not 0 <= max_l * p <= 1:
        warnings.warn(
            "The product  `max_l * p` might not fall between [0,1]."
            "Jaro-Winkler similarity might not be between 0 and 1."
        )
    score = jaro_similarity(s1, s2)
    prefix = 0
    for first, second in zip(s1, s2):
        if first != second:
            break
        prefix += 1
        if prefix == max_l:
            break
    return score + prefix * p * (1 - score)
