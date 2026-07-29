from __future__ import annotations

import warnings

import numpy as np

from .._lib import address, int64, lib


def _pair_ids(first, second) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(first, str) and isinstance(second, str):
        a = np.fromiter(map(ord, first), dtype=np.int64, count=len(first))
        b = np.fromiter(map(ord, second), dtype=np.int64, count=len(second))
        sigma = max(
            int(a.max(initial=-1)), int(b.max(initial=-1))
        ) + 1
        return a, b, max(1, sigma)
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


def _distance_table(s1, s2, substitution_cost=1, transpositions=False):
    substitution_cost = int64(substitution_cost, "substitution_cost")
    first, second, sigma = _pair_ids(s1, s2)
    # A positive substitution above the all-delete/all-insert path can never
    # win. Capping it preserves the result while preventing native addition
    # from overflowing. Very large negative costs cannot be represented by
    # the int64 dynamic-programming table and must fail explicitly.
    substitution_cost = min(substitution_cost, first.size + second.size)
    substitutions = min(first.size, second.size)
    if (
        substitution_cost < 0
        and substitutions
        and substitution_cost < -((1 << 63) // substitutions)
    ):
        raise OverflowError(
            "substitution_cost would overflow the signed 64-bit distance table"
        )
    first_buf = first if first.size else np.zeros(1, dtype=np.int64)
    second_buf = second if second.size else np.zeros(1, dtype=np.int64)
    table = np.empty((first.size + 1, second.size + 1), dtype=np.int64)
    last = np.zeros(sigma, dtype=np.int64)
    result = lib().mnltk_edit_distance(
        address(first_buf, np.int64),
        address(second_buf, np.int64),
        first.size,
        second.size,
        substitution_cost,
        int(transpositions),
        address(table, np.int64, writable=True),
        address(last, np.int64, writable=True),
    )
    return result, table


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    return _distance_table(s1, s2, substitution_cost, transpositions)[0]


def edit_distance_align(s1, s2, substitution_cost=1):
    _, table = _distance_table(s1, s2, substitution_cost, False)
    i, j = len(s1), len(s2)
    alignment = [(i, j)]
    while (i, j) != (0, 0):
        directions = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
        costs = []
        for pi, pj in directions:
            if pi < 0 or pj < 0:
                cost = float("inf")
            elif pi == i - 1 and pj == j - 1:
                cost = table[pi, pj] + (
                    0 if s1[pi] == s2[pj] else substitution_cost
                )
            else:
                cost = table[pi, pj] + 1
            costs.append((cost, (pi, pj)))
        _, (i, j) = min(costs, key=lambda value: value[0])
        alignment.append((i, j))
    return list(reversed(alignment))


def jaro_similarity(s1, s2):
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
