from __future__ import annotations

from collections import deque
from itertools import chain, combinations, islice

MAX_EVERYGRAMS_DEFAULT_LEN = 256


def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def _ngrams_iter(sequence, n, kwargs):
    sequence = pad_sequence(sequence, n, **kwargs)
    iterator = iter(sequence)
    window = deque(islice(iterator, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for item in iterator:
        window.append(item)
        yield tuple(window)


def ngrams(sequence, n, **kwargs):
    padding_keys = {
        "pad_left",
        "pad_right",
        "left_pad_symbol",
        "right_pad_symbol",
    }
    if (
        type(sequence) in (list, tuple, str, range)
        and isinstance(n, int)
        and 0 < n <= 8
        and kwargs.keys() <= padding_keys
        and not kwargs.get("pad_left", False)
        and not kwargs.get("pad_right", False)
    ):
        return zip(*(islice(sequence, offset, None) for offset in range(n)))
    return _ngrams_iter(sequence, n, kwargs)


def bigrams(sequence, **kwargs):
    return ngrams(sequence, 2, **kwargs)


def trigrams(sequence, **kwargs):
    return ngrams(sequence, 3, **kwargs)


def everygrams(
    sequence, min_len=1, max_len=-1, pad_left=False, pad_right=False, **kwargs
):
    if max_len == -1:
        try:
            max_len = len(sequence)
        except TypeError:
            sequence = list(sequence)
            max_len = len(sequence)
        if max_len > MAX_EVERYGRAMS_DEFAULT_LEN:
            raise ValueError(
                "everygrams() called with the default max_len on a sequence of "
                f"{max_len} items: enumerating every n-gram of every length up "
                f"to {max_len} yields O(n**2) tuples (O(n**3) elements) and can "
                "exhaust memory (CWE-770). Pass an explicit max_len (e.g. "
                "max_len=min_len), or raise nltk.util.MAX_EVERYGRAMS_DEFAULT_LEN."
            )
    sequence = pad_sequence(
        sequence, max_len, pad_left, pad_right, **kwargs
    )
    history = list(islice(sequence, max_len))
    while history:
        for ngram_len in range(min_len, len(history) + 1):
            yield tuple(history[:ngram_len])
        try:
            history.append(next(sequence))
        except StopIteration:
            pass
        del history[0]


def skipgrams(sequence, n, k, **kwargs):
    if "pad_left" in kwargs or "pad_right" in kwargs:
        sequence = pad_sequence(sequence, n, **kwargs)
    sentinel = object()
    for ngram in ngrams(
        sequence, n + k, pad_right=True, right_pad_symbol=sentinel
    ):
        head = ngram[:1]
        tail = ngram[1:]
        for skip_tail in combinations(tail, n - 1):
            if skip_tail[-1] is sentinel:
                continue
            yield head + skip_tail
