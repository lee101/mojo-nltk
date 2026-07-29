from __future__ import annotations

import re

import numpy as np

from ._lib import address, lib


class PorterStemmer:
    NLTK_EXTENSIONS = "NLTK_EXTENSIONS"
    MARTIN_EXTENSIONS = "MARTIN_EXTENSIONS"
    ORIGINAL_ALGORITHM = "ORIGINAL_ALGORITHM"

    _MODES = {
        ORIGINAL_ALGORITHM: 0,
        MARTIN_EXTENSIONS: 1,
        NLTK_EXTENSIONS: 2,
    }
    _IRREGULAR = {
        "sky": "sky",
        "skies": "sky",
        "dying": "die",
        "lying": "lie",
        "tying": "tie",
        "news": "news",
        "innings": "inning",
        "inning": "inning",
        "outings": "outing",
        "outing": "outing",
        "cannings": "canning",
        "canning": "canning",
        "howe": "howe",
        "proceed": "proceed",
        "exceed": "exceed",
        "succeed": "succeed",
    }

    def __init__(self, mode=NLTK_EXTENSIONS):
        if mode not in self._MODES:
            raise ValueError(
                "Mode must be one of PorterStemmer.NLTK_EXTENSIONS, "
                "PorterStemmer.MARTIN_EXTENSIONS, or "
                "PorterStemmer.ORIGINAL_ALGORITHM"
            )
        self.mode = mode

    def stem(self, word, to_lowercase=True):
        stem = word.lower() if to_lowercase else word
        if self.mode == self.NLTK_EXTENSIONS and stem in self._IRREGULAR:
            return self._IRREGULAR[stem]
        if self.mode != self.ORIGINAL_ALGORITHM and len(word) <= 2:
            return stem
        source = np.fromiter(map(ord, stem), dtype=np.int64, count=len(stem))
        source_buf = source if source.size else np.zeros(1, dtype=np.int64)
        destination = np.empty(max(8, source.size + 8), dtype=np.int64)
        length = lib().mnltk_porter_stem(
            address(source_buf, np.int64),
            source.size,
            address(destination, np.int64, writable=True),
            self._MODES[self.mode],
        )
        if not 0 <= length <= destination.size:
            raise RuntimeError(f"native stemmer returned invalid length {length}")
        return "".join(chr(int(codepoint)) for codepoint in destination[:length])

    def __repr__(self):
        return "<PorterStemmer>"


class RegexpStemmer:
    def __init__(self, regexp, min=0):
        self._regexp = regexp if hasattr(regexp, "pattern") else re.compile(regexp)
        self._min = min

    def stem(self, word):
        return word if len(word) < self._min else self._regexp.sub("", word)

    def __repr__(self):
        return f"<RegexpStemmer: {self._regexp.pattern!r}>"
