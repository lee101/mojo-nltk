from __future__ import annotations

import ctypes
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
        self._mode_id = self._MODES[mode]
        native = lib()
        self._native_stem_ascii = native.mnltk_porter_stem_ascii
        self._native_stem_unicode = native.mnltk_porter_stem

    def stem(self, word, to_lowercase=True):
        stem = word.lower() if to_lowercase else word
        if self.mode == self.NLTK_EXTENSIONS and stem in self._IRREGULAR:
            return self._IRREGULAR[stem]
        if self.mode != self.ORIGINAL_ALGORITHM and len(word) <= 2:
            return stem
        if not stem:
            return stem
        if stem.isascii():
            word_buffer = bytearray(stem, "ascii")
            length = self._native_stem_ascii(
                ctypes.addressof(ctypes.c_uint8.from_buffer(word_buffer)),
                len(word_buffer),
                self._mode_id,
            )
            if not 0 <= length <= len(word_buffer):
                raise RuntimeError(f"native stemmer returned invalid length {length}")
            return word_buffer[:length].decode("ascii")
        source = np.fromiter(map(ord, stem), dtype=np.int64, count=len(stem))
        destination = np.empty(max(8, source.size + 8), dtype=np.int64)
        length = self._native_stem_unicode(
            address(source, np.int64),
            source.size,
            address(destination, np.int64, writable=True),
            self._mode_id,
        )
        if not 0 <= length <= destination.size:
            raise RuntimeError(f"native stemmer returned invalid length {length}")
        return "".join(map(chr, destination[:length].tolist()))

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
