from __future__ import annotations

import re

import numpy as np

from ._lib import address, lib

_FLAGS = re.UNICODE | re.MULTILINE | re.DOTALL


class TokenizerI:
    def tokenize_sents(self, strings):
        return [self.tokenize(text) for text in strings]

    def span_tokenize_sents(self, strings):
        for text in strings:
            yield list(self.span_tokenize(text))


def _gap_spans(text, regexp):
    left = 0
    for match in regexp.finditer(text):
        if match.start() != left:
            yield left, match.start()
        left = match.end()
    yield left, len(text)


def _string_spans(text, separator):
    if not separator:
        raise ValueError("Token delimiter must not be empty")
    left = 0
    while True:
        right = text.find(separator, left)
        if right < 0:
            if left != len(text):
                yield left, len(text)
            return
        if right != 0:
            yield left, right
        left = right + len(separator)


class RegexpTokenizer(TokenizerI):
    def __init__(
        self, pattern, gaps=False, discard_empty=True, flags=_FLAGS
    ):
        self._pattern = getattr(pattern, "pattern", pattern)
        self._gaps = gaps
        self._discard_empty = discard_empty
        self._flags = flags
        self._regexp = None

    def _check_regexp(self):
        if self._regexp is None:
            self._regexp = re.compile(self._pattern, self._flags)

    def tokenize(self, text):
        self._check_regexp()
        if self._gaps:
            tokens = self._regexp.split(text)
            return [token for token in tokens if token] if self._discard_empty else tokens
        return self._regexp.findall(text)

    def span_tokenize(self, text):
        self._check_regexp()
        if self._gaps:
            for left, right in _gap_spans(text, self._regexp):
                if not (self._discard_empty and left == right):
                    yield left, right
        else:
            for match in self._regexp.finditer(text):
                yield match.span()

    def __repr__(self):
        return "{}(pattern={!r}, gaps={!r}, discard_empty={!r}, flags={!r})".format(
            self.__class__.__name__,
            self._pattern,
            self._gaps,
            self._discard_empty,
            self._flags,
        )


class WordPunctTokenizer(RegexpTokenizer):
    def __init__(self):
        super().__init__(r"\w+|[^\w\s]+")

    def span_tokenize(self, text):
        if not text.isascii():
            yield from super().span_tokenize(text)
            return
        offsets = self._ascii_offsets(text)
        iterator = iter(offsets)
        yield from zip(iterator, iterator)

    @staticmethod
    def _ascii_offsets(text):
        encoded = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        source = encoded if encoded.size else np.zeros(1, dtype=np.uint8)
        spans = np.empty((max(1, len(text)), 2), dtype=np.int64)
        count = lib().mnltk_wordpunct_spans(
            address(source, np.uint8),
            len(text),
            address(spans, np.int64, writable=True),
        )
        if not 0 <= count <= len(text):
            raise RuntimeError(f"native tokenizer returned invalid count {count}")
        return spans.ravel()[: count * 2].tolist()

    def tokenize(self, text):
        if not text.isascii():
            return super().tokenize(text)
        offsets = self._ascii_offsets(text)
        return [
            text[offsets[index] : offsets[index + 1]]
            for index in range(0, len(offsets), 2)
        ]


class WhitespaceTokenizer(RegexpTokenizer):
    def __init__(self):
        super().__init__(r"\s+", gaps=True)


class BlanklineTokenizer(RegexpTokenizer):
    def __init__(self):
        super().__init__(r"\s*\n\s*\n\s*", gaps=True)


class _StringTokenizer(TokenizerI):
    _string = None

    def tokenize(self, text):
        return text.split(self._string)

    def span_tokenize(self, text):
        if self._string is None:
            yield from enumerate(range(1, len(text) + 1))
            return
        yield from _string_spans(text, self._string)


class SpaceTokenizer(_StringTokenizer):
    _string = " "


class TabTokenizer(_StringTokenizer):
    _string = "\t"


class CharTokenizer(_StringTokenizer):
    _string = None

    def tokenize(self, text):
        return list(text)


class LineTokenizer(TokenizerI):
    def __init__(self, blanklines="discard"):
        if blanklines not in ("discard", "keep", "discard-eof"):
            raise ValueError(
                "Blank lines must be one of: discard keep discard-eof"
            )
        self._blanklines = blanklines

    def tokenize(self, text):
        lines = text.splitlines()
        if self._blanklines == "discard":
            lines = [line for line in lines if line.rstrip()]
        elif self._blanklines == "discard-eof" and lines and not lines[-1].strip():
            lines.pop()
        return lines

    def span_tokenize(self, text):
        if self._blanklines == "keep":
            yield from _string_spans(text, r"\n")
        else:
            yield from _gap_spans(text, re.compile(r"\n(\s+\n)*"))


def regexp_tokenize(
    text, pattern, gaps=False, discard_empty=True, flags=_FLAGS
):
    return RegexpTokenizer(pattern, gaps, discard_empty, flags).tokenize(text)


def wordpunct_tokenize(text):
    return WordPunctTokenizer().tokenize(text)


def blankline_tokenize(text):
    return BlanklineTokenizer().tokenize(text)


def line_tokenize(text, blanklines="discard"):
    return LineTokenizer(blanklines).tokenize(text)
