# mojo-nltk

`mojo-nltk` is a standalone Mojo port of the compute-heavy core of a useful
subset of [NLTK](https://www.nltk.org/). It exposes an NLTK-shaped Python API
through `mojonltk`, with native Mojo kernels behind edit distance, Jaro
similarity, Porter stemming, and ASCII word/punctuation boundary scanning.

The project is intentionally a focused port, not a replacement for the full
NLTK distribution. Its behavior is tested directly against NLTK 3.10.0.

## Coverage

Covered:

- `mojonltk.metrics.distance`: `edit_distance`, full Damerau transpositions,
  `edit_distance_align`, `jaro_similarity`, and `jaro_winkler_similarity`
- `mojonltk.stem`: `PorterStemmer` with `ORIGINAL_ALGORITHM`,
  `MARTIN_EXTENSIONS`, and `NLTK_EXTENSIONS`, plus `RegexpStemmer`
- `mojonltk.tokenize`: `WordPunctTokenizer`, `RegexpTokenizer`,
  `WhitespaceTokenizer`, `BlanklineTokenizer`, `SpaceTokenizer`,
  `TabTokenizer`, `CharTokenizer`, and `LineTokenizer`, including span and
  sentence-list methods and the corresponding convenience functions
- `mojonltk.util`: `pad_sequence`, `ngrams`, `bigrams`, `trigrams`,
  `everygrams`, and `skipgrams`
- strings with Unicode code points, list and tuple inputs with hashable
  equality-comparable elements in distance functions, padding symbols, and
  one-shot iterators

Not covered:

- Punkt sentence models or NLTK data downloads
- Treebank, Toktok, Tweet, S-expression, XML, and language-specific tokenizers
- Snowball, Lancaster, ISRI, RSLP, CJK, or WordNet morphology
- NLTK corpus readers, tagging, parsing, classification, language models, and
  metrics outside the distance functions listed above

## Install

Install the pinned Mojo nightly, Python, NLTK, NumPy, and test dependencies,
then build the shared library:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-nltk.so`. The activated Pixi environment adds
`python/` to `PYTHONPATH`.

## Usage

This example runs as written:

```bash
pixi run python - <<'PY'
from mojonltk import PorterStemmer, edit_distance, ngrams, wordpunct_tokenize

text = "Fast tokenizers cost $3.88."
tokens = wordpunct_tokenize(text)

print(tokens)
print([PorterStemmer().stem(token) for token in tokens])
print(list(ngrams(tokens, 2)))
print(edit_distance("kitten", "sitting"))
PY
```

The covered API also supports the upstream-style submodule imports:

```python
from mojonltk.metrics.distance import jaro_winkler_similarity
from mojonltk.stem import PorterStemmer
from mojonltk.tokenize import WordPunctTokenizer
from mojonltk.util import everygrams
```

## Benchmarks

Measured in the final release audit with `pixi run bench` on this machine:
Intel Xeon E5-2697 v4 at 2.30 GHz (72 logical CPUs), Python 3.13.14, and
NLTK 3.10.0. Times are the best of three warm runs and include
Python-to-buffer conversion and result construction. The ratio is NLTK time
divided by mojo-nltk time.

| case | mojo-nltk | NLTK | ratio | result |
|---|---:|---:|---:|---|
| edit_distance (1,400 x 1,400) | 10.20 ms | 1334.77 ms | 130.89x | faster |
| Damerau edit_distance (700 x 700) | 3.22 ms | 437.16 ms | 135.89x | faster |
| jaro_similarity (20,000 chars) | 143.18 ms | 10894.39 ms | 76.09x | faster |
| PorterStemmer.stem (50,000 words) | 1095.65 ms | 599.73 ms | 0.55x | slower |
| wordpunct_tokenize (1.1M chars) | 56.79 ms | 74.72 ms | 1.32x | faster |
| trigrams iterator (300,000 tokens) | 21.69 ms | 56.37 ms | 2.60x | faster |

ASCII tokenization keeps its NumPy buffers zero-copy across the FFI call and
materializes the native offset buffer in bulk before constructing substrings.
Unpadded n-grams over common built-in sequence types use a C-level `zip`
iterator; padded and one-shot iterable inputs retain the general deque path.

No GPU path is provided. The remaining kernels are byte classification,
ordered dynamic programming, suffix branching, or Python object construction,
not arithmetic-intense work above roughly two FLOPs per byte. GPU transfers
would lose here. Token boundaries are stateful and n-gram construction is
GIL-bound, so neither target has a genuinely large independent region that
would repay CPU thread-launch overhead. A SIMD tokenizer scan was measured but
was slower on this short-run workload and was not retained.

Run the verification and benchmark commands with:

```bash
pixi run build
pixi run test
pixi run bench
```

## How it works

Four native functions live in one Mojo compilation unit and are exported with
a C ABI. Python loads the shared object with `ctypes`; checked contiguous NumPy
buffers cross the boundary as pointers. Python keeps every input and output
array alive for the synchronous call and owns every allocation, so the shared
library has no cross-runtime allocator or freeing contract.

Distance inputs become contiguous `int64` symbol IDs. Mojo fills an `int64`
row-major dynamic-programming matrix and, for Damerau distance, a caller-owned
last-seen table. This preserves Python sequence equality and Unicode character
semantics. Jaro uses two caller-owned `int64` match arrays. Porter stemming
operates in place on a caller-owned `int64` Unicode code-point buffer; suffix
rules, measure, vowel handling, and all three upstream modes execute in Mojo.

Word-punctuation tokenization scans ASCII text in Mojo and returns `(start,
end)` offsets in an `int64` array. Unicode text uses Python's Unicode regular
expression engine so `\w` classification remains exactly compatible with
NLTK. Object-producing n-gram APIs and configurable regular expressions stay
in the Python layer.

## License

MIT
