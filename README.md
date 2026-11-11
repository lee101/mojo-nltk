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

| case | before (phase: analyze) | after (phase: unroll) | NLTK | ratio | result |
|---|---:|---:|---:|---:|---|
| edit_distance (1,400 x 1,400) | 8.06 ms | 7.58 ms | 1259.71 ms | 166.11x | faster |
| Damerau edit_distance (700 x 700) | 2.95 ms | 2.93 ms | 371.73 ms | 126.83x | faster |
| jaro_similarity (20,000 chars) | 138.88 ms | 146.30 ms | 8164.87 ms | 55.81x | faster |
| PorterStemmer.stem (50,000 words) | 128.84 ms | 112.84 ms | 589.64 ms | 5.23x | faster |
| wordpunct_tokenize (1.1M chars) | 56.09 ms | 51.78 ms | 76.99 ms | 1.49x | faster |
| trigrams iterator (300,000 tokens) | 20.49 ms | 21.36 ms | 54.12 ms | 2.53x | faster |

The `before` column is the release-audit `pixi run bench` recorded in the
phase before this one; it was not re-run. The `after` column and the NLTK and
ratio columns are one `pixi run bench` on the current code. Three cases are
Python-bound and outside the kernels this phase touched (Jaro, Porter, and
trigrams move by run-to-run noise on this shared machine), so the bench table
alone cannot isolate the kernel change. That is measured separately, by
interleaving the pre-change and post-change shared libraries on identical
inputs in one process, median of 6 paired rounds:

| kernel | input | before | after | speedup |
|---|---|---:|---:|---:|
| Levenshtein table fill (`mnltk_edit_distance_align`) | 1,400 x 1,400 | 5.16 ms | 3.68 ms | 1.41x |
| Damerau table fill (`mnltk_edit_distance`) | 700 x 700 | 2.76 ms | 2.75 ms | 1.00x |
| Jaro match scan | 20,000 chars | 284 ms | 288 ms | 1.00x |
| word/punct scan | 1,050,000 chars | 4.50 ms | 4.50 ms | 1.00x |

Those four rows and the entry-point figures below are medians over 4-6 such
sessions on the final binary; the Levenshtein row's best session was 1.44x and
its worst 1.31x, and the box was shared throughout, so read it as about 1.3-1.4x.

Through the public entry points, with the same Python glue on both sides,
`edit_distance_align` at 800 x 800 falls from 2.56 to 1.99 ms (1.28x) in the
final session (4.3 to 3.6 ms, 1.2x, at the bench's own size). Plain
`edit_distance` at 1,400 x 1,400 measures 1.00-1.03x through the entry point
even though its kernel is 1.4x faster: the unchanged buffer setup (`_pair_ids`
plus a 15.7 MB table allocation the kernel then faults in) dominates at that
size. That gap is Python-layer work and is not addressed in this phase.

Only the Levenshtein table fill changed. Unrolling the Damerau loop measured
1.00-1.07x and hoisting it 0.93-0.96x, so Damerau keeps its original code; the
Jaro and tokenizer unrolls changed results or measured at or under 1.00x and
were reverted whole. The Levenshtein kernel sums integers in the same order as
NLTK, so its table is bit-identical, not just within tolerance; nothing here
reorders a floating-point sum, and the existing 1e-15 float tolerance in the
tests is unchanged.

ASCII tokenization keeps its NumPy buffers zero-copy across the FFI call and
materializes the native offset buffer in bulk before constructing substrings.
ASCII Porter stemming mutates one caller-owned byte buffer in place, avoiding
per-word NumPy arrays, a native input copy, and code-point result construction;
Unicode words retain the exact code-point buffer path.
Unpadded n-grams over common built-in sequence types use a C-level `zip`
iterator; padded and one-shot iterable inputs retain the general deque path.

No GPU path is provided. The remaining kernels are byte classification,
ordered dynamic programming, suffix branching, or Python object construction,
not arithmetic-intense work above roughly two FLOPs per byte. GPU transfers
would lose here. Token boundaries are stateful and n-gram construction is
GIL-bound, while Porter words are tiny and branch-heavy, so these targets have
no genuinely large independent region that would repay CPU thread-launch
overhead. A SIMD tokenizer scan was measured but was slower on this short-run
workload and was not retained.

Run the verification and benchmark commands with:

```bash
pixi run build
pixi run test
pixi run bench
```

## How it works

Five native functions live in one Mojo compilation unit and are exported with
a C ABI. Python loads the shared object with `ctypes`; checked contiguous NumPy
arrays and writable byte buffers cross the boundary as pointers. Python keeps
every input and output buffer alive for the synchronous call and owns every
allocation, so the shared library has no cross-runtime allocator or freeing
contract.

Distance inputs become contiguous `int64` symbol IDs. Mojo fills an `int64`
row-major dynamic-programming matrix and, for Damerau distance, a caller-owned
last-seen table. This preserves Python sequence equality and Unicode character
semantics. The plain (Levenshtein) fill hoists the two row offsets out of the
inner loop and evaluates four columns per step with `comptime for`, leaving a
scalar tail for the last `second_len % 4` columns; the Damerau fill is a plain
loop because unrolling it did not pay. Jaro uses two caller-owned `int64`
match arrays. Porter stemming operates in place on a caller-owned byte buffer
for ASCII and uses caller-owned `int64` code-point buffers for Unicode; suffix
rules, measure, vowel handling, and all three upstream modes execute in Mojo.

Word-punctuation tokenization scans ASCII text in Mojo and returns `(start,
end)` offsets in an `int64` array. Unicode text uses Python's Unicode regular
expression engine so `\w` classification remains exactly compatible with
NLTK. Object-producing n-gram APIs and configurable regular expressions stay
in the Python layer.

## License

MIT
