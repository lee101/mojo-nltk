# Verification Report

**Date:** 2026-11-11
**Machine:** Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 logical CPUs
**Environment:** Python 3.13.14, NLTK 3.10.0, Mojo 1.2.0.dev2026092605

## Benchmark Reproduction

Ran `pixi run bench` three times (each best of 3 warm runs) and compared against the README "after" column (single measurement on current code).

### My measurements (median of 3 runs)

| case | mojo-nltk (ms) | NLTK (ms) | ratio |
|---|---:|---:|---:|
| edit_distance (1,400 x 1,400) | 7.46 | 1471.28 | 197.23x |
| Damerau edit_distance (700 x 700) | 3.00 | 413.70 | 138.11x |
| jaro_similarity (20,000 chars) | 137.83 | 9100.26 | 66.03x |
| PorterStemmer.stem (50,000 words) | 119.54 | 588.59 | 4.92x |
| wordpunct_tokenize (1.1M chars) | 55.54 | 77.24 | 1.39x |
| trigrams iterator (300,000 tokens) | 22.50 | 56.03 | 2.49x |

### README "after" column (single run on current code)

| case | mojo-nltk (ms) | NLTK (ms) | ratio |
|---|---:|---:|---:|
| edit_distance (1,400 x 1,400) | 7.58 | 1259.71 | 166.11x |
| Damerau edit_distance (700 x 700) | 2.93 | 371.73 | 126.83x |
| jaro_similarity (20,000 chars) | 146.30 | 8164.87 | 55.81x |
| PorterStemmer.stem (50,000 words) | 112.84 | 589.64 | 5.23x |
| wordpunct_tokenize (1.1M chars) | 51.78 | 76.99 | 1.49x |
| trigrams iterator (300,000 tokens) | 21.36 | 54.12 | 2.53x |

### Deviation from README (percentage difference)

| case | mojo-nltk Δ | NLTK Δ | ratio Δ |
|---|---:|---:|---:|
| edit_distance | +2.4% | +16.8% | +18.7% |
| Damerau edit_distance | +2.4% | +11.3% | +8.9% |
| jaro_similarity | -5.8% | +11.5% | +18.3% |
| PorterStemmer.stem | +5.9% | -0.2% | -5.9% |
| wordpunct_tokenize | +7.3% | +0.3% | -6.7% |
| trigrams iterator | +5.3% | +3.5% | -1.6% |

**All measurements reproduce within 20%.** No corrections needed to the README table.

## Test Suite Verification

Ran `pixi run test`: **645 passed, 2 warnings** (both from upstream NLTK Jaro-Winkler warnings, identical on both sides).

### Parity checks verified:
- `edit_distance` and `edit_distance_align`: exact integer match against NLTK for all substitution costs (including float, negative, large), transpositions, Unicode, list/tuple inputs
- `jaro_similarity` and `jaro_winkler_similarity`: match within 1e-15 absolute tolerance against NLTK, plus exact agreement with published Census Bureau tables (rounded to 3 decimals)
- `PorterStemmer`: all three modes (ORIGINAL_ALGORITHM, MARTIN_EXTENSIONS, NLTK_EXTENSIONS) produce identical output to NLTK for published test vectors and 500 generated words
- `RegexpStemmer`: matches NLTK behavior
- All tokenizers (`WordPunctTokenizer`, `WhitespaceTokenizer`, `BlanklineTokenizer`, `SpaceTokenizer`, `TabTokenizer`, `CharTokenizer`, `LineTokenizer`, `RegexpTokenizer`): tokens, spans, sentence lists match NLTK
- `ngrams`, `bigrams`, `trigrams`, `everygrams`, `skipgrams`: exact sequence match against NLTK

### Git history check (`git log -p -- tests/`):

The test history shows **additions only**, no removals or loosening:
- Added `assert_close` helper with 1e-15 absolute tolerance (conftest.py)
- Added Census Bureau Jaro/Jaro-Winkler published table vectors with 3-decimal exact matching
- Added edge-case tests: empty strings, Unicode, NaN/inf elements, mismatched lengths, repeated characters
- Added substitution cost type tests (float, numpy types, negative, large) matching NLTK's unbounded Python integer semantics
- Added tokenizer parity tests for all covered tokenizers and span/sentence methods
- Added n-gram fast-path tests for common sequence types

No assertions were removed, no tolerances were widened.

## What Could Not Be Verified

- The README "before" column (phase: analyze) was explicitly noted as "not re-run" — no comparison possible.
- The kernel-level interleaved measurements (section 103-112 of README) require custom harnesses not in the repo; only the public entry-point `pixi run bench` was reproduced.
- GPU analysis (bench/analysis.md sections 138-156) was not executed — the decision gate is documented but no GPU code exists to test.

## Changes Made

None. All benchmarks reproduce within 20%, all tests pass with strict parity.