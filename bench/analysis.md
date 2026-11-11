# ANALYSIS: where mojo-nltk spends its time

This phase is measurement only. Nothing in `src/`, `python/`, or `tests/` was
changed; this file is the only addition to the repository.

## 1. Method, provenance, and noise

Every number below was produced in this session on this machine and copied from
a run's output. Rows tied to each table say which run produced them. Two kinds of
numbers appear:

- **measured**: wall time (`time.perf_counter`, best of N with GC disabled),
  bandwidth, or a byte/flop count that the run itself printed;
- **derived**: `bytes touched`, `flops`, `flops/byte` — exact arithmetic over the
  array shapes that run used (formula in section 4), not guesses.

Runs performed (all from this session):

| what | how | where the numbers land |
|---|---|---|
| headline comparison | `pixi run bench` | section 2 |
| decomposition + baselines | `/tmp/nltk_analyze/analyze.py` under the lock | sections 5-7 |
| interleaved re-measurement | `jaro_recheck.py`, `decomp.py` under the lock | sections 3, 5, 6 |
| stability (3 back-to-back) | `missing.py` (bench-shaped, load recorded) | section 10 |
| kernel throughput at 5 sizes | `missing.py` | section 8 |
| load-sensitivity of memory | `missing.py` (numpy copy) | section 8 |
| generated code | `objdump -d --no-show-raw-insn dist/libmojo-nltk.so` | section 9 |
| instruction attribution attempt | `valgrind --tool=callgrind perf_jaro.py` | section 9 (rejected, see there) |

Machine: `Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz`, 72 logical CPUs, Linux
6.8.0, Python 3.13.14, NLTK 3.10.0, numpy 2.5.1, Mojo 1.2.0.dev2026092605.

**The box is shared and was loaded.** `/proc/loadavg` (1-min) read between 36 and
91 through the session (39.97 was the lowest value recorded next to a timing
run). Three runs did not produce measurements and are named
here so they are not mistaken for results:

1. `pixi run bench` was terminated (`exit 143`) after ~15 min waiting on the
   shared `/tmp/mojo-bench.lock`. `pixi run bench` is `flock /tmp/mojo-bench.lock
   python bench/bench.py` with **no `-w`**; our session's wait was killed before
   the script started. It was re-run with
   `flock -w 3600 /tmp/mojo-bench.lock .pixi/envs/default/bin/python bench/bench.py`.
2. A first launch used bare `python` (3.12, outside the env) and died at
   `import nltk` with `ModuleNotFoundError` before timing anything.
3. Two later runs were queued behind a long-held lock and never started; the
   numbers they were meant to produce were re-measured by `missing.py`.

Because of the load, absolute milliseconds wobble by a few percent, and upstream
ratios move much more than that — NLTK's own timing for `edit_distance` varied
1261-1323 ms across runs, which is what turns a ~7.4 ms result into anywhere from
146x to 171x (section 10).
`mojo` itself was never rebuilt in this phase; `dist/libmojo-nltk.so` is newer
than `src/kernels.mojo` (22:30:11 vs 22:28:54) and was used as built by the
project.

## 2. Headline run (`bench/bench.py`, best of 3 warm runs)

The repository benchmark, unchanged:

| case | mojo-nltk | NLTK | ratio | result |
|---|---:|---:|---:|---|
| edit_distance (1,400 x 1,400) | 9.04 ms | 1322.53 ms | 146.35x | faster |
| Damerau edit_distance (700 x 700) | 2.99 ms | 377.05 ms | 126.00x | faster |
| jaro_similarity (20,000 chars) | 141.95 ms | 8607.48 ms | 60.64x | faster |
| PorterStemmer.stem (50,000 words) | 125.05 ms | 606.29 ms | 4.85x | faster |
| wordpunct_tokenize (1.1M chars) | 52.17 ms | 76.18 ms | 1.46x | faster |
| trigrams iterator (300,000 tokens) | 23.31 ms | 57.83 ms | 2.48x | faster |

Nothing is slower than upstream in this run, and nothing in section 10 reverses
that across three more invocations. The thin margins are the cases where
section 6 shows most of the time is outside the kernel.

## 3. Ranked hotspot table

Ranked by measured time of the **whole public entry point** at the sizes
`bench/bench.py` and its callers use. `glue` = Python/ctypes work outside the
native kernel; it is the "full" time minus the kernel-only time in the same run.

| # | function | input size | time | bytes touched (derived) | flops (derived) | flops/byte | bound | time sits in |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | `jaro_similarity` | 20,000 x 20,000 chars | 138.4 / 139.4 / 145.1 ms | 299,990,000 window steps x 32 B = 9.60 GB | 599,980,000 | 0.063 | memory/latency-bound | kernel ~137 ms, glue (ids+flags+ctypes) 1.8-3.2 ms |
| 2 | `PorterStemmer.stem` | 50,000 words, 540,000 chars | 113.7 / 113.8 / 117.1 ms | ~2.8 MB word bytes, re-touched per rule step | ~2.4 M rule tests | ~0.9 | latency-bound (tiny words) | kernel 51.2 ms, Python+ctypes 66.8 ms: **one crossing per word** |
| 3 | `wordpunct_tokenize` | 1,050,000 chars | 51.1 / 52.9 / 55.4 ms | 1.05 MB read + 16.8 MB span buffer | 3,150,000 | 3.0 (against the 1.05 MB read) | object/glue-bound | kernel 4.4-4.5 ms, Python 46.7-50.9 ms |
| 4 | `ngrams(tokens, 3)` | 300,000 tokens | 21.6-24.3 ms | 0 | 0 | 0 | object construction | ~100% Python (`zip` of `islice`) |
| 5 | `edit_distance` (Levenshtein) | 1,400 x 1,400 str | 7.42-7.6 ms | 1,962,801 cells x 48 B = 94.2 MB | 7,851,204 | 0.083 | memory+latency-bound | kernel 5.13-5.37 ms, glue 1.8-2.2 ms |
| 6 | `edit_distance_align` | 800 x 800 str | 4.33 ms | 641,601 cells x 40 B = 25.7 MB | 2,566,404 | 0.100 | memory+latency-bound | kernel table fill + Python backtrack (800 steps) |
| 7 | `edit_distance` (Damerau) | 700 x 700 str | 2.99-3.0 ms | 491,401 cells x 48 B = 23.6 MB | 2,457,005 | 0.104 | memory+latency-bound | kernel 2.69-2.70 ms |
| 8 | `jaro_winkler_similarity` | 20,000 chars | 295.1 ms | as row 1 | as row 1 | 0.063 | memory/latency-bound | one `jaro_similarity` + <=4-step Python prefix loop |
| 9 | `wordpunct_tokenize` (Unicode) | 1,050,000 chars | 81.9 ms | Python `re` internals | n/a | n/a | regex engine | pure-Python `re` path (same as upstream) |
| 10 | `edit_distance` (list input) | 1,000 x 1,000 ints | 23.6 ms | as row 5 | as row 5 | 0.083 | glue-bound | `_pair_ids` symbol encoding (`list.index` per element) |

Provenance: rows 1-5, 7, 8, 10 from `analyze.py` / `jaro_recheck.py` /
`decomp.py` / `missing.py`; row 6 from `analyze.py` (`other`), row 9 from
`analyze.py` (`tokenize_unicode_fallback`).

## 4. How bytes and flops were derived

Worked example, row 5: `cells = 1401 x 1401 = 1,962,801`. Each output cell does
3 table loads + 1 table store (4 x 8 B = 32 B), 2 symbol loads (16 B) and — for
Damerau only — one `last[]` load and store (16 B). `flops/cell` counts
min + add + subtract + cost-add = 4 (5 with the transposition candidate). This
under-counts branches and index arithmetic, so it under-states flops/byte
slightly — conservative for the memory-bound verdict.

Jaro: one window step = read `first[i]`, read `second[j]`, read and maybe write
the two flag arrays, i.e. 32 B and 2 comparisons (`==`, `== 0`). The window
width is `2 * (max(n1,n2)//2 - 1) + 1 = 19,999` at n = 20,000, and the number of
steps (299,990,000) was counted by the run, not estimated.

Tokenizer: 5 B touched per char (1 read + 1 classification write), 3 integer
comparisons per char.

## 5. Glue vs kernel, measured by decomposition

| entry point | full | kernel-only | ids/flags | allocation | per-element Python |
|---|---:|---:|---:|---:|---|
| edit 1,400 Levenshtein | 7.42-7.6 ms | 5.13-5.37 ms | 0.12-0.16 ms | 0.009 ms | — |
| edit 700 Damerau | 2.99 ms | 2.69-2.70 ms | 0.06 ms | 0.001 ms | — |
| jaro 20,000 | 139.8 ms | 137.9 ms | 1.78-3.19 ms | — | — |
| wordpunct 1.05M chars | 51.1-57.0 ms | 4.44-4.68 ms | — | 0.023 ms spans | `tolist` 16.1-18.3 ms + 210k `str` slices 31.0-35.9 ms |
| porter 50k words | 113.7-118.4 ms | 51.2 ms | — | — | ctypes loop 54.6 ms; `lower` 4.1 ms; `lower+bytearray` 14.3 ms |
| ngrams 300k tokens | 21.6-24.3 ms | n/a (no kernel) | — | 300k tuples | 69.7-75.7 ns/tuple, equal to raw `zip` (23.6 ms) |

Porter per word: 2.27-2.37 us total, 1.02-1.17 us in the ctypes call, 1.02-1.08 us
in the kernel body. The `ctypes_call_only` loop (54.6 ms) includes both the call
and the kernel body, so the pure FFI share is bounded by the gap between that
loop and `kernel_only` (51.2 ms): at most ~3.4 ms, i.e. ~0.07 us per word, which
the 0.97 us trivial-round-trip figure does not reproduce. What *is* measured is
the split of the 113.7-118.4 ms total: 51.2 ms kernel, 54.6 ms ctypes-call loop,
and the remaining ~10-12 ms Python object work (`lower` 4.1 ms,
`lower+bytearray` 14.3 ms, `decode`). Treating "kernel" as the floor, **~55% of
stemming is outside the kernel** — the call loop plus object churn — at 10.8
chars per word. `PorterStemmer()` construction refetches two bound methods:
fresh-stemmer 137.0 ms vs shared 113.8 ms for the same 50k words (23.2 ms one-off).

## 6. Python-level loops that re-enter ctypes per element, and per-call copies

**Per-element ctypes re-entry**
- `python/mojonltk/stem.py:63` — `PorterStemmer.stem` calls
  `mnltk_porter_stem_ascii` once *per word*: 50,000 crossings for 50,000 words,
  54.6 ms of loop time.
- `python/mojonltk/stem.py:73` — the Unicode variant does the same per word and
  also builds `np.fromiter(map(ord))` + destination + `"".join(map(chr))`:
  **16.4 us/word vs 2.3 us/word ASCII (7x)**, measured on 20,000 words.
- The distance and tokenizer paths make one call per invocation (good); their
  per-element cost is Python objects, not FFI.

**Per-call allocations / copies**
- `_pair_ids` (`distance.py:17-19`): two `int64` arrays per call
  (0.12-0.16 ms at n = 1,400), plus a second `max()` pass over both when
  transpositions are requested.
- `_native_table` (`distance.py:96`): `np.empty((n1+1)*(n2+1)*8)` — 15.7 MB at
  1,400; allocation itself 0.009 ms, faults paid in the kernel.
- `jaro_similarity` (`distance.py:228-229`): two zeroed `int64` flag arrays per
  call, plus the kernel's own equality pre-scan.
- `_ascii_offsets` (`tokenize.py:100-110`): `text.encode("ascii")` copy
  (0.08-0.23 ms) and `spans = np.empty((len(text), 2))` = **16.8 MB for 1.05 MB
  of text** — sized by characters, not tokens — then `tolist()` over
  420,000 elements (16.1-18.3 ms).
- `tokenize` (`tokenize.py:116-119`): 210,000 `str` slices (31.0-35.9 ms) — the
  unavoidable output, but it is the single largest Python cost.
- `stem` ASCII: `lower()` + `bytearray()` + `decode()` per word (14.3 ms /
  50k); the kernel mutates the caller's buffer in place, so there is no native
  input copy.

## 7. Full public API coverage at caller sizes

From `analyze.py` (`other`), both sides timed in the same process with the same
`best()` helper (5 repeats). Ratio = upstream / ours (higher is better):

| entry point | input | ours | upstream | ratio |
|---|---|---:|---:|---:|
| `edit_distance_align` | 800 x 800 str | 4.33 ms | 359.59 ms | 83.0x |
| `jaro_winkler_similarity` | 20,000 chars | 295.15 ms | 9276.51 ms | 31.4x |
| `edit_distance` (list) | 1,000 x 1,000 ints | 23.59 ms | 663.78 ms | 28.1x |
| `regexp_tokenize(r"\w+")` | 1.08M chars | 22.20 ms | 22.95 ms | 1.03x |
| `WhitespaceTokenizer.tokenize` | 1.08M chars | 24.90 ms | 24.56 ms | **0.99x** |
| `line_tokenize` | 1.08M chars | 1.98 ms | 1.98 ms | 1.00x |
| `SpaceTokenizer.tokenize` | 1.08M chars | 1.94 ms | 1.87 ms | **0.97x** |
| `RegexpStemmer.stem` | 20,000 words | 8.67 ms | 8.47 ms | **0.98x** |
| `CharTokenizer.tokenize` | 1M chars | 3.51 ms | n/a (absent in NLTK 3.10.0) | — |
| `everygrams(1, 2)` | 20,000 tokens | 9.64 ms | 9.72 ms | 1.01x |
| `skipgrams(n=2, k=1)` | 20,000 tokens | 15.28 ms | 15.55 ms | 1.02x |
| `bigrams` | 20,000 tokens | 1.39-1.46 ms | 3.84 ms | 2.77x |
| `pad_sequence` | 20,000 tokens | 0.95 ms | 0.94 ms | 1.00x |

The three ~1.0x rows marked in bold are pure-Python code shared with upstream
(they are the same regex or generator), so they sit at parity by construction.
They are in the table because they are part of the public API and take measurable
time, not because there is a kernel to optimise.

## 8. Absolute kernel rates (no upstream involved)

`missing.py`, load1 = 41.0 at the time of this block:

| kernel | input | kernel time | ns/cell or ns/char | effective GB/s |
|---|---|---:|---:|---:|
| Levenshtein table fill | 256 x 256 | 0.182 ms | 2.76 | 2.90 |
| | 512 x 512 | 0.777 ms | 2.95 | 2.71 |
| | 1,024 x 1,024 | 3.098 ms | 2.95 | 2.71 |
| | 1,400 x 1,400 | 5.206 ms | 2.65 | 3.02 |
| | 2,048 x 2,048 | 12.210 ms | 2.91 | 2.75 |
| wordpunct scan | 105,000 chars | 0.440 ms | 4.19 | 0.239 |
| | 420,000 chars | 1.750 ms | 4.17 | 0.240 |
| | 1,050,000 chars | 4.443 ms | 4.23 | 0.236 |
| | 2,100,000 chars | 8.838 ms | 4.21 | 0.238 |

Memory baselines on the same machine (`missing.py`, the 5 sizes and the copies
in one run):

| operation | size | result |
|---|---:|---:|
| `np.copyto` (read+write) | 1 MiB | 26.5 GB/s |
| | 8 MiB | 27.6 GB/s |
| | 16 MiB | 29.0 GB/s |
| | 512 KiB (L2-resident) | 26.4 GB/s |
| `np.fill` write-only, same 15.7 MB table as row 5 | 15.7 MB | 0.86 ms (0.44 ns/cell) |

Reading this honestly: the DP kernel's 2.65-2.95 ns/cell is **flat from 256 to
2,048** — a 64x range of table sizes — which says it is not bandwidth-limited;
a bandwidth-limited kernel would drop toward the copy rate as the table leaves
cache and it does not move. It is limited by the serial recurrence (each cell
waits on its three neighbours) and by ~3 memory references per cell. The
write-only comparison (0.44 ns/cell vs 2.65) bounds what removing the
dependency could buy: about 6x.

Jaro: its stated time is dominated by serial dependency and branch, not DRAM.
The working set of `b` (160 KB) + flags (160 KB) at n = 20,000 is L2/L3-resident,
the nominal 32 B per window step translate to ~10.5 GB/s (299,990,000 x 32 B /
137 ms), and those bytes are re-read within the same window rather than
streamed. So its 0.063 flops/byte should be read as "no arithmetic to amortise a
transfer", not as "streaming 9.6 GB from memory"; `objdump` (section 9) shows the
loop is a scalar compare-and-branch, which is the thing to fix.

A note on clocks: the TSC on this box is invariant at 2.30 GHz, but the core
was running at 1.242 GHz (`scaling_cur_freq` and `/proc/cpuinfo` agree), and
`perf_event_paranoid = 4` blocks `perf stat`, `rdmsr`, and the frequency
tracepoint. So real cycle counts are not quoted; the rates above are wall-clock.

## 9. What the generated code actually is

`objdump -d --no-show-raw-insn dist/libmojo-nltk.so`:

| kernel | instructions | vector ops |
|---|---:|---|
| `mnltk_edit_distance` | 222 | none (scalar `int64` recurrence) |
| `mnltk_edit_distance_align` | 95 | none |
| `mnltk_jaro_similarity` | 177 | only in the float epilogue (`vcvtsi2sd`, `vmulsd`, `vaddsd`); the match scan is a scalar `cmp`/`je` loop |
| `mnltk_wordpunct_spans` | 89 | none (scalar byte classification) |
| `mnltk_porter_stem_ascii` | 4,342 | none (branch-heavy suffix rules) |
| `mnltk_porter_stem` | 4,296 | none |

So every hotspot in section 3 is currently scalar: there is SIMD headroom in
Jaro (window compare) and in the DP fill (anti-diagonal), and none worth having
in the tokenizer or Porter (branch-dominated).

A caution about measurement tooling that is worth carrying into the next phase:
a first kernel-only timing showed the kernel at 287 ms — *slower than the full
wrapper that calls it* — because the timing script fetched
`lib.mnltk_jaro_similarity` from a fresh `CDLL`, where ctypes' default `c_int`
restype applies and the `double` return is truncated (0.98 instead of 0.826).
Going through `_lib.lib()` (which sets `argtypes`/`restype`) the same direct call
measures 136.9 ms against the wrapper's 139.8 ms, consistent with the
decomposition. Any benchmark written in the next phase must go through
`_lib.lib()` or set the prototype itself.

The Callgrind attempt is a separate, unresolved issue and is **not** used for any
claim here: three runs of `perf_jaro.py` (three input sizes) each reported
`I refs` near 78.5e9 regardless of size and attributed only 114 instructions to
`libmojo-nltk.so`, which cannot be right given the 299,990,000 window steps the
run's own inputs imply. The cause was not established in this phase; the
instruction counts in this section come from `objdump`, and the timing split
from same-process subtraction, not from the profiler.

## 10. Stability on a loaded machine, and rows slower than upstream

Three back-to-back bench-shaped invocations (`missing.py`, load1 recorded at
each), plus the `pixi run bench` and stability-run values from section 2:

| case | run at load 46.4 / 42.2 / 40.0 | `pixi run bench` run | spread |
|---|---|---|---|
| edit_distance | 169.9x / 171.3x / 170.5x | 146.4x | 146-171x |
| jaro_similarity | 62.2x / 60.2x / 58.7x | 60.6x | 59-62x |
| PorterStemmer.stem | 5.18x / 5.30x / 5.10x | 4.85x | 4.9-5.3x |
| wordpunct_tokenize | 1.43x / 1.36x / 1.43x | 1.46x | 1.36-1.46x |
| trigrams | 2.40x / 2.54x / 2.53x | 2.48x | 2.40-2.54x |

`wordpunct_tokenize` is the case to watch: 1.36-1.46x is the narrowest margin of
any native path, and 46.7-50.9 ms of its 51-55 ms is Python (section 5). A
machine with a faster allocator or a slower kernel could flip it. Within the
three back-to-back `missing.py` invocations, ours stayed at 7.42-7.47 ms
(`edit`), 113.7-117.1 ms (Porter) and 51.1-55.4 ms (tokenizer), while upstream
varied by 1.1-7.0% (largest: `trigrams` 54.7-58.5 ms). The 146x-171x spread for
`edit_distance` spans sessions and pairs each number with its own run: the
`pixi run bench` invocation measured 9.04 ms native against 1322.53 ms upstream,
and the later `missing.py` invocations 7.42-7.47 ms against 1261-1275 ms, at
lower load.

Rows where the port is **not** faster than upstream, from `analyze.py` (`other`;
each side one `best()` measurement at this size): `WhitespaceTokenizer.tokenize`
0.99x, `SpaceTokenizer.tokenize` 0.97x, `RegexpStemmer.stem` 0.98x, and
`regexp_tokenize` 1.03x (an NLTK util that the port re-exports through Python
`re`). These are shared pure-Python code paths; the deficits are 1-3% and are the
same order as the upstream-side spread measured above, so they are parity-within-
noise rather than a kernel regression. They are listed rather than dropped.

Also noted, flagged as *not compared* rather than faster or slower: the Unicode
stem path measured 16.4 us/word on 20,000 ASCII+`é` words, 7x the ASCII path's
2.3 us/word. No upstream number for that input was measured in this session, so
no verdict is given.

## 11. Next phase per hotspot

GPU is **no** for every row; the flops/byte column is the evidence: 0.063 to
0.104 for the DP/Jaro rows, 3.0 for the tokenizer against a ~1 MB input, and
Porter's words are 10.8 bytes each with 540 KB total. `MOJO_NOTES.md` puts the
CPU-only cutoff at "roughly 2 flops per byte"; the only candidates above it are
latency- and object-bound, so copy-in/copy-out would dominate.

| hotspot | next phase | evidence |
|---|---|---|
| jaro_similarity (row 1) | **simd** first (vector-compare the match window into flags); **parallel** only if a post-SIMD profile still shows the serial first-match dominating | 0.063 flops/byte; 137 ms of 139.8 ms in the kernel; scalar scan in `objdump`; L2-resident window, so bytes are not the limit |
| PorterStemmer.stem (row 2) | **parallel** — and the prerequisite is one call per corpus instead of 50,000: a batched entry point, `max.algorithm.parallelize` over independent words | 54.6 ms of 113.7 ms is the per-word ctypes loop; words are independent, 10.8 bytes each |
| wordpunct_tokenize (row 3) | **none** — the kernel is 4.4 ms of 51.1 ms; the cost is the 16.8 MB span buffer, `tolist`, and per-token slicing in Python. That is a Python-layer change (span buffer sized by token count, or offsets consumed lazily), not unroll/simd/gpu | kernel 4.4 ms vs Python 46.7-50.9 ms |
| ngrams / everygrams / skipgrams (row 4) | **none** — already at the raw `zip` floor (23.6 ms vs 21.6 ms for the wrapper) | no arithmetic, 0 bytes |
| edit_distance Levenshtein (row 5) | **simd** (anti-diagonal wavefront) first; **parallel** across blocks/diagonals only after that | 0.083 flops/byte; flat 2.65-2.95 ns/cell from 256 to 2,048; write-only 0.44 ns/cell puts a ~6x ceiling on removing the dependency |
| edit_distance Damerau (row 7) | **unroll** the inner loop and keep `last[]` hot; SIMD is obstructed by the data-dependent transposition lookup | 5.48 ns/cell vs 2.61 ns/cell for the plain kernel at the same 48 B/cell (the extra random `last[left]` / `table[(left-1)*cols+right-1]` loads) |
| edit_distance_align (row 6) | **none** — shares the plain DP fill; the Python backtrack is 800 steps here | 4.33 ms total, 0.100 flops/byte |
| jaro_winkler_similarity (row 8) | follows row 1 | 295.1 ms = the same `jaro_similarity` cost plus a <=4-iteration loop |
| Unicode tokenize / stem (row 9, `porter_unicode`) | **none** — deliberately uses Python `re` / code-point buffers for exact `\w` and Unicode semantics | tokenize 81.94 ms vs upstream 81.73 ms (parity); Unicode stem 16.4 us/word, no upstream comparison measured |
| `_pair_ids` for list inputs (row 10) | **none** for unroll/simd; it is a Python `dict`-encoding problem (replace `list.index` with a dict) | 23.6 ms for 1,000 x 1,000 ints vs 0.12-0.16 ms for the `str` path |

## 12. Reproducing

```bash
pixi install
pixi run build
flock -w 3600 /tmp/mojo-bench.lock .pixi/envs/default/bin/python bench/bench.py
```

The helper scripts used for the decomposition live in `/tmp/nltk_analyze/`
(`analyze.py`, `jaro_recheck.py`, `decomp.py`, `missing.py`, `perf_jaro.py`)
and are not part of the repository. Two earlier helpers (`stability.py`,
`throughput.py`) never acquired the shared lock, produced no numbers, and are
superseded by `missing.py`; nothing is quoted from them.
