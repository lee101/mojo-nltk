"""Native kernels for the covered NLTK-compatible API."""

from max.algorithm import parallelize
from std.sys import simd_width_of

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


# Matches nltk.metrics.distance.edit_distance/_edit_dist_step: the table is a
# plain int64 array in the same shape as upstream's Python list of rows. The
# caller proves that no cell can leave int64 before calling (see _fills_int64 in
# the Python wrapper); anything else recomputes with Python integers there.
@export("mnltk_edit_distance")
def mnltk_edit_distance(
    first_addr: Int,
    second_addr: Int,
    first_len: Int,
    second_len: Int,
    substitution_cost: Int,
    transpositions: Int,
    table_addr: Int,
    last_addr: Int,
) abi("C") -> Int:
    var first = ip(first_addr)
    var second = ip(second_addr)
    var table = ip(table_addr)
    var last = ip(last_addr)
    var cols = second_len + 1
    for i in range(first_len + 1):
        table[i * cols] = Int64(i)
    for j in range(second_len + 1):
        table[j] = Int64(j)
    for i in range(1, first_len + 1):
        var last_right_buf = 0
        for j in range(1, second_len + 1):
            var left = Int(last[Int(second[j - 1])])
            var right = last_right_buf
            var same = first[i - 1] == second[j - 1]
            if same:
                last_right_buf = j
            var deletion = Int(table[(i - 1) * cols + j]) + 1
            var insertion = Int(table[i * cols + j - 1]) + 1
            var substitution = Int(table[(i - 1) * cols + j - 1])
            if not same:
                substitution += substitution_cost
            var transpose = substitution + 1
            if transpositions != 0 and left > 0 and right > 0:
                transpose = (
                    Int(table[(left - 1) * cols + right - 1])
                    + i - left + j - right - 1
                )
            table[i * cols + j] = Int64(
                min(deletion, min(insertion, min(substitution, transpose)))
            )
        last[Int(first[i - 1])] = Int64(i)
    return Int(table[first_len * cols + second_len])


comptime W = simd_width_of[DType.int64]()

# Anti-diagonal fill only above this total length. Below it the table is
# cache-resident and the wavefront's separate diagonal-2 stream costs more than
# it saves (see README: it measured slower, so the row fill is kept there).
comptime WAVE_LIMIT = 200


# One W-wide slice of the anti-diagonal i + j = d.
#
# Cells on one anti-diagonal are independent: each reads only the diagonal-1
# (deletion, insertion) and diagonal-2 (substitution) cells. A slice is
# contiguous in the row-major table because stepping one cell along the
# diagonal is `+cols - 1`. All values stay in int64 (the caller proved it).
@no_inline
def diagonal_block(
    first: IPtr,
    second: IPtr,
    table: IPtr,
    cols: Int,
    base: Int,
    i0: Int,
    j0: Int,
    substitution_cost: Int,
) -> None:
    # row k of the slice is i0 - k, column k is j0 + k
    var rows = SIMD[DType.int64, W](0)
    var columns = SIMD[DType.int64, W](0)
    comptime for k in range(W):
        rows[k] = Int64(i0 - k)
        columns[k] = Int64(j0 + k)
    var previous = base - cols - 1
    var deletion = table.load[width=W](previous + 1)
    var insertion = table.load[width=W](previous + cols)
    var diagonal = table.load[width=W](previous)
    var first_symbols = first.load[width=W](i0 - W + 1)
    var second_symbols = second.load[width=W](j0)
    # mismatch cost: 0 on equal lanes. `select` takes its first argument on
    # true lanes; it is the only 0/1-producing comparison path on this
    # toolchain (Bool SIMD has no cast, `&`/`|` on Bool vectors reduce to one
    # scalar Bool).
    var cost = SIMD[DType.int64, W](substitution_cost)
    var zero = SIMD[DType.int64, W](0)
    var mismatch = first_symbols.eq(second_symbols).select(zero, cost)
    var one = SIMD[DType.int64, W](1)
    var substitution = diagonal + mismatch
    var transpose = substitution + one
    var deletion_up = deletion + one
    var insertion_left = insertion + one
    var result = min(min(deletion_up, insertion_left), min(substitution, transpose))
    table.store(base, result)


def edit_distance_wavefront(
    first: IPtr,
    second: IPtr,
    table: IPtr,
    first_len: Int,
    second_len: Int,
    cols: Int,
    substitution_cost: Int,
) -> None:
    # Fill the first `split` anti-diagonals W cells at a time, the remaining
    # (longer than any W slice) ones a row at a time. The cell values are
    # computed by the same expression in both halves, so the table equals the
    # plain row-major fill bit for bit.
    var split = (first_len + second_len) // 2
    var d = 2
    while d <= split:
        var high = min(d - 1, first_len)
        var low_column = d - high
        if low_column > second_len:
            low_column = second_len
            high = d - low_column
        var count = min(first_len - high, low_column - 1) + 1
        var done = 0
        while done + W <= count:
            var i = high - done
            var j = low_column + done
            diagonal_block(
                first, second, table, cols, i * cols + j, i, j,
                substitution_cost,
            )
            done += W
        while done < count:
            var i = high - done
            var j = low_column + done
            var deletion = Int(table[(i - 1) * cols + j]) + 1
            var insertion = Int(table[i * cols + j - 1]) + 1
            var substitution = Int(table[(i - 1) * cols + j - 1])
            if first[i - 1] != second[j - 1]:
                substitution += substitution_cost
            table[i * cols + j] = Int64(
                min(deletion, min(insertion, substitution))
            )
            done += 1
        d += 1
    for i in range(1, first_len + 1):
        var previous = (i - 1) * cols
        var current = i * cols
        var row_char = Int64(first[i - 1])
        var j = max(1, split - i + 1)
        while j <= second_len:
            var deletion = Int(table[previous + j]) + 1
            var insertion = Int(table[current + j - 1]) + 1
            var substitution = Int(table[previous + j - 1])
            if row_char != second[j - 1]:
                substitution += substitution_cost
            table[current + j] = Int64(
                min(deletion, min(insertion, substitution))
            )
            j += 1


# Matches the lev[i][j] table that nltk.metrics.distance.edit_distance_align
# fills before _edit_dist_backtrace (which stays in Python). Same overflow
# contract as mnltk_edit_distance.
@export("mnltk_edit_distance_align")
def mnltk_edit_distance_align(
    first_addr: Int,
    second_addr: Int,
    first_len: Int,
    second_len: Int,
    substitution_cost: Int,
    table_addr: Int,
) abi("C") -> Int:
    var first = ip(first_addr)
    var second = ip(second_addr)
    var table = ip(table_addr)
    var cols = second_len + 1
    for i in range(first_len + 1):
        table[i * cols] = Int64(i)
    for j in range(second_len + 1):
        table[j] = Int64(j)
    if first_len >= 2 and second_len >= 2 and first_len + second_len <= WAVE_LIMIT:
        edit_distance_wavefront(
            first, second, table, first_len, second_len, cols,
            substitution_cost,
        )
        return Int(table[first_len * cols + second_len])
    for i in range(1, first_len + 1):
        var previous = (i - 1) * cols
        var current = i * cols
        var row_char = Int64(first[i - 1])
        var j = 1
        while j + 3 <= second_len:
            comptime for k in range(4):
                var deletion = Int(table[previous + j + k]) + 1
                var insertion = Int(table[current + j + k - 1]) + 1
                var substitution = Int(table[previous + j + k - 1])
                if row_char != second[j + k - 1]:
                    substitution += substitution_cost
                table[current + j + k] = Int64(
                    min(deletion, min(insertion, substitution))
                )
            j += 4
        while j <= second_len:
            var deletion = Int(table[previous + j]) + 1
            var insertion = Int(table[current + j - 1]) + 1
            var substitution = Int(table[previous + j - 1])
            if row_char != second[j - 1]:
                substitution += substitution_cost
            table[current + j] = Int64(
                min(deletion, min(insertion, substitution))
            )
            j += 1
    return Int(table[first_len * cols + second_len])


@export("mnltk_jaro_similarity")
def mnltk_jaro_similarity(
    first_addr: Int,
    second_addr: Int,
    first_len: Int,
    second_len: Int,
    first_flags_addr: Int,
    second_flags_addr: Int,
) abi("C") -> Float64:
    var first = ip(first_addr)
    var second = ip(second_addr)
    var first_flags = ip(first_flags_addr)
    var second_flags = ip(second_flags_addr)
    if first_len == second_len:
        var equal = True
        for i in range(first_len):
            if first[i] != second[i]:
                equal = False
                break
        if equal:
            return 1.0
    if first_len == 0 or second_len == 0:
        return 0.0
    var match_bound = max(first_len, second_len) // 2 - 1
    var matches = 0
    for i in range(first_len):
        var lower = max(0, i - match_bound)
        var upper = min(i + match_bound, second_len - 1)
        for j in range(lower, upper + 1):
            if first[i] == second[j] and second_flags[j] == 0:
                first_flags[i] = 1
                second_flags[j] = 1
                matches += 1
                break
    if matches == 0:
        return 0.0
    var i = 0
    var j = 0
    var transpositions = 0
    while i < first_len and j < second_len:
        while i < first_len and first_flags[i] == 0:
            i += 1
        while j < second_len and second_flags[j] == 0:
            j += 1
        if i < first_len and j < second_len:
            if first[i] != second[j]:
                transpositions += 1
            i += 1
            j += 1
    var m = Float64(matches)
    # Same arithmetic order as nltk.metrics.distance.jaro_similarity:
    # 1/3 * (m/|s1| + m/|s2| + (m - t//2)/m).
    return (
        m / Float64(first_len)
        + m / Float64(second_len)
        + Float64(matches - transpositions // 2) / m
    ) * (1.0 / 3.0)


# --- Parallel Jaro (see README, "Parallelism") -----------------------------
#
# The match pass above is O(n^2) in the match window and runs one row at a time
# on one core. Rows `step` windows apart never have to see each other's marks:
# their windows share fewer columns than their first windows are apart. So the
# rows are cut into blocks of `step` rows, every block scans against its own
# empty second-side flags (in parallel), and a sequential fold then applies the
# serial rule to the whole row order.
#
# A block records, for each of its rows, the first three ascending columns in
# the row's window where its symbol occurs unseen. The fold takes the first of
# those still free, which is the row's first free column: every column the block
# marked below it was taken by an earlier row of the same block, and at most two
# of those can mark inside this row's window. When a row takes a column, the
# serial loop leaves its cursor (`last_right`) on that column and marks the next
# row's window end with it; the fold repeats that from the block's recorded
# cursor.
#
# The export takes three arguments: a longer ctypes argument list crashes this
# toolchain even with a trivial body. Python sizes and zeroes the arena; layout
# in int64 words from the address given:
#   [0] n1  [1] n2  [2] matched count  [3] transpositions  [4] block rows
# then  first ids (n1) | second ids (n2 + 8 guard words) |
#       candidates (3 per row) | block cursors (1 per block) |
#       second flags (n2) | first flags (n1)
# The +8 words after the second-side ids take the serial loop's cursor write at
# the end of the last window, which does not check the length.

# Public calls with fewer first-side rows than this stay serial; it is set from
# the measurements in README ("Parallelism") and must match PY below.
comptime JARO_PARALLEL_MIN_ROWS = 35000
# Measured on the shared box (README, "Parallelism").
comptime JARO_MAX_WORKERS = 4
comptime JARO_CANDIDATES = 3


@always_inline
def jaro_offer(candidates: IPtr, row: Int, value: Int, found: Int) -> Int:
    """Insert one distinct candidate into `row`'s ascending list."""
    var k = 0
    while k < found:
        if Int(candidates[row * 3 + k]) == value:
            return found
        k += 1
    k = found
    while k > 0 and Int(candidates[row * 3 + k - 1]) > value:
        candidates[row * 3 + k] = candidates[row * 3 + k - 1]
        k -= 1
    candidates[row * 3 + k] = Int64(value)
    return found + 1


def jaro_block(
    first: IPtr,
    second: IPtr,
    candidates: IPtr,
    second_flags: IPtr,
    start: Int,
    end: Int,
    second_len: Int,
    match_bound: Int,
) -> Int:
    """Rows [start, end) against this block's own empty flags; returns its cursor."""
    var cursor = 0
    var i = start
    while i < end:
        candidates[i * 3] = 0
        candidates[i * 3 + 1] = 0
        candidates[i * 3 + 2] = 0
        var symbol = first[i]
        var found = 0
        var j = max(0, i - match_bound)
        var hi = min(i + match_bound, second_len - 1)
        while j <= hi and found < 3:
            if symbol == second[j] and second_flags[j] == 0:
                second_flags[j] = 1
                cursor = j
                found = jaro_offer(candidates, i, Int(second[j]), found)
            j += 1
        i += 1
    return cursor


def jaro_fold(
    candidates: IPtr,
    cursors: IPtr,
    second_flags: IPtr,
    first_flags: IPtr,
    n1: Int,
    block_rows: Int,
) -> Int:
    """The serial rule over the whole row order, using the block results.

    The serial loop marks `last_right` (the block's last matched column) at the
    end of every failed row's window; putting that mark back on the next row's
    window end is what the block's cursor records, and a row that matches moves
    the cursor forward from there. The cursor is therefore re-written on every
    row: after a match it is that column, otherwise it is where the block left
    it, which is the column the next row would have marked.
    """
    var matched = 0
    var block = 0
    var start = 0
    var match_bound = 0
    while start < n1:
        var end = min(start + block_rows, n1)
        var cursor = Int(cursors[block])
        var i = start
        while i < end:
            var value = 0
            var r = 0
            while r < 3:
                value = Int(candidates[i * 3 + r])
                if value != 0 and second_flags[value] != 0:
                    second_flags[value] = 0
                    first_flags[i] = 1
                    matched += 1
                    cursor = value
                    break
                r += 1
            second_flags[cursor] = 0
            i += 1
        _ = match_bound
        block += 1
        start = end
    return matched


def jaro_count_transpositions(
    first: IPtr,
    second: IPtr,
    second_flags: IPtr,
    first_flags: IPtr,
    n1: Int,
    n2: Int,
) -> Int:
    # The serial walk, over the mark arrays this pass produced (second: nonzero
    # is free, 0 is matched, the complement of first_flags).
    var transpositions = 0
    var i = 0
    var j = 0
    while i < n1 and j < n2:
        while i < n1 and first_flags[i] == 0:
            i += 1
        while j < n2 and second_flags[j] == 0:
            j += 1
        if i < n1 and j < n2:
            if first[i] != second[j]:
                transpositions += 1
            i += 1
            j += 1
    return transpositions


# Returns 0 when Python must use the serial export, else 1 with [2] and [3] set.
@export("mnltk_jaro_similarity_parallel")
def mnltk_jaro_similarity_parallel(
    workspace_addr: Int, workers: Int, block_rows: Int
) abi("C") -> Int:
    var workspace = ip(workspace_addr)
    var n1 = Int(workspace[0])
    var n2 = Int(workspace[1])
    if (
        workers < 2
        or workers > JARO_MAX_WORKERS
        or block_rows < 1
        or n1 < JARO_PARALLEL_MIN_ROWS
    ):
        return 0
    var first = workspace.unsafe_offset(6)
    var second = first.unsafe_offset(n1)
    var candidates = second.unsafe_offset(n2 + 8)
    var blocks = (n1 + block_rows - 1) // block_rows
    var cursors = candidates.unsafe_offset(n1 * 3)
    var second_flags = cursors.unsafe_offset(blocks)
    var first_flags = second_flags.unsafe_offset(n2)
    var match_bound = max(n1, n2) // 2 - 1
    if match_bound < 0:
        return 0
    var i = 0
    while i < n1:
        first_flags[i] = 0
        i += 1

    def work(index: Int) {imm}:
        var start = index * block_rows
        var got = jaro_block(
            first,
            second,
            candidates,
            second_flags,
            start,
            min(start + block_rows, n1),
            n2,
            max(n1, n2) // 2 - 1,
        )
        cursors[index] = Int64(1000000 + index * 100000 + got)

    parallelize(work, blocks, workers)
    var matched = jaro_fold(
        candidates, cursors, second_flags, first_flags, n1, block_rows
    )
    var transpositions = jaro_count_transpositions(
        first, second, second_flags, first_flags, n1, n2
    )
    workspace[2] = Int64(matched)
    workspace[3] = Int64(transpositions)
    return 1


def helper_write(tags: IPtr, flags: IPtr, start: Int, end: Int) -> Int:
    var acc = 0
    var i = start
    while i < end:
        if flags[i] == 0:
            tags[i] = Int64(i + 1)
            acc += 1
        i += 1
    return acc


@export("mnltk_jaro_probe_layout")
def mnltk_jaro_probe_layout(
    workspace_addr: Int, workers: Int, mode: Int
) abi("C"):
    var workspace = ip(workspace_addr)
    var n1 = Int(workspace[0])
    var n2 = Int(workspace[1])
    var first = workspace.unsafe_offset(6)
    var second = first.unsafe_offset(n1)
    var candidates = second.unsafe_offset(n2 + 8)
    var blocks = (n1 + 63) // 64
    var cursors = candidates.unsafe_offset(n1 * 3)
    var second_flags = cursors.unsafe_offset(blocks)

    def work(index: Int) {imm}:
        var start = index * 64
        var end = min(start + 64, n1)
        var i = start
        var got = 0
        while i < end:
            candidates[i * 3] = 0
            candidates[i * 3 + 1] = 0
            candidates[i * 3 + 2] = 0
            var symbol = first[i]
            var found = 0
            var j = max(0, i - (max(n1, n2) // 2 - 1))
            var hi = min(i + (max(n1, n2) // 2 - 1), n2 - 1)
            while j <= hi and found < 3:
                if symbol == second[j] and second_flags[j] == 0:
                    second_flags[j] = 1
                    got = j
                    found = jaro_offer(candidates, i, Int(second[j]), found)
                j += 1
            i += 1
        cursors[index] = Int64(got)

    parallelize(work, blocks, workers)
    _ = mode


# Matches nltk WordPunctTokenizer(r"\w+|[^\w\s]+"): Python re treats 0x1c-0x1f
# as \s, so they are separators here too.
def ascii_class(c: UInt8) -> Int:
    if (
        c == 9
        or c == 10
        or c == 11
        or c == 12
        or c == 13
        or (c >= 28 and c <= 32)
    ):
        return 0
    if (
        (c >= 48 and c <= 57)
        or (c >= 65 and c <= 90)
        or c == 95
        or (c >= 97 and c <= 122)
    ):
        return 1
    return 2


@export("mnltk_wordpunct_spans")
def mnltk_wordpunct_spans(
    text_addr: Int, text_len: Int, spans_addr: Int
) abi("C") -> Int:
    var text = bp(text_addr)
    var spans = ip(spans_addr)
    var count = 0
    var start = 0
    var state = 0
    for i in range(text_len):
        var current = ascii_class(text[i])
        if current != state:
            if state != 0:
                spans[count * 2] = Int64(start)
                spans[count * 2 + 1] = Int64(i)
                count += 1
            if current != 0:
                start = i
            state = current
    if state != 0:
        spans[count * 2] = Int64(start)
        spans[count * 2 + 1] = Int64(text_len)
        count += 1
    return count


def ends_with[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    suffix: StringSlice,
) -> Bool:
    var bytes = suffix.as_bytes()
    if len(bytes) > n:
        return False
    var offset = n - len(bytes)
    for i in range(len(bytes)):
        if word[offset + i] != Scalar[dtype](bytes[i]):
            return False
    return True


def replace_suffix[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    suffix: StringSlice,
    replacement: StringSlice,
) -> Int:
    var suffix_bytes = suffix.as_bytes()
    var replacement_bytes = replacement.as_bytes()
    var stem_len = n - len(suffix_bytes)
    for i in range(len(replacement_bytes)):
        word[stem_len + i] = Scalar[dtype](replacement_bytes[i])
    return stem_len + len(replacement_bytes)


def is_consonant[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], index: Int
) -> Bool:
    var c = word[index]
    if c == 97 or c == 101 or c == 105 or c == 111 or c == 117:
        return False
    if c == 121:
        var negate = False
        var i = index
        while i > 0 and word[i] == 121:
            negate = not negate
            i -= 1
        var base_is_consonant = not (
            word[i] == 97
            or word[i] == 101
            or word[i] == 105
            or word[i] == 111
            or word[i] == 117
        )
        return base_is_consonant != negate
    return True


def measure[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int
) -> Int:
    var result = 0
    var previous_vowel = False
    for i in range(n):
        var vowel = not is_consonant(word, i)
        if previous_vowel and not vowel:
            result += 1
        previous_vowel = vowel
    return result


def contains_vowel[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int
) -> Bool:
    for i in range(n):
        if not is_consonant(word, i):
            return True
    return False


def ends_double_consonant[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int
) -> Bool:
    return (
        n >= 2
        and word[n - 1] == word[n - 2]
        and is_consonant(word, n - 1)
    )


def ends_cvc[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int, mode: Int
) -> Bool:
    if n >= 3:
        var last = word[n - 1]
        if (
            is_consonant(word, n - 3)
            and not is_consonant(word, n - 2)
            and is_consonant(word, n - 1)
            and last != 119
            and last != 120
            and last != 121
        ):
            return True
    return (
        mode == 2
        and n == 2
        and not is_consonant(word, 0)
        and is_consonant(word, 1)
    )


def step1a[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int, mode: Int
) -> Int:
    if mode == 2 and n == 4 and ends_with(word, n, "ies"):
        return replace_suffix(word, n, "ies", "ie")
    if ends_with(word, n, "sses"):
        return replace_suffix(word, n, "sses", "ss")
    if ends_with(word, n, "ies"):
        return replace_suffix(word, n, "ies", "i")
    if ends_with(word, n, "ss"):
        return n
    if ends_with(word, n, "s"):
        return n - 1
    return n


def step1b[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int, mode: Int
) -> Int:
    if mode == 2 and ends_with(word, n, "ied"):
        if n == 4:
            return replace_suffix(word, n, "ied", "ie")
        return replace_suffix(word, n, "ied", "i")
    if ends_with(word, n, "eed"):
        var stem_len = n - 3
        if measure(word, stem_len) > 0:
            return replace_suffix(word, n, "eed", "ee")
        return n
    var intermediate = n
    if ends_with(word, n, "ed") and contains_vowel(word, n - 2):
        intermediate = n - 2
    elif ends_with(word, n, "ing") and contains_vowel(word, n - 3):
        intermediate = n - 3
    else:
        return n
    if ends_with(word, intermediate, "at"):
        word[intermediate] = Scalar[dtype](101)
        return intermediate + 1
    if ends_with(word, intermediate, "bl"):
        word[intermediate] = Scalar[dtype](101)
        return intermediate + 1
    if ends_with(word, intermediate, "iz"):
        word[intermediate] = Scalar[dtype](101)
        return intermediate + 1
    if ends_double_consonant(word, intermediate):
        var last = word[intermediate - 1]
        if last != 108 and last != 115 and last != 122:
            return intermediate - 1
        return intermediate
    if measure(word, intermediate) == 1 and ends_cvc(word, intermediate, mode):
        word[intermediate] = Scalar[dtype](101)
        return intermediate + 1
    return intermediate


def step1c[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int, mode: Int
) -> Int:
    if not ends_with(word, n, "y"):
        return n
    var stem_len = n - 1
    if mode == 2:
        if stem_len > 1 and is_consonant(word, stem_len - 1):
            word[n - 1] = Scalar[dtype](105)
    elif contains_vowel(word, stem_len):
        word[n - 1] = Scalar[dtype](105)
    return n


def positive_rule[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    suffix: StringSlice,
    replacement: StringSlice,
) -> Int:
    var suffix_len = suffix.byte_length()
    if measure(word, n - suffix_len) > 0:
        return replace_suffix(word, n, suffix, replacement)
    return n


def step2[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int, mode: Int
) -> Int:
    if mode == 2 and ends_with(word, n, "alli"):
        if measure(word, n - 4) > 0:
            return step2(word, replace_suffix(word, n, "alli", "al"), mode)
        return n
    if ends_with(word, n, "ational"):
        return positive_rule(word, n, "ational", "ate")
    if ends_with(word, n, "tional"):
        return positive_rule(word, n, "tional", "tion")
    if ends_with(word, n, "enci"):
        return positive_rule(word, n, "enci", "ence")
    if ends_with(word, n, "anci"):
        return positive_rule(word, n, "anci", "ance")
    if ends_with(word, n, "izer"):
        return positive_rule(word, n, "izer", "ize")
    if mode == 0 and ends_with(word, n, "abli"):
        return positive_rule(word, n, "abli", "able")
    if mode != 0 and ends_with(word, n, "bli"):
        return positive_rule(word, n, "bli", "ble")
    if ends_with(word, n, "alli"):
        return positive_rule(word, n, "alli", "al")
    if ends_with(word, n, "entli"):
        return positive_rule(word, n, "entli", "ent")
    if ends_with(word, n, "eli"):
        return positive_rule(word, n, "eli", "e")
    if ends_with(word, n, "ousli"):
        return positive_rule(word, n, "ousli", "ous")
    if ends_with(word, n, "ization"):
        return positive_rule(word, n, "ization", "ize")
    if ends_with(word, n, "ation"):
        return positive_rule(word, n, "ation", "ate")
    if ends_with(word, n, "ator"):
        return positive_rule(word, n, "ator", "ate")
    if ends_with(word, n, "alism"):
        return positive_rule(word, n, "alism", "al")
    if ends_with(word, n, "iveness"):
        return positive_rule(word, n, "iveness", "ive")
    if ends_with(word, n, "fulness"):
        return positive_rule(word, n, "fulness", "ful")
    if ends_with(word, n, "ousness"):
        return positive_rule(word, n, "ousness", "ous")
    if ends_with(word, n, "aliti"):
        return positive_rule(word, n, "aliti", "al")
    if ends_with(word, n, "iviti"):
        return positive_rule(word, n, "iviti", "ive")
    if ends_with(word, n, "biliti"):
        return positive_rule(word, n, "biliti", "ble")
    if mode == 2 and ends_with(word, n, "fulli"):
        return positive_rule(word, n, "fulli", "ful")
    if ends_with(word, n, "logi"):
        if mode == 2:
            if measure(word, n - 3) > 0:
                return replace_suffix(word, n, "logi", "log")
        elif mode == 1:
            return positive_rule(word, n, "logi", "log")
    return n


def step3[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int
) -> Int:
    if ends_with(word, n, "icate"):
        return positive_rule(word, n, "icate", "ic")
    if ends_with(word, n, "ative"):
        return positive_rule(word, n, "ative", "")
    if ends_with(word, n, "alize"):
        return positive_rule(word, n, "alize", "al")
    if ends_with(word, n, "iciti"):
        return positive_rule(word, n, "iciti", "ic")
    if ends_with(word, n, "ical"):
        return positive_rule(word, n, "ical", "ic")
    if ends_with(word, n, "ful"):
        return positive_rule(word, n, "ful", "")
    if ends_with(word, n, "ness"):
        return positive_rule(word, n, "ness", "")
    return n


def step4_rule[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    suffix: StringSlice,
) -> Int:
    var stem_len = n - suffix.byte_length()
    if measure(word, stem_len) > 1:
        return stem_len
    return n


def step4[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int
) -> Int:
    if ends_with(word, n, "al"):
        return step4_rule(word, n, "al")
    if ends_with(word, n, "ance"):
        return step4_rule(word, n, "ance")
    if ends_with(word, n, "ence"):
        return step4_rule(word, n, "ence")
    if ends_with(word, n, "er"):
        return step4_rule(word, n, "er")
    if ends_with(word, n, "ic"):
        return step4_rule(word, n, "ic")
    if ends_with(word, n, "able"):
        return step4_rule(word, n, "able")
    if ends_with(word, n, "ible"):
        return step4_rule(word, n, "ible")
    if ends_with(word, n, "ant"):
        return step4_rule(word, n, "ant")
    if ends_with(word, n, "ement"):
        return step4_rule(word, n, "ement")
    if ends_with(word, n, "ment"):
        return step4_rule(word, n, "ment")
    if ends_with(word, n, "ent"):
        return step4_rule(word, n, "ent")
    if ends_with(word, n, "ion"):
        var stem_len = n - 3
        if (
            measure(word, stem_len) > 1
            and stem_len > 0
            and (word[stem_len - 1] == 115 or word[stem_len - 1] == 116)
        ):
            return stem_len
        return n
    if ends_with(word, n, "ou"):
        return step4_rule(word, n, "ou")
    if ends_with(word, n, "ism"):
        return step4_rule(word, n, "ism")
    if ends_with(word, n, "ate"):
        return step4_rule(word, n, "ate")
    if ends_with(word, n, "iti"):
        return step4_rule(word, n, "iti")
    if ends_with(word, n, "ous"):
        return step4_rule(word, n, "ous")
    if ends_with(word, n, "ive"):
        return step4_rule(word, n, "ive")
    if ends_with(word, n, "ize"):
        return step4_rule(word, n, "ize")
    return n


def step5a[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int, mode: Int
) -> Int:
    if ends_with(word, n, "e"):
        var stem_len = n - 1
        var m = measure(word, stem_len)
        if m > 1 or (m == 1 and not ends_cvc(word, stem_len, mode)):
            return stem_len
    return n


def step5b[dtype: DType](
    word: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int
) -> Int:
    if ends_with(word, n, "ll") and measure(word, n - 1) > 1:
        return n - 1
    return n


@export("mnltk_porter_stem")
def mnltk_porter_stem(
    source_addr: Int, source_len: Int, destination_addr: Int, mode: Int
) abi("C") -> Int:
    var source = ip(source_addr)
    var word = ip(destination_addr)
    for i in range(source_len):
        word[i] = source[i]
    var n = source_len
    n = step1a(word, n, mode)
    n = step1b(word, n, mode)
    n = step1c(word, n, mode)
    n = step2(word, n, mode)
    n = step3(word, n)
    n = step4(word, n)
    n = step5a(word, n, mode)
    n = step5b(word, n)
    return n


@export("mnltk_porter_stem_ascii")
def mnltk_porter_stem_ascii(
    word_addr: Int, word_len: Int, mode: Int
) abi("C") -> Int:
    var word = bp(word_addr)
    var n = word_len
    n = step1a(word, n, mode)
    n = step1b(word, n, mode)
    n = step1c(word, n, mode)
    n = step2(word, n, mode)
    n = step3(word, n)
    n = step4(word, n)
    n = step5a(word, n, mode)
    n = step5b(word, n)
    return n
