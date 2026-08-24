"""Native kernels for the covered NLTK-compatible API."""

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def min4(a: Int, b: Int, c: Int, d: Int) -> Int:
    return min(min(a, b), min(c, d))


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
                min4(deletion, insertion, substitution, transpose)
            )
        last[Int(first[i - 1])] = Int64(i)
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
    return (
        m / Float64(first_len)
        + m / Float64(second_len)
        + Float64(matches - transpositions // 2) / m
    ) / 3.0


def ascii_class(c: UInt8) -> Int:
    if c == 9 or c == 10 or c == 11 or c == 12 or c == 13 or c == 32:
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
