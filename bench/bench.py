from __future__ import annotations

import math
import os
import platform
import random
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojonltk as ours
import nltk
from nltk.metrics import distance as nltk_distance
from nltk.stem import PorterStemmer as NltkPorterStemmer
from nltk.tokenize import wordpunct_tokenize as nltk_wordpunct_tokenize
from nltk.util import ngrams as nltk_ngrams


def best_time(function, repeats=3):
    function()
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def distance_case(size, transpositions=False):
    rng = random.Random(17)
    first = "".join(rng.choices("abcdefghijklmnop", k=size))
    second = list(first)
    for index in range(7, size, 19):
        second[index] = rng.choice("qrstuvwxyz")
    second = "".join(second)
    return (
        lambda: ours.edit_distance(first, second, 1, transpositions),
        lambda: nltk_distance.edit_distance(
            first, second, 1, transpositions
        ),
    )


def jaro_case():
    rng = random.Random(23)
    first = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=20_000))
    second = list(first)
    for index in range(13, len(second), 47):
        second[index] = "Z"
    second = "".join(second)
    return (
        lambda: ours.jaro_similarity(first, second),
        lambda: nltk_distance.jaro_similarity(first, second),
    )


def porter_case():
    roots = [
        "relational",
        "conditional",
        "digitizer",
        "vietnamization",
        "decisiveness",
        "sensibiliti",
        "triplicate",
        "electrical",
        "replacement",
        "bowdlerize",
    ]
    words = roots * 5_000
    mojo_stemmer = ours.PorterStemmer()
    nltk_stemmer = NltkPorterStemmer()
    return (
        lambda: [mojo_stemmer.stem(word) for word in words],
        lambda: [nltk_stemmer.stem(word) for word in words],
    )


def tokenizer_case():
    paragraph = (
        "Good muffins cost $3.88 in New York. Please buy two; "
        "tokenization should keep words_and_digits together!\n"
    )
    text = paragraph * 10_000
    return (
        lambda: ours.wordpunct_tokenize(text),
        lambda: nltk_wordpunct_tokenize(text),
    )


def ngram_case():
    tokens = list(range(300_000))
    return (
        lambda: sum(1 for _ in ours.ngrams(tokens, 3)),
        lambda: sum(1 for _ in nltk_ngrams(tokens, 3)),
    )


CASES = [
    ("edit_distance (1,400 x 1,400)", lambda: distance_case(1_400)),
    (
        "Damerau edit_distance (700 x 700)",
        lambda: distance_case(700, True),
    ),
    ("jaro_similarity (20,000 chars)", jaro_case),
    ("PorterStemmer.stem (50,000 words)", porter_case),
    ("wordpunct_tokenize (1.1M chars)", tokenizer_case),
    ("trigrams iterator (300,000 tokens)", ngram_case),
]


def main():
    print(f"Machine: {cpu_name()}; {os.cpu_count()} logical CPUs")
    print(
        f"Software: Python {platform.python_version()}, "
        f"NLTK {nltk.__version__}; best of 3 warm runs"
    )
    print()
    print("| case | mojo-nltk | NLTK | ratio | result |")
    print("|---|---:|---:|---:|---|")
    for name, make_functions in CASES:
        mojo_function, nltk_function = make_functions()
        mojo_time = best_time(mojo_function)
        nltk_time = best_time(nltk_function)
        ratio = nltk_time / mojo_time
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{nltk_time * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
