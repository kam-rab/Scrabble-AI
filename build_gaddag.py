#!/usr/bin/env python3
"""Build the GADDAG from the word list and save it to disk.

Run once (after download_wordlist.py):
    python build_gaddag.py

On subsequent program starts the solver loads the cached gaddag.pkl
directly instead of rebuilding from scratch.
"""

import os
import sys
import time

from gaddag import GADDAG

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WORDLIST_PATH = os.path.join(DATA_DIR, 'wordlist.txt')
GADDAG_PATH = os.path.join(DATA_DIR, 'gaddag.pkl')

# Words whose presence/absence we use to sanity-check the dictionary.
# Adjust if you use a non-TWL word list.
VALIDATION = [
    ('AA',       True),   # short valid Scrabble word
    ('SCRABBLE', True),
    ('QUIZ',     True),
    ('ZYMURGY',  True),
    ('HELLO',    True),
    ('QQQQQ',    False),  # nonsense – must not be in any dictionary
    ('AAAAAAAA', False),  # 8 A's – not a word
]


def hr(label: str = '') -> None:
    width = 60
    if label:
        print(f"{'─' * 3} {label} {'─' * (width - len(label) - 5)}")
    else:
        print('─' * width)


def main() -> None:
    hr('word list')
    if not os.path.exists(WORDLIST_PATH):
        print(f"Word list not found:\n  {WORDLIST_PATH}")
        print("\nFetch it first:\n  python download_wordlist.py")
        sys.exit(1)
    size_kb = os.path.getsize(WORDLIST_PATH) / 1024
    print(f"  {WORDLIST_PATH} ({size_kb:,.0f} KB)")

    hr('build')
    t0 = time.perf_counter()
    gaddag = GADDAG()
    count = gaddag.build_from_file(WORDLIST_PATH)
    build_time = time.perf_counter() - t0
    print(f"  {count:,} words inserted in {build_time:.2f} s")

    hr('save')
    t1 = time.perf_counter()
    gaddag.save(GADDAG_PATH)
    save_time = time.perf_counter() - t1
    size_mb = os.path.getsize(GADDAG_PATH) / 1024 / 1024
    print(f"  {GADDAG_PATH}")
    print(f"  {size_mb:.1f} MB written in {save_time:.2f} s")

    hr('verify – round-trip load')
    t2 = time.perf_counter()
    loaded = GADDAG.load(GADDAG_PATH)
    load_time = time.perf_counter() - t2
    print(f"  Loaded in {load_time:.2f} s  →  {loaded}")

    hr('verify – word lookups')
    all_ok = True
    for word, expected in VALIDATION:
        result = loaded.contains(word)
        status = 'OK  ' if result == expected else 'FAIL'
        print(f"  [{status}]  {word}: {result}")
        if status.startswith('FAIL'):
            all_ok = False

    hr()
    if all_ok:
        print("GADDAG is ready.  Use GADDAG.load(GADDAG_PATH) in the solver.")
    else:
        print("WARNING: some checks failed – verify the word list source.")
    print()


if __name__ == '__main__':
    main()
