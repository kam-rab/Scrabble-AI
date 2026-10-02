#!/usr/bin/env python3
"""Download a Scrabble word list to data/wordlist.txt.

Three sources are available (choose with --source):

  twl06    – North American Tournament Word List (default)
  sowpods  – International Collins Scrabble Words
  enable   – ENABLE word list (public domain, ~172 K words)

Example
-------
    python download_wordlist.py              # TWL06
    python download_wordlist.py --source sowpods
"""

import argparse
import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WORDLIST_PATH = os.path.join(DATA_DIR, 'wordlist.txt')

SOURCES: dict[str, tuple[str, str]] = {
    'twl06': (
        'https://raw.githubusercontent.com/redbo/scrabble/master/dictionary.txt',
        'TWL06 — North American Scrabble Tournament Word List',
    ),
    'sowpods': (
        'https://raw.githubusercontent.com/jesstess/Scrabble/master/scrabble/sowpods.txt',
        'SOWPODS — Collins International Scrabble Words',
    ),
    'enable': (
        'https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt',
        'ENABLE — Enhanced North American Benchmark LExicon (public domain)',
    ),
}


def download(url: str, dest: str) -> int:
    """Fetch *url* and write its content to *dest*; return byte count."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'ScrabbleAI/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(dest, 'wb') as fh:
        fh.write(data)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--source',
        choices=list(SOURCES),
        default='twl06',
        help='Word list to download (default: twl06)',
    )
    args = parser.parse_args()

    url, description = SOURCES[args.source]
    print(f"Source : {description}")
    print(f"URL    : {url}")
    print(f"Dest   : {WORDLIST_PATH}\n")

    try:
        nbytes = download(url, WORDLIST_PATH)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "\nAlternatively, place any word list (one word per line) at:\n"
            f"  {WORDLIST_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(WORDLIST_PATH, 'r', encoding='utf-8', errors='replace') as fh:
        valid = [ln.strip() for ln in fh if ln.strip().isalpha()]

    print(f"Downloaded : {nbytes:>10,} bytes")
    print(f"Valid words: {len(valid):>10,}")
    print("\nNext step:\n  python build_gaddag.py")


if __name__ == '__main__':
    main()
