"""Scrape Kanshudo collection vocab into CSV.

Usage:
  python scripts/scrape_kanshudo_collection.py \
    "https://www.kanshudo.com/collections/wikipedia_jlpt" \
    --out content/raw/WPJLPT-N1-1.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

try:
    from bs4 import BeautifulSoup
except Exception as exc:  # pragma: no cover - simple runtime guard
    raise SystemExit("BeautifulSoup not available. Install with: pip install beautifulsoup4") from exc


KANA_RE = re.compile(r"^[ぁ-ゟ゠-ヿー]+$")
KANJI_RE = re.compile(r"[一-龯々〆ヵヶ]")
PITCH_RE = re.compile(r"^[ぁ-ゟ゠-ヿー\s]+[0-9]+$")

POS_HINTS = (
    "noun",
    "verb",
    "adjective",
    "adverb",
    "interjection",
    "conjunction",
    "suffix",
    "prefix",
    "transitive",
    "intransitive",
    "ichidan",
    "godan",
    "suru",
    "na adjective",
    "no adjective",
)

SKIP_LINES = {
    "QUICK STUDY",
    "FLASHCARDS",
    "DOWNLOAD",
    "Favorites",
    "OK",
    "×",
}


def is_kana(line: str) -> bool:
    return bool(KANA_RE.fullmatch(line))


def has_kanji(line: str) -> bool:
    return bool(KANJI_RE.search(line))


def is_pitch_line(line: str) -> bool:
    return bool(PITCH_RE.fullmatch(line)) or (
        " " in line and all(ch == " " or KANA_RE.fullmatch(ch) for ch in line.replace(" ", ""))
    )


def is_pos_line(line: str) -> bool:
    low = line.lower()
    return any(h in low for h in POS_HINTS) and re.fullmatch(r"[a-zA-Z',\-\s()]+", line) is not None


def normalize_meaning(line: str) -> str:
    return re.sub(r"^\d+\.\s*", "", line).strip()


def extract_entries(text: str) -> list[dict]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Start after the DOWNLOAD marker if present
    if "DOWNLOAD" in lines:
        start = lines.index("DOWNLOAD") + 1
    else:
        start = 0

    entries: list[dict] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line in SKIP_LINES or line.startswith("(click the word"):
            i += 1
            continue

        if is_kana(line) and i + 1 < len(lines) and has_kanji(lines[i + 1]):
            reading = line
            surface = lines[i + 1]
            i += 2

            # skip "Most common form" line(s)
            while i < len(lines) and lines[i].startswith("Most common form:"):
                i += 1

            # skip pitch/accent line(s)
            while i < len(lines) and is_pitch_line(lines[i]):
                i += 1

            # skip POS lines
            while i < len(lines) and is_pos_line(lines[i]):
                i += 1

            meanings: list[str] = []
            while i < len(lines):
                cur = lines[i]
                if cur in SKIP_LINES:
                    i += 1
                    continue
                if cur.startswith("(click the word"):
                    i += 1
                    break
                if is_kana(cur) and i + 1 < len(lines) and has_kanji(lines[i + 1]):
                    break
                if cur.startswith("Please ") or cur.startswith("Wherever you see"):
                    i += 1
                    continue
                if is_pos_line(cur):
                    i += 1
                    continue
                meanings.append(normalize_meaning(cur))
                i += 1

            if meanings:
                entries.append(
                    {
                        "surface": surface,
                        "reading": reading,
                        "meaning": "; ".join(meanings),
                    }
                )
            continue

        i += 1
    return entries


RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def collect_range_links(index_url: str, soup: BeautifulSoup) -> list[tuple[int, int, str]]:
    title = "Wikipedia JLPT N1 Vocab (3362 words)"
    if soup.find(string=lambda s: isinstance(s, str) and title in s) is None:
        print(f"Warning: title not found on index page: {title}")

    links: list[tuple[int, int, str]] = []
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip()
        href = a.get("href") or ""
        m = RANGE_RE.match(text)
        if not m:
            continue
        if "WPJLPT-N1-" not in href:
            continue
        start = int(m.group(1))
        end = int(m.group(2))
        links.append((start, end, urljoin(index_url, href)))
    links.sort(key=lambda x: x[0])
    return links


def fetch_soup(url: str) -> BeautifulSoup:
    res = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; JapaneseMastery_App scraper)",
        },
        timeout=30,
    )
    res.raise_for_status()
    return BeautifulSoup(res.text, "html.parser")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Kanshudo collection vocab to CSV.")
    parser.add_argument("url", help="Kanshudo collection index URL")
    parser.add_argument("--out", default="content/raw/WPJLPT-N1-1.csv", help="Output CSV path")
    parser.add_argument("--sleep", type=float, default=0.6, help="Delay between page fetches (seconds)")
    args = parser.parse_args()

    index_soup = fetch_soup(args.url)
    ranges = collect_range_links(args.url, index_soup)
    if not ranges:
        raise SystemExit("No range links found on index page.")

    all_entries: list[dict] = []
    for start, end, url in ranges:
        page_soup = fetch_soup(url)
        text = page_soup.get_text("\n")
        entries = extract_entries(text)
        for e in entries:
            e["source_url"] = url
            e["range"] = f"{start}-{end}"
        all_entries.extend(entries)
        time.sleep(args.sleep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["surface", "reading", "meaning", "range", "source_url"])
        for e in all_entries:
            w.writerow([e["surface"], e["reading"], e["meaning"], e["range"], e["source_url"]])

    print(f"Wrote {len(all_entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
