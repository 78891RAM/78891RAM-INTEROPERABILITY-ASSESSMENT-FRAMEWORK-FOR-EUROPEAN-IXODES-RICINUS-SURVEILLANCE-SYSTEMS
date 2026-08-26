"""
Re-fetches the official source page for every system in data/evidence.csv and
archives the plain-text content under research/snapshots/, so the citations
backing data/systems.csv can be re-checked for link rot or content drift.

This is NOT an auto-scorer. Deciding whether e.g. a licence counts as "open"
or a schema as "complete" needs a human (or an LLM doing research, as was
done to build the current data/systems.csv) reading the page in context —
that judgment lives in systems.csv, not in this script. What this script
gives you is a reproducible, timestamped record of what each source actually
said, so those judgment calls can be audited or redone later.

Usage:
    python research/fetch_sources.py            # fetch every system's official_website
    python research/fetch_sources.py SYS06       # fetch just one system
"""

from __future__ import annotations

import sys
import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_CSV = ROOT / "data" / "evidence.csv"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

USER_AGENT = (
    "Mozilla/5.0 (compatible; TickHarmonixResearchBot/1.0; "
    "+academic dissertation research, non-commercial, one-off fetch)"
)


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text: strips tags/scripts/styles, keeps visible text."""

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.chunks.append(text)


def _slugify(system_id: str, url: str) -> str:
    host = re.sub(r"^https?://", "", url).split("/")[0]
    host = unicodedata.normalize("NFKD", host).encode("ascii", "ignore").decode()
    return f"{system_id}_{re.sub(r'[^a-zA-Z0-9.-]', '_', host)}"


def fetch_one(system_id: str, url: str) -> Path | None:
    if not url:
        print(f"  {system_id}: no official_website URL on file — skipped")
        return None
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode(resp.headers.get_content_charset() or "utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"  {system_id}: FETCH FAILED ({exc}) — {url}")
        return None

    parser = _TextExtractor()
    parser.feed(raw)
    text = "\n".join(parser.chunks)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{_slugify(system_id, url)}.txt"
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_path.write_text(
        f"# Snapshot of {url}\n# system_id: {system_id}\n# fetched_at: {fetched_at}\n\n{text}",
        encoding="utf-8",
    )
    print(f"  {system_id}: saved {len(text)} chars -> {out_path.relative_to(ROOT)}")
    return out_path


def main() -> None:
    evidence = pd.read_csv(EVIDENCE_CSV, dtype={"system_id": str})
    targets = sys.argv[1:] or evidence["system_id"].tolist()

    print(f"Fetching {len(targets)} system source page(s) into {SNAPSHOT_DIR.relative_to(ROOT)}/ ...")
    for system_id in targets:
        row = evidence.loc[evidence["system_id"] == system_id]
        if row.empty:
            print(f"  {system_id}: not found in evidence.csv")
            continue
        fetch_one(system_id, str(row.iloc[0]["official_website"] or "").strip())


if __name__ == "__main__":
    main()
