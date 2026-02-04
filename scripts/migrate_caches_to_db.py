"""
One-off migration: ingest JSON caches into SQLite.

Usage (from repo root):
  python -m scripts.migrate_caches_to_db

Requires: sqlmodel (pip install sqlmodel)
"""

import json
import hashlib
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.db_models import (
    ENGINE,
    SentenceCache,
    VocabEntryCache,
    init_db,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_CACHE = ROOT / "analysis_cache"
VOCAB_CACHE = ROOT / "vocab_cache"


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def migrate_sentences(session: Session):
    for path in ANALYSIS_CACHE.glob("*.json"):
        data = load_json(path)
        sentence_hash = path.stem
        sentence_text = data.get("sentence") or ""  # fallback empty
        hiragana = data.get("hiragana")
        audio_path = data.get("audio_path")
        hard_items = data.get("hard_items", [])
        record = SentenceCache(
            sentence_hash=sentence_hash,
            sentence_text=sentence_text,
            hiragana=hiragana,
            audio_path=audio_path,
            hard_items_json=json.dumps(hard_items, ensure_ascii=False),
        )
        session.merge(record)
    session.commit()


def migrate_vocab(session: Session):
    for path in VOCAB_CACHE.glob("*.json"):
        data = load_json(path)
        surface = data.get("surface", "")
        surface_hash = path.stem or hashlib.sha256(surface.encode("utf-8")).hexdigest()
        base_form = data.get("base_form") or surface
        brief = data.get("brief")
        detail = data.get("detail")
        examples = data.get("examples", [])
        record = VocabEntryCache(
            surface_hash=surface_hash,
            surface=surface,
            base_form=base_form,
            brief=brief,
            detail=detail,
            examples_json=json.dumps(examples, ensure_ascii=False),
        )
        session.merge(record)
    session.commit()


def main():
    init_db()
    with Session(ENGINE) as session:
        migrate_sentences(session)
        migrate_vocab(session)
    print("Migration complete -> app.db")


if __name__ == "__main__":
    main()
