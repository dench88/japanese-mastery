"""
Migrate data from SQLite (app.db) to PostgreSQL.

Usage (from repo root):
  DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname python -m scripts.migrate_sqlite_to_postgres

Prerequisites:
  - The PostgreSQL database must exist and have the schema applied (via Alembic or init_db).
  - The SQLite file (app.db) must be present at the repo root.
"""

import asyncio
import json
import sqlite3
from pathlib import Path

from app.db_models import (
    ENGINE,
    SentenceCache,
    VocabEntryCache,
    SourceMaterial,
    UserSettings,
    async_session,
    init_db,
)

ROOT = Path(__file__).resolve().parents[1]
SQLITE_PATH = ROOT / "app.db"


def read_sqlite_table(conn: sqlite3.Connection, table: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {table}")  # noqa: S608 — trusted table name
        return [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return []


async def migrate():
    if not SQLITE_PATH.exists():
        print(f"SQLite file not found at {SQLITE_PATH}")
        return

    conn = sqlite3.connect(SQLITE_PATH)
    await init_db()

    # --- Sentence caches ---
    rows = read_sqlite_table(conn, "sentencecache")
    print(f"Migrating {len(rows)} sentence cache entries...")
    async with async_session() as session:
        for row in rows:
            existing = await session.get(SentenceCache, row["sentence_hash"])
            if existing:
                continue
            hard_items = json.loads(row.get("hard_items_json") or "[]")
            translations = json.loads(row.get("translations_json") or "{}")
            session.add(SentenceCache(
                sentence_hash=row["sentence_hash"],
                sentence_text=row.get("sentence_text", ""),
                hiragana=row.get("hiragana"),
                audio_path=row.get("audio_path"),
                hard_items_json=hard_items,
                translations_json=translations,
            ))
        await session.commit()
    print("  done.")

    # --- Vocab caches ---
    rows = read_sqlite_table(conn, "vocabentrycache")
    print(f"Migrating {len(rows)} vocab cache entries...")
    async with async_session() as session:
        for row in rows:
            existing = await session.get(VocabEntryCache, row["surface_hash"])
            if existing:
                continue
            examples = json.loads(row.get("examples_json") or "[]")
            session.add(VocabEntryCache(
                surface_hash=row["surface_hash"],
                surface=row.get("surface", ""),
                base_form=row.get("base_form"),
                brief=row.get("brief"),
                detail=row.get("detail"),
                examples_json=examples,
            ))
        await session.commit()
    print("  done.")

    # --- Source materials ---
    rows = read_sqlite_table(conn, "sourcematerial")
    print(f"Migrating {len(rows)} source materials...")
    async with async_session() as session:
        for row in rows:
            existing = await session.get(SourceMaterial, row["source_hash"])
            if existing:
                continue
            session.add(SourceMaterial(
                source_hash=row["source_hash"],
                title_ja=row.get("title_ja"),
                title_en=row.get("title_en"),
                title_ko=row.get("title_ko"),
                title_zh=row.get("title_zh"),
                text_path=row.get("text_path"),
                audio_path=row.get("audio_path"),
                category=row.get("category"),
            ))
        await session.commit()
    print("  done.")

    # --- User settings ---
    rows = read_sqlite_table(conn, "usersettings")
    print(f"Migrating {len(rows)} user settings entries...")
    async with async_session() as session:
        for row in rows:
            existing = await session.get(UserSettings, row["user_id"])
            if existing:
                continue
            session.add(UserSettings(
                user_id=row["user_id"],
                default_voice=row.get("default_voice"),
                tts_speed=row.get("tts_speed"),
                show_fg=bool(row.get("show_fg")) if row.get("show_fg") is not None else None,
                show_hg=bool(row.get("show_hg")) if row.get("show_hg") is not None else None,
                default_source=row.get("default_source"),
                font_scale=row.get("font_scale"),
                native_lang=row.get("native_lang"),
            ))
        await session.commit()
    print("  done.")

    conn.close()
    await ENGINE.dispose()
    print("\nMigration complete: SQLite -> PostgreSQL")


if __name__ == "__main__":
    asyncio.run(migrate())
