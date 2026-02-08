import os
import json
from typing import Optional, Any, Iterable

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# ---------------------------------------------------------------------------
# Database URL – expects a Postgres asyncpg connection string.
# Example: postgresql+asyncpg://user:pass@host:5432/dbname
# Falls back to a local SQLite async URL for development if not set.
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/japanese_mastery",
)

# Railway (and others) sometimes provide postgres:// instead of postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

ENGINE = create_async_engine(DATABASE_URL, echo=False, pool_size=5, max_overflow=10)
async_session = async_sessionmaker(ENGINE, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    display_name: Optional[str] = None
    is_active: bool = Field(default=True)


class SentenceCache(SQLModel, table=True):
    sentence_hash: str = Field(primary_key=True)
    sentence_text: str = Field(sa_column=Column(Text))
    hiragana: Optional[str] = Field(default=None, sa_column=Column(Text))
    audio_path: Optional[str] = None
    hard_items_json: Optional[str] = Field(default=None, sa_column=Column(JSONB))
    translations_json: Optional[str] = Field(default=None, sa_column=Column(JSONB))
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)


class VocabEntryCache(SQLModel, table=True):
    surface_hash: str = Field(primary_key=True)
    surface: str
    base_form: Optional[str] = None
    brief: Optional[str] = Field(default=None, sa_column=Column(Text))
    detail: Optional[str] = Field(default=None, sa_column=Column(Text))
    examples_json: Optional[str] = Field(default=None, sa_column=Column(JSONB))
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)


class SourceMaterial(SQLModel, table=True):
    source_hash: str = Field(primary_key=True)
    title_ja: Optional[str] = None
    title_en: Optional[str] = None
    title_ko: Optional[str] = None
    title_zh: Optional[str] = None
    text_path: Optional[str] = None
    audio_path: Optional[str] = None
    category: Optional[str] = None


class UserSettings(SQLModel, table=True):
    user_id: str = Field(primary_key=True)
    default_voice: Optional[str] = None
    tts_speed: Optional[float] = None
    show_fg: Optional[bool] = None
    show_hg: Optional[bool] = None
    default_source: Optional[str] = None
    font_scale: Optional[float] = None
    native_lang: Optional[str] = None


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------

async def init_db():
    async with ENGINE.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


def get_session() -> AsyncSession:
    """Return a new async session (caller must use `async with`)."""
    return async_session()


# ---------------------------------------------------------------------------
# Helper accessors (now work with raw dicts from JSONB columns)
# ---------------------------------------------------------------------------

def sentence_to_dict(row: SentenceCache | None) -> Optional[dict[str, Any]]:
    if not row:
        return None
    hard_items = row.hard_items_json if isinstance(row.hard_items_json, list) else json.loads(row.hard_items_json or "[]")
    translations = row.translations_json if isinstance(row.translations_json, dict) else json.loads(row.translations_json or "{}")
    return {
        "sentence_hash": row.sentence_hash,
        "sentence_text": row.sentence_text,
        "hiragana": row.hiragana,
        "audio_path": row.audio_path,
        "hard_items": hard_items,
        "translations": translations,
    }


def vocab_to_dict(row: VocabEntryCache | None) -> Optional[dict[str, Any]]:
    if not row:
        return None
    examples = row.examples_json if isinstance(row.examples_json, list) else json.loads(row.examples_json or "[]")
    return {
        "surface_hash": row.surface_hash,
        "surface": row.surface,
        "base_form": row.base_form,
        "brief": row.brief,
        "detail": row.detail,
        "examples": examples,
    }


async def upsert_sentence(
    session: AsyncSession,
    sentence_hash: str,
    sentence_text: str,
    hiragana: Optional[str],
    audio_path: Optional[str],
    hard_items: Iterable[Any],
    translations: dict[str, Any],
    user_id: Optional[int] = None,
):
    hard_items_data = list(hard_items or [])
    translations_data = translations or {}
    existing = await session.get(SentenceCache, sentence_hash)
    if existing:
        existing.sentence_text = sentence_text
        existing.hiragana = hiragana
        existing.audio_path = audio_path
        existing.hard_items_json = hard_items_data
        existing.translations_json = translations_data
        if user_id is not None:
            existing.user_id = user_id
    else:
        session.add(
            SentenceCache(
                sentence_hash=sentence_hash,
                sentence_text=sentence_text,
                hiragana=hiragana,
                audio_path=audio_path,
                hard_items_json=hard_items_data,
                translations_json=translations_data,
                user_id=user_id,
            )
        )


async def upsert_vocab(
    session: AsyncSession,
    surface_hash: str,
    surface: str,
    base_form: Optional[str],
    brief: Optional[str],
    detail: Optional[str],
    examples: Iterable[Any],
    user_id: Optional[int] = None,
):
    examples_data = list(examples or [])
    existing = await session.get(VocabEntryCache, surface_hash)
    if existing:
        existing.surface = surface
        existing.base_form = base_form
        existing.brief = brief
        existing.detail = detail
        existing.examples_json = examples_data
        if user_id is not None:
            existing.user_id = user_id
    else:
        session.add(
            VocabEntryCache(
                surface_hash=surface_hash,
                surface=surface,
                base_form=base_form,
                brief=brief,
                detail=detail,
                examples_json=examples_data,
                user_id=user_id,
            )
        )
