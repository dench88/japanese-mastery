from pathlib import Path
from typing import Optional, Any, Iterable

from sqlmodel import Field, SQLModel, create_engine, Session
import json

# SQLite location at project root
ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "app.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class SentenceCache(SQLModel, table=True):
    sentence_hash: str = Field(primary_key=True)
    sentence_text: str
    hiragana: Optional[str] = None
    audio_path: Optional[str] = None  # relative path to mp3
    hard_items_json: Optional[str] = None  # JSON string of hard_items list
    translations_json: Optional[str] = None  # JSON map lang -> translation


class VocabEntryCache(SQLModel, table=True):
    surface_hash: str = Field(primary_key=True)
    surface: str
    base_form: Optional[str] = None
    brief: Optional[str] = None
    detail: Optional[str] = None
    examples_json: Optional[str] = None  # JSON string list of examples


class SourceMaterial(SQLModel, table=True):
    source_hash: str = Field(primary_key=True)
    title_ja: Optional[str] = None
    title_en: Optional[str] = None
    title_ko: Optional[str] = None
    title_zh: Optional[str] = None
    text_path: Optional[str] = None  # relative path to source text
    audio_path: Optional[str] = None  # relative path to recorded audio
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


def init_db():
    SQLModel.metadata.create_all(ENGINE)


# ---------- helper accessors ----------

def get_session() -> Session:
    return Session(ENGINE)


def sentence_to_dict(row: SentenceCache | None) -> Optional[dict[str, Any]]:
    if not row:
        return None
    return {
        "sentence_hash": row.sentence_hash,
        "sentence_text": row.sentence_text,
        "hiragana": row.hiragana,
        "audio_path": row.audio_path,
        "hard_items": json.loads(row.hard_items_json or "[]"),
        "translations": json.loads(row.translations_json or "{}"),
    }


def vocab_to_dict(row: VocabEntryCache | None) -> Optional[dict[str, Any]]:
    if not row:
        return None
    return {
        "surface_hash": row.surface_hash,
        "surface": row.surface,
        "base_form": row.base_form,
        "brief": row.brief,
        "detail": row.detail,
        "examples": json.loads(row.examples_json or "[]"),
    }


def upsert_sentence(
    session: Session,
    sentence_hash: str,
    sentence_text: str,
    hiragana: Optional[str],
    audio_path: Optional[str],
    hard_items: Iterable[Any],
    translations: dict[str, Any],
):
    hard_items_json = json.dumps(list(hard_items or []), ensure_ascii=False)
    translations_json = json.dumps(translations or {}, ensure_ascii=False)
    existing = session.get(SentenceCache, sentence_hash)
    if existing:
        existing.sentence_text = sentence_text
        existing.hiragana = hiragana
        existing.audio_path = audio_path
        existing.hard_items_json = hard_items_json
        existing.translations_json = translations_json
    else:
        session.add(
            SentenceCache(
                sentence_hash=sentence_hash,
                sentence_text=sentence_text,
                hiragana=hiragana,
                audio_path=audio_path,
                hard_items_json=hard_items_json,
                translations_json=translations_json,
            )
        )


def upsert_vocab(
    session: Session,
    surface_hash: str,
    surface: str,
    base_form: Optional[str],
    brief: Optional[str],
    detail: Optional[str],
    examples: Iterable[Any],
):
    examples_json = json.dumps(list(examples or []), ensure_ascii=False)
    existing = session.get(VocabEntryCache, surface_hash)
    if existing:
        existing.surface = surface
        existing.base_form = base_form
        existing.brief = brief
        existing.detail = detail
        existing.examples_json = examples_json
    else:
        session.add(
            VocabEntryCache(
                surface_hash=surface_hash,
                surface=surface,
                base_form=base_form,
                brief=brief,
                detail=detail,
                examples_json=examples_json,
            )
        )
