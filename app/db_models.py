from pathlib import Path
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine

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


class VocabEntryCache(SQLModel, table=True):
    surface_hash: str = Field(primary_key=True)
    surface: str
    base_form: Optional[str] = None
    brief: Optional[str] = None
    detail: Optional[str] = None
    examples_json: Optional[str] = None  # JSON string list of examples


def init_db():
    SQLModel.metadata.create_all(ENGINE)
