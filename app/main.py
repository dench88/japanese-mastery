from pathlib import Path
import os
import re
import json
import hashlib
import shutil
import sqlite3
import urllib.request
import urllib.error
import subprocess
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import OpenAI
from sqlmodel import Session, select

from app.db_models import (
    init_db,
    get_session,
    sentence_to_dict,
    vocab_to_dict,
    audio_to_dict,
    upsert_sentence,
    upsert_vocab,
    upsert_audio_transcript,
    SentenceCache,
    VocabEntryCache,
    AudioTranscript,
    SourceMaterial,
    UserSettings,
    DB_PATH,
)

# --- paths & env ---
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)  # load env from repo root
SPEED_DEFAULT = float(os.getenv("TTS_SPEED_DEFAULT", "1.1"))

# --- load and split book text ---
BOOKS_DIR = ROOT_DIR / "content" / "source_materials"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_BOOK = BOOKS_DIR / "sample-writing-cafune-1.txt"
# always seed default from legacy cafune text if present
legacy = ROOT_DIR / "content" / "raw" / "sample-writing-cafune-1.txt"
if legacy.exists():
    DEFAULT_BOOK.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
elif not DEFAULT_BOOK.exists():
    DEFAULT_BOOK.write_text("これはサンプル文章です。", encoding="utf-8")
BOOK_TEXT = DEFAULT_BOOK.read_text(encoding="utf-8")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*")  # Japanese sentence split

# --- caches ---
TTS_CACHE_DIR = ROOT_DIR / "tts_cache"
TTS_CACHE_DIR.mkdir(exist_ok=True)
ANALYSIS_CACHE_DIR = ROOT_DIR / "analysis_cache"
ANALYSIS_CACHE_DIR.mkdir(exist_ok=True)
VOCAB_CACHE_DIR = ROOT_DIR / "vocab_cache"
VOCAB_CACHE_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR = ROOT_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
RAW_DIR = ROOT_DIR / "content" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
_podcast_dir = RAW_DIR / "podcast"
_podcasts_dir = RAW_DIR / "podcasts"
if _podcast_dir.exists():
    AUDIO_DIR = _podcast_dir
elif _podcasts_dir.exists():
    AUDIO_DIR = _podcasts_dir
else:
    AUDIO_DIR = _podcast_dir
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    _podcasts_dir = AUDIO_DIR
RAW_ALLOWED_EXTS = {".pdf", ".txt", ".md", ".docx", ".zip"}
AUDIO_ALLOWED_EXTS = {".mp3", ".m4a", ".wav", ".webm", ".ogg"}
AUDIO_CHUNK_SECONDS = 300

# --- user handling (single-user stub) ---
def get_current_user_id() -> str:
    return "local"

# --- OpenAI client (single instance) ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment; set it in .env or shell.")
client = OpenAI(api_key=api_key)

app = FastAPI()
init_db()
# ensure legacy DB has translations column
def _ensure_translations_column():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(sentencecache)")
        cols = [c[1] for c in cur.fetchall()]
        if "translations_json" not in cols:
            cur.execute("ALTER TABLE sentencecache ADD COLUMN translations_json TEXT")
            conn.commit()
    except Exception as e:
        print("DB column check error:", repr(e))
    finally:
        try:
            conn.close()
        except Exception:
            pass

_ensure_translations_column()


def _ensure_user_settings_columns():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(usersettings)")
        cols = [c[1] for c in cur.fetchall()]
        if "show_native_defs" not in cols:
            cur.execute("ALTER TABLE usersettings ADD COLUMN show_native_defs INTEGER")
            conn.commit()
        if "user_name" not in cols:
            cur.execute("ALTER TABLE usersettings ADD COLUMN user_name TEXT")
            conn.commit()
    except Exception as e:
        print("DB user settings column check error:", repr(e))
    finally:
        try:
            conn.close()
        except Exception:
            pass


_ensure_user_settings_columns()

# serve static files
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")
app.mount("/tts_cache", StaticFiles(directory=TTS_CACHE_DIR), name="tts-cache")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")
app.mount("/images", StaticFiles(directory=ROOT_DIR / "content" / "images"), name="images")


class TTSRequest(BaseModel):
    text: str
    voice: str | None = "nova"
    model: str | None = "gpt-4o-mini-tts"
    speed: float | None = None


class AnalyzeRequest(BaseModel):
    text: str
    native_lang: str | None = None


class AnalyzeDeleteRequest(BaseModel):
    text: str
    surface: str


class AnalyzeResponse(BaseModel):
    reading_hiragana: str | None = None
    reading_ruby: str | None = None
    items: list[dict]
    audio_path: str | None = None


class DeepDiveRequest(BaseModel):
    sentence: str
    item: dict


class DeepDiveResponse(BaseModel):
    explanation: str
    examples: list[dict]


class ReadingWordRequest(BaseModel):
    surface: str


# simple in-memory cache for word readings to avoid extra calls in one run
READING_CACHE: dict[str, str] = {}


class SourceRecordResponse(BaseModel):
    name: str
    path: str
    audio_path: str | None = None
    category: str | None = None


class SettingsResponse(BaseModel):
    default_voice: str = "nova"
    tts_speed: float = SPEED_DEFAULT
    show_fg: bool = False
    show_hg: bool = False
    default_source: str | None = None
    font_scale: float = 1.0
    native_lang: str | None = "en"
    show_native_defs: bool = True
    user_name: str | None = None


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    translation: str


class DeleteSentenceRequest(BaseModel):
    text: str


class CleanSourceRequest(BaseModel):
    path: str


class TranslateDeepDiveRequest(BaseModel):
    explanation: str
    examples: list[dict] | None = None
    target_lang: str = "en"

class ManualBriefRequest(BaseModel):
    sentence: str
    surface: str
    native_lang: str | None = None


class RealtimeTokenRequest(BaseModel):
    voice: str | None = None
    model: str | None = None
    user_name: str | None = None


class AudioTranscribeRequest(BaseModel):
    name: str


class VocabEntry(BaseModel):
    surface: str
    base_form: str | None = None
    brief: str | None = None
    detail: str | None = None
    examples: list[dict] | None = None


class TTSExampleRequest(BaseModel):
    text: str
    surface: str | None = None
    voice: str | None = "nova"
    speed: float | None = None


# --- helpers ---
def sentence_cache_path(sentence: str) -> Path:
    normalized_text = sentence.strip()
    key_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    return ANALYSIS_CACHE_DIR / f"{key_hash}.json"


def vocab_cache_path(surface: str) -> Path:
    key_hash = hashlib.sha256(surface.strip().encode("utf-8")).hexdigest()
    return VOCAB_CACHE_DIR / f"{key_hash}.json"


def sentence_hash(sentence: str) -> str:
    return hashlib.sha256(sentence.strip().encode("utf-8")).hexdigest()


def vocab_hash(surface: str) -> str:
    return hashlib.sha256(surface.strip().encode("utf-8")).hexdigest()


def translate_glosses(terms: list[str], target_lang: str) -> list[str]:
    target = (target_lang or "").lower()
    if not terms or not target or target == "ja":
        return ["" for _ in terms]
    prompt = (
        f"Provide a concise dictionary-style gloss in {target} for each Japanese term. "
        "Use the dictionary form directly; do NOT translate any explanatory sentence. "
        "Return short noun/verb/adjective phrases only (no full sentences), no parentheses, no trailing period. "
        "If a grammar label is implied, render it as a natural short gloss (e.g., 'compared to'). "
        'Return ONLY JSON with this schema: { "translations": ["..."] }'
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": f"{prompt}\nTerms: {json.dumps(terms, ensure_ascii=False)}"},
            ],
        )
        raw = chat.choices[0].message.content or "{}"
        data = json.loads(raw)
        translations = data.get("translations", [])
        if not isinstance(translations, list):
            return ["" for _ in terms]
        # pad/trim to expected length
        if len(translations) < len(terms):
            translations.extend([""] * (len(terms) - len(translations)))
        return translations[: len(terms)]
    except Exception as e:
        print("translate_glosses error:", repr(e))
        return ["" for _ in terms]


def ensure_native_hints(items: list[dict], native_lang: str | None) -> list[dict]:
    target = (native_lang or "").lower()
    if not target or target == "ja":
        return items
    missing = [i for i, it in enumerate(items) if it.get("hint_native_lang") != target or not it.get("hint_native")]
    if not missing:
        return items
    terms = [items[i].get("base_form") or items[i].get("surface") or "" for i in missing]
    translations = translate_glosses(terms, target)
    for idx, trans in zip(missing, translations):
        items[idx]["hint_native"] = trans or ""
        items[idx]["hint_native_lang"] = target
    return items


def load_sentence_cache(sentence: str) -> dict:
    """DB first, fallback to JSON. Returns dict with hiragana, hard_items, audio_path, sentence."""
    h = sentence_hash(sentence)
    data = {"sentence": sentence, "hiragana": None, "reading_ruby": None, "hard_items": [], "audio_path": None, "translations": {}}
    with get_session() as session:
        row = session.get(SentenceCache, h)
        if row:
            dbd = sentence_to_dict(row) or {}
            data.update(
                {
                    "hiragana": dbd.get("hiragana"),
                    "reading_ruby": None,  # DB schema has no ruby column; only available via JSON cache
                    "hard_items": dbd.get("hard_items", []),
                    "audio_path": dbd.get("audio_path"),
                    "translations": dbd.get("translations", {}),
                }
            )
            return data
    # fallback JSON
    path = sentence_cache_path(sentence)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            data.update(
                {
                    "hiragana": cached.get("hiragana"),
                    "reading_ruby": cached.get("reading_ruby"),
                    "hard_items": cached.get("hard_items", []),
                    "audio_path": cached.get("audio_path"),
                    "translations": cached.get("translations", {}),
                }
            )
        except Exception:
            pass
    return data


def save_sentence_cache(sentence: str, hiragana=None, reading_ruby=None, hard_items=None, audio_path=None, translations=None):
    """Persist to DB and mirror JSON for backward compatibility."""
    h = sentence_hash(sentence)
    # merge with existing values
    existing = load_sentence_cache(sentence)
    if hiragana is not None:
        existing["hiragana"] = hiragana
    if reading_ruby is not None:
        existing["reading_ruby"] = reading_ruby
    if hard_items is not None:
        existing["hard_items"] = hard_items
    if audio_path is not None:
        existing["audio_path"] = audio_path
    if translations is not None:
        existing["translations"] = translations
    with get_session() as session:
        try:
            upsert_sentence(
                session,
                sentence_hash=h,
                sentence_text=sentence,
                hiragana=existing.get("hiragana"),
                audio_path=existing.get("audio_path"),
                hard_items=existing.get("hard_items") or [],
                translations=existing.get("translations") or {},
            )
            session.commit()
        except Exception as e:
            # if DB schema missing column, skip DB write but continue JSON cache
            print("save_sentence_cache DB write error:", repr(e))
    # mirror JSON
    path = sentence_cache_path(sentence)
    try:
        path.write_text(
            json.dumps(
                {
                    "sentence": sentence,
                    "hiragana": existing.get("hiragana"),
                    "reading_ruby": existing.get("reading_ruby"),
                    "hard_items": existing.get("hard_items") or [],
                    "audio_path": existing.get("audio_path"),
                    "translations": existing.get("translations") or {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def delete_sentence_cache(sentence: str):
    """Remove cached data for one sentence (DB + JSON)."""
    h = sentence_hash(sentence)
    # delete DB row
    with get_session() as session:
        row = session.get(SentenceCache, h)
        if row:
            session.delete(row)
            session.commit()
    # delete JSON cache file if present
    path = sentence_cache_path(sentence)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def load_vocab_cache(surface: str) -> dict:
    """DB first, fallback JSON."""
    h = vocab_hash(surface)
    data = {"surface": surface, "base_form": surface, "brief": None, "detail": None, "examples": []}
    with get_session() as session:
        row = session.get(VocabEntryCache, h)
        if row:
            dbd = vocab_to_dict(row) or {}
            data.update(
                {
                    "base_form": dbd.get("base_form") or surface,
                    "brief": dbd.get("brief"),
                    "detail": dbd.get("detail"),
                    "examples": dbd.get("examples", []),
                }
            )
            return data
    path = vocab_cache_path(surface)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            data.update(
                {
                    "base_form": cached.get("base_form") or surface,
                    "brief": cached.get("brief"),
                    "detail": cached.get("detail"),
                    "examples": cached.get("examples", []),
                }
            )
        except Exception:
            pass
    return data


def save_vocab_cache(surface: str, base_form=None, brief=None, detail=None, examples=None):
    h = vocab_hash(surface)
    existing = load_vocab_cache(surface)
    if base_form is not None:
        existing["base_form"] = base_form
    if brief is not None:
        existing["brief"] = brief
    if detail is not None:
        existing["detail"] = detail
    if examples is not None:
        existing["examples"] = examples
    with get_session() as session:
        upsert_vocab(
            session,
            surface_hash=h,
            surface=surface,
            base_form=existing.get("base_form"),
            brief=existing.get("brief"),
            detail=existing.get("detail"),
            examples=existing.get("examples") or [],
        )
        session.commit()
    path = vocab_cache_path(surface)
    try:
        path.write_text(
            json.dumps(
                {
                    "surface": surface,
                    "base_form": existing.get("base_form"),
                    "brief": existing.get("brief"),
                    "detail": existing.get("detail"),
                    "examples": existing.get("examples") or [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def tts_filename(text: str, voice: str, speed: float) -> str:
    key = f"{text}|{voice}|{speed}"
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{key_hash}.mp3"


def list_sources():
    sources = []
    # DB-driven entries
    with get_session() as session:
        rows = session.exec(select(SourceMaterial)).all()
        for r in rows:
            sources.append(
                {
                    "name": r.title_ja or (Path(r.text_path).name if r.text_path else r.source_hash),
                    "path": r.text_path,
                    "audio_path": r.audio_path,
                    "category": r.category,
                }
            )
    # File-based entries (fallback / legacy)
    for p in BOOKS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            rel = p.relative_to(ROOT_DIR).as_posix()
            if not any(s.get("path") == rel for s in sources):
                sources.append({"name": p.name, "path": rel, "audio_path": None, "category": None})
    return sorted(sources, key=lambda x: x["name"])


def load_source_text(source_name: str | None) -> str:
    if source_name:
        candidate = Path(source_name)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        # fallback to name under BOOKS_DIR
        candidate2 = BOOKS_DIR / source_name
        if candidate2.exists() and candidate2.is_file():
            return candidate2.read_text(encoding="utf-8")
        raise HTTPException(status_code=404, detail="Source not found")
    # fallback
    return BOOK_TEXT


def split_sentences(text: str):
    return [s for s in SENTENCE_PATTERN.split(text.strip()) if s]


def ensure_text_path(text: str, preferred_name: str | None = None) -> str:
    """Create a new text file in BOOKS_DIR with provided content and return relative path."""
    stem = preferred_name or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    candidate = BOOKS_DIR / f"{stem}.txt"
    # avoid overwrite
    counter = 1
    while candidate.exists():
        candidate = BOOKS_DIR / f"{stem}_{counter}.txt"
        counter += 1
    candidate.write_text(text, encoding="utf-8")
    return candidate.relative_to(ROOT_DIR).as_posix()


def clean_japanese_spacing(text: str) -> str:
    """Normalize superfluous spaces around Japanese punctuation/quotes."""
    # collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # remove spaces before Japanese punctuation and quotes
    text = re.sub(r"\s+([、。！？「」『』（）〔〕［］])", r"\1", text)
    # remove spaces after opening quotes/brackets
    text = re.sub(r"([「『（〔［])\s+", r"\1", text)
    # collapse spaces around ASCII punctuation commonly seen in PDFs
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([\(])\s+", r"\1", text)
    # trim extraneous spaces at line starts/ends
    text = "\n".join([ln.strip() for ln in text.splitlines()])
    return text


# --- shared analysis helper (reading + hard items) ---
def analyze_sentence_internal(sentence: str):
    normalized_text = sentence.strip()
    cached = load_sentence_cache(normalized_text)
    if cached.get("hiragana") is not None and isinstance(cached.get("hard_items"), list):
        return cached["hiragana"], cached.get("reading_ruby"), cached.get("hard_items", [])

    prompt = (
        "You are a Japanese language coach for advanced learners (JLPT N1+). Given ONE Japanese sentence, do two things: "
        "1) Convert the entire sentence into hiragana (no romaji), preserving punctuation; leave katakana loanwords as katakana. "
        "2) Return ruby markup for the sentence: each word with kanji should be wrapped as <ruby><rb>漢字</rb><rt>よみ</rt></ruby>; kana-only spans stay plain. "
        "3) List the 3 most difficult items (word or grammar) for an N1+ learner. Ignore anything below N2 unless nothing harder exists. If nothing suitable, return an empty list. "
        "For each item's hint_ja, use the most concise dictionary-style meaning only. Do NOT repeat the surface or base form. "
        "Avoid full sentences or polite forms; no quotes, no trailing period.\n"
        "Respond ONLY with JSON matching exactly this schema (all explanations in Japanese, no English words):\n"
        '{ "reading_hiragana": "<sentence rendered fully in hiragana>", '
        '"reading_ruby": "<sentence with ruby tags>", '
        '"items": [ { "type": "word" | "grammar", "surface": "<exact span from the sentence>", '
        '"base_form": "<dictionary form or grammar label>", "reading": "<hiragana reading of the surface>", "difficulty": 1, '
        '"hint_ja": "<short explanation in Japanese>", "reason": "<why this is difficult (Japanese)>" } ] }'
    )

    reading = None
    reading_ruby = None
    items: list[dict] = []

    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": f"{prompt}\nSentence: {normalized_text}"},
            ],
        )
        raw = chat.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned.split("\n", 1)[-1]
            data = json.loads(cleaned)
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        reading = data.get("reading_hiragana")
        reading_ruby = data.get("reading_ruby")
        save_sentence_cache(normalized_text, hiragana=reading, reading_ruby=reading_ruby, hard_items=items)
    except Exception as e:
        print("Analyze error:", repr(e))
        reading = None
        reading_ruby = None
        items = []

    return reading, reading_ruby, items


@app.get("/")
async def root():
    # open sources page first
    return RedirectResponse(url="/static/sources.html", status_code=302)


@app.get("/api/book")
async def get_book(source: str | None = None):
    """Return full book text for a given source filename (optional)."""
    text = load_source_text(source)
    return {"text": text}


@app.get("/api/sentence/{index}")
async def get_sentence(index: int, source: str | None = None):
    """Return the sentence at the given index."""
    text = load_source_text(source)
    sentences = split_sentences(text)
    total = len(sentences)
    if index < 0 or index >= total:
        raise HTTPException(status_code=404, detail="No more sentences.")
    return {"index": index, "sentence": sentences[index], "total": total}


@app.get("/api/sentences")
async def get_sentences(source: str | None = None):
    """Return all sentences with their indices (for client-side navigation)."""
    text = load_source_text(source)
    sentences = split_sentences(text)
    return {
        "sentences": [{"index": i, "sentence": s} for i, s in enumerate(sentences)],
        "total": len(sentences),
    }


@app.post("/api/tts")
async def tts(req: TTSRequest):
    model = req.model or "gpt-4o-mini-tts"
    voice = req.voice or "nova"
    speed = req.speed or SPEED_DEFAULT
    normalized_text = req.text.strip()

    # deterministic cache key
    key_str = f"{model}|{voice}|{speed}|{normalized_text}"
    key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    cache_path = TTS_CACHE_DIR / f"{key_hash}.mp3"
    audio_rel = f"/tts_cache/{cache_path.name}"

    # cache hit: stream file
    if cache_path.exists():
        save_sentence_cache(normalized_text, audio_path=audio_rel)
        return FileResponse(cache_path, media_type="audio/mpeg")

    # cache miss: call API and stream to disk
    try:
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=normalized_text,
            speed=speed,
            response_format="mp3",
        ) as response:
            response.stream_to_file(cache_path)

        save_sentence_cache(normalized_text, audio_path=audio_rel)

        return FileResponse(cache_path, media_type="audio/mpeg")
    except Exception as e:
        print("TTS error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze_sentence", response_model=AnalyzeResponse)
async def analyze_sentence(req: AnalyzeRequest):
    reading, reading_ruby, items = analyze_sentence_internal(req.text)
    items = ensure_native_hints(items, req.native_lang)
    if items:
        save_sentence_cache(req.text, hard_items=items)
    return {"reading_hiragana": reading, "items": items, "reading_ruby": reading_ruby}


@app.post("/api/reading_sentence")
async def reading_sentence(req: AnalyzeRequest):
    cached = load_sentence_cache(req.text)
    reading = cached.get("hiragana")
    reading_ruby = cached.get("reading_ruby")
    if reading is None:
        reading, reading_ruby, items = analyze_sentence_internal(req.text)
        if reading is not None:
            save_sentence_cache(req.text, hiragana=reading, reading_ruby=reading_ruby, hard_items=items)
    return {"reading_hiragana": reading, "reading_ruby": reading_ruby}


@app.post("/api/hard_items")
async def hard_items(req: AnalyzeRequest):
    cached = load_sentence_cache(req.text)
    items = cached.get("hard_items", [])
    if not items:
        reading, reading_ruby, items = analyze_sentence_internal(req.text)
        save_sentence_cache(req.text, hiragana=reading, reading_ruby=reading_ruby, hard_items=items)
    items = ensure_native_hints(items, req.native_lang)
    if items:
        save_sentence_cache(req.text, hard_items=items)
    return {"items": items}


@app.post("/api/hard_item/delete")
async def delete_hard_item(req: AnalyzeDeleteRequest):
    """Remove a specific item (by surface) from the cached analysis for this sentence."""
    try:
        data = load_sentence_cache(req.text)
        items = data.get("hard_items", [])
        surface = req.surface
        if not surface:
            return {"removed": False}
        new_items = [i for i in items if i.get("surface") != surface]
        save_sentence_cache(req.text, hard_items=new_items)
        return {"removed": True}
    except Exception as e:
        print("Delete hard item error:", repr(e))
        return {"removed": False}


@app.post("/api/manual_brief")
async def manual_brief(req: ManualBriefRequest):
    """Return a short Japanese hint for a user-selected surface in the sentence."""
    cached = load_vocab_cache(req.surface)
    if cached.get("brief"):
        hint_ja = cached.get("brief")
        hint_native = ""
        hint_native_lang = (req.native_lang or "").lower()
        if hint_ja and hint_native_lang and hint_native_lang != "ja":
            hint_native = translate_glosses([req.surface], hint_native_lang)[0]
        return {"hint_ja": hint_ja, "hint_native": hint_native, "hint_native_lang": hint_native_lang}

    prompt = (
        "あなたは日本語上級学習者(JLPT N1+)向けのコーチです。与えられた文中の指定語について、最小限の辞書的意味だけを日本語で返してください。"
        "表層語や基本形の繰り返しはしないでください。引用符は不要。文ではなく語義のみを出してください。"
        "丁寧文や説明文は避け、短い語句で。句点などの末尾記号は付けないでください。"
        'JSONのみで返してください。形式: { "hint_ja": "<語義のみ (日本語)>" }'
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": f"{prompt}\n文: {req.sentence}\n語: {req.surface}"},
            ],
        )
        raw = chat.choices[0].message.content or "{}"
        data = json.loads(raw)
        hint = data.get("hint_ja") or ""
        hint_native_lang = (req.native_lang or "").lower()
        hint_native = ""
        if hint and hint_native_lang and hint_native_lang != "ja":
            hint_native = translate_glosses([req.surface], hint_native_lang)[0]
        # cache per-word
        save_vocab_cache(
            surface=req.surface,
            base_form=req.surface,
            brief=hint,
            detail=None,
            examples=[],
        )
        # update sentence cache hard_items list
        s_data = load_sentence_cache(req.sentence)
        hard_items = s_data.get("hard_items", [])
        if not any(i.get("surface") == req.surface for i in hard_items):
            hard_items.append(
                {
                    "surface": req.surface,
                    "base_form": req.surface,
                    "type": "word",
                    "difficulty": 1,
                    "hint_ja": hint,
                    "hint_native": hint_native,
                    "hint_native_lang": hint_native_lang,
                }
            )
        save_sentence_cache(req.sentence, hard_items=hard_items)
        return {"hint_ja": hint, "hint_native": hint_native, "hint_native_lang": hint_native_lang}
    except Exception as e:
        print("Manual brief error:", repr(e))
        return {"hint_ja": "", "hint_native": "", "hint_native_lang": (req.native_lang or "").lower()}


@app.post("/api/realtime_token")
async def realtime_token(req: RealtimeTokenRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing")
    user_name = (req.user_name or "Rusty").strip() or "Rusty"
    voice = (req.voice or "marin").lower()
    if voice not in {"alloy", "ash", "ballad", "coral", "echo", "fable", "marin", "sage", "shimmer", "verse"}:
        voice = "marin"
    instructions = (
        "あなたは上級日本語学習者のための専門コーチです。ユーザーはあなたと日本語で会話練習したいと思っています。"
        f"接続でき次第すぐに「こんにちは、{user_name}さん。今日は何を勉強していますか？」と始めてください。"
        "ユーザーが話す量があなたより多くなるようにしてください。返答は短く、要点だけで簡潔にし、"
        "話題への理解を深めるための関連質問を中心にしてください。"
        "会話が詰まったら「最近、日本語でどんな話題を勉強していますか？」などの開かれた質問をしてください。"
        "それでも詰まったら「今日はどんな単語を勉強しましたか？」と聞き、その単語を軸に会話を進めてください。"
        "語彙や文法の意味に関する質問には答えてください。"
        "基本は自然な日本語で話してください。"
        "ただしユーザーが理解に苦しんでいる場合は、短い文や易しい語彙、必要に応じて少しゆっくり話してください。"
        "ユーザーは上級学習者なので、適度に難しい日本語で挑戦させてください。"
    )
    session = {
        "type": "realtime",
        "model": req.model or "gpt-realtime",
        "audio": {"output": {"voice": voice}},
        "instructions": instructions,
    }
    payload = json.dumps({"session": session}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/realtime/client_secrets",
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        print("Realtime token HTTP error:", e.code, raw)
        detail = "Failed to create realtime token"
        try:
            payload = json.loads(raw)
            detail = payload.get("error", {}).get("message") or payload.get("detail") or detail
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        print("Realtime token error:", repr(e))
        raise HTTPException(status_code=500, detail="Failed to create realtime token")

    client_secret = None
    if isinstance(data, dict):
        if isinstance(data.get("client_secret"), dict):
            client_secret = data["client_secret"].get("value")
        elif isinstance(data.get("client_secret"), str):
            client_secret = data.get("client_secret")
        if not client_secret:
            client_secret = data.get("value")
    if not client_secret:
        raise HTTPException(status_code=500, detail="Missing client_secret in response")
    return {
        "client_secret": client_secret,
        "value": client_secret,
        "expires_at": (data.get("client_secret") or {}).get("expires_at") if isinstance(data, dict) else None,
    }


@app.post("/api/deep_dive", response_model=DeepDiveResponse)
async def deep_dive(req: DeepDiveRequest):
    item = req.item or {}
    surface = item.get("surface") or ""
    base_form = item.get("base_form") or surface
    item_type = item.get("type") or "word"

    # vocab cache lookup
    cached = load_vocab_cache(surface)
    if cached.get("detail"):
        return {"explanation": cached.get("detail"), "examples": (cached.get("examples") or [])[:2]}

    prompt = (
        "Given a Japanese sentence and a target item (word or grammar), provide a concise deep dive for an advanced learner (JLPT N1+). "
        "Use JAPANESE ONLY in all fields. Do NOT include English. "
        'Return ONLY JSON with keys: explanation (short paragraph, Japanese only), examples (array of 2 objects with keys "jp" (Japanese sentence) and "en" (leave empty string if needed)). '
        "Keep examples natural and highlight the target usage.\n"
        f"Sentence: {req.sentence}\n"
        f"Target item: type={item_type}, surface={surface}, base_form={base_form}"
    )

    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. Use Japanese only; no English."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = chat.choices[0].message.content or "{}"
        data = json.loads(raw)
        explanation = data.get("explanation", "")
        examples = data.get("examples", [])
        if not isinstance(examples, list):
            examples = []
        save_vocab_cache(
            surface=surface,
            base_form=base_form,
            brief=explanation if explanation else None,
            detail=explanation,
            examples=examples,
        )
    except Exception as e:
        print("Deep dive error:", repr(e))
        raise HTTPException(status_code=500, detail="Deep dive failed.")

    return {"explanation": explanation, "examples": examples[:2]}


@app.get("/api/vocab_list")
async def vocab_list():
    entries: list[dict] = []
    try:
        with get_session() as session:
            rows = session.exec(select(VocabEntryCache)).all()
            for r in rows:
                entries.append(
                    {
                        "surface": r.surface,
                        "base_form": r.base_form or r.surface,
                        "brief": r.brief,
                        "detail": r.detail,
                    }
                )
    except Exception as e:
        print("DB vocab_list error:", repr(e))
    # fallback to JSON if DB empty
    if not entries:
        for path in VOCAB_CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries.append(
                    {
                        "surface": data.get("surface", ""),
                        "base_form": data.get("base_form") or data.get("surface", ""),
                        "brief": data.get("brief"),
                        "detail": data.get("detail"),
                    }
                )
            except Exception:
                continue
    # sort alphabetically by surface for stability
    entries.sort(key=lambda x: x.get("surface", ""))
    return {"entries": entries}


@app.post("/api/tts_example")
async def tts_example(req: TTSExampleRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    voice = req.voice or "nova"
    speed = req.speed or SPEED_DEFAULT

    filename = tts_filename(text, voice, speed)
    cache_path = TTS_CACHE_DIR / filename

    # generate if missing
    if not cache_path.exists():
        try:
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                speed=speed,
                response_format="mp3",
            ) as response:
                response.stream_to_file(cache_path)
        except Exception as e:
            print("TTS example error:", repr(e))
            raise HTTPException(status_code=500, detail="TTS failed")

    # update vocab cache for surface if provided
    if req.surface:
        try:
            data = load_vocab_cache(req.surface)
            examples = data.get("examples", [])
            # find matching example jp text and attach audio_path
            updated = False
            for ex in examples:
                if ex.get("jp") == text:
                    ex["audio_path"] = f"/tts_cache/{filename}"
                    updated = True
            if not updated:
                examples.append({"jp": text, "en": "", "audio_path": f"/tts_cache/{filename}"})
            data["examples"] = examples
            save_vocab_cache(
                surface=data.get("surface") or req.surface,
                base_form=data.get("base_form") or req.surface,
                brief=data.get("brief"),
                detail=data.get("detail"),
                examples=examples,
            )
        except Exception as e:
            print("Update vocab with audio failed:", repr(e))

    return {"audio_url": f"/tts_cache/{filename}"}


@app.post("/api/reading_word")
async def reading_word(req: ReadingWordRequest):
    surface = req.surface.strip()
    if not surface:
        raise HTTPException(status_code=400, detail="surface required")
    # in-memory cache first
    if surface in READING_CACHE:
        return {"reading": READING_CACHE[surface]}
    # vocab cache (json/db) fallback
    cached = load_vocab_cache(surface)
    if cached.get("brief") and "読み" in (cached.get("brief") or ""):
        # heuristic skip; not reliable
        pass
    reading = None
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return ONLY JSON like {\"reading\":\"<hiragana>\"}. Use hiragana only."},
                {"role": "user", "content": f"次の日本語の語の読みをひらがなだけで返してください: {surface}"},
            ],
        )
        raw = chat.choices[0].message.content or "{}"
        data = json.loads(raw)
        reading = data.get("reading")
    except Exception as e:
        print("reading_word error:", repr(e))
        reading = None

    if reading:
        READING_CACHE[surface] = reading
    return {"reading": reading}


@app.post("/api/translate_sentence", response_model=TranslateResponse)
async def translate_sentence(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    target = (req.target_lang or "en").lower()
    cache = load_sentence_cache(text)
    translations = cache.get("translations") or {}
    if target in translations:
        return {"translation": translations[target]}
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": f"Translate the following text into {target}. Return only the translation."},
                {"role": "user", "content": text},
            ],
        )
        translation = chat.choices[0].message.content or ""
        translations[target] = translation
        save_sentence_cache(text, translations=translations)
        return {"translation": translation}
    except Exception as e:
        print("translate error:", repr(e))
        raise HTTPException(status_code=500, detail="translation failed")


@app.post("/api/translate_deep_dive")
async def translate_deep_dive(req: TranslateDeepDiveRequest):
    target = (req.target_lang or "en").lower()
    explanation = (req.explanation or "").strip()
    examples = req.examples or []
    jp_examples = [ex.get("jp", "") for ex in examples if isinstance(ex, dict)]
    if not explanation and not jp_examples:
        raise HTTPException(status_code=400, detail="content required")
    if target == "ja":
        return {"explanation": explanation, "examples": jp_examples}

    prompt = (
        f"Translate the following Japanese explanation and example sentences into {target}. "
        "Return ONLY JSON with this schema:\n"
        '{ "explanation": "<translated explanation>", "examples": ["<translated example 1>", "<translated example 2>"] }'
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": f"{prompt}\nExplanation: {explanation}\nExamples: {json.dumps(jp_examples, ensure_ascii=False)}"},
            ],
        )
        raw = chat.choices[0].message.content or "{}"
        data = json.loads(raw)
        translated_expl = data.get("explanation", "")
        translated_examples = data.get("examples", [])
        if not isinstance(translated_examples, list):
            translated_examples = []
        # pad/trim to match input length
        if len(translated_examples) < len(jp_examples):
            translated_examples.extend([""] * (len(jp_examples) - len(translated_examples)))
        translated_examples = translated_examples[: len(jp_examples)]
        return {"explanation": translated_expl, "examples": translated_examples}
    except Exception as e:
        print("translate_deep_dive error:", repr(e))
        raise HTTPException(status_code=500, detail="translation failed")


@app.post("/api/sentence/delete")
async def delete_sentence(req: DeleteSentenceRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    delete_sentence_cache(text)
    return {"deleted": True}


@app.post("/api/clean_source")
async def clean_source(req: CleanSourceRequest):
    rel_path = req.path.strip()
    if not rel_path:
        raise HTTPException(status_code=400, detail="path is required")
    target = ROOT_DIR / rel_path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        original = target.read_text(encoding="utf-8")
        cleaned = clean_japanese_spacing(original)
        target.write_text(cleaned, encoding="utf-8")
        return {"cleaned": True, "bytes": len(cleaned)}
    except Exception as e:
        print("clean_source error:", repr(e))
        raise HTTPException(status_code=500, detail="clean failed")


@app.get("/api/source_list")
async def source_list():
    return {"sources": list_sources()}


@app.get("/api/raw_list")
async def raw_list():
    files = []
    for p in RAW_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in RAW_ALLOWED_EXTS:
            files.append({"name": p.name, "path": p.relative_to(ROOT_DIR).as_posix(), "size": p.stat().st_size})
    files.sort(key=lambda x: x["name"])
    return {"files": files}


@app.get("/api/raw_file")
async def raw_file(name: str):
    # security: do not allow path traversal
    target = RAW_DIR / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.suffix.lower() not in RAW_ALLOWED_EXTS:
        raise HTTPException(status_code=403, detail="extension not allowed")
    return FileResponse(target)


@app.post("/api/audio_transcribe")
async def audio_transcribe(req: AudioTranscribeRequest):
    target = AUDIO_DIR / req.name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.suffix.lower() not in AUDIO_ALLOWED_EXTS:
        raise HTTPException(status_code=403, detail="extension not allowed")
    if target.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="empty audio")

    mtime = target.stat().st_mtime
    cached = load_audio_transcript_cached(req.name, mtime)
    if cached:
        return {"text": cached.get("text", ""), "segments": cached.get("segments", [])}

    try:
        chunks = split_audio_chunks(target, AUDIO_CHUNK_SECONDS)
    except HTTPException:
        raise
    except Exception as e:
        print("audio split error:", repr(e))
        raise HTTPException(status_code=500, detail="audio split failed")

    all_segments: list[dict] = []
    text_parts: list[str] = []
    offset = 0.0
    for idx, ch in enumerate(chunks):
        try:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=(ch["name"], ch["bytes"]),
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        except Exception as e:
            print("audio transcribe chunk error:", repr(e))
            raise HTTPException(status_code=500, detail="transcription failed")

        if isinstance(transcription, dict):
            chunk_text = transcription.get("text") or ""
            segments = transcription.get("segments") or []
        else:
            chunk_text = getattr(transcription, "text", "") or ""
            segments = getattr(transcription, "segments", []) or []

        if chunk_text:
            text_parts.append(chunk_text.strip())

        for seg in segments:
            try:
                start = float(seg.get("start", 0)) + offset
                end = float(seg.get("end", 0)) + offset
                seg_text = (seg.get("text") or "").strip()
                if seg_text:
                    all_segments.append({"start": start, "end": end, "text": seg_text})
            except Exception:
                continue

        offset += float(ch.get("duration", AUDIO_CHUNK_SECONDS))

    text = "\n".join([t for t in text_parts if t]).strip()
    with get_session() as session:
        upsert_audio_transcript(
            session,
            audio_name=req.name,
            audio_path=target.relative_to(ROOT_DIR).as_posix(),
            audio_mtime=mtime,
            text=text,
            segments=all_segments,
        )
        session.commit()

    return {"text": text, "segments": all_segments}


@app.get("/api/audio_list")
async def audio_list():
    files = []
    for p in AUDIO_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_ALLOWED_EXTS:
            files.append(
                {
                    "name": p.name,
                    "path": p.relative_to(ROOT_DIR).as_posix(),
                    "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                }
            )
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return {"files": files}


@app.get("/api/audio_file")
async def audio_file(name: str):
    # security: do not allow path traversal
    target = AUDIO_DIR / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.suffix.lower() not in AUDIO_ALLOWED_EXTS:
        raise HTTPException(status_code=403, detail="extension not allowed")
    return FileResponse(target)


def load_audio_transcript_cached(name: str, mtime: float) -> dict | None:
    try:
        with get_session() as session:
            row = session.get(AudioTranscript, name)
            if not row:
                return None
            cached = audio_to_dict(row) or {}
            cached_mtime = cached.get("audio_mtime")
            if cached_mtime is None:
                return None
            if abs(float(cached_mtime) - float(mtime)) > 0.0001:
                return None
            return cached
    except Exception:
        return None


def probe_duration_seconds(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def split_audio_chunks(path: Path, chunk_seconds: int) -> list[dict]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=500, detail="ffmpeg is required for chunked transcription")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pattern = tmp_path / f"chunk_%03d{path.suffix.lower()}"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            str(pattern),
        ]
        subprocess.run(cmd, check=True)
        chunks = sorted(tmp_path.glob(f"chunk_*{path.suffix.lower()}"))
        if not chunks:
            chunks = [path]
        out = []
        for ch in chunks:
            audio_bytes = ch.read_bytes()
            dur = probe_duration_seconds(ch)
            out.append(
                {
                    "name": ch.name,
                    "bytes": audio_bytes,
                    "duration": dur if dur is not None else float(chunk_seconds),
                }
            )
        return out


def get_user_settings(session: Session, user_id: str) -> SettingsResponse:
    row = session.get(UserSettings, user_id)
    if not row:
        return SettingsResponse(user_name="Rusty")
    return SettingsResponse(
        default_voice=row.default_voice or "nova",
        tts_speed=row.tts_speed if row.tts_speed is not None else SPEED_DEFAULT,
        show_fg=bool(row.show_fg) if row.show_fg is not None else False,
        show_hg=bool(row.show_hg) if row.show_hg is not None else False,
        default_source=row.default_source,
        font_scale=row.font_scale if row.font_scale is not None else 1.0,
        native_lang=row.native_lang or "en",
        show_native_defs=bool(getattr(row, "show_native_defs", True)) if getattr(row, "show_native_defs", None) is not None else True,
        user_name=row.user_name or "Rusty",
    )


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings():
    user_id = get_current_user_id()
    with get_session() as session:
        return get_user_settings(session, user_id)


@app.post("/api/settings", response_model=SettingsResponse)
async def save_settings(settings: SettingsResponse):
    user_id = get_current_user_id()
    with get_session() as session:
        row = session.get(UserSettings, user_id)
        if not row:
            row = UserSettings(user_id=user_id)
            session.add(row)
        row.default_voice = settings.default_voice
        row.tts_speed = settings.tts_speed
        row.show_fg = settings.show_fg
        row.show_hg = settings.show_hg
        row.default_source = settings.default_source
        row.font_scale = settings.font_scale
        row.native_lang = settings.native_lang
        row.show_native_defs = settings.show_native_defs
        if settings.user_name and settings.user_name.strip():
            row.user_name = settings.user_name.strip()
        elif row.user_name is None:
            row.user_name = "Rusty"
        session.commit()
        return get_user_settings(session, user_id)


@app.post("/api/source_record", response_model=SourceRecordResponse)
async def source_record(
    file: UploadFile = File(...),
    category: str | None = Form(None),
    title: str | None = Form(None),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    # save audio
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    audio_path = RECORDINGS_DIR / f"{audio_hash}.webm"
    audio_path.write_bytes(audio_bytes)
    audio_rel = audio_path.relative_to(ROOT_DIR).as_posix()

    # transcribe to text
    transcript_text = ""
    try:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=(file.filename or "audio.webm", audio_bytes),
            response_format="text",
        )
        transcript_text = transcription or ""
    except Exception as e:
        print("transcription error:", repr(e))
        raise HTTPException(status_code=500, detail="transcription failed")

    # translate to Japanese
    ja_text = transcript_text
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Translate the following into natural Japanese only. Return only the Japanese text."},
                {"role": "user", "content": transcript_text},
            ],
        )
        ja_text = chat.choices[0].message.content or transcript_text
    except Exception as e:
        print("translate error:", repr(e))
        ja_text = transcript_text

    # save text to source_materials
    text_rel = ensure_text_path(ja_text, preferred_name=f"recording_{audio_hash[:8]}")

    source_hash = hashlib.sha256(ja_text.encode("utf-8")).hexdigest()
    with get_session() as session:
        session.merge(
            SourceMaterial(
                source_hash=source_hash,
                title_ja=title or ja_text[:30],
                title_en=None,
                title_ko=None,
                title_zh=None,
                text_path=text_rel,
                audio_path=audio_rel,
                category=category,
            )
        )
        session.commit()

    return {
        "name": title or ja_text[:30],
        "path": text_rel,
        "audio_path": audio_rel,
        "category": category,
    }
