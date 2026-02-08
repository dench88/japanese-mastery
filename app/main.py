from pathlib import Path
import os
import re
import json
import hashlib

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import select

from dotenv import load_dotenv
from openai import OpenAI

from app.db_models import (
    init_db,
    get_session,
    sentence_to_dict,
    vocab_to_dict,
    upsert_sentence,
    upsert_vocab,
    SentenceCache,
    VocabEntryCache,
    SourceMaterial,
    UserSettings,
    User,
)
from app.auth import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    oauth2_scheme,
)

# --- paths & env ---
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)
SPEED_DEFAULT = float(os.getenv("TTS_SPEED_DEFAULT", "1.1"))

# --- load and split book text ---
BOOKS_DIR = ROOT_DIR / "content" / "source_materials"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_BOOK = BOOKS_DIR / "sample-writing-cafune-1.txt"
legacy = ROOT_DIR / "content" / "raw" / "sample-writing-cafune-1.txt"
if legacy.exists():
    DEFAULT_BOOK.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
elif not DEFAULT_BOOK.exists():
    DEFAULT_BOOK.write_text("これはサンプル文章です。", encoding="utf-8")
BOOK_TEXT = DEFAULT_BOOK.read_text(encoding="utf-8")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*")

# --- caches ---
TTS_CACHE_DIR = ROOT_DIR / "tts_cache"
TTS_CACHE_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR = ROOT_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
RAW_DIR = ROOT_DIR / "content" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_ALLOWED_EXTS = {".pdf", ".txt", ".md", ".docx", ".zip"}

# --- OpenAI client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment; set it in .env or shell.")
client = OpenAI(api_key=api_key)

app = FastAPI()


@app.on_event("startup")
async def startup():
    await init_db()


# serve static files
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")
app.mount("/tts_cache", StaticFiles(directory=TTS_CACHE_DIR), name="tts-cache")
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")


# --- Pydantic schemas ---

class TTSRequest(BaseModel):
    text: str
    voice: str | None = "nova"
    model: str | None = "gpt-4o-mini-tts"
    speed: float | None = None


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeDeleteRequest(BaseModel):
    text: str
    surface: str


class AnalyzeResponse(BaseModel):
    reading_hiragana: str | None = None
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


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    translation: str


class DeleteSentenceRequest(BaseModel):
    text: str


class CleanSourceRequest(BaseModel):
    path: str


class ManualBriefRequest(BaseModel):
    sentence: str
    surface: str


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


# simple in-memory cache for word readings
READING_CACHE: dict[str, str] = {}


# --- helpers ---
def sentence_hash(sentence: str) -> str:
    return hashlib.sha256(sentence.strip().encode("utf-8")).hexdigest()


def vocab_hash(surface: str) -> str:
    return hashlib.sha256(surface.strip().encode("utf-8")).hexdigest()


async def load_sentence_cache(sentence: str) -> dict:
    """Load cached sentence data from DB."""
    h = sentence_hash(sentence)
    data = {"sentence": sentence, "hiragana": None, "reading_ruby": None, "hard_items": [], "audio_path": None, "translations": {}}
    async with get_session() as session:
        row = await session.get(SentenceCache, h)
        if row:
            dbd = sentence_to_dict(row) or {}
            data.update(
                {
                    "hiragana": dbd.get("hiragana"),
                    "reading_ruby": None,
                    "hard_items": dbd.get("hard_items", []),
                    "audio_path": dbd.get("audio_path"),
                    "translations": dbd.get("translations", {}),
                }
            )
    return data


async def save_sentence_cache(sentence: str, hiragana=None, reading_ruby=None, hard_items=None, audio_path=None, translations=None):
    """Persist sentence data to DB."""
    h = sentence_hash(sentence)
    existing = await load_sentence_cache(sentence)
    if hiragana is not None:
        existing["hiragana"] = hiragana
    if hard_items is not None:
        existing["hard_items"] = hard_items
    if audio_path is not None:
        existing["audio_path"] = audio_path
    if translations is not None:
        existing["translations"] = translations
    async with get_session() as session:
        await upsert_sentence(
            session,
            sentence_hash=h,
            sentence_text=sentence,
            hiragana=existing.get("hiragana"),
            audio_path=existing.get("audio_path"),
            hard_items=existing.get("hard_items") or [],
            translations=existing.get("translations") or {},
        )
        await session.commit()


async def delete_sentence_cache(sentence: str):
    """Remove cached data for one sentence."""
    h = sentence_hash(sentence)
    async with get_session() as session:
        row = await session.get(SentenceCache, h)
        if row:
            await session.delete(row)
            await session.commit()


async def load_vocab_cache(surface: str) -> dict:
    """Load cached vocab data from DB."""
    h = vocab_hash(surface)
    data = {"surface": surface, "base_form": surface, "brief": None, "detail": None, "examples": []}
    async with get_session() as session:
        row = await session.get(VocabEntryCache, h)
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


async def save_vocab_cache(surface: str, base_form=None, brief=None, detail=None, examples=None):
    h = vocab_hash(surface)
    existing = await load_vocab_cache(surface)
    if base_form is not None:
        existing["base_form"] = base_form
    if brief is not None:
        existing["brief"] = brief
    if detail is not None:
        existing["detail"] = detail
    if examples is not None:
        existing["examples"] = examples
    async with get_session() as session:
        await upsert_vocab(
            session,
            surface_hash=h,
            surface=surface,
            base_form=existing.get("base_form"),
            brief=existing.get("brief"),
            detail=existing.get("detail"),
            examples=existing.get("examples") or [],
        )
        await session.commit()


def tts_filename(text: str, voice: str, speed: float) -> str:
    key = f"{text}|{voice}|{speed}"
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{key_hash}.mp3"


async def list_sources():
    sources = []
    async with get_session() as session:
        result = await session.exec(select(SourceMaterial))
        rows = result.all()
        for r in rows:
            sources.append(
                {
                    "name": r.title_ja or (Path(r.text_path).name if r.text_path else r.source_hash),
                    "path": r.text_path,
                    "audio_path": r.audio_path,
                    "category": r.category,
                }
            )
    # File-based fallback for text files not yet in DB
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
        candidate2 = BOOKS_DIR / source_name
        if candidate2.exists() and candidate2.is_file():
            return candidate2.read_text(encoding="utf-8")
        raise HTTPException(status_code=404, detail="Source not found")
    return BOOK_TEXT


def split_sentences(text: str):
    return [s for s in SENTENCE_PATTERN.split(text.strip()) if s]


def ensure_text_path(text: str, preferred_name: str | None = None) -> str:
    stem = preferred_name or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    candidate = BOOKS_DIR / f"{stem}.txt"
    counter = 1
    while candidate.exists():
        candidate = BOOKS_DIR / f"{stem}_{counter}.txt"
        counter += 1
    candidate.write_text(text, encoding="utf-8")
    return candidate.relative_to(ROOT_DIR).as_posix()


def clean_japanese_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([、。！？「」『』（）〔〕［］])", r"\1", text)
    text = re.sub(r"([「『（〔［])\s+", r"\1", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([\(])\s+", r"\1", text)
    text = "\n".join([ln.strip() for ln in text.splitlines()])
    return text


# --- shared analysis helper ---
async def analyze_sentence_internal(sentence: str):
    normalized_text = sentence.strip()
    cached = await load_sentence_cache(normalized_text)
    if cached.get("hiragana") is not None and isinstance(cached.get("hard_items"), list):
        return cached["hiragana"], cached.get("reading_ruby"), cached.get("hard_items", [])

    prompt = (
        "You are a Japanese language coach for advanced learners (JLPT N1+). Given ONE Japanese sentence, do two things: "
        "1) Convert the entire sentence into hiragana (no romaji), preserving punctuation; leave katakana loanwords as katakana. "
        "2) Return ruby markup for the sentence: each word with kanji should be wrapped as <ruby><rb>漢字</rb><rt>よみ</rt></ruby>; kana-only spans stay plain. "
        "3) List the 3 most difficult items (word or grammar) for an N1+ learner. Ignore anything below N2 unless nothing harder exists. If nothing suitable, return an empty list. "
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
        await save_sentence_cache(normalized_text, hiragana=reading, reading_ruby=reading_ruby, hard_items=items)
    except Exception as e:
        print("Analyze error:", repr(e))
        reading = None
        reading_ruby = None
        items = []

    return reading, reading_ruby, items


# ==========================================================================
# AUTH ENDPOINTS
# ==========================================================================

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    async with get_session() as session:
        result = await session.exec(select(User).where(User.email == req.email))
        existing = result.first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        user = User(
            email=req.email,
            hashed_password=hash_password(req.password),
            display_name=req.display_name,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        token = create_access_token(user.id)
        return TokenResponse(access_token=token)


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    async with get_session() as session:
        result = await session.exec(select(User).where(User.email == req.email))
        user = result.first()
        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account disabled")
        token = create_access_token(user.id)
        return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserResponse)
async def auth_me(current_user: User = Depends(get_current_user)):
    return UserResponse(id=current_user.id, email=current_user.email, display_name=current_user.display_name)


# ==========================================================================
# APP ENDPOINTS (protected by auth)
# ==========================================================================

@app.get("/")
async def root():
    return RedirectResponse(url="/static/login.html", status_code=302)


@app.get("/api/book")
async def get_book(source: str | None = None, current_user: User = Depends(get_current_user)):
    text = load_source_text(source)
    return {"text": text}


@app.get("/api/sentence/{index}")
async def get_sentence(index: int, source: str | None = None, current_user: User = Depends(get_current_user)):
    text = load_source_text(source)
    sentences = split_sentences(text)
    total = len(sentences)
    if index < 0 or index >= total:
        raise HTTPException(status_code=404, detail="No more sentences.")
    return {"index": index, "sentence": sentences[index], "total": total}


@app.get("/api/sentences")
async def get_sentences(source: str | None = None, current_user: User = Depends(get_current_user)):
    text = load_source_text(source)
    sentences = split_sentences(text)
    return {
        "sentences": [{"index": i, "sentence": s} for i, s in enumerate(sentences)],
        "total": len(sentences),
    }


@app.post("/api/tts")
async def tts(req: TTSRequest, current_user: User = Depends(get_current_user)):
    model = req.model or "gpt-4o-mini-tts"
    voice = req.voice or "nova"
    speed = req.speed or SPEED_DEFAULT
    normalized_text = req.text.strip()

    key_str = f"{model}|{voice}|{speed}|{normalized_text}"
    key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    cache_path = TTS_CACHE_DIR / f"{key_hash}.mp3"
    audio_rel = f"/tts_cache/{cache_path.name}"

    if cache_path.exists():
        await save_sentence_cache(normalized_text, audio_path=audio_rel)
        return FileResponse(cache_path, media_type="audio/mpeg")

    try:
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=normalized_text,
            speed=speed,
            response_format="mp3",
        ) as response:
            response.stream_to_file(cache_path)

        await save_sentence_cache(normalized_text, audio_path=audio_rel)
        return FileResponse(cache_path, media_type="audio/mpeg")
    except Exception as e:
        print("TTS error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze_sentence", response_model=AnalyzeResponse)
async def analyze_sentence(req: AnalyzeRequest, current_user: User = Depends(get_current_user)):
    reading, reading_ruby, items = await analyze_sentence_internal(req.text)
    return {"reading_hiragana": reading, "items": items, "reading_ruby": reading_ruby}


@app.post("/api/reading_sentence")
async def reading_sentence(req: AnalyzeRequest, current_user: User = Depends(get_current_user)):
    cached = await load_sentence_cache(req.text)
    reading = cached.get("hiragana")
    reading_ruby = cached.get("reading_ruby")
    if reading is None:
        reading, reading_ruby, items = await analyze_sentence_internal(req.text)
        if reading is not None:
            await save_sentence_cache(req.text, hiragana=reading, reading_ruby=reading_ruby, hard_items=items)
    return {"reading_hiragana": reading, "reading_ruby": reading_ruby}


@app.post("/api/hard_items")
async def hard_items(req: AnalyzeRequest, current_user: User = Depends(get_current_user)):
    cached = await load_sentence_cache(req.text)
    items = cached.get("hard_items", [])
    if not items:
        reading, reading_ruby, items = await analyze_sentence_internal(req.text)
        await save_sentence_cache(req.text, hiragana=reading, hard_items=items)
    return {"items": items}


@app.post("/api/hard_item/delete")
async def delete_hard_item(req: AnalyzeDeleteRequest, current_user: User = Depends(get_current_user)):
    try:
        data = await load_sentence_cache(req.text)
        items = data.get("hard_items", [])
        surface = req.surface
        if not surface:
            return {"removed": False}
        new_items = [i for i in items if i.get("surface") != surface]
        await save_sentence_cache(req.text, hard_items=new_items)
        return {"removed": True}
    except Exception as e:
        print("Delete hard item error:", repr(e))
        return {"removed": False}


@app.post("/api/manual_brief")
async def manual_brief(req: ManualBriefRequest, current_user: User = Depends(get_current_user)):
    cached = await load_vocab_cache(req.surface)
    if cached.get("brief"):
        return {"hint_ja": cached.get("brief")}

    prompt = (
        "あなたは日本語上級学習者(JLPT N1+)向けのコーチです。与えられた文中の指定語について、短く要点だけ日本語で解説してください。"
        'JSONのみで返してください。形式: { "hint_ja": "<短い説明 (日本語)>" }'
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
        await save_vocab_cache(
            surface=req.surface,
            base_form=req.surface,
            brief=hint,
            detail=None,
            examples=[],
        )
        s_data = await load_sentence_cache(req.sentence)
        hard_items_list = s_data.get("hard_items", [])
        if not any(i.get("surface") == req.surface for i in hard_items_list):
            hard_items_list.append(
                {
                    "surface": req.surface,
                    "base_form": req.surface,
                    "type": "word",
                    "difficulty": 1,
                    "hint_ja": hint,
                }
            )
        await save_sentence_cache(req.sentence, hard_items=hard_items_list)
        return {"hint_ja": hint}
    except Exception as e:
        print("Manual brief error:", repr(e))
        return {"hint_ja": ""}


@app.post("/api/deep_dive", response_model=DeepDiveResponse)
async def deep_dive(req: DeepDiveRequest, current_user: User = Depends(get_current_user)):
    item = req.item or {}
    surface = item.get("surface") or ""
    base_form = item.get("base_form") or surface
    item_type = item.get("type") or "word"

    cached = await load_vocab_cache(surface)
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
        await save_vocab_cache(
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
async def vocab_list(current_user: User = Depends(get_current_user)):
    entries: list[dict] = []
    async with get_session() as session:
        result = await session.exec(select(VocabEntryCache))
        rows = result.all()
        for r in rows:
            entries.append(
                {
                    "surface": r.surface,
                    "base_form": r.base_form or r.surface,
                    "brief": r.brief,
                    "detail": r.detail,
                }
            )
    entries.sort(key=lambda x: x.get("surface", ""))
    return {"entries": entries}


@app.post("/api/tts_example")
async def tts_example(req: TTSExampleRequest, current_user: User = Depends(get_current_user)):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    voice = req.voice or "nova"
    speed = req.speed or SPEED_DEFAULT

    filename = tts_filename(text, voice, speed)
    cache_path = TTS_CACHE_DIR / filename

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

    if req.surface:
        try:
            data = await load_vocab_cache(req.surface)
            examples = data.get("examples", [])
            updated = False
            for ex in examples:
                if ex.get("jp") == text:
                    ex["audio_path"] = f"/tts_cache/{filename}"
                    updated = True
            if not updated:
                examples.append({"jp": text, "en": "", "audio_path": f"/tts_cache/{filename}"})
            data["examples"] = examples
            await save_vocab_cache(
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
async def reading_word(req: ReadingWordRequest, current_user: User = Depends(get_current_user)):
    surface = req.surface.strip()
    if not surface:
        raise HTTPException(status_code=400, detail="surface required")
    if surface in READING_CACHE:
        return {"reading": READING_CACHE[surface]}
    reading = None
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": 'Return ONLY JSON like {"reading":"<hiragana>"}. Use hiragana only.'},
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
async def translate_sentence(req: TranslateRequest, current_user: User = Depends(get_current_user)):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    target = (req.target_lang or "en").lower()
    cache = await load_sentence_cache(text)
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
        await save_sentence_cache(text, translations=translations)
        return {"translation": translation}
    except Exception as e:
        print("translate error:", repr(e))
        raise HTTPException(status_code=500, detail="translation failed")


@app.post("/api/sentence/delete")
async def delete_sentence(req: DeleteSentenceRequest, current_user: User = Depends(get_current_user)):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    await delete_sentence_cache(text)
    return {"deleted": True}


@app.post("/api/clean_source")
async def clean_source(req: CleanSourceRequest, current_user: User = Depends(get_current_user)):
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
async def source_list(current_user: User = Depends(get_current_user)):
    return {"sources": await list_sources()}


@app.get("/api/raw_list")
async def raw_list(current_user: User = Depends(get_current_user)):
    files = []
    for p in RAW_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in RAW_ALLOWED_EXTS:
            files.append({"name": p.name, "path": p.relative_to(ROOT_DIR).as_posix(), "size": p.stat().st_size})
    files.sort(key=lambda x: x["name"])
    return {"files": files}


@app.get("/api/raw_file")
async def raw_file(name: str, current_user: User = Depends(get_current_user)):
    target = RAW_DIR / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.suffix.lower() not in RAW_ALLOWED_EXTS:
        raise HTTPException(status_code=403, detail="extension not allowed")
    return FileResponse(target)


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings(current_user: User = Depends(get_current_user)):
    async with get_session() as session:
        row = await session.get(UserSettings, str(current_user.id))
        if not row:
            return SettingsResponse()
        return SettingsResponse(
            default_voice=row.default_voice or "nova",
            tts_speed=row.tts_speed if row.tts_speed is not None else SPEED_DEFAULT,
            show_fg=bool(row.show_fg) if row.show_fg is not None else False,
            show_hg=bool(row.show_hg) if row.show_hg is not None else False,
            default_source=row.default_source,
            font_scale=row.font_scale if row.font_scale is not None else 1.0,
            native_lang=row.native_lang or "en",
        )


@app.post("/api/settings", response_model=SettingsResponse)
async def save_settings(settings: SettingsResponse, current_user: User = Depends(get_current_user)):
    user_id = str(current_user.id)
    async with get_session() as session:
        row = await session.get(UserSettings, user_id)
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
        await session.commit()
        return SettingsResponse(
            default_voice=row.default_voice or "nova",
            tts_speed=row.tts_speed if row.tts_speed is not None else SPEED_DEFAULT,
            show_fg=bool(row.show_fg) if row.show_fg is not None else False,
            show_hg=bool(row.show_hg) if row.show_hg is not None else False,
            default_source=row.default_source,
            font_scale=row.font_scale if row.font_scale is not None else 1.0,
            native_lang=row.native_lang or "en",
        )


@app.post("/api/source_record", response_model=SourceRecordResponse)
async def source_record(
    file: UploadFile = File(...),
    category: str | None = Form(None),
    title: str | None = Form(None),
    current_user: User = Depends(get_current_user),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    audio_path = RECORDINGS_DIR / f"{audio_hash}.webm"
    audio_path.write_bytes(audio_bytes)
    audio_rel = audio_path.relative_to(ROOT_DIR).as_posix()

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

    text_rel = ensure_text_path(ja_text, preferred_name=f"recording_{audio_hash[:8]}")

    source_hash_val = hashlib.sha256(ja_text.encode("utf-8")).hexdigest()
    async with get_session() as session:
        existing = await session.get(SourceMaterial, source_hash_val)
        if existing:
            existing.title_ja = title or ja_text[:30]
            existing.text_path = text_rel
            existing.audio_path = audio_rel
            existing.category = category
        else:
            session.add(
                SourceMaterial(
                    source_hash=source_hash_val,
                    title_ja=title or ja_text[:30],
                    text_path=text_rel,
                    audio_path=audio_rel,
                    category=category,
                )
            )
        await session.commit()

    return {
        "name": title or ja_text[:30],
        "path": text_rel,
        "audio_path": audio_rel,
        "category": category,
    }
