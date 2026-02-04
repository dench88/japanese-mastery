from pathlib import Path
import os
import re
import json
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import OpenAI

# --- paths & env ---
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)  # load env from repo root
SPEED_DEFAULT = float(os.getenv("TTS_SPEED_DEFAULT", "1.1"))

# --- load and split book text ---
BOOK_PATH = ROOT_DIR / "content" / "raw" / "sample-writing-cafune-1.txt"
BOOK_TEXT = BOOK_PATH.read_text(encoding="utf-8")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*")  # Japanese sentence split
SENTENCES = [s for s in SENTENCE_PATTERN.split(BOOK_TEXT.strip()) if s]

# --- caches ---
TTS_CACHE_DIR = ROOT_DIR / "tts_cache"
TTS_CACHE_DIR.mkdir(exist_ok=True)
ANALYSIS_CACHE_DIR = ROOT_DIR / "analysis_cache"
ANALYSIS_CACHE_DIR.mkdir(exist_ok=True)
VOCAB_CACHE_DIR = ROOT_DIR / "vocab_cache"
VOCAB_CACHE_DIR.mkdir(exist_ok=True)

# --- OpenAI client (single instance) ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment; set it in .env or shell.")
client = OpenAI(api_key=api_key)

app = FastAPI()

# serve static files
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")


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


class ManualBriefRequest(BaseModel):
    sentence: str
    surface: str


class VocabEntry(BaseModel):
    surface: str
    base_form: str | None = None
    brief: str | None = None
    detail: str | None = None
    examples: list[dict] | None = None


# --- helpers ---
def sentence_cache_path(sentence: str) -> Path:
    normalized_text = sentence.strip()
    key_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    return ANALYSIS_CACHE_DIR / f"{key_hash}.json"


def vocab_cache_path(surface: str) -> Path:
    key_hash = hashlib.sha256(surface.strip().encode("utf-8")).hexdigest()
    return VOCAB_CACHE_DIR / f"{key_hash}.json"


# --- shared analysis helper (reading + hard items) ---
def analyze_sentence_internal(sentence: str):
    normalized_text = sentence.strip()
    cache_path = sentence_cache_path(normalized_text)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            reading = cached.get("hiragana")
            items = cached.get("hard_items", [])
            if reading is not None and isinstance(items, list):
                return reading, items
        except Exception:
            pass  # fall through to fresh call

    prompt = (
        "You are a Japanese language coach for advanced learners (JLPT N1+). Given ONE Japanese sentence, do two things: "
        "1) Convert the entire sentence into hiragana (no romaji), preserving punctuation; leave katakana loanwords as katakana. "
        "2) List the 3 most difficult items (word or grammar) for an N1+ learner. Ignore anything below N2 unless nothing harder exists. If nothing suitable, return an empty list. "
        "Respond ONLY with JSON matching exactly this schema (all explanations in Japanese, no English words):\n"
        '{ "reading_hiragana": "<sentence rendered fully in hiragana>", '
        '"items": [ { "type": "word" | "grammar", "surface": "<exact span from the sentence>", '
        '"base_form": "<dictionary form or grammar label>", "difficulty": 1, '
        '"hint_ja": "<short explanation in Japanese>", "reason": "<why this is difficult (Japanese)>" } ] }'
    )

    reading = None
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
        try:
            cache_path.write_text(
                json.dumps(
                    {"hiragana": reading, "hard_items": items},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            print("Analyze cache write error:", repr(e))
    except Exception as e:
        print("Analyze error:", repr(e))
        reading = None
        items = []

    return reading, items


@app.get("/")
async def root():
    return FileResponse(ROOT_DIR / "static" / "index.html")


@app.get("/api/book")
async def get_book():
    """Return full book text."""
    return {"text": BOOK_TEXT}


@app.get("/api/sentence/{index}")
async def get_sentence(index: int):
    """Return the sentence at the given index."""
    total = len(SENTENCES)
    if index < 0 or index >= total:
        raise HTTPException(status_code=404, detail="No more sentences.")
    return {
        "index": index,
        "sentence": SENTENCES[index],
        "total": total,
    }


@app.get("/api/sentences")
async def get_sentences():
    """Return all sentences with their indices (for client-side navigation)."""
    return {
        "sentences": [
            {"index": i, "sentence": s}
            for i, s in enumerate(SENTENCES)
        ],
        "total": len(SENTENCES),
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
    sentence_cache = sentence_cache_path(normalized_text)
    # ensure sentence cache exists with defaults
    if not sentence_cache.exists():
        sentence_cache.write_text(
            json.dumps({"hiragana": None, "hard_items": [], "audio_path": None}, ensure_ascii=False),
            encoding="utf-8",
        )

    # cache hit: stream file
    if cache_path.exists():
        # write audio path reference
        try:
            data = json.loads(sentence_cache.read_text(encoding="utf-8"))
            data["audio_path"] = str(cache_path.relative_to(ROOT_DIR))
            sentence_cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
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

        try:
            data = {}
            if sentence_cache.exists():
                data = json.loads(sentence_cache.read_text(encoding="utf-8"))
            data.setdefault("hiragana", None)
            data.setdefault("hard_items", [])
            data["audio_path"] = str(cache_path.relative_to(ROOT_DIR))
            sentence_cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        return FileResponse(cache_path, media_type="audio/mpeg")
    except Exception as e:
        print("TTS error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze_sentence", response_model=AnalyzeResponse)
async def analyze_sentence(req: AnalyzeRequest):
    reading, items = analyze_sentence_internal(req.text)
    return {"reading_hiragana": reading, "items": items}


@app.post("/api/reading_sentence")
async def reading_sentence(req: AnalyzeRequest):
    cache_path = sentence_cache_path(req.text)
    reading = None
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            reading = data.get("hiragana")
        except Exception:
            pass
    if reading is None:
        reading, items = analyze_sentence_internal(req.text)
        # update cache with hiragana if available
        if reading is not None:
            try:
                data = {}
                if cache_path.exists():
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                data["hiragana"] = reading
                data.setdefault("hard_items", items or [])
                cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
    return {"reading_hiragana": reading}


@app.post("/api/hard_items")
async def hard_items(req: AnalyzeRequest):
    cache_path = sentence_cache_path(req.text)
    items = []
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            items = data.get("hard_items", [])
        except Exception:
            pass
    if not items:
        reading, items = analyze_sentence_internal(req.text)
        try:
            data = {}
            if cache_path.exists():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            data.setdefault("hiragana", reading)
            data["hard_items"] = items
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return {"items": items}


@app.post("/api/hard_item/delete")
async def delete_hard_item(req: AnalyzeDeleteRequest):
    """Remove a specific item (by surface) from the cached analysis for this sentence."""
    cache_path = sentence_cache_path(req.text)
    if not cache_path.exists():
        return {"removed": False}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        items = data.get("hard_items", [])
        surface = req.surface
        if not surface:
            return {"removed": False}
        new_items = [i for i in items if i.get("surface") != surface]
        data["hard_items"] = new_items
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {"removed": True}
    except Exception as e:
        print("Delete hard item error:", repr(e))
        return {"removed": False}


@app.post("/api/manual_brief")
async def manual_brief(req: ManualBriefRequest):
    """Return a short Japanese hint for a user-selected surface in the sentence."""
    cache_path = vocab_cache_path(req.surface)
    # serve cached brief if present
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            brief = data.get("brief")
            if brief:
                return {"hint_ja": brief}
        except Exception:
            pass

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
        try:
            # cache per-word
            cache_path.write_text(
                json.dumps(
                    {
                        "surface": req.surface,
                        "base_form": req.surface,
                        "brief": hint,
                        "detail": None,
                        "examples": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            # update sentence cache hard_items list
            s_path = sentence_cache_path(req.sentence)
            s_data = {"hiragana": None, "hard_items": []}
            if s_path.exists():
                try:
                    s_data = json.loads(s_path.read_text(encoding="utf-8"))
                except Exception:
                    s_data = {"hiragana": None, "hard_items": []}
            hard_items = s_data.get("hard_items", [])
            if not any(i.get("surface") == req.surface for i in hard_items):
                hard_items.append(
                    {
                        "surface": req.surface,
                        "base_form": req.surface,
                        "type": "word",
                        "difficulty": 1,
                        "hint_ja": hint,
                    }
                )
            s_data["hard_items"] = hard_items
            s_path.write_text(json.dumps(s_data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return {"hint_ja": hint}
    except Exception as e:
        print("Manual brief error:", repr(e))
        return {"hint_ja": ""}


@app.post("/api/deep_dive", response_model=DeepDiveResponse)
async def deep_dive(req: DeepDiveRequest):
    item = req.item or {}
    surface = item.get("surface") or ""
    base_form = item.get("base_form") or surface
    item_type = item.get("type") or "word"

    # vocab cache lookup
    vpath = vocab_cache_path(surface)
    if vpath.exists():
        try:
            data = json.loads(vpath.read_text(encoding="utf-8"))
            detail = data.get("detail")
            examples = data.get("examples", [])
            if detail:
                return {"explanation": detail, "examples": examples[:2]}
        except Exception:
            pass

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
        try:
            vpath.write_text(
                json.dumps(
                    {
                        "surface": surface,
                        "base_form": base_form,
                        "brief": explanation if explanation else None,
                        "detail": explanation,
                        "examples": examples,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    except Exception as e:
        print("Deep dive error:", repr(e))
        raise HTTPException(status_code=500, detail="Deep dive failed.")

    return {"explanation": explanation, "examples": examples[:2]}


@app.get("/api/vocab_list")
async def vocab_list():
    entries: list[dict] = []
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
