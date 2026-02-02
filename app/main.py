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


class AnalyzeResponse(BaseModel):
    items: list[dict]


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

    # cache hit: stream file
    if cache_path.exists():
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

        return FileResponse(cache_path, media_type="audio/mpeg")
    except Exception as e:
        print("TTS error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze_sentence", response_model=AnalyzeResponse)
async def analyze_sentence(req: AnalyzeRequest):
    normalized_text = req.text.strip()
    # cache key only depends on sentence text (analysis is voice/speed agnostic)
    key_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    cache_path = ANALYSIS_CACHE_DIR / f"{key_hash}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            items = cached.get("items", [])
            if isinstance(items, list):
                return {"items": items}
        except Exception:
            # fall through to fresh call on corrupt cache
            pass

    prompt = (
        "You are a Japanese language coach. Given ONE Japanese sentence, list the 3 most difficult items "
        "(word or grammar) for a JLPT N3–N2 learner. Ignore basic N5–N4 material unless nothing harder exists. "
        "Respond ONLY with JSON matching exactly this schema:\n"
        '{ "items": [ { "type": "word" | "grammar", "surface": "<exact span from the sentence>", '
        '"base_form": "<dictionary form or grammar label>", "difficulty": 1, '
        '"english_hint": "<short explanation in English>", "reason": "<why this is difficult>" } ] }'
    )

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
            # handle if model wrapped in code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned.split("\n", 1)[-1]
            data = json.loads(cleaned)
        items = data.get("items", [])
        if not isinstance(items, list):
            items = []
        # cache the successful response
        try:
            cache_path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print("Analyze cache write error:", repr(e))
    except Exception as e:
        print("Analyze error:", repr(e))
        items = []

    return {"items": items}




