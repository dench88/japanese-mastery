# app/main.py
from pathlib import Path
import re
import os

from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import OpenAI

# --- paths & env ---
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)  # harmless if you prefer setting env manually

# OpenAI client - relies on OPENAI_API_KEY being in environment
# client = OpenAI()

# --- load and split book text ---
BOOK_PATH = ROOT_DIR / "content" / "raw" / "sample-writing-cafune-1.txt"
BOOK_TEXT = BOOK_PATH.read_text(encoding="utf-8")

# Split on Japanese sentence-ending punctuation: 。！？ plus ascii !?
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*")
SENTENCES = [s for s in SENTENCE_PATTERN.split(BOOK_TEXT.strip()) if s]

app = FastAPI()

# serve static files
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")


class TTSRequest(BaseModel):
    text: str
    voice: str | None = "coral"


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
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    audio_bytes = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=req.voice,
        input=req.text,
        response_format="mp3",
    )
    return Response(content=audio_bytes, media_type="audio/mpeg")
