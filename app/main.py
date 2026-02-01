# app/main.py
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os


# load_dotenv()  # loads OPENAI_API_KEY from .env
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# client = OpenAI() 
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI()

# Serve static files (your HTML, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


class TTSRequest(BaseModel):
    text: str
    voice: str | None = "coral"  # default voice; you can change


@app.get("/")
async def root():
    # Serve the main HTML page
    return FileResponse("static/index.html")


@app.post("/api/tts")
async def tts(req: TTSRequest):
    """
    Generate Japanese audio from text and return as MP3.
    """
    # Call OpenAI TTS; non-streaming, returns bytes in current client
    audio_bytes = client.audio.speech.create(
        model="gpt-4o-mini-tts",  # adjust if model name changes
        voice=req.voice,
        input=req.text,
        response_format="mp3",
        # instructions can be baked into the text or added here later if needed
    )

    # FastAPI will send this as audio/mpeg
    return Response(content=audio_bytes, media_type="audio/mpeg")


# # with open('content/raw/sample-writing-cafune-1.txt', 'r', encoding='utf-8') as f:
# #     book_text = f.read()

# from pathlib import Path
# import re
# from openai import OpenAI
# from dotenv import load_dotenv
# import pygame
# import time

# load_dotenv()  # take environment variables from .env


# # Load text and grab the first Japanese sentence (ending with 。！？)
# txt = Path("content/raw/sample-writing-cafune-1.txt").read_text(encoding="utf-8")
# first_sentence = re.split(r"(?<=[。！？!?])\s*", txt.strip(), maxsplit=1)[0]

# client = OpenAI()  # uses OPENAI_API_KEY

# out_path = Path("content/raw/cafune-first-sentence.mp3")
# text_out_path = Path("content/raw/cafune-first-sentence.txt")
# text_out_path.write_text(first_sentence, encoding="utf-8")
# with client.audio.speech.with_streaming_response.create(
#     model="gpt-4o-mini-tts",      # latest high-quality TTS
#     voice="coral",                # pick any listed voice
#     input=first_sentence,
#     instructions="自然な日本語で、落ち着いたトーンで話してください。",
#     response_format="mp3",
# ) as resp:
#     resp.stream_to_file(out_path)

# print(f"Wrote {out_path}")

# # Play the generated audio

# pygame.mixer.init()
# pygame.mixer.music.load("example.mp3")
# pygame.mixer.music.play()

# # Keep the program alive while the music plays
# while pygame.mixer.music.get_busy():
#     time.sleep(0.1)

