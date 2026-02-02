from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
print("Key present:", os.getenv("OPENAI_API_KEY") is not None)

client = OpenAI()

try:
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",  # or the model you’re using
        voice="alloy",
        input="こんにちは、テストです。"
    ) as response:
        response.stream_to_file("test_output.mp3")

    print("TTS OK, wrote test_output.mp3")
except Exception as e:
    print("TTS error in test script:", repr(e))
