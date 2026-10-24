"""Call the OpenAI-compatible server with the official openai package and with plain HTTP.

    lure serve --port 8880 &                 # with LURE_MODEL set
    lure voice add willow ref.wav "transcript of ref.wav"
    python examples/openai_client.py
"""
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8880/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
    model="lure", voice="willow", input="(laugh) Okay, that actually worked.",
    instructions="Amused, relaxed.", response_format="wav",
) as resp:
    resp.stream_to_file("openai_client.wav")

# Extra fields (seed, sampling, language) go in the JSON body.
body = {"model": "lure", "voice": "willow", "input": "Same voice, fixed seed.", "seed": 7,
        "sampling": {"temperature": 0.9, "top_k": 50}, "response_format": "pcm"}
with httpx.stream("POST", "http://127.0.0.1:8880/v1/audio/speech", json=body, timeout=120) as r:
    r.raise_for_status()
    pcm = b"".join(r.iter_bytes())  # raw s16le, 24 kHz, mono
print(f"wrote openai_client.wav; pcm request returned {len(pcm) // 2 / 24000:.2f} s of audio")
