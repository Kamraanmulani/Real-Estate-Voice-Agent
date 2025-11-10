"""
terminal_agent/tts_service.py
Text-to-Speech using ElevenLabs (voiceId-only). OpenAI fallback is disabled by request.
Returns path to produced WAV file (for easy playback).
"""
import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import requests  # For REST fallback
except Exception:
    requests = None  # type: ignore

load_dotenv()

# Use temp_calls directory for all TTS outputs (consistent with webrtc_bridge)
OUT_DIR = Path(__file__).resolve().parent / "temp_calls"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _tts_openai(text: str, session_id: str) -> Optional[str]:
    """
    Convert text to speech using OpenAI gpt-4o-mini-tts model.

    Args:
        text: Text (Hindi/English/Hinglish)
        session_id: Session identifier for filename grouping

    Returns:
        Path to generated MP3 file, or None on failure
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None

    try:
        client = OpenAI(api_key=api_key)
        voice = os.getenv("OPENAI_TTS_VOICE", "alloy")

        # Non-streaming TTS generation
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="wav",  # use response_format for wider SDK compatibility
        )

        out_path = OUT_DIR / f"reply_{session_id}_{int(time.time())}.wav"
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return str(out_path)
    except Exception as e:
        print("OpenAI TTS error:", e)
        return None


def _tts_elevenlabs(text: str, session_id: str) -> Optional[str]:
    """Convert text to speech using ElevenLabs REST (voiceId-only).

    Env vars (required):
      - ELEVENLABS_API_KEY
      - ELEVENLABS_VOICE_ID
    Optional:
      - ELEVENLABS_MODEL (default: 'eleven_multilingual_v2')
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id:
        print("❌ ElevenLabs not configured. Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID.")
        return None

    model = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

    out_path = OUT_DIR / f"reply_{session_id}_{int(time.time())}.wav"

    # REST call (requires voice_id and requests)
    try:
        if requests is None or not voice_id:
            return None
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "accept": "audio/wav",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return str(out_path)
    except Exception as e:
        print("ElevenLabs REST TTS error:", e)
        return None


def tts_from_text(text: str, session_id: str = "local") -> Optional[str]:
    """TTS using ElevenLabs only. OpenAI fallback is disabled by request."""
    path = _tts_elevenlabs(text, session_id)
    if path:
        return path
    # OpenAI fallback intentionally disabled; uncomment to re-enable:
    # return _tts_openai(text, session_id)
    print("❌ TTS failed: ElevenLabs did not return audio. Ensure ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are set and valid.")
    return None
