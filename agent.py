#!/usr/bin/env python3
"""
terminal_agent/agent.py
Main CLI loop: greet -> record -> STT -> LLM -> TTS -> memory store
"""
import os
import uuid
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Try to use provided utils; fallback to local implementations if missing
try:
    from utils.audio_utils import (
        record_audio as _record_audio,
        play_audio as _play_audio,
    )
except Exception:
    _record_audio = None
    _play_audio = None

from stt_service import transcribe_multilingual
from llm_service import chat_with_llm
from tts_service import tts_from_text

# Optional MemoryStore import; fallback to simple JSONL store
try:
    from memory import MemoryStore as _MemoryStore
except Exception:
    _MemoryStore = None


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR.parent / "demos"
DEMO_DIR.mkdir(parents=True, exist_ok=True)


def print_div():
    print("=" * 60)


def _fallback_record_audio(
    duration: int = 4, out_path: str = str(DEMO_DIR / "record.wav")
) -> str:
    """Low-latency microphone capture with simple VAD early-stop.

    Keeps audio settings the same but stops as soon as the user finishes speaking
    (short silence), rather than waiting the full window. This dramatically
    reduces round-trip time without changing fs/channels/record_duration values.
    """
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import threading
    import time as _time

    # Do not change these settings (per user request)
    fs = 16000
    channels = 1
    record_duration = 10  # upper bound only; we stop earlier on detected silence

    # VAD parameters (tuned for conversational speech at 16 kHz)
    frame_ms = 50  # analyze in ~50 ms chunks
    frame_samples = int(fs * frame_ms / 1000)
    silence_after_speech_sec = 0.35  # end after ~350 ms of silence
    min_speech_sec = 0.25  # need at least this much speech before we allow stop
    energy_threshold = 0.008  # simple RMS threshold; adjust if too sensitive

    print("🎤 Listening… speak now (auto-stops on pause)")

    collected: list[np.ndarray] = []
    speaking = False
    speech_start_time = None
    last_voice_time = None
    stop_event = threading.Event()

    def _rms(x: np.ndarray) -> float:
        x = x.astype(np.float32)
        return float(np.sqrt(np.mean(np.square(x))))

    def callback(indata, frames, time_info, status):  # sd.InputStream callback
        nonlocal speaking, speech_start_time, last_voice_time
        if status:
            # Drop status logs to keep loop tight
            pass
        # Copy to avoid referencing the same buffer
        chunk = np.copy(indata)
        collected.append(chunk)

        # Compute RMS over this chunk
        e = _rms(chunk)
        now = _time.time()

        if e >= energy_threshold:
            if not speaking:
                speech_start_time = now
                speaking = True
            last_voice_time = now
        else:
            # If we've started speaking and we have enough speech, allow early stop
            if speaking and last_voice_time is not None and speech_start_time is not None:
                if (now - last_voice_time) >= silence_after_speech_sec and (now - speech_start_time) >= min_speech_sec:
                    stop_event.set()

    # Run a bounded stream; we'll also enforce a hard cap of record_duration seconds
    start = _time.time()
    try:
        with sd.InputStream(
            samplerate=fs,
            channels=channels,
            dtype="float32",
            blocksize=frame_samples,
            callback=callback,
        ):
            while not stop_event.is_set():
                _time.sleep(frame_ms / 1000.0)
                if (_time.time() - start) >= record_duration:
                    break
    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted")
    except Exception as e:
        # Fallback to a short fixed recording if stream fails for any reason
        try:
            recording = sd.rec(int(min(duration, 2) * fs), samplerate=fs, channels=channels, dtype="float32")
            sd.wait()
            collected = [recording]
        except Exception:
            # Last resort: generate silence
            collected = [np.zeros((fs, channels), dtype=np.float32)]

    # Concatenate collected chunks
    if len(collected) == 0:
        audio = np.zeros((fs, channels), dtype=np.float32)
    else:
        audio = np.concatenate(collected, axis=0)

    # Save the recording
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, fs)

    effective_len = audio.shape[0] / float(fs)
    print(f"✅ Recording complete ({effective_len:.2f}s)")
    return out_path


def _fallback_play_audio(file_path: str) -> None:
    """Local audio playback using sounddevice/soundfile as a fallback."""
    import sounddevice as sd
    import soundfile as sf

    if not os.path.exists(file_path):
        print(f"⚠️  Audio file not found: {file_path}")
        return
    data, sr = sf.read(file_path)
    sd.play(data, sr)
    sd.wait()


# Choose active record/play functions
record_audio = _record_audio or _fallback_record_audio
play_audio = _play_audio or _fallback_play_audio


class SimpleMemoryStore:
    """Minimal JSONL memory store (id, text, reply, ts, session_id)."""

    def __init__(self, file_path: os.PathLike):
        self.file_path = str(file_path)
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.file_path):
            open(self.file_path, "w", encoding="utf-8").close()

    def upsert(self, item: Dict) -> None:
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            print("❌ Memory write error:", e)

    def retrieve_relevant(self, _query: str, k: int = 3) -> List[Dict]:
        try:
            rows: List[Dict] = []
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows[-k:] if len(rows) > k else rows
        except Exception:
            return []


# Pick MemoryStore implementation
MemoryStore = _MemoryStore or SimpleMemoryStore


def greet_user() -> str:
    """Return a contextual greeting based on local time.

    Rules:
      - Before 11:00 -> morning tea check
      - 12:00 to <16:00 -> lunch check
      - 16:00 to <18:00 -> evening tea check
      - Otherwise fallback to a generic greeting
    """
    from datetime import datetime

    now = datetime.now()
    hour = now.hour

    if hour < 11:
        return "Namaste ji! Chai pee li? Bataiye, aaj kis baare mein jaankari chaiye apko"
    if 12 <= hour < 16:
        return "Namaste ji! Khana ho gaya apka ? Bataiye, aaj kis baare mein jaankari chaiye apko"
    if 16 <= hour < 18:
        return "Namaste ji! Chai pee li? Bataiye, aaj kis baare mein jaankari chaiye apko"
    # Fallback (evening/night or 11:00-11:59 which wasn't explicitly defined)
    return "Namaste! Main Miss Riverwood hoon. Aaj main aapki kya madad kar sakti hoon?"


def main_loop():
    print("📞 Miss Riverwood — Voice Call Agent")
    print("═" * 60)
    mem = MemoryStore()
    system_prompt = """
You are Miss Riverwood — a warm, professional construction assistant for Riverwood Estate.

GOAL:
- Build a friendly bond, not just answer questions.
- Remember and reuse personal details naturally (name, preferences, visits, plot IDs).
- Keep responses concise and voice-friendly (< 120 words).

TONE & LANGUAGE:
- Speak naturally like a human. Prefer Hindi or Hinglish for Indian users; switch to English if the user speaks English.
- Mirror the user’s language and formality.
- Use short, clear sentences suitable for TTS. No emojis.
- Natural examples you can use when appropriate:
  - "Namaste Sir, aaj aapne chai pi?"
  - "Kal bola tha weekend visit karne wale the — kaisa raha experience?"
  - "Aapne last time 2BHK clubhouse ke paas pasand kiya tha, sahi yaad hai na?"

MEMORY & CONTEXT:
- If relevant memories are provided, reference them smoothly:
  - "Kal aapne P-23 ke foundation ke baare mein pucha tha..."
  - "Aapki preference 2BHK near clubhouse thi — main wahi options dikhati hoon."
- Never invent facts. If unsure, say you’ll check.

CONSTRUCTION FACTS:
- If plot status is requested: clearly state stage and progress if known.
- Avoid committing exact dates/prices without confirmation. Offer to verify.

RELATIONSHIP BUILDING:
- Remember and reference past conversations: 'Kal aapne bola tha weekend visit karne wale the - kaisa raha?
- Ask personal, caring questions: 'Aaj chai pi? Family sab theek?
- Show genuine interest in their life, preferences, and concerns
- Celebrate small moments: 'Wah! Bahut accha laga sunke!
- Use their name if you know it, build familiarity over time

 CONVERSATION STYLE:
- Keep responses natural and concise (under 100 words for voice)
- Use memory context to make conversations feel continuous
- Be empathetic, warm, and emotionally intelligent
- Don't just answer - engage: 'Aur bataiye, aapka din kaisa guzar raha hai?
- Balance professionalism with friendliness - you're helpful but also human

OUTPUT RULES:
- <= 120 words, short sentences, natural pauses via commas.
- No markdown formatting. No bullet points unless requested.
- Keep it personable, empathetic, and helpful.
- Keep the text of the in Hindi , Hinglish or English.
- Do not chnage in any other language.
- If user talks in Hindi respond in Hindi and if English then respond in English

Remember: You're not just an assistant, you're building lasting relationships. 
Every interaction should feel personal, warm, and memorable.
"""

    # Greet at startup (print + TTS)
    greeting = greet_user()
    print("\n🤖 Assistant:", greeting)
    try:
        audio_greet = tts_from_text(greeting, session_id="local")
        if audio_greet:
            play_audio(audio_greet)
    except Exception:
        pass
    print("═" * 60)

    turn = 0
    try:
        while True:
            turn += 1
            print(f"\n[Turn {turn}] Your turn to speak...")

            # Auto-record with VAD (no Enter key needed)
            wav_path = record_audio(duration=4, out_path=str(DEMO_DIR / "record.wav"))
            print("→ Processing speech...")
            # Use multilingual STT with helpful prompt for faster/robust results
            transcript = transcribe_multilingual(wav_path) or ""

            if (
                not transcript.strip()
                or transcript == "[local-fallback] (simulated transcript)"
            ):
                print("(Silent or no speech detected)")
                time.sleep(0.5)
                continue

            print(f"👤 You: {transcript}")

            # Check for exit keywords
            if any(
                word in transcript.lower()
                for word in ["goodbye", "bye", "alvida", "exit", "end call"]
            ):
                farewell = (
                    "Thank you for calling Riverwood Estate! Have a great day. Goodbye!"
                )
                print(f"🤖 Assistant: {farewell}")
                audio_out = tts_from_text(farewell, session_id="local")
                if audio_out:
                    play_audio(audio_out)
                print("\n📞 Call ended.\n")
                break

            # Retrieve relevant memory
            relevant = mem.retrieve_relevant(transcript, k=3)
            if relevant and len(relevant) > 0:
                print(f"💭 Using {len(relevant)} past conversation(s) for context")

            # Generate response
            messages = [{"role": "user", "content": transcript}]
            # Lower max_tokens + temperature for faster LLM turnaround (shorter, focused replies)
            reply = chat_with_llm(
                system_prompt,
                messages,
                relevant_memories=relevant,
                temperature=0.3,
                max_tokens=160,
            )
            print(f"🤖 Assistant: {reply}")

            # TTS and playback
            audio_out = tts_from_text(reply, session_id="local")
            if audio_out:
                play_audio(audio_out)

            # Save to memory
            mem_item = {
                "id": str(uuid.uuid4()),
                "text": transcript,
                "reply": reply,
                "ts": time.time(),
                "session_id": "local",
            }
            mem.upsert(mem_item)

    except KeyboardInterrupt:
        print("\n\n📞 Call interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main_loop()
