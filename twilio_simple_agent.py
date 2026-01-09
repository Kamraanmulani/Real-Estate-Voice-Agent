#!/usr/bin/env python3
"""
Realestate Voice Agent — Twilio Phone Call Agent (Groq + Memory)

This file merges:
  - The conversational logic, memory usage, and relationship-building style
    from your previous terminal-based agent.py
  - The Twilio call handling from twilio_simple_agent.py

FEATURES:
  - Inbound / outbound phone calls via Twilio (using /voice webhook)
  - Uses Groq-backed LLM via chat_with_llm()
  - Uses MemoryStore to remember past conversations and reuse details
  - Hindi / Hinglish / English responses (as per your system prompt)
  - Short, voice-friendly replies for phone calls

REQUIREMENTS:
  - pip install flask twilio python-dotenv
  - GROQ_API_KEY in .env
  - TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN in .env (for calls)
  - MemoryStore implementation in memory.py (optional; will fallback to JSONL)

RUN:
  1. python agent.py
  2. ngrok http 5000
  3. Set Twilio Voice webhook to: https://YOUR_NGROK_URL/voice
  4. Use make_call_simple.py to place outbound calls
"""

import os
import uuid
import time
import json
import traceback
import threading
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv
from flask import Flask, request, Response, send_from_directory
from twilio.twiml.voice_response import VoiceResponse, Gather

from llm_service import chat_with_llm
from call_memories import call_memory_manager  # Qdrant + Neo4j unified memory manager

# Optional MemoryStore import; fallback to simple JSONL store
try:
    from memory import MemoryStore as ExternalMemoryStore
except Exception:
    ExternalMemoryStore = None

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class JsonlMemoryStore:
    """Minimal JSONL memory store (id, text, reply, ts, session_id)."""

    def __init__(self, file_path: str = str(MEMORY_DIR / "real_estate_voice_agent_twilio.jsonl")):
        self.file_path = file_path
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


# Choose MemoryStore implementation
MemoryStore = ExternalMemoryStore or JsonlMemoryStore

app = Flask(__name__)
memory = MemoryStore()

# Per-call session storage (in-memory)
sessions: Dict[str, Dict] = {}

# ---- Configuration (optimized for Twilio 15s timeout) ----
RESPONSE_MAX_TOKENS = 65          # Reduced for speed
TIME_WARN_THRESHOLD = 10.0        # Warn if > 10s

def _now() -> float:
    return time.time()

def _log_duration(label: str, start: float):
    elapsed = time.time() - start
    print(f"⏱ {label} took {elapsed:.2f}s")


def greet_user() -> str:
    """Return a contextual greeting based on local time (same logic as old agent.py)."""
    from datetime import datetime

    now = datetime.now()
    hour = now.hour

    if hour < 11:
        return "Namaste ji! Chai pee li? Bataiye, aaj kis baare mein jaankari chaiye apko"
    if 12 <= hour < 16:
        return "Namaste ji! Khana ho gaya apka? Bataiye, aaj kis baare mein jaankari chaiye apko"
    if 16 <= hour < 18:
        return "Namaste ji! Chai pee li? Bataiye, aaj kis baare mein jaankari chaiye apko"
    return "Namaste! Main Real estate Voice Agent hoon. Aaj main aapki kya madad kar sakti hoon?"


def generate_elevenlabs_audio(text: str, voice_id: str, api_key: str) -> bytes:
    """Generate audio from text using Eleven Labs API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2"
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Eleven Labs error: {response.status_code} - {response.text}")
        return None


# Main system prompt from your agent.py (slightly adapted for phone calls)
SYSTEM_PROMPT = """
You are Realestate Voice Agent — a warm, professional real estate assistant for Real Estate Agency.

GOAL:
- Build a friendly bond, not just answer questions.
- Remember and reuse personal details naturally (name, preferences, visits, plot IDs).
- Keep responses concise and voice-friendly (under 40 words).

TONE & LANGUAGE:
- Speak naturally like a human. Prefer Hindi or Hinglish for Indian users; switch to English if the user speaks English.
- Mirror the user’s language and formality.
- Use short, clear sentences suitable for phone TTS. No emojis.
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
- Remember and reference past conversations: "Kal aapne bola tha weekend visit karne wale the, kaisa raha?"
- Ask caring questions: "Aaj chai pi? Family sab theek?"
- Show genuine interest in their life, preferences, and concerns.
- Celebrate small moments: "Wah! Bahut accha laga sunke!"
- Use their name if you know it, build familiarity over time.

CONVERSATION STYLE:
- Keep responses natural and concise (under 40 words for voice).
- Use memory context to make conversations feel continuous.
- Be empathetic, warm, and emotionally intelligent.
- Don't just answer, engage: "Aur bataiye, aapka din kaisa guzar raha hai?"
- Balance professionalism with friendliness.

OUTPUT RULES:
- <= 40 words, short sentences, natural pauses via commas.
- No markdown formatting. No bullet points unless requested.
- Keep the text in Hindi, Hinglish or English only.
- If user talks in Hindi respond in Hindi and if English then respond in English.

Remember: You're not just an assistant, you're building lasting relationships.
Every interaction should feel personal, warm, and memorable.
"""


@app.route("/")
def index():
    """Health check / basic info."""
    return {
        "status": "online",
        "service": "Real estate Voice Agent Twilio Phone Agent",
        "groq_available": bool(os.getenv("GROQ_API_KEY")),
        "twilio_configured": bool(os.getenv("TWILIO_ACCOUNT_SID")),
        "memory_backend": type(memory).__name__,
    }


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    return send_from_directory(AUDIO_DIR, filename)


@app.route("/voice", methods=["POST"])
def voice():
    """
    Main Twilio webhook endpoint.
    Handles incoming (or outbound-connected) calls and asks first question.
    """
    response = VoiceResponse()
    call_sid = request.form.get("CallSid", "unknown")
    from_number = request.form.get("From", "unknown")

    print(f"\n📞 Incoming call: {call_sid} from {from_number}")

    # Initialize session if new call
    if call_sid not in sessions:
        sessions[call_sid] = {
            "session_id": f"call_{call_sid}",
            "from_number": from_number,
            "messages": [],
            "turn_count": 0,
        }

    # Contextual greeting
    greeting = greet_user()

    # Generate Eleven Labs audio for greeting
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    base_url = os.getenv("BASE_URL")
    audio_content = generate_elevenlabs_audio(greeting, voice_id, api_key)
    audio_url = None
    if audio_content:
        filename = f"{uuid.uuid4()}.mp3"
        filepath = AUDIO_DIR / filename
        with open(filepath, "wb") as f:
            f.write(audio_content)
        audio_url = f"{base_url}/audio/{filename}"
        print(f"🎵 Greeting audio generated: {audio_url}")
    else:
        print("⚠️ Greeting audio generation failed, falling back to Twilio TTS")

    # Use Gather to collect user speech
    gather = Gather(
        input="speech",
        action="/process_speech",
        method="POST",
        language="hi-IN",
        speech_timeout=3,
        timeout=10,
    )

    if audio_url:
        gather.play(audio_url)
    else:
        gather.say(
            greeting,
            voice="Polly.Aditi",
            language="hi-IN",
        )

    response.append(gather)

    # If no speech detected, fallback prompt
    response.say(
        "Kuch nahi suna. Phir se boliye.",
        voice="Polly.Aditi",
        language="hi-IN",
    )

    return Response(str(response), mimetype="text/xml")


@app.route("/process_speech", methods=["POST"])
def process_speech():
    """
    Process user speech recognized by Twilio and generate LLM response.
    """
    response = VoiceResponse()
    call_sid = request.form.get("CallSid", "unknown")
    user_speech = request.form.get("SpeechResult", "") or ""

    print(f"👤 User said: {user_speech}")

    if not user_speech.strip():
        # No speech detected: ask again
        response.say(
            "Kuch nahi suna, kripya thoda zor se boliye.",
            voice="Polly.Aditi",
            language="hi-IN",
        )
        response.redirect("/voice")
        return Response(str(response), mimetype="text/xml")

    # Get session (should already exist from /voice)
    session = sessions.get(call_sid)
    if session is None:
        # If somehow missing, recreate minimal session
        session = {
            "session_id": f"call_{call_sid}",
            "from_number": request.form.get("From", "unknown"),
            "messages": [],
            "turn_count": 0,
        }
        sessions[call_sid] = session

    session_id = session["session_id"]
    messages = session["messages"]

    # Check for exit keywords (same logic as terminal agent)
    lower = user_speech.lower()
    if any(word in lower for word in ["goodbye", "bye", "alvida", "exit", "end call"]):
        farewell = (
            "Thank you for calling Real estate! Have a great day. Goodbye!"
        )
        print(f"🤖 Assistant (farewell): {farewell}")
        
        # Generate Eleven Labs audio for farewell
        voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        api_key = os.getenv("ELEVENLABS_API_KEY")
        base_url = os.getenv("BASE_URL")
        audio_content = generate_elevenlabs_audio(farewell, voice_id, api_key)
        if audio_content:
            filename = f"{uuid.uuid4()}.mp3"
            filepath = AUDIO_DIR / filename
            with open(filepath, "wb") as f:
                f.write(audio_content)
            audio_url = f"{base_url}/audio/{filename}"
            response.play(audio_url)
        else:
            response.say(
                farewell,
                voice="Polly.Aditi",
                language="hi-IN",
            )
        
        response.hangup()

        # Clean up session
        if call_sid in sessions:
            del sessions[call_sid]
        return Response(str(response), mimetype="text/xml")

    # Add user message to session history
    messages.append({"role": "user", "content": user_speech})

    # Ultra-fast path: NO memory retrieval, direct LLM
    t_start = _now()
    agent_response = ""
    
    print("⚡ Fast path: skipping memory for speed")
    
    try:
        agent_response = chat_with_llm(
            system_prompt=SYSTEM_PROMPT,
            messages=messages,
            relevant_memories=[],  # Empty for maximum speed
            temperature=0.2,
            max_tokens=RESPONSE_MAX_TOKENS,
            use_groq=True,
        )
    except TypeError:
        try:
            agent_response = chat_with_llm(
                SYSTEM_PROMPT,
                messages,
                relevant_memories=[],
                temperature=0.2,
                max_tokens=RESPONSE_MAX_TOKENS,
            )
        except Exception as e:
            print(f"❌ LLM failed: {e}")
    except Exception as e:
        print(f"❌ LLM failed: {e}")
    
    if not agent_response:
        agent_response = "Namaste, main aapki kaise madad kar sakti hoon?"
    
    total_time = _now() - t_start
    _log_duration("Total response", t_start)
    
    if total_time > TIME_WARN_THRESHOLD:
        print(f"⚠️ Response {total_time:.2f}s > {TIME_WARN_THRESHOLD}s")

    print(f"🤖 Assistant: {agent_response}")

    messages.append({"role": "assistant", "content": agent_response})
    session["messages"] = messages[-10:]
    session["turn_count"] += 1

    # Generate Eleven Labs audio
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    base_url = os.getenv("BASE_URL")
    audio_content = generate_elevenlabs_audio(agent_response, voice_id, api_key)
    audio_url = None
    if audio_content:
        filename = f"{uuid.uuid4()}.mp3"
        filepath = AUDIO_DIR / filename
        with open(filepath, "wb") as f:
            f.write(audio_content)
        audio_url = f"{base_url}/audio/{filename}"
        print(f"🎵 Audio generated: {audio_url}")
    else:
        print("⚠️ Audio generation failed, falling back to Twilio TTS")

    # Async memory storage (happens AFTER response sent)
    def _bg_store():
        try:
            memory.upsert({
                "id": str(uuid.uuid4()),
                "text": user_speech,
                "reply": agent_response,
                "ts": time.time(),
                "session_id": session_id,
            })
            call_memory_manager.store_interaction_async(session_id, user_speech, agent_response)
            print("📝 Background storage complete")
        except Exception as e:
            print(f"⚠️ Background storage failed: {e}")
    
    threading.Thread(target=_bg_store, daemon=True).start()

    # Prepare next turn: speak reply + gather again
    gather = Gather(
        input="speech",
        action="/process_speech",
        method="POST",
        language="hi-IN",
        speech_timeout=3,  # Fixed timeout
        timeout=10,  # Total gather timeout
    )

    if audio_url:
        gather.play(audio_url)
    else:
        gather.say(
            agent_response,
            voice="Polly.Aditi",
            language="hi-IN",
        )

    response.append(gather)

    # If user says nothing after this, politely end
    response.say(
        "Dhanyavaad! Call band kar rahe hain.",
        voice="Polly.Aditi",
        language="hi-IN",
    )
    response.hangup()

    return Response(str(response), mimetype="text/xml")


@app.route("/status", methods=["POST"])
def status():
    """Call status callback from Twilio."""
    call_sid = request.form.get("CallSid", "unknown")
    call_status = request.form.get("CallStatus", "unknown")

    print(f"📊 Call {call_sid} status: {call_status}")

    # Clean up session when call ends
    if call_status in ["completed", "failed", "busy", "no-answer", "canceled"]:
        if call_sid in sessions:
            print(f"🗑️  Cleaning up session {call_sid}")
            del sessions[call_sid]

    return Response("OK", mimetype="text/plain")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Starting Real estate Voice Agent Twilio Phone Agent")
    print("=" * 70)

    required_vars = {
        "TWILIO_ACCOUNT_SID": os.getenv("TWILIO_ACCOUNT_SID"),
        "TWILIO_AUTH_TOKEN": os.getenv("TWILIO_AUTH_TOKEN"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    }

    missing_vars = [k for k, v in required_vars.items() if not v]

    if missing_vars:
        print("\n⚠️  WARNING: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nAdd these to your .env file!")

    print("\n✅ Configuration:")
    print(f"   Groq API:   {'✓' if os.getenv('GROQ_API_KEY') else '✗'}")
    print(f"   Twilio:     {'✓' if os.getenv('TWILIO_ACCOUNT_SID') else '✗'}")
    print(f"   Memory:     {'✓' if memory else '✗'}")

    print("\n📝 Next Steps:")
    print("   1. Run ngrok: ngrok http 5000")
    print("   2. Set Twilio Voice webhook to: https://YOUR_NGROK_URL/voice")
    print("   3. Use make_call_simple.py to start an outbound call to your number")

    print("\n🌐 Server starting on http://0.0.0.0:5000")
    print("=" * 70 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
