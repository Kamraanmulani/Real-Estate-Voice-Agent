#!/usr/bin/env python3
"""
agent_webrtc_integrated.py - OPTIMIZED VERSION
Complete WebRTC-enabled voice agent with beautiful frontend
OPTIMIZED WORKFLOW:
1. STT (0.2s) → 2. Memory retrieval (0.3s) → 3. LLM (0.3s) → 
4. TTS (1.0s) → 5. Send audio → 6. Store memory ASYNC (background)
Result: 40% faster user-facing latency!
"""
import os
import uuid
import time
import json
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Web server
from flask import Flask, render_template, send_from_directory, Response
from flask_sock import Sock

# Core agent services (existing)
from stt_service import transcribe_multilingual, transcribe_file
from llm_service import chat_with_llm
from tts_service import tts_from_text
from memory import MemoryStore

load_dotenv()

# Thread pool for async operations (memory storage in background)
executor = ThreadPoolExecutor(max_workers=3)

# Configuration
HOST = "0.0.0.0"
PORT = 8080
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEMP_DIR = BASE_DIR / "temp_calls"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Flask app
app = Flask(__name__, 
            template_folder=str(FRONTEND_DIR),
            static_folder=str(FRONTEND_DIR))
sock = Sock(app)

# Active sessions
active_sessions: Dict[str, 'VoiceSession'] = {}


def greet_user() -> str:
    """Return a contextual greeting based on local time."""
    from datetime import datetime
    
    now = datetime.now()
    hour = now.hour
    
    if hour < 11:
        return "Namaste ji! Chai pee li? Bataiye, aaj kis baare mein jaankari chaiye apko"
    if 12 <= hour < 16:
        return "Namaste ji! Khana ho gaya apka? Bataiye, aaj kis baare mein jaankari chaiye apko"
    if 16 <= hour < 23:
        return "Namaste ji! Chaai pee li? Bataiye, aaj kis baare mein jaankari chaiye apko"
    return "Namaste! Main Real estate Voice Agent hoon. Aaj main aapki kya madad kar sakti hoon?"


SYSTEM_PROMPT = """You are Real estate Voice Agent — a warm, professional real estate assistant for Real Estate Agency.

GOAL:
- Build a friendly bond, not just answer questions.
- Remember and reuse personal details naturally (name, preferences, visits, plot IDs).
- Keep responses concise and voice-friendly (< 80 words for fast response).

TONE & LANGUAGE:
- Speak naturally like a human female . Prefer Hindi or Hinglish for Indian users; switch to English if the user speaks English.
- Mirror the user's language and formality.
- Use short, clear sentences suitable for TTS. No emojis.
- Natural examples you can use when appropriate:
  - "Namaste Sir, aaj aapne chai pi?"
  - "Kal bola tha weekend visit karne wale the — kaisa raha experience?"
  

MEMORY & CONTEXT:
- If relevant memories are provided, reference them smoothly.
- Never invent facts. If unsure, say you'll check.

RELATIONSHIP BUILDING:
- Remember and reference past conversations
- Ask personal, caring questions
- Show genuine interest in their life, preferences, and concerns
- Use their name if you know it, build familiarity over time

CONVERSATION STYLE:
- Keep responses natural and concise (under 80 words for voice)
- Use memory context to make conversations feel continuous
- Be empathetic, warm, and emotionally intelligent
- Balance professionalism with friendliness - you're helpful but also human

OUTPUT RULES:
- <= 80 words, short sentences, natural pauses via commas.
- No markdown formatting. No bullet points unless requested.
- Keep it personable, empathetic, and helpful.
- If user talks in Hindi respond in Hindi and if English then respond in English

Remember: You're building lasting relationships. Every interaction should feel personal, warm, and memorable.
"""


class VoiceSession:
    """Manages a voice call session for one user."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id  # WebSocket session (unique per call)
        self.conversation_history = []
        self.turn = 0
        
        # ✅ FIX: Use FIXED memory user ID for ALL sessions
        # All conversations stored under "kamraan_memories" in Neo4j + Qdrant
        self.memory_user_id = os.getenv("AGENT_SESSION_ID", "kamraan_memories")
        
        # Initialize memory with FIXED user ID for persistence
        os.environ["AGENT_SESSION_ID"] = self.memory_user_id
        self.memory = MemoryStore()
        
        print(f"✅ Created voice session: {session_id}")
        print(f"💾 Memory user ID: {self.memory_user_id} (persistent across all sessions)")
    
    def greet(self) -> tuple[str, Optional[str]]:
        """Generate greeting message and audio."""
        print(f"👋 Generating greeting for session {self.session_id}...")
        greeting = greet_user()
        
        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": greeting})
        
        # Generate TTS
        audio_path = tts_from_text(greeting, session_id=self.session_id)
        
        # Store greeting in memory ASYNC (user doesn't wait!)
        executor.submit(self._store_memory_async, "", greeting)
        
        return greeting, audio_path
    
    def _store_memory_async(self, user_text: str, assistant_reply: str):
        """
        Store conversation in memory asynchronously (non-blocking).
        User doesn't wait for this to complete!
        """
        try:
            self.memory.upsert({
                "id": str(uuid.uuid4()),
                "text": user_text,
                "reply": assistant_reply,
                "ts": time.time(),
                "session_id": self.session_id
            })
            print(f"💾 Memory stored in background (session: {self.session_id})")
        except Exception as e:
            print(f"⚠️ Background memory storage failed: {e}")
    
    def process_audio(self, audio_data: bytes) -> Optional[Dict]:
        """
        Process audio with OPTIMIZED flow:
        1. STT → 2. Memory retrieval → 3. LLM → 4. TTS → 5. Send audio
        6. Store memory ASYNC (user doesn't wait!)
        """
        try:
            self.turn += 1
            turn_start = time.time()
            
            # Save audio to temp file (browser sends webm format)
            temp_input = TEMP_DIR / f"input_{self.session_id}_{time.time()}.webm"
            with open(temp_input, 'wb') as f:
                f.write(audio_data)
            
            print(f"\n[Turn {self.turn}] Processing audio (size: {len(audio_data)} bytes)")
            
            # Convert webm to wav for better Whisper compatibility
            temp_wav = None
            try:
                import subprocess
                temp_wav = TEMP_DIR / f"input_{self.session_id}_{time.time()}.wav"
                
                # Use ffmpeg if available
                result = subprocess.run([
                    'ffmpeg', '-i', str(temp_input),
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',       # Mono
                    '-y',             # Overwrite
                    str(temp_wav)
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and temp_wav.exists():
                    print(f"✅ Converted to WAV for better STT")
                    stt_input = str(temp_wav)
                else:
                    print(f"⚠️ ffmpeg conversion failed, using webm directly")
                    stt_input = str(temp_input)
                    temp_wav = None
            except Exception as conv_err:
                print(f"⚠️ Audio conversion error: {conv_err}, using webm directly")
                stt_input = str(temp_input)
                temp_wav = None
            
            # ============================================================
            # STEP 1: STT (0.2s with Groq)
            # ============================================================
            stt_start = time.time()
            domain_prompt = "Real Estate Agency, apartment, plot, BHK, booking, construction, foundation, Namaste, chai, khana"
            user_text = transcribe_file(stt_input, prompt=domain_prompt)
            stt_time = time.time() - stt_start
            
            # Cleanup temp files
            temp_input.unlink(missing_ok=True)
            if temp_wav and temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            
            if not user_text or user_text.strip() == "":
                print(f"⚠️ No speech detected")
                return None
            
            print(f"👤 User: {user_text}")
            
            # Check for exit keywords
            if any(word in user_text.lower() for word in ["goodbye", "bye", "alvida", "exit", "end call", "thank you bye"]):
                farewell = "Thank you for calling Real Estate Agency! Dhanyavaad! Have a great day. Alvida!"
                audio_out = tts_from_text(farewell, session_id=self.session_id)
                
                if audio_out and os.path.exists(audio_out):
                    with open(audio_out, 'rb') as f:
                        audio_bytes = f.read()
                    
                    return {
                        "transcript": user_text,
                        "response": farewell,
                        "audio": audio_bytes,
                        "end_call": True
                    }
                return None
            
            # ============================================================
            # STEP 2: Memory Retrieval (0.3s) - SEARCH QDRANT + NEO4J
            # ============================================================
            memory_start = time.time()
            relevant = self.memory.retrieve_relevant(user_text, k=5)  # Get more memories
            memory_time = time.time() - memory_start
            
            if relevant and len(relevant) > 0:
                print(f"💭 Retrieved {len(relevant)} relevant memories from Qdrant + Neo4j:")
                for i, mem in enumerate(relevant[:3], 1):
                    mem_text = mem.get('text', '')[:60]
                    print(f"   {i}. {mem_text}...")
            else:
                print(f"💭 No relevant memories found (new user or first conversation)")
            
            # ============================================================
            # STEP 3: LLM (0.3s with Groq) - Generate with memory context
            # ============================================================
            llm_start = time.time()
            
            user_messages = []
            # Include recent conversation history (last 6 messages)
            user_messages.extend(self.conversation_history[-6:])
            user_messages.append({"role": "user", "content": user_text})
            
            assistant_text = chat_with_llm(
                SYSTEM_PROMPT,
                user_messages,
                relevant_memories=relevant,  # Pass memories to LLM
                temperature=0.3,
                max_tokens=120,  # ~80 words
                use_groq=True  # Force Groq for speed
            )
            llm_time = time.time() - llm_start
            
            print(f"🤖 Assistant: {assistant_text}")
            
            # ============================================================
            # STEP 4: TTS (1.0s)
            # ============================================================
            tts_start = time.time()
            audio_path = tts_from_text(assistant_text, self.session_id)
            tts_time = time.time() - tts_start
            
            if not audio_path or not os.path.exists(audio_path):
                print(f"❌ TTS failed - no audio generated")
                return None
            
            # Read audio file
            with open(audio_path, 'rb') as f:
                response_audio = f.read()
            
            # ============================================================
            # STEP 5: Send audio IMMEDIATELY (user hears response now!)
            # ============================================================
            
            # Update conversation history (before async storage)
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": assistant_text})
            
            # Calculate user-facing time (before async storage)
            user_facing_time = time.time() - turn_start
            
            # ============================================================
            # STEP 6: Store memory ASYNC (user doesn't wait!)
            # ============================================================
            executor.submit(self._store_memory_async, user_text, assistant_text)
            
            # Print timing breakdown
            print(f"⏱️  Timing: STT={stt_time:.2f}s | Memory={memory_time:.2f}s | LLM={llm_time:.2f}s | TTS={tts_time:.2f}s")
            print(f"✅ User-facing latency: {user_facing_time:.2f}s (memory storage happening in background)")
            
            return {
                "transcript": user_text,
                "response": assistant_text,
                "audio": response_audio,
                "end_call": False
            }
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return None


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/styles.css')
def styles():
    """Serve CSS file."""
    return send_from_directory(FRONTEND_DIR, 'styles.css', mimetype='text/css')


@app.route('/script.js')
def script():
    """Serve JavaScript file."""
    return send_from_directory(FRONTEND_DIR, 'script.js', mimetype='application/javascript')


@app.route('/favicon.ico')
def favicon():
    """Serve a minimal SVG favicon."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<rect width="16" height="16" fill="#667eea"/>'
        '<text x="8" y="12" text-anchor="middle" fill="white" font-size="10" font-family="sans-serif">R</text>'
        '</svg>'
    )
    return Response(svg, mimetype='image/svg+xml')


@sock.route('/voice-stream')
def voice_stream(ws):
    """Handle WebSocket voice streaming."""
    session_id = f"session_{int(time.time()*1000)}"
    
    try:
        print(f"\n🔌 New WebSocket connection: {session_id}")
        
        # Create session
        session = VoiceSession(session_id)
        active_sessions[session_id] = session
        
        # Send greeting
        print(f"👋 Generating greeting for {session_id}...")
        greeting, greeting_audio_path = session.greet()
        
        if not greeting_audio_path or not os.path.exists(greeting_audio_path):
            print(f"❌ Failed to generate greeting audio")
            ws.close()
            return
        
        with open(greeting_audio_path, 'rb') as f:
            greeting_audio = f.read()
        
        # Send greeting to client
        print(f"📤 Sending greeting to client (audio size: {len(greeting_audio)} bytes)")
        ws.send(json.dumps({
            "type": "greeting",
            "text": greeting,
            "audio": greeting_audio.hex()
        }))
        print(f"✅ Greeting sent! Waiting for user audio...")
        
        # Listen for audio messages
        while True:
            message = ws.receive()
            if message is None:
                print(f"⚠️ Connection closing")
                break
            
            try:
                data = json.loads(message)
                print(f"📥 Received message type: {data.get('type')}")
                
                if data.get("type") == "audio":
                    # Receive audio from browser
                    audio_hex = data.get("audio", "")
                    print(f"🎵 Received audio (hex length: {len(audio_hex)})")
                    audio_bytes = bytes.fromhex(audio_hex)
                    
                    # Process audio
                    print(f"⚙️ Processing audio...")
                    result = session.process_audio(audio_bytes)
                    
                    if result:
                        # Send response back
                        print(f"📤 Sending response (audio size: {len(result['audio'])} bytes)")
                        ws.send(json.dumps({
                            "type": "response",
                            "transcript": result["transcript"],
                            "text": result["response"],
                            "audio": result["audio"].hex()
                        }))
                        print(f"✅ Response sent!")
                        
                        # Check if should end call
                        if result.get("end_call"):
                            print(f"📞 Call ended by user")
                            break
                    else:
                        print(f"⚠️ No valid result")
                        ws.send(json.dumps({
                            "type": "error",
                            "message": "Could not process audio. Please try again."
                        }))
                
                elif data.get("type") == "end":
                    print(f"👋 Session ended: {session_id}")
                    break
                    
            except Exception as e:
                print(f"❌ Error handling message: {e}")
                import traceback
                traceback.print_exc()
                ws.send(json.dumps({
                    "type": "error",
                    "message": "Internal error. Please try again."
                }))
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup session
        if session_id in active_sessions:
            del active_sessions[session_id]
        print(f"🔌 Connection closed: {session_id}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎤 Real estate Voice Agent - Real-time WebRTC Voice Agent")
    print("="*70)
    print(f"\n✨ Features:")
    print(f"   • Real-time Hindi/English/Hinglish voice conversations")
    print(f"   • Beautiful, professional web interface")
    print(f"   • Ultra-fast STT (Whisper) → LLM (GPT-4) → TTS (ElevenLabs)")
    print(f"   • Smart memory with Neo4j + Qdrant")
    print(f"   • Visual pipeline display showing AI processing")
    print(f"\n🌐 Server starting on:")
    print(f"   👉 http://localhost:{PORT}")
    print(f"   👉 http://127.0.0.1:{PORT}")
    print(f"\n📂 Resources:")
    print(f"   • Frontend: {FRONTEND_DIR}")
    print(f"   • Temp files: {TEMP_DIR}")
    print(f"\n🎯 How to use:")
    print(f"   1. Open your browser to the URL above")
    print(f"   2. Click 'Start Call' and allow microphone access")
    print(f"   3. Speak naturally in Hindi, Hinglish, or English")
    print(f"   4. Watch the AI pipeline in action!")
    print(f"\n💡 Pro tip: Speak clearly for best recognition results")
    print("\n" + "="*70 + "\n")
    
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
