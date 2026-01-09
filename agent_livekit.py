#!/usr/bin/env python3
"""
agent_livekit.py
Real-time voice agent using LiveKit WebRTC for ultra-fast Hindi/English conversations
Integrates with existing STT, LLM, TTS, and Memory services
"""
import os
import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# LiveKit imports
try:
    from livekit import rtc, api
    from livekit.agents import (
        AutoSubscribe,
        JobContext,
        WorkerOptions,
        cli,
        llm,
    )
    from livekit.agents.pipeline import VoicePipelineAgent
    from livekit.plugins import openai, silero
except ImportError:
    print("❌ LiveKit not installed. Run: pip install livekit livekit-agents livekit-plugins-openai livekit-plugins-silero")
    exit(1)

# Your existing services
from stt_service import transcribe_multilingual, transcribe_file
from llm_service import chat_with_llm
from tts_service import tts_from_text
from memory import MemoryStore

load_dotenv()

# Configuration
TEMP_DIR = Path(__file__).resolve().parent / "temp_calls"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def greet_user() -> str:
    """Return a contextual greeting based on local time."""
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


SYSTEM_PROMPT = """You are Real estate Voice Agent — a warm, professional real estate assistant for Real Estate Agency.

GOAL:
- Build a friendly bond, not just answer questions.
- Remember and reuse personal details naturally (name, preferences, visits, plot IDs).
- Keep responses concise and voice-friendly (< 80 words for fast TTS).

TONE & LANGUAGE:
- Speak naturally like a human. Prefer Hindi or Hinglish for Indian users; switch to English if the user speaks English.
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


class LiveKitVoiceAgent:
    """Real-time voice agent using LiveKit with existing services."""
    
    def __init__(self, room: rtc.Room, participant: rtc.Participant):
        self.room = room
        self.participant = participant
        self.session_id = f"livekit_{int(time.time()*1000)}"
        
        # Initialize memory
        os.environ["AGENT_SESSION_ID"] = self.session_id
        self.memory = MemoryStore()
        
        # Conversation state
        self.conversation_history = []
        self.turn = 0
        
        print(f"✅ LiveKit agent initialized: {self.session_id}")
    
    async def greet(self):
        """Send greeting to user."""
        greeting = greet_user()
        
        # Store in memory
        self.memory.upsert({
            "id": str(uuid.uuid4()),
            "text": "",
            "reply": greeting,
            "ts": time.time(),
            "session_id": self.session_id
        })
        
        # Generate TTS
        audio_path = await asyncio.to_thread(tts_from_text, greeting, self.session_id)
        
        if audio_path:
            # Send audio to room
            await self._send_audio(audio_path)
        
        return greeting
    
    async def process_transcript(self, transcript: str) -> Optional[str]:
        """Process user transcript and generate response."""
        try:
            self.turn += 1
            print(f"\n[Turn {self.turn}] Processing: {transcript}")
            
            if not transcript or not transcript.strip():
                return None
            
            # Check for exit keywords
            if any(word in transcript.lower() for word in ["goodbye", "bye", "alvida", "exit", "end call"]):
                farewell = "Thank you for calling! Have a great day. Alvida!"
                await self._generate_and_send_response(farewell)
                return None
            
            # Retrieve memory
            relevant = await asyncio.to_thread(
                self.memory.retrieve_relevant, transcript, 3
            )
            
            if relevant and len(relevant) > 0:
                print(f"💭 Using {len(relevant)} past conversation(s) for context")
            
            # Generate LLM response (async)
            messages = [{"role": "user", "content": transcript}]
            reply = await asyncio.to_thread(
                chat_with_llm,
                SYSTEM_PROMPT,
                messages,
                relevant_memories=relevant,
                temperature=0.3,
                max_tokens=120  # Shorter for faster response
            )
            
            print(f"🤖 Assistant: {reply}")
            
            # Store in memory (async)
            mem_item = {
                "id": str(uuid.uuid4()),
                "text": transcript,
                "reply": reply,
                "ts": time.time(),
                "session_id": self.session_id
            }
            await asyncio.to_thread(self.memory.upsert, mem_item)
            
            # Generate and send TTS
            await self._generate_and_send_response(reply)
            
            return reply
            
        except Exception as e:
            print(f"❌ Error processing transcript: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _generate_and_send_response(self, text: str):
        """Generate TTS and send to room."""
        # Generate TTS (async)
        audio_path = await asyncio.to_thread(tts_from_text, text, self.session_id)
        
        if audio_path:
            await self._send_audio(audio_path)
    
    async def _send_audio(self, audio_path: str):
        """Send audio file to LiveKit room."""
        try:
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Get audio track from room
            audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
            track = rtc.LocalAudioTrack.create_audio_track("agent_voice", audio_source)
            
            # Publish track
            options = rtc.TrackPublishOptions()
            publication = await self.room.local_participant.publish_track(track, options)
            
            # Stream audio data
            # TODO: Implement proper audio streaming with frame conversion
            
            print(f"📤 Sent audio to room (size: {len(audio_data)} bytes)")
            
        except Exception as e:
            print(f"❌ Error sending audio: {e}")


async def entrypoint(ctx: JobContext):
    """LiveKit agent entrypoint."""
    print(f"🚀 Agent starting for room: {ctx.room.name}")
    
    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Wait for participant
    participant = await ctx.wait_for_participant()
    print(f"👤 Participant joined: {participant.identity}")
    
    # Create agent
    agent = LiveKitVoiceAgent(ctx.room, participant)
    
    # Send greeting
    await agent.greet()
    
    # Use LiveKit's built-in pipeline with OpenAI for real-time streaming
    # This provides the fastest Hindi/English STT and TTS
    try:
        # Initialize OpenAI pipeline for ultra-fast streaming
        pipeline_agent = VoicePipelineAgent(
            vad=silero.VAD.load(),  # Voice Activity Detection
            stt=openai.STT(
                model="whisper-1",
                language="hi",  # Hindi/Hinglish support
            ),
            llm=openai.LLM(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            ),
            tts=openai.TTS(
                model="tts-1",
                voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
            ),
        )
        
        # Override LLM function to use our custom logic
        original_llm = pipeline_agent._llm_stream
        
        async def custom_llm_stream(transcript: str):
            """Custom LLM with memory integration."""
            response = await agent.process_transcript(transcript)
            if response:
                yield response
        
        pipeline_agent._llm_stream = custom_llm_stream
        
        # Start the pipeline
        pipeline_agent.start(ctx.room, participant)
        
        print("✅ Pipeline started - agent is now listening and responding in real-time!")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        # Fallback to manual processing
        print("⚠️ Using fallback manual processing...")
        
        # Manual audio processing (slower but more reliable)
        @ctx.room.on("track_subscribed")
        async def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            """Handle incoming audio tracks."""
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                print(f"🎤 Subscribed to audio track: {track.sid}")
                
                audio_stream = rtc.AudioStream(track)
                
                async for frame in audio_stream:
                    # Save frame to file
                    temp_file = TEMP_DIR / f"frame_{time.time()}.wav"
                    
                    # TODO: Convert frame to WAV and process with STT
                    # This is a simplified version - you'd need to properly convert frames
                    
                    # Process with STT
                    transcript = await asyncio.to_thread(
                        transcribe_multilingual, str(temp_file)
                    )
                    
                    if transcript:
                        await agent.process_transcript(transcript)


if __name__ == "__main__":
    """Run LiveKit agent."""
    print("\n" + "="*60)
    print("🎤 Real estate Voice Agent - LiveKit WebRTC Agent")
    print("="*60)
    print("\n✨ Features:")
    print("   • Ultra-fast real-time Hindi/English voice streaming")
    print("   • LiveKit WebRTC for low-latency audio")
    print("   • Existing STT → LLM → TTS → Memory pipeline")
    print("   • Voice Activity Detection for natural conversations")
    print("\n📋 Setup:")
    print("   1. Set LIVEKIT_URL and LIVEKIT_API_KEY in .env")
    print("   2. Run this agent")
    print("   3. Connect from web frontend")
    print("\n" + "="*60 + "\n")
    
    # Run agent with LiveKit CLI
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
