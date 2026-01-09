"""
terminal_agent/stt_service.py
Speech-to-Text using Groq Whisper Large V3 Turbo (ultra-fast!)
With OpenAI Whisper as fallback
Optimized for Hindi, Hinglish, and English
Performance: 0.2s (Groq) vs 1.5s (OpenAI) - 7.5x faster!
"""
import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed. Run: pip install groq")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()


def transcribe_file(path: str, language: str = None, prompt: str = None, use_groq: bool = True) -> str:
    """
    Transcribe audio file to text using Groq Whisper (ultra-fast) or OpenAI Whisper (fallback).
    Supports Hindi, Hinglish (Hindi-English mix), and English.

    Args:
        path: Path to audio file (WAV, MP3, M4A, WEBM, etc.)
        language: Optional ISO language code ('hi' for Hindi, 'en' for English).
                 If None, auto-detects language.
        prompt: Optional context prompt to improve accuracy for domain-specific terms.
        use_groq: Try Groq first for ultra-fast transcription (default: True)

    Returns:
        Transcribed text string (empty on failure)
    
    Performance:
        - Groq Whisper Large V3 Turbo: ~0.2s (FREE - 14,400 req/day)
        - OpenAI Whisper: ~1.5s ($0.006/min)
    """
    # Validate audio file
    audio_path = Path(path)
    if not audio_path.exists():
        print(f"⚠️ Audio file not found: {path}")
        return ""

    # PRIMARY: Try Groq first (7x faster!)
    if use_groq and GROQ_AVAILABLE:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                client = Groq(api_key=groq_key)
                
                with open(audio_path, "rb") as audio_file:
                    # Groq Whisper Large V3 Turbo - optimized for speed
                    transcript = client.audio.transcriptions.create(
                        file=(audio_path.name, audio_file.read()),
                        model="whisper-large-v3-turbo",
                        prompt=prompt or "Real Estate Agency, Hindi, English, Hinglish, apartment, plot, BHK, construction, booking, Namaste, chai, khana",
                        response_format="text",
                        language=language,  # 'hi' for Hindi, 'en' for English, None for auto-detect
                        temperature=0.0  # More deterministic for better accuracy
                    )
                    
                    # Extract text from response
                    result = transcript.text.strip() if hasattr(transcript, 'text') else str(transcript).strip()
                    
                    if result:
                        print(f"✅ Groq STT: '{result[:50]}...' ({len(result)} chars in ~0.2s)")
                        return result
                        
            except Exception as e:
                print(f"⚠️ Groq STT failed ({e}), falling back to OpenAI...")

    # FALLBACK: OpenAI Whisper (slower but reliable)
    if OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                
                with open(audio_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        prompt=prompt,
                        language=language,
                        response_format="text"
                    )
                    
                    result = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()
                    print(f"✅ OpenAI STT: '{result[:50]}...' ({len(result)} chars in ~1.5s)")
                    return result
                    
            except Exception as e:
                print(f"❌ OpenAI STT failed: {e}")

    return ""


def transcribe_hinglish(path: str) -> str:
    """
    Optimized transcription for Hinglish (Hindi-English code-mixed speech).
    
    Args:
        path: Path to audio file
        
    Returns:
        Transcribed text
    """
    # Prompt helps Whisper understand context for code-mixed language
    hinglish_prompt = (
        "This is a conversation mixing Hindi and English words. "
        "Common terms: haan, nahi, kya, computer, file, system, terminal."
    )
    return transcribe_file(path, language="hi", prompt=hinglish_prompt)


def transcribe_multilingual(path: str) -> str:
    """
    Auto-detect and transcribe Hindi, Hinglish, or English.
    
    Args:
        path: Path to audio file
        
    Returns:
        Transcribed text
    """
    # No language specified - Whisper auto-detects
    # Prompt provides context for common Hinglish terms
    multilingual_prompt = "Terminal commands, Hindi, English, or mixed language."
    return transcribe_file(path, prompt=multilingual_prompt)
