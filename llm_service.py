"""
terminal_agent/llm_service.py
LLM service using Groq LLaMA 3.3 70B (ultra-fast!) with OpenAI GPT-4o-mini as fallback
Optimized for Hindi, Hinglish, and English conversations
Performance: 0.3s (Groq) vs 1.5s (OpenAI) - 5x faster!
"""
import os
from typing import List, Dict, Optional
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

# Model configuration
GROQ_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def build_prompt(system_prompt: str, user_messages: List[Dict], relevant_memories: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Build chat prompt with system instructions, document context, memory context, and user messages.
    
    Args:
        system_prompt: System instruction for the LLM
        user_messages: List of {"role": "user/assistant", "content": "..."}
        relevant_memories: Optional memory entries to inject (includes documents + user memories)
        
    Returns:
        Complete message list for chat API
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Separate document context from user memories for better prompt structure
    if relevant_memories:
        documents = [m for m in relevant_memories if m.get("type") == "document"]
        memories = [m for m in relevant_memories if m.get("type") == "memory"]
        
        # Inject document context first (property information)
        if documents:
            doc_text = "\n".join([
                f"📄 [{m.get('collection', 'doc')}] {m.get('text', '')}" 
                for m in documents if m.get("text")
            ])
            if doc_text.strip():
                messages.append({
                    "role": "system", 
                    "content": f"📚 Property Information (from documents):\n{doc_text}\n\nUse this information to answer user queries about pricing, specifications, or FAQs."
                })
        
        # Inject user memory context (conversation history + entities)
        if memories:
            mem_text = "\n".join([f"- {m.get('text','')}" for m in memories if m.get("text")])
            if mem_text.strip():
                messages.append({
                    "role": "system", 
                    "content": f"💭 User Memory (conversation history):\n{mem_text}"
                })
    
    messages.extend(user_messages)
    return messages


def chat_with_llm(
    system_prompt: str, 
    messages: List[Dict], 
    relevant_memories: Optional[List[Dict]] = None, 
    temperature: float = 0.6, 
    max_tokens: int = 400,
    use_groq: bool = True
) -> str:
    """
    Generate chat response using Groq LLaMA (fast) or OpenAI GPT (fallback).
    
    Args:
        system_prompt: System instruction for the LLM
        messages: List of conversation messages
        relevant_memories: Optional memory context
        temperature: Randomness (0.0-1.0, higher = more creative)
        max_tokens: Maximum response length
        use_groq: Try Groq first for ultra-fast inference (default: True)
        
    Returns:
        Generated response text
        
    Performance:
        - Groq LLaMA 3.3 70B: ~0.3s (FREE - 14,400 req/day)
        - OpenAI GPT-4o-mini: ~1.5s ($0.15/1M tokens)
    """
    prompt_messages = build_prompt(system_prompt, messages, relevant_memories=relevant_memories)
    
    # PRIMARY: Try Groq first (5x faster!)
    if use_groq and GROQ_AVAILABLE:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                client = Groq(api_key=groq_key)
                
                # Groq LLaMA 3.3 70B - optimized for speed and quality
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=prompt_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9,  # Nucleus sampling for better quality
                    stream=False  # Full response (can enable streaming later)
                )
                
                result = resp.choices[0].message.content.strip()
                print(f"✅ Groq LLM: '{result[:50]}...' ({len(result)} chars in ~0.3s)")
                return result
                
            except Exception as e:
                print(f"⚠️ Groq LLM failed ({e}), falling back to OpenAI...")
    
    # FALLBACK: OpenAI GPT (slower but reliable)
    if OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=prompt_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                result = resp.choices[0].message.content.strip()
                print(f"✅ OpenAI LLM: '{result[:50]}...' ({len(result)} chars in ~1.5s)")
                return result
                
            except Exception as e:
                print(f"❌ OpenAI LLM failed: {e}")
    
    # Last resort fallback
    return "⚠️ LLM service unavailable. Please check API keys (GROQ_API_KEY or OPENAI_API_KEY)."
