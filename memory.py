"""Minimal Qdrant + Neo4j memory for the terminal agent using mem0.

- Uses mem0 Memory client for automatic entity extraction and graph updates
- Per-user memory via AGENT_SESSION_ID env var
- Stores conversation turns with automatic relationship building in Neo4j
- Document retrieval from property collections (pricing, specifications, FAQ)
"""

import os, json, time, uuid, re
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Optional mem0 for advanced memory with graph support
try:
    from mem0 import Memory
except Exception:
    Memory = None  # type: ignore

# Fallback to basic OpenAI if mem0 unavailable
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Document retrieval from Qdrant collections
try:
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client import QdrantClient
    DOCUMENT_RETRIEVAL_AVAILABLE = True
except Exception:
    DOCUMENT_RETRIEVAL_AVAILABLE = False

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536


def _uid():
    try:
        return str(uuid.uuid4())
    except Exception:
        return str(int(time.time()*1000))


class _JsonlFallback:
    def __init__(self, path: os.PathLike):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "w", encoding="utf-8").close()

    def upsert(self, item: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def retrieve_relevant(self, _q: str, k: int = 3) -> List[Dict]:
        rows: List[Dict] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            return []
        return rows[-k:] if len(rows) > k else rows



class MemoryStore:
    """Memory store using mem0 for automatic entity extraction and Neo4j graph updates + document retrieval."""

    def __init__(self, file_path: Optional[os.PathLike] = None):
        self.session = os.getenv("AGENT_SESSION_ID", "local") or "local"
        self._jsonl = _JsonlFallback(file_path) if file_path else None
        self._mem_client = None
        self._doc_client = None
        self._doc_embeddings = None
        
        # Initialize mem0 Memory client with Qdrant + Neo4j
        if Memory is not None:
            try:
                config = {
                    "version": "v1.1",
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "api_key": os.getenv("OPENAI_API_KEY"),
                            "model": EMBED_MODEL
                        },
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "api_key": os.getenv("OPENAI_API_KEY"),
                            "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                            "temperature": 0.7
                        }
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "host": os.getenv("QDRANT_HOST", "localhost"),
                            "port": int(os.getenv("QDRANT_PORT", "6333")),
                        },
                    },
                    "graph_store": {
                        "provider": "neo4j",
                        "config": {
                            "url": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                            "username": os.getenv("NEO4J_USER", "neo4j"),
                            "password": os.getenv("NEO4J_PASSWORD", "changeme")
                        },
                    },
                }
                self._mem_client = Memory.from_config(config)
                print("✅ mem0 Memory client initialized with Qdrant + Neo4j graph store")
            except Exception as e:
                print(f"⚠️ mem0 initialization failed: {e}")
                self._mem_client = None
        
        # Initialize document retrieval client (separate Qdrant client for property docs)
        if DOCUMENT_RETRIEVAL_AVAILABLE:
            try:
                self._doc_client = QdrantClient(
                    host=os.getenv("QDRANT_HOST", "localhost"),
                    port=int(os.getenv("QDRANT_PORT", "6333"))
                )
                self._doc_embeddings = OpenAIEmbeddings(
                    model=EMBED_MODEL,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                print("✅ Document retrieval client initialized (property_pricing, property_specifications, riverwood_faq)")
            except Exception as e:
                print(f"⚠️ Document retrieval initialization failed: {e}")
                self._doc_client = None
                self._doc_embeddings = None
        
        # Fallback to basic OpenAI client if mem0 unavailable
        if self._mem_client is None and OpenAI is not None:
            try:
                self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                self._openai = None

    # Public API
    def upsert(self, item: Dict) -> None:
        """Store conversation turn; mem0 automatically extracts entities and updates Neo4j graph."""
        # Optional JSONL audit trail
        if self._jsonl is not None:
            try:
                self._jsonl.upsert(item)
            except Exception:
                pass

        user_text = (item.get("text") or "").strip()
        reply = (item.get("reply") or "").strip()

        # Use mem0 to add conversation - it handles entity extraction and graph updates automatically
        if self._mem_client is not None:
            try:
                # mem0 expects conversation format
                conversation = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply}
                ]
                self._mem_client.add(conversation, user_id=self.session)
                print(f"💾 mem0: Stored conversation for user '{self.session}' with automatic entity extraction")
            except Exception as e:
                print(f"⚠️ mem0 add failed: {e}")

    def _get_document_collection(self, query: str) -> Optional[str]:
        """
        Determine which document collection to search based on query keywords.
        
        Returns:
            Collection name or None if no document search needed
        """
        query_lower = query.lower()
        
        # Check for pricing keywords
        if re.search(r'\b(price|pricing|cost|payment|booking|amount|pay|bhk|crore|lakh|lac|₹|rs)\b', query_lower):
            return "property_pricing"
        
        # Check for specification keywords
        if re.search(r'\b(spec|specification|floor|sqft|area|size|plan|layout|design|construction|feature|amenity|amenities)\b', query_lower):
            return "property_specifications"
        
        # Check for FAQ/question keywords
        if re.search(r'\b(question|faq|query|ask|how|what|when|where|why|tell me|info|information)\b', query_lower):
            return "riverwood_faq"
        
        return None
    
    def _retrieve_from_documents(self, query: str, collection: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant documents from specified Qdrant collection.
        
        Args:
            query: Search query
            collection: Collection name (property_pricing, property_specifications, riverwood_faq)
            k: Number of results to return
            
        Returns:
            List of document chunks with metadata
        """
        if not self._doc_client or not self._doc_embeddings:
            return []
        
        try:
            # Generate query embedding
            query_vector = self._doc_embeddings.embed_query(query)
            
            # Search in specified collection
            results = self._doc_client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=k
            )
            
            # Convert to agent format
            docs: List[Dict] = []
            for result in results:
                docs.append({
                    "text": result.payload.get("page_content", ""),
                    "score": result.score,
                    "type": "document",
                    "collection": collection,
                    "source": result.payload.get("source_file", "")
                })
            
            if docs:
                print(f"📄 Retrieved {len(docs)} documents from '{collection}' collection")
            return docs
            
        except Exception as e:
            print(f"⚠️ Document retrieval from '{collection}' failed: {e}")
            return []

    def retrieve_relevant(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant context from:
        1. Property documents (if query matches pricing/specs/FAQ keywords)
        2. User memories (from mem0 + Neo4j graph)
        
        Returns combined results with document context prioritized when available.
        """
        if not query:
            return []
        
        all_results: List[Dict] = []
        
        # STEP 1: Check if query requires document retrieval
        doc_collection = self._get_document_collection(query)
        if doc_collection:
            doc_results = self._retrieve_from_documents(query, doc_collection, k)
            all_results.extend(doc_results)
            print(f"🔍 Document search triggered: '{doc_collection}' collection")

        # STEP 2: Always retrieve user memories (conversation history + graph entities)
        if self._mem_client is not None:
            try:
                result = self._mem_client.search(query=query, user_id=self.session, limit=k)
                memories = result.get("results", [])
                
                # Convert mem0 format to agent format
                for mem in memories:
                    mem_text = mem.get("memory", "")
                    if mem_text:
                        all_results.append({
                            "text": mem_text,
                            "score": mem.get("score", 0.0),
                            "type": "memory"
                        })
                
                if memories:
                    print(f"🧠 Retrieved {len(memories)} memories (from Qdrant + Neo4j graph)")
            except Exception as e:
                print(f"⚠️ mem0 search failed: {e}")
        
        # STEP 3: Fallback to JSONL if mem0 unavailable
        if not all_results and self._jsonl is not None:
            all_results = self._jsonl.retrieve_relevant(query, k)
        
        return all_results


if __name__ == "__main__":
    # Smoke test with mem0
    print("🧪 Testing mem0-based MemoryStore...")
    store = MemoryStore()
    
    # Test 1: Store a conversation about a plot
    item1 = {
        "id": _uid(),
        "text": "My name is Rohan and I want to know about P-23 status",
        "reply": "Hello Rohan! Plot P-23 is currently at Foundation stage with 80% progress.",
        "ts": time.time(),
        "session_id": os.getenv("AGENT_SESSION_ID", "local")
    }
    print("\n1️⃣ Storing conversation about Rohan and P-23...")
    store.upsert(item1)
    
    # Test 2: Retrieve relevant memories
    print("\n2️⃣ Searching for 'foundation progress'...")
    results = store.retrieve_relevant("foundation progress", 3)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # Test 3: Search for name
    print("\n3️⃣ Searching for 'Rohan'...")
    results2 = store.retrieve_relevant("Rohan", 3)
    print(json.dumps(results2, ensure_ascii=False, indent=2))
    
    print("\n✅ Smoke test complete. Check Neo4j browser for graph updates!")
