"""
load_property_docs.py - Load property documents into Qdrant
Run once to populate knowledge base with property information

This script loads documents (PDFs, TXT, CSV) from the 'document_loader' folder
and stores each file in its own Qdrant collection based on filename.

Usage:
1. Activate venv: .venv\\Scripts\\activate
2. Install dependencies: pip install -r requirements_docs.txt
3. Run: python load_property_docs.py
4. Agent will automatically retrieve from these collections!
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import traceback
import re

# LangChain document loaders
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Qdrant
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client import QdrantClient
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install -r requirements_docs.txt")
    traceback.print_exc()
    sys.exit(1)

load_dotenv()


def sanitize_collection_name(filename: str) -> str:
    """
    Convert filename to valid Qdrant collection name.
    Collection names must start with letter/underscore and contain only letters, digits, underscores, hyphens.
    
    Args:
        filename: Original filename (e.g., "Pricing_Inventory.csv")
        
    Returns:
        Valid collection name (e.g., "pricing_inventory")
    """
    # Remove extension
    name = Path(filename).stem
    
    # Convert to lowercase
    name = name.lower()
    
    # Replace spaces and special chars with underscores
    name = re.sub(r'[^a-z0-9_-]', '_', name)
    
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Ensure starts with letter or underscore
    if name and not (name[0].isalpha() or name[0] == '_'):
        name = '_' + name
    
    return name or "documents"


def load_property_documents(docs_folder: str = "document_loader"):
    """
    Load all property documents into Qdrant.
    Each file gets its own collection based on filename.
    
    Supports: PDF, TXT, CSV
    
    Args:
        docs_folder: Path to folder containing documents
    """
    print("\n" + "="*70)
    print("🏠 Real estate - Document Loader")
    print("="*70 + "\n")
    
    # Create docs folder if it doesn't exist
    docs_path = Path(docs_folder)
    if not docs_path.exists():
        print(f"📁 Creating folder: {docs_path}")
        docs_path.mkdir(parents=True)
        print(f"\n💡 Please add your documents to: {docs_path.absolute()}")
        print("   Supported formats: PDF, TXT, CSV")
        print("\n   Then run this script again!")
        return False
    
    print(f"📂 Scanning folder: {docs_path.absolute()}\n")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY in .env file")
        return False
    
    # Initialize embeddings
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
        print("✅ OpenAI Embeddings initialized")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return False
    
    # Initialize Qdrant client
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        
        qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Test connection
        collections = qdrant.get_collections()
        print(f"✅ Qdrant connected ({qdrant_host}:{qdrant_port})")
        print(f"   Existing collections: {[c.name for c in collections.collections]}\n")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("   Make sure Qdrant is running (docker-compose up -d)")
        return False
    
    # Text splitter (chunks for better retrieval)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 500 chars per chunk
        chunk_overlap=50,  # 50 char overlap for context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    total_collections = 0
    total_chunks = 0
    
    # Process each file type separately
    all_files = []
    
    # Find all supported files
    all_files.extend([(f, "pdf") for f in docs_path.glob("*.pdf")])
    all_files.extend([(f, "txt") for f in docs_path.glob("*.txt")])
    all_files.extend([(f, "csv") for f in docs_path.glob("*.csv")])
    
    if not all_files:
        print("\n⚠️ No documents found in folder!")
        print(f"   Add PDF, TXT, or CSV files to: {docs_path.absolute()}")
        return False
    
    print(f"📚 Found {len(all_files)} file(s) to process\n")
    print("="*70)
    
    # Process each file
    for file_path, file_type in all_files:
        print(f"\n📄 Processing: {file_path.name}")
        print(f"   Type: {file_type.upper()}")
        
        try:
            # Load document based on type
            if file_type == "pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load_and_split(splitter)
            elif file_type == "txt":
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load_and_split(splitter)
            elif file_type == "csv":
                loader = CSVLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
            else:
                continue
            
            if not docs:
                print(f"   ⚠️ No content extracted")
                continue
            
            # Add metadata
            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["file_type"] = file_type
            
            print(f"   ✅ Loaded {len(docs)} chunks")
            
            # Create collection name from filename
            collection_name = sanitize_collection_name(file_path.name)
            print(f"   📦 Collection: '{collection_name}'")
            
            # Delete existing collection if it exists
            try:
                qdrant.delete_collection(collection_name)
                print(f"   🗑️  Deleted existing collection")
            except:
                pass
            
            # Create new collection with documents
            vectorstore = Qdrant.from_documents(
                documents=docs,
                embedding=embeddings,
                collection_name=collection_name,
                url=f"http://{qdrant_host}:{qdrant_port}"
            )
            
            print(f"   ✅ Created collection with {len(docs)} chunks")
            
            # Test retrieval
            test_query = "price" if "price" in file_path.name.lower() else "information"
            results = vectorstore.similarity_search(test_query, k=1)
            
            if results:
                print(f"   🔍 Test query '{test_query}': ✅ Retrieved")
            
            total_collections += 1
            total_chunks += len(docs)
            
        except Exception as e:
            print(f"   ❌ Error loading {file_path.name}: {e}")
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("✅ DOCUMENT LOADING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total collections created: {total_collections}")
    print(f"   Total chunks stored: {total_chunks}")
    print(f"\n📚 Collections created:")
    
    # List all collections
    collections = qdrant.get_collections()
    for collection in collections.collections:
        if collection.name in ['pricing_inventory', 'property_specifications', 'real_estate_faq']:
            count = qdrant.count(collection.name)
            print(f"   - {collection.name}: {count.count} chunks")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Update your agent to retrieve from these collections")
    print(f"   2. Test with queries like:")
    print(f"      - 'What is the price of 2BHK?'")
    print(f"      - 'Tell me about specifications'")
    print(f"      - 'FAQ about amenities'")
    print(f"\n💡 Each document is in its own collection for organized retrieval!")
    print(f"\n")
    
    return True


if __name__ == "__main__":
    try:
        success = load_property_documents()
        
        if success:
            print("✅ Done! Your documents are ready to use.\n")
        else:
            print("⚠️ Document loading incomplete. Check errors above.\n")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Exiting...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
