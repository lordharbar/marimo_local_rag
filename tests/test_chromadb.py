#!/usr/bin/env python3
"""Test ChromaDB installation and basic functionality."""

import chromadb
from pathlib import Path

def test_chromadb():
    """Test ChromaDB basic operations."""
    print("Testing ChromaDB...")
    
    # Create a temporary directory for testing
    test_dir = Path("./test_chromadb")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create a client
        print("1. Creating ChromaDB client...")
        client = chromadb.PersistentClient(path=str(test_dir))
        print("   ✅ Client created successfully")
        
        # Create a collection
        print("2. Creating collection...")
        collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )
        print("   ✅ Collection created successfully")
        
        # Add some test data
        print("3. Adding test data...")
        collection.add(
            ids=["test1", "test2"],
            embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            documents=["Test document 1", "Test document 2"],
            metadatas=[{"source": "test1"}, {"source": "test2"}]
        )
        print("   ✅ Data added successfully")
        
        # Query the collection
        print("4. Querying collection...")
        results = collection.query(
            query_embeddings=[[1.0, 2.0, 3.0]],
            n_results=2
        )
        print(f"   ✅ Query successful, found {len(results['ids'][0])} results")
        
        # Clean up
        client.delete_collection("test_collection")
        print("5. Cleanup completed")
        
        print("\n✅ All ChromaDB tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test directory
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_chromadb()
