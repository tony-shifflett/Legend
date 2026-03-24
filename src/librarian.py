import os
from typing import List
import chromadb
# We keep the import just in case, but we won't force it in __init__
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class LoreLibrarian:
    def __init__(self, db_path: str = None):
        # 1. Path Logic: Priority to explicit path, fallback to repo-relative
        if db_path:
            self.persistent_path = os.path.abspath(os.path.expanduser(db_path))
        else:
            # Climb up from src/ to root, then into data/lore_db
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.persistent_path = os.path.join(base_dir, "data/lore_db")

        print(f"DEBUG: Librarian connecting to ChromaDB at: {self.persistent_path}")

        # 2. Initialize Client
        self.client = chromadb.PersistentClient(path=self.persistent_path)
        
        # 3. Handle Embedding Function Conflict
        # By setting this to None, we allow ChromaDB to use the 'persisted' 
        # embedding function already stored in your dnd_lore collection.
        self.embedding_function = None 

        # 4. Connect to Collection (Smarter Check)
        try:
            # Get list of actual objects to bypass hidden character issues
            collections = self.client.list_collections()
            collection_names = [c.name for c in collections]
            
            target = "dnd_lore"
            
            if target in collection_names:
                active_name = target
            elif len(collection_names) == 1:
                # If only one exists, it's almost certainly the one we want
                active_name = collection_names[0]
                print(f"⚠️ Precise 'dnd_lore' not found. Using only available collection: '{active_name}'")
            else:
                raise ValueError(f"Collection '{target}' not found. Available: {collection_names}")

            # Connect without forcing the embedding_function to avoid the ValueError
            self.collection = self.client.get_collection(name=active_name)
            print(f"✅ Successfully bound to collection: {active_name}")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Librarian failed to bind to a collection at {self.persistent_path}")
            print(f"Raw Error: {e}")
            raise e

    def search(self, query: str, n_results: int = 3) -> str:
        # ChromaDB will use the stored embedding function automatically here
        results = self.collection.query(query_texts=[query], n_results=n_results)
        
        # Safe extraction of documents
        documents = results.get("documents", [[]])[0]
        return "\n\n".join(documents) if documents else ""