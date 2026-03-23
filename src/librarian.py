import os
from typing import List
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class LoreLibrarian:
    def __init__(self, db_path: str = None):
        # 1. Path Logic: Priority to explicit path, fallback to repo-relative
        if db_path:
            self.persistent_path = os.path.abspath(os.path.expanduser(db_path))
        else:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.persistent_path = os.path.join(base_dir, "data/lore_db")

        print(f"DEBUG: Librarian connecting to ChromaDB at: {self.persistent_path}")

        # 2. Initialize Client
        self.client = chromadb.PersistentClient(path=self.persistent_path)
        
        # 3. Embedding Function (Ensure name matches ingestion)
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 4. Connect to Collection (Fail if not found to avoid 'Ghost DBs')
        try:
            self.collection = self.client.get_collection(
                name="dnd_lore",
                embedding_function=self.embedding_function,
            )
        except Exception as e:
            print(f"❌ ERROR: Could not find collection 'dnd_lore' at {self.persistent_path}")
            print(f"Available collections: {self.client.list_collections()}")
            raise e

    def search(self, query: str, n_results: int = 3) -> str:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        # Ensure we return a string even if no results are found
        documents = results.get("documents", [[]])[0]
        return "\n\n".join(documents) if documents else ""