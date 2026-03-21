import os
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class LoreLibrarian:
    def __init__(self, db_path: str = None):
        # If no path is provided, use the relative repo path (for VS Code)
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.persistent_path = os.path.join(base_dir, "data/lore_db")
        else:
            # Use the explicit path provided (for Colab)
            self.persistent_path = db_path

        self.client = chromadb.PersistentClient(path=self.persistent_path)
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="dnd_lore",
            embedding_function=self.embedding_function,
        )

    def search(self, query: str, n_results: int = 3) -> str:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        documents: List[str] = (results.get("documents") or [[]])[0]
        return "\n\n".join(documents)
