import os
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class LoreLibrarian:
    def __init__(self, db_path: str = "data/lore_db"):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        persistent_path = os.path.join(base_dir, db_path)

        self.client = chromadb.PersistentClient(path=persistent_path)
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
