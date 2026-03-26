import os
import re
import chromadb

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

        # 5. Retrieval quality controls (tune if needed)
        # For cosine distance in Chroma, lower is better.
        self.distance_threshold = 1.0
        self.min_keyword_overlap = 1
        self.max_candidates = 8

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", (text or "").lower()))

    def _keyword_overlap(self, query: str, doc: str) -> int:
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(doc)
        # Ignore tiny/common tokens to reduce accidental matches
        query_tokens = {t for t in query_tokens if len(t) > 2}
        return len(query_tokens & doc_tokens)

    def search(self, query: str, n_results: int = 3) -> str:
        """Return only high-confidence lore snippets.

        Filters candidates by vector distance and keyword overlap to avoid
        injecting unrelated context that can trigger hallucinations.
        """
        if not query or not query.strip():
            return ""

        # Pull extra candidates, then filter down.
        results = self.collection.query(
            query_texts=[query],
            n_results=max(n_results, self.max_candidates),
            include=["documents", "distances"]
        )

        documents = results.get("documents", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        filtered = []
        for i, doc in enumerate(documents):
            if not doc:
                continue

            distance = distances[i] if i < len(distances) else None
            overlap = self._keyword_overlap(query, doc)

            distance_ok = (distance is None) or (distance <= self.distance_threshold)
            overlap_ok = overlap >= self.min_keyword_overlap

            if distance_ok and overlap_ok:
                filtered.append(doc.strip())

        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for doc in filtered:
            key = doc[:200]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)

        return "\n\n".join(deduped[:n_results]) if deduped else ""