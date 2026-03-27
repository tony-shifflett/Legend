import os
import re
import chromadb
from sentence_transformers import CrossEncoder

class LoreLibrarian:
    def __init__(self, db_path: str = None, ranker_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
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
        
        # 4. Initialize Cross-Encoder for re-ranking
        print(f"DEBUG: Loading CrossEncoder model: {ranker_model_name}")
        self.ranker = CrossEncoder(ranker_model_name)
        print(f"✅ CrossEncoder loaded successfully") 

        # 5. Connect to Collection (Smarter Check)
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

        # 6. Retrieval quality controls (tune if needed)
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

    def expand_query(self, query: str, llm_pipe) -> list:
        """Generate 3 technical D&D 5e search variations using the LLM pipeline.
        
        Args:
            query: The original user query
            llm_pipe: The transformers text-generation pipeline (e.g., Gemma 3)
        
        Returns:
            List of expanded query strings including the original
        """
        if not query or not query.strip():
            return [query]
        
        expansion_prompt = (
            f"Given this D&D 5e query, generate 3 alternative technical search queries "
            f"that would find similar lore. Return them as a numbered list (1., 2., 3.) "
            f"with concise, direct searches. No explanations.\n\n"
            f"Query: {query}\n\nAlternatives:"
        )
        
        try:
            outputs = llm_pipe(
                [{"role": "user", "content": expansion_prompt}],
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            response_text = outputs[0]["generated_text"]
            if isinstance(response_text, list) and response_text:
                last = response_text[-1]
                expansion_content = last.get("content", "") if isinstance(last, dict) else str(last)
            else:
                expansion_content = str(response_text)
            
            # Parse numbered list responses
            expanded_queries = []
            lines = expansion_content.split("\n")
            for line in lines:
                # Match patterns like "1. query text" or "1) query text"
                match = re.search(r"^\d+[.)]\s*(.+)$", line.strip())
                if match:
                    expanded_queries.append(match.group(1).strip())
            
            # Always include original query and limit to 3 expansions
            search_queries = [query] + expanded_queries[:3]
            print(f"DEBUG: Query expansion generated {len(search_queries)} search queries")
            return search_queries
            
        except Exception as e:
            print(f"WARNING: Query expansion failed, falling back to original query: {e}")
            return [query]

    def search(self, query: str, llm_pipe=None, n_results: int = 3) -> str:
        """Return high-confidence lore snippets using query expansion and cross-encoder re-ranking.

        Process:
        1. Expand the query into multiple search variations using the LLM
        2. Query ChromaDB with each variation to collect a larger candidate pool
        3. Re-rank candidates using a cross-encoder model
        4. Filter by distance and keyword overlap
        5. Return the top-N snippets by re-ranker score
        """
        if not query or not query.strip():
            return ""

        # Step 1: Expand query if LLM pipeline is provided
        search_queries = [query]
        if llm_pipe is not None:
            search_queries = self.expand_query(query, llm_pipe)
        
        # Step 2: Collect candidates from all expanded queries
        all_candidates = {}  # key -> document text (deduped by first 200 chars)
        
        for expanded_query in search_queries:
            results = self.collection.query(
                query_texts=[expanded_query],
                n_results=10,  # Larger pool per query
                include=["documents", "distances"]
            )
            
            documents = results.get("documents", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []
            
            for i, doc in enumerate(documents):
                if not doc:
                    continue
                
                key = doc[:200]  # Dedup key
                if key not in all_candidates:
                    distance = distances[i] if i < len(distances) else None
                    all_candidates[key] = {
                        "text": doc.strip(),
                        "distance": distance
                    }
        
        # Step 3: Filter by distance and keyword overlap thresholds
        filtered = []
        for key, data in all_candidates.items():
            doc = data["text"]
            distance = data["distance"]
            overlap = self._keyword_overlap(query, doc)
            
            distance_ok = (distance is None) or (distance <= self.distance_threshold)
            overlap_ok = overlap >= self.min_keyword_overlap
            
            if distance_ok and overlap_ok:
                filtered.append(doc)
        
        if not filtered:
            return ""
        
        # Step 4: Re-rank using cross-encoder against the ORIGINAL query
        try:
            # Prepare pairs: (original_query, candidate_doc)
            query_doc_pairs = [[query, doc] for doc in filtered]
            
            # Get scores from cross-encoder
            scores = self.ranker.predict(query_doc_pairs)
            
            # Sort by score (descending) and get top N
            ranked_docs = sorted(
                zip(filtered, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_docs = [doc for doc, score in ranked_docs[:n_results]]
            print(f"DEBUG: Cross-encoder re-ranked {len(filtered)} candidates to top {len(top_docs)}")
            
        except Exception as e:
            print(f"WARNING: Cross-encoder re-ranking failed, using filtered results: {e}")
            top_docs = filtered[:n_results]
        
        # Step 5: Return deduplicated top results
        deduped = []
        seen = set()
        for doc in top_docs:
            key = doc[:200]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        
        return "\n\n".join(deduped[:n_results]) if deduped else ""