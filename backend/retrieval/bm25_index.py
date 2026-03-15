import re
import heapq
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------- NLTK SETUP ----------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ---------- CONFIG ----------
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = set(stopwords.words('english'))
_STEMMER = PorterStemmer()

def tokenize(text: str) -> List[str]:
    """
    Professional BM25 tokenizer.
    
    Pipeline:
    1. Lowercase
    2. Alphanumeric extraction
    3. Stopword removal
    4. Porter Stemming
    """
    if not text:
        return []
    
    tokens = _TOKEN_PATTERN.findall(text.lower())
    
    # Process: remove stopwords and apply stemming.
    processed = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        
        stemmed = _STEMMER.stem(t)
        if len(stemmed) < 2:
            continue
            
        processed.append(stemmed)
    
    return processed


# ---------- BM25 INDEX ----------
class BM25ChunkIndex:
    """
    BM25 index over semantic chunks.
    Optimized for memory efficiency in notebook environments.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunk_ids: List[str] = []
        self.batch_ids: List[str] = [] # Track batch for filtering
        self.bm25: Optional[BM25Okapi] = None
        
        if not chunks:
            return

        # Use a generator to avoid keeping raw text and tokens in memory simultaneously
        def get_corpus():
            for c in chunks:
                cid = c.get("chunk_id")
                batch = c.get("batch_id") or c.get("metadata", {}).get("batch_id", "default_batch")
                if not cid:
                    continue # Skip chunks with missing IDs
                
                text = c.get("text", "")
                tokens = tokenize(text)
                if tokens:
                    self.chunk_ids.append(str(cid))
                    self.batch_ids.append(str(batch))
                    yield tokens

        print(f"[BM25] Building index with {len(chunks)} chunks...")
        corpus_iterator = get_corpus()
        
        # BM25Okapi internally converts to list, but our generator 
        # minimizes peak memory during the tokenization phase.
        corpus_list = list(corpus_iterator)

        if corpus_list:
            self.bm25 = BM25Okapi(corpus_list)
            print(f"[BM25] Index built successfully. Memory-efficient construction complete.")
        else:
            self.bm25 = None
            print("[BM25] WARNING: No valid tokens found. Index is empty.")

    def to_dict(self) -> Dict:
        """Serializes the index data for persistence."""
        if not self.bm25:
            return {}
        return {
            "chunk_ids": self.chunk_ids,
            "batch_ids": self.batch_ids,
            "doc_freqs": self.bm25.doc_freqs,
            "idf": self.bm25.idf,
            "doc_len": self.bm25.doc_len,
            "avgdl": self.bm25.avgdl,
            "corpus_size": self.bm25.corpus_size,
            "k1": self.bm25.k1,
            "b": self.bm25.b,
            "epsilon": self.bm25.epsilon
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Deserializes the index data from a dictionary."""
        if not data or "doc_freqs" not in data:
            return None
        
        instance = cls([]) # Create empty instance
        instance.chunk_ids = data["chunk_ids"]
        instance.batch_ids = data["batch_ids"]
        
        # Reconstruct BM25Okapi state WITHOUT calling the standard constructor 
        # to avoid the ZeroDivisionError on initialization logic.
        bm25_obj = BM25Okapi([["dummy"]]) # Instantiate with dummy data
        
        # Overwrite internal state with persisted data
        bm25_obj.doc_freqs = data["doc_freqs"]
        bm25_obj.idf = data["idf"]
        bm25_obj.doc_len = data["doc_len"]
        bm25_obj.avgdl = data["avgdl"]
        bm25_obj.corpus_size = data["corpus_size"]
        bm25_obj.k1 = data["k1"]
        bm25_obj.b = data["b"]
        bm25_obj.epsilon = data["epsilon"]
        
        instance.bm25 = bm25_obj
        return instance

    def search(self, query: str, batch_id: str = None, top_k: int = 20) -> Dict[str, float]:
        """
        Sparse retrieval using BM25 with batch isolation.
        """
        if not self.bm25:
            return {}

        tokens = tokenize(query)
        if not tokens:
            return {}

        scores = self.bm25.get_scores(tokens)
        
        # Zip IDs, Batches and Scores for filtering
        all_results = list(zip(self.chunk_ids, self.batch_ids, scores))
        
        if batch_id:
            filtered_results = [
                (cid, score) for cid, chid, score in all_results
                if chid == batch_id
            ]
        else:
            filtered_results = [(cid, score) for cid, chid, score in all_results]

        ranked = heapq.nlargest(
            top_k,
            filtered_results,
            key=lambda x: x[1]
        )

        return {
            chunk_id: float(score)
            for chunk_id, score in ranked
            if score > 0.0
        }
