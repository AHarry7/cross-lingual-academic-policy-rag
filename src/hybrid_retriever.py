from __future__ import annotations

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.retriever import build_or_load_vector_store


# ─────────────────────────────────────────────────────────────────
# RRF constant — k=60 is the standard value from the original
# Reciprocal Rank Fusion paper (Cormack et al., 2009).
# Higher k reduces the impact of very high ranks; 60 is well-validated.
# ─────────────────────────────────────────────────────────────────
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """
    Minimal whitespace tokenizer for BM25.
    Lowercases and splits on whitespace — sufficient for English policy text.
    """
    return text.lower().split()


class HybridRetriever:
    """
    Combines ChromaDB dense retrieval with BM25 sparse retrieval,
    fused via Reciprocal Rank Fusion (RRF).

    Corpus is sourced directly from the existing ChromaDB store so
    BM25 and ChromaDB are always in sync — no re-reading of PDFs needed.
    """

    def __init__(self):
        # ── Load ChromaDB ────────────────────────────────────────────────
        print("[HybridRetriever] Loading ChromaDB vector store...")
        self.db = build_or_load_vector_store()

        # ── Extract all stored chunks from ChromaDB ──────────────────────
        # .get() returns a dict with 'documents' (text) and 'metadatas'
        print("[HybridRetriever] Extracting corpus from ChromaDB...")
        stored = self.db.get()

        self.corpus_texts: list[str] = stored["documents"]
        self.corpus_metadatas: list[dict] = stored["metadatas"]

        if not self.corpus_texts:
            raise ValueError(
                "ChromaDB is empty. Run build_or_load_vector_store() first."
            )

        # ── Build BM25 index from corpus ─────────────────────────────────
        print(f"[HybridRetriever] Building BM25 index over {len(self.corpus_texts)} chunks...")
        tokenized_corpus = [_tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("[HybridRetriever] Ready.")

    def _dense_retrieve(self, query: str, k: int) -> list[tuple[Document, int]]:
        """
        Returns (Document, original_rank) tuples from ChromaDB dense search.
        Rank is 0-indexed (0 = most relevant).
        """
        results = self.db.similarity_search(query, k=k)
        return [(doc, rank) for rank, doc in enumerate(results)]

    def _sparse_retrieve(self, query: str, k: int) -> list[tuple[Document, int]]:
        """
        Returns (Document, original_rank) tuples from BM25 sparse search.
        Rank is 0-indexed (0 = most relevant).
        """
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top-k scores in descending order
        top_k_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for rank, idx in enumerate(top_k_indices):
            doc = Document(
                page_content=self.corpus_texts[idx],
                metadata=self.corpus_metadatas[idx],
            )
            results.append((doc, rank))

        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, int]],
        sparse_results: list[tuple[Document, int]],
        k: int,
    ) -> list[Document]:
        """
        Fuses two ranked lists using Reciprocal Rank Fusion.

        Formula per document:
            RRF_score = Σ 1 / (RRF_K + rank)

        Documents appearing in both lists accumulate scores from both.
        Uses page_content as the deduplication key.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for doc, rank in dense_results:
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_map[key] = doc

        for doc, rank in sparse_results:
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_map[key] = doc

        # Sort by fused score descending and return top-k Document objects
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:k]]

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """
        Main public method. Runs hybrid retrieval and returns top-k fused Documents.

        Args:
            query: Clean English query (post-normalization).
            k:     Number of documents to return after fusion.

        Returns:
            List of top-k LangChain Document objects ranked by RRF score.
        """
        # Fetch more candidates than k before fusion so RRF has enough
        # signal to re-rank from — standard practice is fetch_k = k * 2
        fetch_k = k * 2

        dense_results = self._dense_retrieve(query, fetch_k)
        sparse_results = self._sparse_retrieve(query, fetch_k)

        fused_docs = self._reciprocal_rank_fusion(dense_results, sparse_results, k)

        print(
            f"[HybridRetriever] Dense: {len(dense_results)} | "
            f"Sparse: {len(sparse_results)} | "
            f"Fused top-{k}: {len(fused_docs)}"
        )

        return fused_docs


# ─────────────────────────────────────────────
# Quick test when running this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "grading policy",
        "probation rules GPA",
        "Head of Department Computer Science",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        docs = retriever.retrieve(query, k=3)
        for i, doc in enumerate(docs):
            print(f"\n  Result {i+1}: {doc.page_content[:200]}...")