from ingestion.embedder import LocalEmbedder, OpenAIEmbedder
from retrieval.retriever import retrieve
from orchestration.context_manager import ContextManager
from modes.decision import DecisionMode
from orchestration.mode_router import ModeRouter
from orchestration.mode_guesser import ModeGuesser

class RAGPipeline:
    def __init__(
        self,
        index,
        chunks,
        metas,
        embedder_type="local",
        k=4
    ):
        self.index = index
        self.chunks = chunks
        self.metas = metas
        self.k = k

        self.embedder = (
            LocalEmbedder() if embedder_type == "local"
            else OpenAIEmbedder()
        )

        self.context_manager = ContextManager()
        self.mode_router = ModeRouter()
        self.mode_guesser = ModeGuesser()

    # -------------------------
    # Core pipeline entry point
    # -------------------------
    def run(self, user_query: str, mode=None):
        # 1️⃣ Update and enrich query using context
        self.context_manager.update(user_query)
        enriched_query = self.context_manager.enrich_query(user_query)
        
        if mode is None:
            mode = self.mode_guesser.guess(user_query)
        # 2️⃣ Retrieve evidence from FAISS
        ids, scores = retrieve(
            self.index,
            self.embedder.embed,
            enriched_query,
            self.k
        )

        # 3️⃣ Safety check
        if not scores or max(scores) < 0.30:
            return {
                "status": "REFUSED",
                "message": "Insufficient evidence to answer reliably.",
                "mode": mode
            }

        # 4️⃣ Build structured evidence
        evidence = []
        for i, s in zip(ids, scores):
            if i < 0:
                continue

            evidence.append({
                "text": self.chunks[i],
                "score": float(s),
                "metadata": self.metas[i]
            })
        if not evidence:
            return {
                "status": "REFUSED",
                "message": "No usable evidence found."
            }

        # 5️⃣ Compute confidence
        confidence = self._compute_confidence(evidence)

        # 6️⃣ Route to mode
        mode_handler = self.mode_router.get(mode)
        result = mode_handler.run(
            user_query,
            evidence,
            confidence
        )
        result["mode"] = mode
        return result

        

    # -------------------------
    # Confidence logic (global)
    # -------------------------
    def _compute_confidence(self, evidence):
        avg_score = sum(e["score"] for e in evidence) / len(evidence)

        if avg_score >= 0.7:
            return "HIGH"
        elif avg_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
