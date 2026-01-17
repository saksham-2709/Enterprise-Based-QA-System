from .base import BaseMode

class SearchMode(BaseMode):
    name = "search"

    def run(self, query: str, evidence: list, confidence: str):
        """
        Search Mode:
        Returns ranked evidence only (no reasoning).
        """
        return {
            "results": [
                {
                    "text": e["text"],
                    "score": e["score"],
                    "source": e["metadata"].get("source"),
                    "preview": e["text"][:120].replace("\n", " ")
                }
                for e in evidence
            ],
            "confidence": confidence
        }
