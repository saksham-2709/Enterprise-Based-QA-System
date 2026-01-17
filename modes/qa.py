from .base import BaseMode
from .summarizer import PolicySummarizer
IMPORTANT_TERMS = [
    "paid",
    "rolling",
    "covid",
    "caregiver",
    "public holiday",
    "12",
    "does not reduce"
]


def select_best_chunk(evidence):
    for e in evidence:
        text = e["text"].lower()
        if any(term in text for term in IMPORTANT_TERMS):
            return e["text"]
    return evidence[0]["text"]


class QAMode(BaseMode):
    name = "qa"

    def run(self, query: str, evidence: list, confidence: str):
        answer = select_best_chunk(evidence)
        summary  = answer

        return {
            "answer": summary,
            "confidence": confidence,
            "evidence": evidence
        }
