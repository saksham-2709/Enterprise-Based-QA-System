from .base import BaseMode
from collections import defaultdict

class ComparisonMode(BaseMode):
    name = "comparison"
    top_k = 12

    def run(self, query, retriever):
        chunks = retriever.search(query, top_k=self.top_k)

        grouped = defaultdict(list)
        for c in chunks:
            grouped[c.metadata["doc_name"]].append(c.text)

        if len(grouped) < 2:
            return {
                "result": "Insufficient documents for comparison",
                "confidence": 0.2
            }

        comparison_prompt = f"""
        Compare the following policy sections.
        Highlight:
        - Added rules
        - Removed rules
        - Contradictions

        {grouped}
        """

        diff = retriever.llm_raw(comparison_prompt)

        return {
            "differences": diff,
            "documents": list(grouped.keys()),
            "confidence": 0.6
        }
