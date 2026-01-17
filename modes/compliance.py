
from modes.base import BaseMode


class ComplianceMode(BaseMode):
    name = "compliance"
    top_k = 10

    def run(self, query, retriever):
        chunks = retriever.search(query, top_k=self.top_k)

        prompt = f"""
        Identify:
        - Missing clauses
        - Conflicts with standard policies
        - Ambiguous wording

        DO NOT give legal advice.
        Content:
        {chunks}
        """

        risks = retriever.llm_raw(prompt)

        return {
            "risks": risks,
            "confidence": 0.5,
            "disclaimer": "Not legal advice"
        }
