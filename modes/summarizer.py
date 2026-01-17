import os
from openai import OpenAI


class PolicySummarizer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=api_key)

    def summarize(self, text: str) -> str:
        prompt = f"""
Summarize the following policy sentence clearly and concisely.
Do NOT add new information.
Do NOT change numbers or rules.

Policy text:
{text}

Summary:
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarize HR policies accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()
