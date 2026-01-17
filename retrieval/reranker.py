import openai
import os


def rerank_with_llm(query: str, chunks):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
Question:
{query}

Candidates:
{chunks}

Return ONLY the most precise answer.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp["choices"][0]["message"]["content"]
