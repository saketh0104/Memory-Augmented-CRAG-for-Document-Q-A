import json
import re
from rag.openrouter_llm import OpenRouterLLM


class LLMIntentRouter:
    def __init__(self):
        self.llm = OpenRouterLLM()

        self.system_prompt = """
You are an intent classification agent for a document-grounded RAG system.

Classify the query based on how evidence should be retrieved from enterprise documents.

Choose exactly one intent:

FACT_LOOKUP
Use for explicit facts such as values, names, dates, counts, percentages, or direct statements usually found in one localized chunk.

Policy:
threshold = 0.30
allow_soft_aggregation = false


GLOBAL_SUMMARY
Use for broad summaries, business overviews, company purpose, strategy, or themes requiring multiple chunks.

Policy:
threshold = 0.20
allow_soft_aggregation = true


PROCEDURAL
Use for processes, workflows, methods, steps, or how something is performed.

Policy:
threshold = 0.25
allow_soft_aggregation = true


EXPLORATORY
Use for open-ended reasoning, causes, impacts, risks, comparisons, trends, or interpretive questions.

Policy:
threshold = 0.15
allow_soft_aggregation = true


Return ONLY valid JSON:

{
  "intent": "...",
  "threshold": number,
  "allow_soft_aggregation": true
}

No explanation.
No markdown.
No extra text.
"""

    def classify(self, query: str) -> dict:

        user_prompt = f"User Query:\n{query}"

        response = self.llm.generate(self.system_prompt, user_prompt)

        print("\n--- RAW INTENT LLM OUTPUT ---")
        print(response)

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                raise ValueError("No JSON found")

        except Exception:
            return {
                "intent": "FACT_LOOKUP",
                "threshold": 0.30,
                "allow_soft_aggregation": False
            }
