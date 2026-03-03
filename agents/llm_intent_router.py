import json
import re
from rag.openrouter_llm import OpenRouterLLM


class LLMIntentRouter:
    def __init__(self):
        self.llm = OpenRouterLLM()

        self.system_prompt = """
You are an intent classification agent for a document-grounded question answering system.

Your job is to classify a user query based on HOW the answer must be obtained from a document.

IMPORTANT DEFINITIONS:

FACT_LOOKUP:
- The query asks for a single, explicit, localized fact.

GLOBAL_SUMMARY:
- The query asks for the overall purpose, aim, mission, or theme.

PROCEDURAL:
- The query asks for steps or methods.

EXPLORATORY:
- The query is open-ended.

CRAG CONTROL POLICY:
- FACT_LOOKUP → threshold ≈ 0.35, allow_soft_aggregation = false
- PROCEDURAL → threshold ≈ 0.30, allow_soft_aggregation = true
- GLOBAL_SUMMARY → threshold ≈ 0.25, allow_soft_aggregation = true
- EXPLORATORY → threshold ≈ 0.20, allow_soft_aggregation = true

Return ONLY valid JSON in this format:
{
  "intent": "...",
  "threshold": number,
  "allow_soft_aggregation": true | false
}

Do not explain.
"""

    def classify(self, query: str) -> dict:

        user_prompt = f"User Query:\n{query}"

        response = self.llm.generate(self.system_prompt, user_prompt)

        print("\n--- RAW INTENT LLM OUTPUT ---")
        print(response)

        try:
            # 🔥 Extract first JSON object safely
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                raise ValueError("No JSON found")

        except Exception:
            # Ultra-safe fallback
            return {
                "intent": "FACT_LOOKUP",
                "threshold": 0.35,
                "allow_soft_aggregation": False
            }
