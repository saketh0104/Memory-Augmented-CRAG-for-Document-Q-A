import re
from rag.openrouter_llm import OpenRouterLLM


class LLMIntentRouter:
    def __init__(self):
        self.llm = OpenRouterLLM()

        # -------- CLEAN PROMPT --------
        self.system_prompt = """
You are an intent classification agent for a document-grounded RAG system.

Classify the user query into ONE of the following categories:

FACT_LOOKUP:
- Specific facts (numbers, names, dates, values)

GLOBAL_SUMMARY:
- High-level summaries, themes, business overview

PROCEDURAL:
- Steps, workflows, instructions

EXPLORATORY:
- Reasoning, comparisons, analysis, risks

Rules:
- Return ONLY one label
- No explanation
- No extra text

Valid outputs:
FACT_LOOKUP
GLOBAL_SUMMARY
PROCEDURAL
EXPLORATORY
"""

        # -------- HARD POLICY MAP --------
        self.policy_map = {
            "FACT_LOOKUP": {
                "threshold": 0.30,
                "allow_soft_aggregation": False,
                "bm25_weight": 0.5,
                "semantic_weight": 0.5,
                "top_k": 5
            },
            "GLOBAL_SUMMARY": {
                "threshold": 0.20,
                "allow_soft_aggregation": True,
                "bm25_weight": 0.2,
                "semantic_weight": 0.8,
                "top_k": 12
            },
            "PROCEDURAL": {
                "threshold": 0.25,
                "allow_soft_aggregation": True,
                "bm25_weight": 0.3,
                "semantic_weight": 0.7,
                "top_k": 10
            },
            "EXPLORATORY": {
                "threshold": 0.15,
                "allow_soft_aggregation": True,
                "bm25_weight": 0.2,
                "semantic_weight": 0.8,
                "top_k": 15
            }
        }

    # -------- MAIN FUNCTION --------
    def classify(self, query: str) -> dict:

        response = self.llm.generate(self.system_prompt, query)

        print("\n--- RAW INTENT OUTPUT ---")
        print(response)

        intent = self._extract_intent(response)

        # fallback safety
        if intent not in self.policy_map:
            intent = "FACT_LOOKUP"

        config = self.policy_map[intent]

        return {
            "intent": intent,
            **config
        }

    # -------- INTENT PARSER --------
    def _extract_intent(self, response: str) -> str:
        response = response.strip().upper()

        for intent in ["FACT_LOOKUP", "GLOBAL_SUMMARY", "PROCEDURAL", "EXPLORATORY"]:
            if intent in response:
                return intent

        return "FACT_LOOKUP"