import re
from rag.openrouter_llm import OpenRouterLLM


class QueryRefiner:

    def __init__(self):
        self.llm = OpenRouterLLM()

        self.stop_words = {
            "please", "tell", "me", "about", "can", "you",
            "is", "are", "was", "were", "the", "a", "an",
            "what", "which", "who", "when", "how"
        }

        self.doc_entities = {
            "company", "financial", "revenue", "sales",
            "employees", "risk", "market", "shares",
            "governance", "board", "operations",
            "products", "services", "segment",
            "income", "assets", "liabilities",
            "cash", "stock", "ceo", "director",
            "dividend", "profit", "loss"
        }

    def clean_text(self, query):
        query = query.lower()
        query = re.sub(r"[^a-zA-Z0-9\s]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def extract_keywords(self, words):
        return [
            w for w in words
            if w not in self.stop_words and len(w) > 2
        ]

    def rule_refine(self, query):
        cleaned = self.clean_text(query)
        words = cleaned.split()

        keywords = self.extract_keywords(words)

        matched = [w for w in keywords if w in self.doc_entities]

        if matched:
            return " ".join(matched)

        return cleaned

    def llm_refine(self, query):

        system_prompt = """
You improve failed search queries for enterprise annual-report retrieval.

Rules:
- Rewrite into short semantic search query
- Preserve original intent
- Focus on finance, governance, operations, risk
- Remove noisy words
- Keep concise

Return only refined query.
"""

        try:
            refined = self.llm.generate(system_prompt, query).strip()
            return refined if refined else query
        except:
            return query

    def refine(self, query: str):

        # Stage 1: rule-based filtering
        rule_query = self.rule_refine(query)

        # Stage 2: LLM semantic reformulation
        final_query = self.llm_refine(rule_query)

        return final_query