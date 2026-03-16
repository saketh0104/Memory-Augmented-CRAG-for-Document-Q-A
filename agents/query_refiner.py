class QueryRefiner:
    def refine(self, query: str) -> str:
        return f"Find exact financial, governance, or operational details related to: {query}"