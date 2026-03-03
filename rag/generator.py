from .openrouter_llm import OpenRouterLLM

class RAGGenerator:
    def __init__(self):
        self.llm = OpenRouterLLM()

    def generate_answer(self, query, retrieved_chunks, intent, episodic_memory=None):

        context = "\n\n".join(
            f"[{c['source_file']} | chunk {c['chunk_id']}]\n{c['text']}"
            for c in retrieved_chunks
        )

        memory_context = ""
        if episodic_memory:
            for mem in episodic_memory:
                if isinstance(mem, dict):
                    memory_context += f"Q: {mem.get('query','')}\nA: {mem.get('answer','')}\n\n"


        if intent.lower() == "global_summary":
            system_prompt = """
You are a document-grounded assistant.

Rules:
- Use ONLY the provided excerpts.
- The answer may be implied across multiple excerpts.
- You must synthesize across passages if needed.
- Do NOT introduce external knowledge.
- Only say "The document does not contain sufficient information." if absolutely no relevant evidence exists.
"""

        else:
            system_prompt = """
You are a document-grounded assistant.

Rules:
- Answer ONLY if explicitly supported by provided excerpts.
- You may reference previous validated Q/A only if relevant.
- Do NOT infer beyond context.
- If not found, say: "The document does not contain sufficient information."
"""

        user_prompt = f"""
{memory_context}

Context:
{context}

Question:
{query}

Provide a grounded answer.
"""

        return self.llm.generate(system_prompt, user_prompt)
