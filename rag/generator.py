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
You are a document-grounded enterprise assistant.

Rules:
- Summarize only from provided excerpts.
- Combine evidence across excerpts carefully.
- Preserve factual correctness.
- Do not introduce external interpretation.
"""

        else:
            system_prompt = """
You are a document-grounded enterprise assistant.

Rules:
- Use only provided document excerpts.
- Prefer exact factual wording when possible.
- Preserve financial values exactly.
- Do not invent missing facts.
- If evidence is insufficient, say:
  'The document does not contain sufficient information.'
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
