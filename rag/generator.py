from .openrouter_llm import OpenRouterLLM


class RAGGenerator:
    def __init__(self):
        self.llm = OpenRouterLLM()

    def generate_answer(self, query, retrieved_chunks, intent, episodic_memory=None):

        # ---------- Retrieval Context ----------
        context = "\n\n".join(
            f"[{c['source_file']} | chunk {c['chunk_id']}]\n{c['text']}"
            for c in retrieved_chunks
        )

        # ---------- Smart Episodic Memory ----------
        memory_context = ""
        if episodic_memory:
            # take only last 2–3 relevant interactions
            recent_memory = episodic_memory[-3:]

            for mem in recent_memory:
                if isinstance(mem, dict):
                    memory_context += (
                        f"User previously asked: {mem.get('query','')}\n"
                        f"Assistant answered: {mem.get('answer','')}\n\n"
                    )

        # ---------- Intent Routing ----------
        intent = intent.lower().strip()

        if intent == "fact_lookup":
            system_prompt = """
You are a high-precision enterprise assistant.

Task:
Answer the question directly using the provided document evidence.

Memory Usage:
- Use conversation history ONLY to understand what the user refers to.
- DO NOT treat memory as factual source.

Evidence Priority:
- Retrieved context is the ONLY source of truth.

Behavior:
- Answer immediately.
- Preserve numbers, names, and values exactly.
- No phrases like "based on the context".
- Be concise and precise.
"""

        elif intent == "procedural":
            system_prompt = """
You are an enterprise workflow assistant.

Task:
Explain processes using document evidence.

Memory Usage:
- Use history only to resolve references (e.g., "that process").
- Do not rely on it for factual correctness.

Behavior:
- Provide step-by-step explanation.
- Improve clarity.
- Keep structured and professional.
"""

        elif intent == "global_summary":
            system_prompt = """
You are an enterprise summarization assistant.

Task:
Generate a high-quality summary from document evidence.

Memory Usage:
- Use conversation context to understand scope.
- Do not prioritize it over retrieved evidence.

Behavior:
- Merge information across sources.
- Write clean, structured summary.
- Avoid repetition.
"""

        else:
            system_prompt = """
You are an enterprise analytical assistant.

Task:
Answer complex questions using document evidence.

Memory Usage:
- Use memory to understand intent progression.
- Do not treat memory as ground truth.

Behavior:
- Provide reasoning, insights, and interpretation.
- Stay grounded in retrieved context.
"""

        # ---------- Final Prompt ----------
        user_prompt = f"""
Conversation History (for context only):
{memory_context}

Retrieved Evidence:
{context}

Current Question:
{query}

Instructions:
- Use retrieved evidence as the primary source.
- Use conversation history only for context understanding.
- Do not hallucinate.
- If insufficient evidence, say so clearly.

Provide the best possible answer.
"""

        return self.llm.generate(system_prompt, user_prompt).strip()