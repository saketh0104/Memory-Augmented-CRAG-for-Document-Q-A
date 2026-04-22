from .openrouter_llm import OpenRouterLLM


class RAGGenerator:
    def __init__(self):
        self.llm = OpenRouterLLM()

    def generate_answer(self, query, retrieved_chunks, intent, episodic_memory=None):

        # ---------- Build Retrieval Context ----------
        context = "\n\n".join(
            f"[{c['source_file']} | chunk {c['chunk_id']}]\n{c['text']}"
            for c in retrieved_chunks
        )

        # ---------- Episodic Memory ----------
        memory_context = ""
        if episodic_memory:
            for mem in episodic_memory:
                if isinstance(mem, dict):
                    memory_context += (
                        f"Previous Question: {mem.get('query','')}\n"
                        f"Previous Answer: {mem.get('answer','')}\n\n"
                    )

        intent = intent.lower().strip()
        
        if intent == "fact_lookup":

            system_prompt = """
You are a high-precision enterprise assistant.

Task:
Answer the question directly using the provided context as evidence.

Behavior:
- Understand the question first.
- Use context to identify the correct fact.
- Respond naturally and clearly.
- Start with the answer immediately.
- No phrases like:
  "According to the document"
  "Based on the context"
- Preserve numbers, names, dates, percentages exactly.
- If multiple values exist, choose the most relevant one.
- Keep concise but readable.
- Plain text only.
"""
        elif intent == "procedural":

            system_prompt = """
You are an enterprise workflow assistant.

Task:
Explain how something works using the provided context.

Behavior:
- Use the context as reference material.
- Present the answer in logical numbered steps.
- Make steps clear, readable, and professional.
- Improve wording where needed.
- If context is partial, infer a reasonable structure only from available evidence.
- No markdown.
- No bold text.
- No unnecessary introduction.
"""
        elif intent == "global_summary":

            system_prompt = """
You are an enterprise summarization assistant.

Task:
Create a high-quality summary using the context as source material.

Behavior:
- Understand the question theme.
- Merge related evidence across chunks.
- Write a polished multi-paragraph summary.
- Cover major points such as business model, operations, strategy, products, governance, financial themes, or risks when relevant.
- Avoid repeating raw chunk wording.
- Rewrite into clean human-readable language.
- Use rich but concise detail.
- No markdown.
"""

        else:

            system_prompt = """
You are an enterprise analytical assistant.

Task:
Answer open-ended business questions using the context as supporting evidence.

Behavior:
- First understand what the user is really asking.
- Use all relevant context to form a thoughtful answer.
- Explain reasons, impacts, comparisons, incentives, risks, relationships, or conclusions when relevant.
- Do not merely repeat chunk text.
- Interpret the evidence into a polished readable response.
- If context is partial, still provide the best supported answer.
- Be intelligent, grounded, and professional.
- No markdown.
"""

        user_prompt = f"""
{memory_context}

Use the following retrieved context as supporting evidence.

Context:
{context}

Question:
{query}

Generate the best final answer.
"""

        return self.llm.generate(system_prompt, user_prompt).strip()