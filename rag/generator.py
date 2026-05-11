from .openrouter_llm import OpenRouterLLM

class RAGGenerator:

    def __init__(self):
        self.llm = OpenRouterLLM()

    # BUILD RETRIEVAL CONTEXT
    def _build_context(self, retrieved_chunks):

        contexts = []

        for chunk in retrieved_chunks:

            text = chunk.get("text", "").strip()

            if not text:
                continue

            source = chunk.get("source_file", "unknown")
            chunk_id = chunk.get("chunk_id", "N/A")

            contexts.append(
                f"[Source: {source} | Chunk: {chunk_id}]\n{text}"
            )

        return "\n\n".join(contexts)

    # BUILD MEMORY CONTEXT
    def _build_memory(self, episodic_memory):

        if not episodic_memory:
            return ""

        memory_lines = []

        # keep only recent conversational continuity
        recent_memory = episodic_memory[-3:]

        for mem in recent_memory:

            if not isinstance(mem, dict):
                continue

            q = mem.get("query", "").strip()
            a = mem.get("answer", "").strip()

            if not q or not a:
                continue

            memory_lines.append(
                f"User: {q}\nAssistant: {a}"
            )

        return "\n\n".join(memory_lines)

    # SYSTEM PROMPTS
    def _get_system_prompt(self, intent):

        intent = intent.lower().strip()

        # FACT LOOKUP
        if intent == "fact_lookup":

            return """
You are a high-precision enterprise document assistant.

Your role is to answer factual questions accurately using retrieved evidence.

GROUNDING HIERARCHY:
1. Retrieved evidence is the ONLY factual source.
2. Conversation memory is ONLY for conversational continuity.
3. Never invent unsupported information.

STRICT RULES:
- Never hallucinate values, dates, names, metrics, or claims.
- Preserve numbers and entities exactly as written.
- If evidence is insufficient, clearly say:
  "I could not find sufficient evidence for that in the retrieved documents."

FORMATTING RULES:
- Use concise professional markdown.
- Start directly with the answer.
- Use bullet points where useful.
- Avoid unnecessary introductions.
- Avoid repeating evidence verbatim.
- Keep answers precise and grounded.
"""

        # PROCEDURAL
        elif intent == "procedural":

            return """
You are an enterprise workflow assistant.

Your role is to explain processes clearly using retrieved evidence.

GROUNDING HIERARCHY:
1. Retrieved evidence is authoritative.
2. Memory is only for conversational reference.
3. Never fabricate undocumented steps.

STRICT RULES:
- Do not invent process stages.
- If evidence is partial, explicitly mention limitations.
- Keep explanations operational and readable.

FORMATTING RULES:
- Use markdown numbered lists.
- Group related steps logically.
- Use short paragraphs.
- Keep explanations clean and structured.
"""

        # GLOBAL SUMMARY
        elif intent == "global_summary":

            return """
You are an enterprise summarization assistant.

Your role is to synthesize retrieved evidence into a coherent,
professional summary.

GROUNDING HIERARCHY:
1. Retrieved evidence is the factual foundation.
2. Memory only helps maintain topic continuity.
3. Never introduce unsupported business claims.

STRICT RULES:
- Summarize only supported information.
- Merge related evidence intelligently.
- Avoid raw chunk repetition.
- Avoid exaggerated language.

FORMATTING RULES:
- Use professional markdown formatting.
- Use section headings where appropriate.
- Use concise paragraphs and bullet points.
- Keep the tone analytical and polished.
"""

        # EXPLORATORY / ANALYTICAL
        else:

            return """
You are an enterprise analytical assistant.

Your role is to answer analytical questions using retrieved evidence.

GROUNDING HIERARCHY:
1. Retrieved evidence is authoritative.
2. Memory only provides conversational continuity.
3. Unsupported assumptions must never be presented as facts.

STRICT RULES:
- Clearly separate evidence from interpretation.
- Acknowledge uncertainty when evidence is incomplete.
- Do not overstate conclusions.

FORMATTING RULES:
- Use professional markdown formatting.
- Use clear sections and bullet points.
- Keep reasoning structured and readable.
- Avoid filler language.
"""

    # GENERATE ANSWER
    def generate_answer(
        self,
        query,
        retrieved_chunks,
        intent,
        episodic_memory=None
    ):

        # BUILD CONTEXTS
        retrieval_context = self._build_context(
            retrieved_chunks
        )

        memory_context = self._build_memory(
            episodic_memory
        )

        # HANDLE EMPTY RETRIEVAL
        if not retrieval_context.strip():

            return (
                "I could not find sufficient evidence "
                "in the uploaded documents to answer "
                "that question."
            )

        # SYSTEM PROMPT
        system_prompt = self._get_system_prompt(intent)


        # USER PROMPT
        user_prompt = f"""
Conversation Memory
(For conversational continuity only — NOT factual evidence):

{memory_context}


Retrieved Evidence
(Primary factual source):

{retrieval_context}


Current User Question:

{query}


FINAL INSTRUCTIONS:

- Retrieved evidence is the primary source of truth.
- Use memory only to understand references or topic continuity.
- Never hallucinate unsupported claims.
- If evidence is weak or incomplete, explicitly acknowledge it.
- Keep the response professional, grounded, and readable.
- Use clean markdown formatting.
- Prefer concise clarity over verbosity.
"""

        # GENERATE
        answer = self.llm.generate(
            system_prompt,
            user_prompt
        )

        # CLEANUP
        if not answer:
            return (
                "I could not generate a reliable answer "
                "from the retrieved evidence."
            )

        return answer.strip()