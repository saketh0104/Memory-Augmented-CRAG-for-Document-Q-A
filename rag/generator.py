from pathlib import Path

from .openrouter_llm import OpenRouterLLM


class RAGGenerator:

    MAX_CHARS_PER_CHUNK = 1200
    MAX_MEMORY_ITEMS = 3

    def __init__(self):

        self.llm = OpenRouterLLM()

        self.prompt_dir = (
            Path(__file__).parent / "prompts"
        )

    # LOAD PROMPT FILE
    def _load_prompt(self, filename):

        path = self.prompt_dir / filename

        with open(path, "r", encoding="utf-8") as f:
            return f.read()


    # BUILD RETRIEVAL CONTEXT
    def _build_context(self, retrieved_chunks):

        contexts = []

        for chunk in retrieved_chunks:

            text = chunk.get("text", "").strip()

            if not text:
                continue

            # CONTEXT COMPRESSION
            text = text[:self.MAX_CHARS_PER_CHUNK]

            source = chunk.get(
                "source_file",
                "unknown"
            )

            chunk_id = chunk.get(
                "chunk_id",
                "N/A"
            )

            contexts.append(
                f"[Source: {source} | Chunk: {chunk_id}]\n{text}"
            )

        return "\n\n".join(contexts)

    
    # BUILD MEMORY CONTEXT
    def _build_memory(self, episodic_memory):

        if not episodic_memory:
            return ""

        memory_lines = []

        recent_memory = episodic_memory[
            -self.MAX_MEMORY_ITEMS:
        ]

        for mem in recent_memory:

            if not isinstance(mem, dict):
                continue

            query = mem.get(
                "query",
                ""
            ).strip()

            if not query:
                continue

            # MEMORY SUMMARIZATION
            short_query = (
                query[:120] + "..."
                if len(query) > 120
                else query
            )

            memory_lines.append(
                f"- Prior topic: {short_query}"
            )

        return "\n".join(memory_lines)

    
    # SYSTEM PROMPTS
    def _get_system_prompt(self, intent):

        intent = intent.lower().strip()

        prompt_map = {
            "fact_lookup": "fact_lookup.txt",
            "procedural": "procedural.txt",
            "global_summary": "global_summary.txt",
            "exploratory": "exploratory.txt"
        }

        filename = prompt_map.get(intent,"exploratory.txt")

        return self._load_prompt(filename)


    # GENERATE ANSWER
    def generate_answer(
        self,
        query,
        retrieved_chunks,
        intent,
        episodic_memory=None,
        retrieval_confidence=None
    ):

        # BUILD CONTEXTS
        retrieval_context = self._build_context(
            retrieved_chunks
        )

        memory_context = self._build_memory(
            episodic_memory
        )

        # EMPTY RETRIEVAL
        if not retrieval_context.strip():

            return (
                "I could not find sufficient evidence "
                "in the uploaded documents to answer "
                "that question."
            )

        # CONFIDENCE WARNING

        confidence_note = ""

        if (
            retrieval_confidence is not None
            and retrieval_confidence < 0.45
        ):

            confidence_note = (
                "WARNING: The retrieved evidence may be incomplete "
                "or weakly relevant. Avoid strong conclusions "
                "unless clearly supported."
            )


        # SYSTEM PROMPT
        system_prompt = self._get_system_prompt(
            intent
        )

        # USER PROMPT
        user_prompt = f"""Conversation Reference (for continuity only):
{memory_context}

Retrieved Evidence (primary factual source):
{retrieval_context}

Current User Question:
{query}

{confidence_note}

FINAL INSTRUCTIONS:

GROUNDING RULES:
- Retrieved evidence is the ONLY factual source.
- Never use external world knowledge unless explicitly present in the evidence.
- Never hallucinate unsupported facts, metrics, dates, or claims.
- If evidence is insufficient, clearly acknowledge limitations.

NUMERICAL SAFETY:
- Never alter numerical values.
- Never estimate missing numbers.
- Preserve currencies, units, percentages, and dates exactly.

MEMORY RULES:
- Use memory only for conversational continuity.
- Do not treat memory as factual evidence.

FORMATTING RULES:
- Default to natural chat-style answers.
- Use short paragraphs with minimal spacing.
- Use bullet points ONLY when listing multiple items.
- Use headings ONLY for long summaries or complex explanations.
- Do NOT insert unnecessary blank lines.
- Do NOT format every answer like a report.
- Use tables ONLY when numerical comparison genuinely helps.
- Avoid unnecessary introductions.
- Never dump raw chunk text.

ANSWER STYLE:
- Be professional and grounded.
- Prefer precision over verbosity.
- Clearly distinguish facts from interpretation.

RESPONSE SHAPE:
- For simple factual answers: answer in 1–3 compact paragraphs.
- For lists: use bullets with tight spacing.
- For procedural questions: use numbered steps.
- For summaries: use headings only if truly needed.
"""

        # GENERATE
        answer = self.llm.generate(
            system_prompt,
            user_prompt
        )

        # FALLBACK
        if not answer:

            return (
                "I could not generate a reliable answer "
                "from the retrieved evidence."
            )

        return answer.strip()