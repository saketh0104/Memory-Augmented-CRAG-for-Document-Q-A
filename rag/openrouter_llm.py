from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class OpenRouterLLM:
    def __init__(self, model_name="openai/gpt-oss-120b:free", base_url="https://openrouter.ai/api/v1"):
        api_key = os.getenv("OPENROUTER_API_KEY") if base_url != "openai" else os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")


        if base_url == "openai":
            self.client = OpenAI(
                api_key=api_key
            )
        else:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )

        self.model_name = model_name if base_url != "openai" else "gpt-5.4-mini"

    def generate(self, system_prompt: str, user_prompt: str) -> str:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        return response.choices[0].message.content.strip()