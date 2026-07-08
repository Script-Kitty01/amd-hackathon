"""Thin Google Generative AI wrapper.

All inference MUST go through a Google API. This is the only place that
talks to the network, so token accounting and retry policy live here.
"""

from __future__ import annotations

from dataclasses import dataclass

import google.generativeai as genai

from .config import Config


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GoogleClient:
    def __init__(self, cfg: Config) -> None:
        genai.configure(api_key=cfg.api_key)
        # Note: Google Generative AI client usually handles base_url internally
        # However, keeping it for consistency if a custom endpoint is needed.
        # For now, base_url from config is not directly used by genai.configure.

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        """Single deterministic chat completion. Raises on API error."""
        try:
            print(f"GoogleClient: Calling API with model={model}, max_tokens={max_tokens}")

            # For Google Generative AI, system instructions are typically part of the prompt
            # or handled differently depending on the model/API version.
            # For basic use, we'll prepend the system instruction to the user prompt.
            full_prompt = f"{system}\n\n{user}"

            model_client = genai.GenerativeModel(model)
            resp = model_client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0,
                ),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )

            # Handle edge cases where the response might be blocked or empty
            text_response = ""
            if resp.parts:
                text_response = resp.text.strip()
            else:
                text_response = "Safety filter block or empty response."

            # Google's API returns usage in a different format. Estimate tokens for now.
            # This might need refinement based on actual Google API response structure for token counts.
            prompt_tokens = model_client.count_tokens(full_prompt).total_tokens
            completion_tokens = model_client.count_tokens(text_response).total_tokens if text_response else 0
            total_tokens = prompt_tokens + completion_tokens

            print(f"GoogleClient: API call successful. Total tokens: {total_tokens}")
            return LLMResult(
                text=text_response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            print(f"GoogleClient: API call failed with exception: {e}")
            raise
