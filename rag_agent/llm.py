import os
import sys
import re
from typing import Dict, Any, Iterable
from openai import OpenAI

# NVIDIA Inference API via OpenAI-compatible client.
# Uses environment variables for configuration:
# - NVIDIA_API_KEY
# - NVIDIA_BASE_URL (default: https://integrate.api.nvidia.com/v1)
# - NVIDIA_MODEL (default: nvidia/llama-3.3-nemotron-super-49b-v1.5)

NVIDIA_BASE = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
DEFAULT_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")


def sanitize_output(text: str) -> str:
    # Remove any accidental chain-of-thought disclosures like <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # Trim extra whitespace
    return text.strip()


class NvidiaChatClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or NVIDIA_API_KEY
        self.base_url = (base_url or NVIDIA_BASE).rstrip("/")
        self.model = model or DEFAULT_MODEL
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set. Export it before running.")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.25, top_p: float = 0.95,
             max_tokens: int = 2200, stream: bool = True) -> Iterable[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=0.2,
            presence_penalty=0.2,
        )
        if stream:
            for chunk in resp:
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                except Exception:
                    continue
        else:
            try:
                yield resp.choices[0].message.content or ""
            except Exception:
                yield ""

    def chat_text(self, messages: list[dict[str, str]], temperature: float = 0.25, top_p: float = 0.95,
                  max_tokens: int = 2200) -> str:
        # Use non-stream when stdout is not a tty (e.g., piped to head) to avoid BrokenPipe
        stream = sys.stdout.isatty()
        chunks: list[str] = []
        for piece in self.chat(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stream=stream):
            chunks.append(piece)
        return sanitize_output("".join(chunks))


def generate_legal_brief(prompt: str, client: NvidiaChatClient, temperature: float = 0.3, max_tokens: int = 1600) -> str:
    system_rules = (
        "You are a concise Indian constitutional law analyst. "
        "Do not reveal chain-of-thought or internal reasoning. "
        "Never output <think> blocks or similar. "
        "Avoid boilerplate, disclaimers, and repetition. "
        "Use only the requested section headings. "
        "Prefer 1–3 tight bullets per list; keep each bullet under ~25 words. "
        "Cite only the provided titles and years; no footnotes or URLs. "
        "Output plain text only — no Markdown, no code fences, no formatting syntax."
    )
    messages = [
        {"role": "system", "content": system_rules},
        {"role": "user", "content": prompt},
    ]
    return client.chat_text(messages=messages, temperature=temperature, max_tokens=max_tokens)
