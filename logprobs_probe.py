"""
v3 probe (optional, try after v1/v2 are working): pull top_logprobs at
generation time to approximate a soft distribution over your closed
vocabulary, rather than a hard argmax pick -- relevant if Stage 2 wants
probability-weighted slot embeddings instead of one-hot.

Uses Ollama's NATIVE API (/api/generate) directly, not the OpenAI-compat
/v1/chat/completions layer. As of mid-2026 the OpenAI-compat layer was
still dropping logprobs even when requested (ollama/ollama#16117) --
check whether that's been fixed before relying on the compat layer here.

Requires Ollama >= 0.12.11. Check with: ollama --version
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.6:27b"


def generate_with_logprobs(prompt: str, fmt: dict = None, top_logprobs: int = 10) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "options": {"temperature": 0},
    }
    if fmt is not None:
        payload["format"] = fmt

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # Quick smoke test against a generic prompt first, before pointing
    # this at the schema-constrained decomposer prompts.
    out = generate_with_logprobs("Why is the sky blue?")
    print(json.dumps(out.get("logprobs", []), indent=2)[:2000])
