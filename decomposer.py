"""
Stage 1 decomposer -- Regime A (frozen LLM, no fine-tuning).

Three modes, meant to be tried in order:
  v0 - free text, no constraints            (semantic sanity check)
  v1 - single-shot, JSON-schema constrained  (first real baseline)
  v2 - sequential, constrained per row       (closer to true SayCan)

Requires:
    pip install ollama
    ollama pull qwen3.6:27b   (already done before running this)
"""

import json
import ollama

from vocab import PHASES
from schema import build_full_sequence_schema, build_single_row_schema
from prompts import build_v0_prompt, build_v1_prompt, build_v2_step_prompt

MODEL = "qwen3.6:27b"


def run_v0(task_text: str, model: str = MODEL) -> str:
    """Free-text generation. Returns raw text for manual inspection --
    no parsing, this is just checking whether the model understands the
    task semantically before you add any scaffolding."""
    resp = ollama.generate(
        model=model,
        prompt=build_v0_prompt(task_text),
        options={"temperature": 0},
    )
    return resp["response"]


def run_v1(task_text: str, model: str = MODEL) -> dict:
    """Single-shot, all 5 rows in one call, JSON-schema constrained.
    Guaranteed-valid output: every slot value is from your closed vocab."""
    resp = ollama.generate(
        model=model,
        prompt=build_v1_prompt(task_text),
        format=build_full_sequence_schema(),
        options={"temperature": 0},
    )
    return json.loads(resp["response"])


def run_v2(task_text: str, model: str = MODEL) -> dict:
    """Sequential, one phase at a time, each call schema-constrained.
    Each phase's prompt includes the phases already committed, so later
    decisions can condition on earlier ones."""
    committed_rows = []
    schema = build_single_row_schema()
    for phase in PHASES:
        prompt = build_v2_step_prompt(task_text, committed_rows, phase)
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            format=schema,
            options={"temperature": 0},
        )
        row = json.loads(resp["response"])
        row["phase"] = phase
        committed_rows.append(row)
    return {"steps": committed_rows}


if __name__ == "__main__":
    task = "open the box"

    print("=== v0 (free text) ===")
    print(run_v0(task))

    print("\n=== v1 (single-shot, constrained) ===")
    print(json.dumps(run_v1(task), indent=2))

    print("\n=== v2 (sequential, constrained) ===")
    print(json.dumps(run_v2(task), indent=2))
