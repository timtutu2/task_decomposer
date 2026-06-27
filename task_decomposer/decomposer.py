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

import argparse
import json
from json import JSONDecodeError

from schema import (
    build_full_sequence_schema,
    build_phase_plan_schema,
    build_single_row_schema,
)
from prompts import (
    build_phase_plan_prompt,
    build_v0_prompt,
    build_v1_prompt,
    build_v2_step_prompt,
)

MODEL = "qwen3:8b"
DEFAULT_TASK = "pick the object and rotate the object"


def _ollama():
    import ollama

    return ollama


def _ollama_value(resp, key: str, default=None):
    if hasattr(resp, key):
        return getattr(resp, key)
    try:
        return resp[key]
    except (KeyError, TypeError):
        return default


def _ollama_keys(resp) -> str:
    if hasattr(resp, "keys"):
        return ", ".join(sorted(resp.keys()))
    if hasattr(resp, "model_dump"):
        return ", ".join(sorted(resp.model_dump().keys()))
    return type(resp).__name__


def _generate_constrained_json(prompt: str, schema: dict, model: str = MODEL) -> dict:
    """Call Ollama with schema-constrained output and parse the JSON body.

    Qwen reasoning models can put their reasoning in a separate field on newer
    Ollama versions. Passing think=False keeps the structured JSON in
    ``response`` when the installed client supports it; older clients are
    retried without that argument.
    """
    kwargs = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "options": {"temperature": 0},
    }
    try:
        resp = _ollama().generate(**kwargs, think=False)
    except TypeError:
        resp = _ollama().generate(**kwargs)

    text = _ollama_value(resp, "response", "")
    if not text.strip():
        response_keys = _ollama_keys(resp)
        thinking = _ollama_value(resp, "thinking")
        thinking_note = " Ollama returned a separate 'thinking' field." if thinking else ""
        raise ValueError(
            "Ollama returned an empty structured response. "
            f"Response keys: {response_keys}.{thinking_note}"
        )

    try:
        return json.loads(text)
    except JSONDecodeError as exc:
        preview = text[:500].replace("\n", "\\n")
        raise ValueError(
            "Ollama did not return valid JSON for the constrained call. "
            f"First 500 chars: {preview!r}"
        ) from exc


def run_v0(task_text: str, model: str = MODEL) -> str:
    """Free-text generation. Returns raw text for manual inspection --
    no parsing, this is just checking whether the model understands the
    task semantically before you add any scaffolding."""
    resp = _ollama().generate(
        model=model,
        prompt=build_v0_prompt(task_text),
        options={"temperature": 0},
    )
    return resp["response"]


def run_v1(task_text: str, model: str = MODEL) -> dict:
    """Single-shot, all variable-length rows in one constrained call.
    Guaranteed-valid output: every slot value is from your closed vocab."""
    return _generate_constrained_json(
        prompt=build_v1_prompt(task_text),
        schema=build_full_sequence_schema(),
        model=model,
    )


def run_v2(task_text: str, model: str = MODEL) -> dict:
    """Plan free-form phases, then generate each row with slot constraints.

    Each phase's prompt includes the phases already committed, so later
    decisions can condition on earlier ones."""
    plan = _generate_constrained_json(
        prompt=build_phase_plan_prompt(task_text),
        schema=build_phase_plan_schema(),
        model=model,
    )
    committed_rows = []
    schema = build_single_row_schema()
    for phase in plan["phases"]:
        prompt = build_v2_step_prompt(task_text, committed_rows, phase)
        row = _generate_constrained_json(prompt=prompt, schema=schema, model=model)
        row["phase"] = phase
        committed_rows.append(row)
    return {"steps": committed_rows}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 decomposer for a hand-object task."
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=DEFAULT_TASK,
        help=f'Task text to decompose. Defaults to "{DEFAULT_TASK}".',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task = args.task

    print(f"Task: {task}\n")

    print("=== v0 (free text) ===")
    print(run_v0(task))

    print("\n=== v1 (single-shot, constrained) ===")
    print(json.dumps(run_v1(task), indent=2))

    print("\n=== v2 (sequential, constrained) ===")
    print(json.dumps(run_v2(task), indent=2))
