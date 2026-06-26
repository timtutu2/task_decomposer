"""
Per-slot accuracy harness against a gold eval set.

Gold set format (JSON list) -- see example_tasks.json:
[
  {
    "task": "open the box",
    "gold_steps": [
      {"phase": "approach", "target_part": "lid", "contact": "none", "motion": "reach"},
      ...
    ]
  },
  ...
]

This is the number that tells you whether Regime A (this whole script)
is good enough, or whether you need to move to Regime B (fine-tuning).
Run it every time you change the prompt, schema, or vocab -- not just
once at the end.
"""

import json
from decomposer import run_v1, run_v2

SLOTS = ["target_part", "contact", "motion"]


def load_gold(path: str):
    with open(path) as f:
        return json.load(f)


def score_one(pred_steps, gold_steps):
    """Per-slot accuracy for a single task. Assumes matching phase
    order/count between prediction and gold (true while PHASES is fixed
    at 5; revisit if you move to variable-length rows)."""
    correct = {s: 0 for s in SLOTS}
    total = {s: 0 for s in SLOTS}
    for pred_row, gold_row in zip(pred_steps, gold_steps):
        for s in SLOTS:
            total[s] += 1
            if pred_row.get(s) == gold_row.get(s):
                correct[s] += 1
    return correct, total


def evaluate(gold_path: str, mode: str = "v1"):
    runner = run_v1 if mode == "v1" else run_v2
    gold = load_gold(gold_path)

    agg_correct = {s: 0 for s in SLOTS}
    agg_total = {s: 0 for s in SLOTS}

    for item in gold:
        pred = runner(item["task"])
        correct, total = score_one(pred["steps"], item["gold_steps"])
        for s in SLOTS:
            agg_correct[s] += correct[s]
            agg_total[s] += total[s]

    print(f"Mode: {mode}")
    for s in SLOTS:
        acc = agg_correct[s] / agg_total[s] if agg_total[s] else 0.0
        print(f"  {s}: {acc:.1%} ({agg_correct[s]}/{agg_total[s]})")


if __name__ == "__main__":
    evaluate("example_tasks.json", mode="v1")
    evaluate("example_tasks.json", mode="v2")
