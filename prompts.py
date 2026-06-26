"""
Few-shot prompt construction, SayCan-style: show the model 1-2 worked
examples of the task -> structured decomposition mapping, then ask it
to do the same for a new task. Add more worked examples here as you
build out your gold set -- SayCan's own ablations showed planning
accuracy is sensitive to the number of in-context examples.
"""

FEWSHOT_HEADER = """You are decomposing a hand-object manipulation task into a fixed sequence of 5 generic phases: approach, contact, grasp, open, release.

For each phase, fill in:
- target_part: which part of the object is involved (or "none")
- contact: the hand-object contact relation at this phase (or "none")
- motion: the motion happening at this phase (or "none")

Use ONLY the vocabulary you've been given. If a phase doesn't apply meaningfully to this task, use "none" for its slots rather than guessing.

Example:

Task: open the box

1. approach - target_part: lid - contact: none - motion: reach
2. contact - target_part: lid_edge - contact: fingertip_touch - motion: none
3. grasp - target_part: lid_edge - contact: index_thumb_to_lid_edge - motion: none
4. open - target_part: lid_edge - contact: index_thumb_to_lid_edge - motion: object_part_rotation
5. release - target_part: lid - contact: none - motion: release

Now do the same for the new task.
"""


def build_v0_prompt(task_text: str) -> str:
    """Free-text, no schema. For the v0 sanity check only."""
    return f"{FEWSHOT_HEADER}\nTask: {task_text}\n\n"


def build_v1_prompt(task_text: str) -> str:
    """Used alongside JSON-schema constrained decoding (v1)."""
    return (
        f"{FEWSHOT_HEADER}\n"
        f"Task: {task_text}\n\n"
        f"Respond with the 5-phase decomposition as JSON matching the schema."
    )


def build_v2_step_prompt(task_text: str, committed_rows: list, next_phase: str) -> str:
    """Sequential per-row prompt (v2): decide one phase at a time,
    conditioning on phases already committed -- closer to SayCan's
    actual iterative 'I would: 1. X, 2. ___' pattern."""
    if committed_rows:
        committed_str = "\n".join(
            f"{i + 1}. {r['phase']} - target_part: {r['target_part']} - "
            f"contact: {r['contact']} - motion: {r['motion']}"
            for i, r in enumerate(committed_rows)
        )
        committed_block = f"\nDecided so far:\n{committed_str}\n"
    else:
        committed_block = ""

    return (
        f"{FEWSHOT_HEADER}\n"
        f"Task: {task_text}\n"
        f"{committed_block}\n"
        f"Now decide ONLY the '{next_phase}' phase. "
        f"Respond with JSON for target_part, contact, and motion for this phase."
    )
