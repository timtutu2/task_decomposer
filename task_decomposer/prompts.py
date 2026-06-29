"""
Few-shot prompt construction, SayCan-style: show the model 1-2 worked
examples of the task -> structured decomposition mapping, then ask it
to do the same for a new task. Add more worked examples here as you
build out your gold set -- SayCan's own ablations showed planning
accuracy is sensitive to the number of in-context examples.
"""

FEWSHOT_HEADER = """You are decomposing a hand-object manipulation task into a sequence of task-specific phases.

Choose as many phases as the task actually needs. Use concise phase names and
do not add empty padding phases.

For each phase, fill in:
- target_part: which part of the object is involved (or "none")
- hand_part: which part of the hand is contacting the object (or "none")
- motion: the motion happening at this phase (or "none")

Use ONLY the vocabulary you've been given. If a phase doesn't apply meaningfully to this task, use "none" for its slots rather than guessing.

Example:

Task: pick the object and rotate the object

1. approach - target_part: object - hand_part: none - motion: reach
2. touch - target_part: object_surface - hand_part: fingertip - motion: none
3. grasp - target_part: object - hand_part: power_grasp - motion: grasp_close
4. pick - target_part: object - hand_part: power_grasp - motion: lift
5. rotate - target_part: object - hand_part: power_grasp - motion: object_rotation

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
        f"Respond with the complete variable-length decomposition as JSON matching the schema."
    )


def build_phase_plan_prompt(task_text: str) -> str:
    """Choose the task-specific phase names used by sequential v2."""
    return (
        "Choose the ordered phases needed to perform this hand-object task. "
        "Use as many phases as the task actually requires — one phase if it "
        "describes a single ongoing action, more if the task has distinct "
        "sequential steps. Use concise phase names and no padding phases.\n\n"
        "Examples:\n\n"
        'Task: the hand is rotating the object with fingertips\n'
        '{"phases": ["rotate"]}\n\n'
        'Task: the hand has pinched the carton and is rotating it via wrist rotation\n'
        '{"phases": ["rotate"]}\n\n'
        'Task: pick the object and place it on the shelf\n'
        '{"phases": ["reach", "grasp", "lift", "place"]}\n\n'
        'Task: open the drawer and take out the bottle\n'
        '{"phases": ["reach", "pull_drawer", "grasp_bottle", "lift_bottle", "retract"]}\n\n'
        f"Task: {task_text}\n"
        'Respond as JSON with a single "phases" array.'
    )


def build_v2_step_prompt(task_text: str, committed_rows: list, next_phase: str) -> str:
    """Sequential per-row prompt (v2): decide one phase at a time,
    conditioning on phases already committed -- closer to SayCan's
    actual iterative 'I would: 1. X, 2. ___' pattern."""
    if committed_rows:
        committed_str = "\n".join(
            f"{i + 1}. {r['phase']} - target_part: {r['target_part']} - "
            f"hand_part: {r['hand_part']} - motion: {r['motion']}"
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
        f"Respond with JSON for target_part, hand_part, and motion for this phase. "
        f"hand_part is the region of the hand contacting the object (e.g. fingertip, power_grasp), "
        f"NOT the joint driving the motion."
    )
