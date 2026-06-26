"""
Closed-vocabulary tables for the Stage 1 step decomposer.

IMPORTANT: These are PLACEHOLDER vocabularies, seeded only from the single
"open the box" worked example. Before running this at scale, replace
TARGET_PART / CONTACT / MOTION with the tables derived from HanDyVQA's
Parts / Action / Process answer-frequency distributions, per the dataset
construction plan (Step 1: build vocab tables empirically from HanDyVQA's
top-N verb/part frequency stats, not by hand).

PHASES is the fixed 5-slot template (approach/contact/grasp/open/release).
This assumes every task gets exactly 5 rows with "none" filling slots that
don't apply. If you hit a task that genuinely needs repeated phases
(e.g. a regrasp mid-task), this fixed-cardinality assumption breaks and
you'll need a variable-length design instead -- flag it if you see it.
"""

PHASES = ["approach", "contact", "grasp", "open", "release"]

TARGET_PART = [
    "lid",
    "lid_edge",
    "handle",
    "body",
    "base",
    "none",
]

CONTACT = [
    "none",
    "fingertip_touch",
    "index_thumb_to_lid_edge",
    "palm_to_body",
    "full_grasp",
]

MOTION = [
    "none",
    "reach",
    "object_part_rotation",
    "release",
    "lift",
    "slide",
]
