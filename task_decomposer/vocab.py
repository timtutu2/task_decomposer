"""
Closed-vocabulary slot values for the "pick and rotate an object" task.

Phase names are intentionally not defined here: the model chooses the number
and names of phases for each task. These small slot tables are suitable for
this task, but should still be replaced with dataset-derived tables before
the decomposer is used for a broad range of tasks.
"""

TARGET_PART = [
    "object",
    "object_surface",
    "none",
]

CONTACT = [
    "none",
    "fingertip_touch",
    "full_grasp",
]

MOTION = [
    "none",
    "reach",
    "lift",
    "object_rotation",
]
