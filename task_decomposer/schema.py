"""
Builds the JSON schema passed as `format=` to ollama.generate()/chat().
Ollama grammar-constrains output to match this schema, so target_part /
hand_part / motion can only ever be values from your closed vocabulary --
no parsing, no hallucinated slot values.
"""

from vocab import TARGET_PART, HAND_PART, MOTION

MAX_STEPS = 20


def build_full_sequence_schema() -> dict:
    """Schema for a variable-length sequence generated in one call (v1)."""
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_STEPS,
                "items": {
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "string",
                            "minLength": 1,
                            "description": "A concise task-specific phase name.",
                        },
                        "target_part": {
                            "type": "string",
                            "enum": TARGET_PART,
                            "description": "The object or object region being acted on.",
                        },
                        "hand_part": {
                            "type": "string",
                            "enum": HAND_PART,
                            "description": "Which part of the hand is contacting the object.",
                        },
                        "motion": {
                            "type": "string",
                            "enum": MOTION,
                            "description": "The hand or object motion in this phase.",
                        },
                    },
                    "required": ["phase", "target_part", "hand_part", "motion"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


def build_single_row_schema() -> dict:
    """Schema for generating one row at a time (v2, sequential)."""
    return {
        "type": "object",
        "properties": {
            "target_part": {"type": "string", "enum": TARGET_PART},
            "hand_part": {"type": "string", "enum": HAND_PART},
            "motion": {"type": "string", "enum": MOTION},
        },
        "required": ["target_part", "hand_part", "motion"],
        "additionalProperties": False,
    }


def build_phase_plan_schema() -> dict:
    """Schema for choosing a variable-length phase plan before v2 rows."""
    return {
        "type": "object",
        "properties": {
            "phases": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_STEPS,
                "items": {"type": "string", "minLength": 1},
            }
        },
        "required": ["phases"],
        "additionalProperties": False,
    }
