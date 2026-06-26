"""
Builds the JSON schema passed as `format=` to ollama.generate()/chat().
Ollama grammar-constrains output to match this schema, so target_part /
contact / motion can only ever be values from your closed vocabulary --
no parsing, no hallucinated slot values.
"""

from vocab import PHASES, TARGET_PART, CONTACT, MOTION


def build_full_sequence_schema() -> dict:
    """Schema for generating all PHASES rows in a single call (v1)."""
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": len(PHASES),
                "maxItems": len(PHASES),
                "items": {
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string", "enum": PHASES},
                        "target_part": {"type": "string", "enum": TARGET_PART},
                        "contact": {"type": "string", "enum": CONTACT},
                        "motion": {"type": "string", "enum": MOTION},
                    },
                    "required": ["phase", "target_part", "contact", "motion"],
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
            "contact": {"type": "string", "enum": CONTACT},
            "motion": {"type": "string", "enum": MOTION},
        },
        "required": ["target_part", "contact", "motion"],
        "additionalProperties": False,
    }
