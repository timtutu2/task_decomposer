"""
Closed-vocabulary slot values for hand-object manipulation tasks.

Phase names are intentionally not defined here: the model chooses the number
and names of phases for each task. Expand these tables as new task types are
added -- the StructuredTaskEncoder uses hash buckets so adding values here
does not change model shapes.
"""

TARGET_PART = [
    "none",
    # whole object
    "object",
    "object_surface",
    "body",
    "outer_surface",
    "inner_surface",
    # functional sub-parts
    "handle",
    "grip_zone",
    "rim",
    "edge",
    "corner",
    "lid",
    "cap",
    "base",
    "foot",
    "drawer",
    "door",
    "flap",
    "hinge",
    "button",
    "trigger",
    "knob",
    "dial",
    "lever",
    "switch",
    "spout",
    "nozzle",
    "neck",
    "stem",
    "blade",
    "tip",
    # faces / orientations
    "top_face",
    "bottom_face",
    "front_face",
    "back_face",
    "side_face",
    "left_face",
    "right_face",
]

HAND_PART = [
    "none",
    # individual fingertips
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
    "fingertip",            # unspecified fingertip(s)
    "fingertip_touch",      # legacy: light fingertip contact
    # finger pads (distal phalanx pad)
    "thumb_pad",
    "index_pad",
    "finger_pads",          # multiple finger pads together
    # mid-finger / proximal regions
    "finger_side",          # lateral edge of a finger
    "index_lateral",        # lateral (radial) side of index — used in lateral pinch
    # palm regions
    "palm",
    "palm_center",
    "palm_edge",
    "thenar",               # thumb base / thenar eminence
    "hypothenar",           # pinky base / hypothenar eminence
    # wrist
    "wrist",
    # ---- grasp configurations ----
    "pinch_2f",             # 2-finger pinch: thumb tip + index tip
    "pinch_3f",             # tripod pinch: thumb + index + middle
    "lateral_pinch",        # key/chuck pinch: thumb pad vs. index lateral
    "precision_grip",       # pad-to-pad 3-finger precision
    "power_grasp",          # full-hand cylindrical/spherical wrap
    "full_grasp",           # legacy alias for power_grasp
    "hook_grip",            # fingers hooked around edge, no thumb
    "palmar_grasp",         # object resting in cupped palm
    "lumbrical_grasp",      # fingers extended, object resting on metacarpals
]

MOTION = [
    "none",
    # transport / approach
    "reach",                # hand moves toward target
    "retract",              # hand moves away after action
    "approach",             # slow controlled close-in before contact
    "withdraw",             # controlled pull-back after release
    # vertical
    "lift",                 # raise object upward
    "lower",                # bring object down toward surface
    "place",                # set object onto a surface
    # horizontal / planar
    "push",                 # push object away from body
    "pull",                 # pull object toward body
    "slide",                # translate object along a surface
    "drag",                 # drag with maintained contact
    "sweep",                # broad lateral sweep
    # rotational
    "object_rotation",      # generic object rotation (legacy)
    "rotate_cw",            # clockwise rotation (top-down view)
    "rotate_ccw",           # counter-clockwise rotation
    "wrist_rotation",       # forearm/wrist pronation-supination
    "flip",                 # 180-degree flip about a horizontal axis
    "tilt",                 # small angle tilt
    "pivot",                # in-plane pivot about a contact point
    "spin",                 # fast rotation about object's own axis
    "swing",                # arc motion about a proximal joint
    "stir",                 # circular stirring motion
    # in-hand / finger
    "grasp_close",          # fingers closing into grasp
    "grasp_open",           # fingers opening to release
    "pinch_close",          # pinch fingers closing
    "pinch_open",           # pinch fingers opening
    "reposition",           # re-adjust hand pose without losing contact
    "roll",                 # roll object between fingers / palm
    # contact actions
    "press",                # apply normal force downward
    "poke",                 # single-finger jab
    "squeeze",              # bilateral compression
    "twist",                # torsion about object long axis
    "shake",                # rapid back-and-forth
    "tap",                  # brief repeated press
    "scrape",               # edge contact drag
    # end-effector specific
    "pour",                 # tilt to decant contents
    "scoop",                # underhand cupping motion
    "insert",               # push into a cavity
    "extract",              # pull out of a cavity
]
