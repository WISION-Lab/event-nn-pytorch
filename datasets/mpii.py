# This file doesn't actually define the MPII dataset, it just contains
# some metadata useful for working with models trained on MPII.

JOINT_NAMES = [
    "right ankle",
    "right knee",
    "right hip",
    "left hip",
    "left knee",
    "left ankle",
    "pelvis",
    "thorax",
    "upper neck",
    "head top",
    "right wrist",
    "right elbow",
    "right shoulder",
    "left shoulder",
    "left elbow",
    "left wrist",
]

# For visualization purposes
JOINT_PAIRS = [
    ("head top", "upper neck"),
    ("upper neck", "thorax"),
    ("thorax", "right shoulder"),
    ("thorax", "left shoulder"),
    ("right shoulder", "right elbow"),
    ("left shoulder", "left elbow"),
    ("right elbow", "right wrist"),
    ("left elbow", "left wrist"),
    ("thorax", "pelvis"),
    ("pelvis", "right hip"),
    ("pelvis", "left hip"),
    ("right hip", "right knee"),
    ("left hip", "left knee"),
    ("right knee", "right ankle"),
    ("left knee", "left ankle"),
]
JOINT_PAIRS = [
    (JOINT_NAMES.index(s_1), JOINT_NAMES.index(s_2)) for s_1, s_2 in JOINT_PAIRS
]

N_JOINTS = 16
