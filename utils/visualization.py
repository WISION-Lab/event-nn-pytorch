import cv2

from utils.image import as_uint8
from utils.misc import as_numpy


def draw_pose(
    image,
    joints,
    scores,
    joint_pairs=None,
    color=(255, 0, 0),
    threshold=0.0,
    thickness=2,
    radius=4,
):
    image = as_numpy(as_uint8(image)).transpose(1, 2, 0).copy()
    n = joints.shape[0]

    # Draw limbs.
    for i_1, i_2 in [] if joint_pairs is None else joint_pairs:
        if scores[i_1] > threshold and scores[i_2] > threshold:
            p_1 = tuple(int(z) for z in joints[i_1])
            p_2 = tuple(int(z) for z in joints[i_2])
            cv2.line(image, p_1, p_2, color=color, thickness=thickness)

    # Draw joints.
    for i in range(n):
        if scores[i] > threshold:
            p = tuple(int(z) for z in joints[i])
            cv2.circle(image, p, radius, color=color, thickness=-1)

    return image.transpose(2, 0, 1)
