"""Reduce net and fence interference before pose estimation.

Nets are where cricketers practise, so "shot through mesh" is the normal case rather than an
edge case. Mesh hurts pose estimation twice: the cords occlude parts of the body outright,
and their high-contrast edges give the model spurious structure to latch onto.

The capture protocol already requires a **static camera**, and that is what makes this
tractable: anything stationary across the clip is scene, not player. A temporal median
therefore isolates the net, and the resulting mask can be inpainted out of every frame.

This is a genuine image-restoration step, not a confidence fudge. Whether it helps is an
empirical question - :func:`measure_benefit` exists to answer it on real footage rather than
letting the idea stand on plausibility.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Structuring-element size for the black-hat transform that isolates thin dark cords.
#: Must exceed the cord width but stay well under limb width, or the mask eats the player.
CORD_KERNEL = 5

#: Black-hat response above which a pixel is called mesh. Lower catches more cord and more
#: player; this is deliberately conservative, since removing body detail costs more than
#: leaving some net behind.
MESH_RESPONSE_THRESHOLD = 22

#: Dilation applied to the mask so inpainting covers each cord's soft edges too.
MASK_DILATION = 2

#: Inpainting radius, in pixels. Small: we are filling thin cords, not holes.
INPAINT_RADIUS = 3

#: Frames sampled to build the static-scene estimate.
MEDIAN_SAMPLE_FRAMES = 15


def static_scene(frames: Sequence[np.ndarray], *, sample: int = MEDIAN_SAMPLE_FRAMES) -> np.ndarray:
    """Temporal median of sampled frames: everything that does not move.

    With a static camera this is the net, the ground and the background, with the player
    averaged away.
    """
    step = max(1, len(frames) // sample)
    stack = np.stack([frames[i] for i in range(0, len(frames), step)][:sample])
    return np.median(stack, axis=0).astype(np.uint8)


def build_mesh_mask(frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Binary mask of net cords, or ``None`` if no mesh-like structure is found.

    Uses a black-hat transform on the static scene, which responds to dark structures
    thinner than the kernel - exactly what a net cord is, and not what a limb, pad or bat is.
    Deriving it from the static scene rather than per-frame means the player can never
    contribute to the mask, however dark their kit.
    """
    import cv2

    if len(frames) < 3:
        return None

    scene = static_scene(frames)
    grey = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CORD_KERNEL, CORD_KERNEL))
    blackhat = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)

    mask = (blackhat > MESH_RESPONSE_THRESHOLD).astype(np.uint8) * 255
    if MASK_DILATION:
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_RECT, (MASK_DILATION, MASK_DILATION))
        )

    coverage = float(np.count_nonzero(mask)) / mask.size
    # Too little is not mesh; too much means the transform latched onto scene texture and
    # inpainting would smear the whole frame.
    if coverage < 0.01 or coverage > 0.45:
        log.info("No usable mesh mask (coverage %.3f)", coverage)
        return None

    log.info("Mesh mask covers %.1f%% of the frame", coverage * 100)
    return mask


def remove_mesh(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint masked cords out of a single frame."""
    import cv2

    return cv2.inpaint(frame, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def measure_benefit(
    frames: Sequence[np.ndarray], backend, *, sample_every: int = 8
) -> Tuple[float, float, Optional[np.ndarray]]:
    """Compare tracking confidence with and without mesh removal.

    Returns ``(baseline_confidence, cleaned_confidence, mask)``.

    Exists so the pipeline can decide per clip on evidence rather than on the assumption that
    inpainting must help. It can easily not: on fine mesh the cords carry little of the body,
    and blurring the player to remove them is a net loss.
    """
    mask = build_mesh_mask(frames)
    if mask is None:
        return 0.0, 0.0, None

    indices = list(range(0, len(frames), sample_every))
    baseline, cleaned = [], []

    for i in indices:
        state_a: dict = {}
        _, conf_a, _ = backend.process_frame(frames[i], 0, state_a)
        baseline.append(float(np.mean(conf_a)))

        state_b: dict = {}
        _, conf_b, _ = backend.process_frame(remove_mesh(frames[i], mask), 0, state_b)
        cleaned.append(float(np.mean(conf_b)))

    return float(np.mean(baseline)), float(np.mean(cleaned)), mask
