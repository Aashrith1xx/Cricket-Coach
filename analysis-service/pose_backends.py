"""Pose estimation backends.

Separated from :mod:`pose_extraction` so the rest of the pipeline can be developed, tested
and validated without model weights present. The ONNX backend is the production path; the
replay backend loads previously-extracted keypoints and is what the test suite and the
calibration harness run against.

Licensing, which constrains what may be added here
--------------------------------------------------
RTMPose is top-down and needs a person detector. Pair it **only with RTMDet** - both are
Apache 2.0 via OpenMMLab. Do not add an Ultralytics YOLOv5/v8 detector: those are AGPL-3.0
and would require open-sourcing this application or buying a commercial licence. It is the
most convenient detector in most tutorials, so this is easy to do by accident.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

N_KEYPOINTS = 17

# RTMDet defaults, verified against the mmdeploy end2end export's own signature rather than
# assumed: input [1, 3, 320, 320], outputs dets [1, N, 5] and labels [1, N].
DETECTOR_INPUT_SIZE = (320, 320)
PERSON_CLASS_ID = 0
DETECTION_SCORE_THRESHOLD = 0.35

# RTMPose-m. Note the (width, height) ordering - mmpose configs state 192x256 as (w, h)
# while the ONNX input is [batch, 3, 256, 192] as (h, w). Transposing this silently produces
# plausible-looking but wrong keypoints, so it is verified at load time below.
POSE_INPUT_SIZE = (192, 256)

#: SimCC bins per pixel. Confirmed from the export: simcc_x is 384 = 192 * 2 and simcc_y is
#: 512 = 256 * 2.
SIMCC_SPLIT_RATIO = 2.0

# Normalisation is NOT baked into these graphs - the mmdeploy SDK applies it in its own
# pipeline. Feeding raw 0-255 pixels produces confident nonsense rather than an error, which
# is the worst kind of failure, so these constants matter.
DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)   # BGR order
DET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB order
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass
class Detection:
    bbox: np.ndarray  # xyxy
    score: float


class PoseBackend(ABC):
    """Produces per-frame keypoints and confidences for a single tracked person."""

    @abstractmethod
    def run(self, frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(xy, confidence, bboxes)``.

        ``xy`` is ``(n_frames, 17, 2)`` in pixels, ``confidence`` is ``(n_frames, 17)``,
        ``bboxes`` is ``(n_frames, 4)`` xyxy.

        Frames where nothing was detected must be filled with NaN and zero confidence -
        never with a plausible guess. :mod:`smoothing` decides whether a gap is bridgeable,
        and it can only do that if gaps are visible.
        """


# ---------------------------------------------------------------------------
# Geometry shared by top-down pose models
# ---------------------------------------------------------------------------


def bbox_to_center_scale(
    bbox: np.ndarray, *, padding: float = 1.25, aspect: float = POSE_INPUT_SIZE[0] / POSE_INPUT_SIZE[1]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an xyxy box to the centre/scale form top-down models expect.

    The box is expanded to the model's aspect ratio *before* padding, so the crop does not
    squash the subject - a squashed crop shifts every joint slightly and the error is
    systematic rather than noisy, which is the kind that survives averaging.
    """
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    w, h = x2 - x1, y2 - y1

    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect

    scale = np.array([w, h], dtype=np.float32) * padding
    return center, scale


def warp_matrix(
    center: np.ndarray, scale: np.ndarray, output_size: Tuple[int, int]
) -> np.ndarray:
    """Affine matrix mapping the source crop onto the model input.

    Rotation is deliberately unsupported. Rotating the crop would help an upside-down
    subject, but a cricket clip that needs it has failed the capture gate already, and
    supporting it would add an inverse-transform path that is easy to get subtly wrong.
    """
    out_w, out_h = output_size
    src_w, src_h = float(scale[0]), float(scale[1])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0] = center
    src[1] = center + np.array([0.0, -src_h * 0.5], dtype=np.float32)
    src[2] = center + np.array([src_w * 0.5, 0.0], dtype=np.float32)

    dst[0] = np.array([out_w * 0.5, out_h * 0.5], dtype=np.float32)
    dst[1] = np.array([out_w * 0.5, out_h * 0.5 - out_h * 0.5], dtype=np.float32)
    dst[2] = np.array([out_w * 0.5 + out_w * 0.5, out_h * 0.5], dtype=np.float32)

    import cv2

    return cv2.getAffineTransform(src, dst)


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, *, split_ratio: float = SIMCC_SPLIT_RATIO
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode RTMPose SimCC heads to keypoints in model-input coordinates.

    SimCC treats localisation as classification over discretised x and y bins, so decoding
    is an argmax per axis rather than a heatmap centroid.

    Confidence is the *smaller* of the two axis maxima. A joint the model is confident about
    horizontally but not vertically is not a confident joint, and taking the mean would hide
    exactly that case - which matters here because the failure mode we care about is a
    plausible-looking wrong joint, not a missing one.
    """
    x_locs = np.argmax(simcc_x, axis=-1).astype(np.float32)
    y_locs = np.argmax(simcc_y, axis=-1).astype(np.float32)

    x_conf = np.max(simcc_x, axis=-1)
    y_conf = np.max(simcc_y, axis=-1)
    confidence = np.minimum(x_conf, y_conf)

    keypoints = np.stack([x_locs, y_locs], axis=-1) / split_ratio
    return keypoints, confidence


def keypoints_to_image(
    keypoints: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Map keypoints from model-input space back to original image pixels."""
    out_w, out_h = output_size
    scale_factor = np.array([scale[0] / out_w, scale[1] / out_h], dtype=np.float32)
    return keypoints * scale_factor + center - scale * 0.5


# ---------------------------------------------------------------------------
# ONNX backend
# ---------------------------------------------------------------------------


class OnnxPoseBackend(PoseBackend):
    """RTMDet person detection followed by RTMPose keypoint estimation.

    Detection runs on every ``detect_every`` frames and boxes are interpolated between,
    because the player does not leave frame mid-shot and detection is a large share of total
    cost. On a fractional-vCPU instance that difference is minutes, not seconds.
    """

    def __init__(
        self,
        detector_path: str,
        pose_path: str,
        *,
        detect_every: int = 5,
        providers: Optional[List[str]] = None,
    ) -> None:
        import onnxruntime as ort

        for path, what in ((detector_path, "detector"), (pose_path, "pose model")):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{what} weights not found at {path}. Download the RTMDet and RTMPose "
                    f"ONNX exports from OpenMMLab (Apache 2.0) into analysis-service/models/. "
                    f"See README.md."
                )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Single-threaded: the deployment target is a fractional vCPU, where extra threads
        # contend rather than help.
        opts.intra_op_num_threads = 1

        self.det = ort.InferenceSession(
            detector_path, sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self.pose = ort.InferenceSession(
            pose_path, sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self.detect_every = detect_every

        # Read the real input geometry rather than trusting the constants. A silent
        # height/width transpose here yields keypoints that look plausible and are wrong.
        det_shape = self.det.get_inputs()[0].shape
        self.det_size = (int(det_shape[3]), int(det_shape[2]))  # (w, h)

        pose_shape = self.pose.get_inputs()[0].shape
        self.pose_size = (int(pose_shape[3]), int(pose_shape[2]))  # (w, h)
        if self.pose_size != POSE_INPUT_SIZE:
            log.warning(
                "Pose model expects %s (w, h), module default is %s - using the model's",
                self.pose_size,
                POSE_INPUT_SIZE,
            )

        simcc_x_bins = self.pose.get_outputs()[0].shape[-1]
        self.split_ratio = float(simcc_x_bins) / self.pose_size[0]

    def _detect(
        self, frame: np.ndarray, previous: Optional[np.ndarray] = None
    ) -> Optional[Detection]:
        import cv2

        h, w = frame.shape[:2]
        in_w, in_h = self.det_size
        # Letterbox, preserving aspect ratio: squashing the frame to a square would distort
        # every box and, downstream, every crop the pose model sees.
        ratio = min(in_w / w, in_h / h)
        resized = cv2.resize(frame, (int(w * ratio), int(h * ratio)))

        padded = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized

        normalised = (padded.astype(np.float32) - DET_MEAN) / DET_STD
        blob = normalised.transpose(2, 0, 1)[None]
        outputs = self.det.run(None, {self.det.get_inputs()[0].name: blob})

        dets, labels = outputs[0], outputs[1]
        dets, labels = np.squeeze(dets, 0), np.squeeze(labels, 0)

        people = [
            Detection(bbox=d[:4] / ratio, score=float(d[4]))
            for d, lab in zip(dets, labels)
            if int(lab) == PERSON_CLASS_ID and d[4] >= DETECTION_SCORE_THRESHOLD
        ]
        if not people:
            return None

        def area(d: Detection) -> float:
            return (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])

        # Prefer continuity with the previous frame over raw size. Picking the largest box
        # every frame lets the tracker jump to a net partner, a bowler or an umpire the
        # moment they step nearer the camera, and the resulting skeleton is a blend of two
        # people rather than an obviously broken one.
        if previous is not None and np.isfinite(previous).all():
            prev_centre = np.array(
                [(previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2]
            )
            prev_area = max((previous[2] - previous[0]) * (previous[3] - previous[1]), 1.0)

            def continuity(d: Detection) -> float:
                centre = np.array(
                    [(d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2]
                )
                drift = float(np.linalg.norm(centre - prev_centre)) / max(
                    previous[3] - previous[1], 1.0
                )
                size_change = abs(area(d) - prev_area) / prev_area
                return drift + size_change

            best = min(people, key=continuity)
            # Only accept the continuous choice if it really is the same subject; a large
            # jump means the original person left frame, so fall back to the largest.
            if continuity(best) < 1.0:
                return best

        return max(people, key=area)

    def _estimate(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import cv2

        aspect = self.pose_size[0] / self.pose_size[1]
        center, scale = bbox_to_center_scale(bbox, aspect=aspect)
        matrix = warp_matrix(center, scale, self.pose_size)
        crop = cv2.warpAffine(frame, matrix, self.pose_size, flags=cv2.INTER_LINEAR)

        # RTMPose was trained on RGB; the frame arrives BGR from OpenCV.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        normalised = (rgb.astype(np.float32) - POSE_MEAN) / POSE_STD
        blob = normalised.transpose(2, 0, 1)[None]

        simcc_x, simcc_y = self.pose.run(None, {self.pose.get_inputs()[0].name: blob})

        keypoints, confidence = decode_simcc(
            np.squeeze(simcc_x, 0), np.squeeze(simcc_y, 0), split_ratio=self.split_ratio
        )
        return keypoints_to_image(keypoints, center, scale, self.pose_size), confidence

    def process_frame(
        self, frame: np.ndarray, index: int, state: dict
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Detect and pose a single frame, carrying tracker state between calls.

        Split out from :meth:`run` so the same code can drive a streaming pass that never
        holds more than one frame in memory - which is what lets pose run at the source
        resolution instead of a downscaled copy.
        """
        last_box = state.get("box")
        if index % self.detect_every == 0 or last_box is None:
            det = self._detect(frame, previous=last_box)
            if det is not None:
                last_box = det.bbox
                state["box"] = last_box

        if last_box is None:
            return (
                np.full((N_KEYPOINTS, 2), np.nan, dtype=np.float32),
                np.zeros(N_KEYPOINTS, dtype=np.float32),
                None,
            )

        try:
            kp, kc = self._estimate(frame, last_box)
            return kp, kc, last_box
        except Exception:  # noqa: BLE001
            log.warning("Pose estimation failed on frame %d", index, exc_info=True)
            return (
                np.full((N_KEYPOINTS, 2), np.nan, dtype=np.float32),
                np.zeros(N_KEYPOINTS, dtype=np.float32),
                last_box,
            )

    def run(self, frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(frames)
        xy = np.full((n, N_KEYPOINTS, 2), np.nan, dtype=np.float32)
        conf = np.zeros((n, N_KEYPOINTS), dtype=np.float32)
        boxes = np.full((n, 4), np.nan, dtype=np.float32)

        state: dict = {}
        for i, frame in enumerate(frames):
            kp, kc, box = self.process_frame(frame, i, state)
            xy[i], conf[i] = kp, kc
            if box is not None:
                boxes[i] = box

        return xy, conf, boxes


# ---------------------------------------------------------------------------
# Replay backend
# ---------------------------------------------------------------------------


class ReplayPoseBackend(PoseBackend):
    """Replays keypoints from a JSON file instead of running inference.

    Two uses. It lets the pipeline be tested and the calibration harness be run without
    model weights, and - because keypoints are retained permanently while video mostly is
    not - it is how a stored clip gets re-analysed after a scoring change without needing
    the original footage back.
    """

    def __init__(self, keypoints_path: str) -> None:
        data = json.loads(Path(keypoints_path).read_text(encoding="utf-8"))
        self.xy = np.asarray(data["xy"], dtype=np.float32)
        self.conf = np.asarray(data["confidence"], dtype=np.float32)
        self.boxes = np.asarray(
            data.get("bboxes", np.full((len(self.xy), 4), np.nan)), dtype=np.float32
        )

        if self.xy.shape[1:] != (N_KEYPOINTS, 2):
            raise ValueError(f"expected (n, {N_KEYPOINTS}, 2), got {self.xy.shape}")

    def run(self, frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.xy, self.conf, self.boxes


class ArrayPoseBackend(PoseBackend):
    """Wraps in-memory arrays. Used by tests to drive the pipeline with synthetic motion."""

    def __init__(self, xy: np.ndarray, confidence: np.ndarray) -> None:
        self.xy = np.asarray(xy, dtype=np.float32)
        self.conf = np.asarray(confidence, dtype=np.float32)
        if self.xy.shape[1:] != (N_KEYPOINTS, 2):
            raise ValueError(f"expected (n, {N_KEYPOINTS}, 2), got {self.xy.shape}")
        if self.conf.shape != self.xy.shape[:2]:
            raise ValueError("confidence shape must match xy")

    def run(self, frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.xy)
        return self.xy, self.conf, np.full((n, 4), np.nan, dtype=np.float32)
