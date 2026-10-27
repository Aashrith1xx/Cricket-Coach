"""Post-analysis retention: what is kept, for how long, and under whose consent.

**Status: specified stub.** Policy is real; the storage calls are TODO.

Two forces pull in opposite directions here.

*Keep everything*: every analysed clip is a weakly-labelled training example, so retention
turns the app into a data-collection mechanism for a future model at no extra effort.

*Keep nothing*: video is expensive. At ~15 MB per 3-second 120 fps clip and 5 clips per
session, Appwrite's 2 GB free tier holds roughly 25 sessions - exhausted during testing.

The resolution is that **the video is not the valuable part**. Extracted keypoints are about
1000x smaller (~150 KB vs ~15 MB, far less gzipped) and are what model training actually
consumes. So keypoints are retained permanently and cheaply; the original video is kept only
where the player has consented, and is transcoded to a compressed proxy after successful
analysis.

Consent is not implied by upload. ``consentForTraining`` is an explicit opt-in captured at
signup and editable afterwards. Without it, the original is discarded once analysis
completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RetentionState(str, Enum):
    ORIGINAL = "original"
    PROXY_ONLY = "proxy_only"
    KEYPOINTS_ONLY = "keypoints_only"


#: Proxy transcode settings. 720p is enough to review a skeleton overlay, which is all the
#: retained video is for once metrics are extracted.
PROXY_MAX_HEIGHT = 720
PROXY_CRF = 28

#: Days an original is kept for a consenting user before being transcoded to a proxy. Short,
#: because the analysis is already complete and the keypoints are already stored.
ORIGINAL_RETENTION_DAYS = 30


@dataclass
class RetentionOutcome:
    state: RetentionState
    proxy_path: Optional[str] = None
    bytes_freed: int = 0
    note: str = ""


def target_state(consent_for_training: bool) -> RetentionState:
    """Retention state a clip should end up in. Implemented - it is pure policy."""
    return (
        RetentionState.PROXY_ONLY if consent_for_training else RetentionState.KEYPOINTS_ONLY
    )


def apply(
    *,
    clip_id: str,
    video_path: str,
    consent_for_training: bool,
    storage: Optional["RetentionStorage"] = None,
) -> RetentionOutcome:
    """Apply the retention policy after a successful analysis.

    ``storage`` is injected rather than imported so this module has no Appwrite dependency
    and the policy stays testable. With no storage configured the policy is *decided and
    returned* but nothing is deleted - the safe direction, since an accidental delete is
    unrecoverable whereas an accidental keep costs pennies.

    Ordering rule for any implementation of ``RetentionStorage``: keypoints, metrics and
    capture metadata must be persisted and confirmed **before** anything is deleted. Losing
    the cheap, valuable artefact while keeping the expensive one would be the worst possible
    outcome, and it is the natural failure mode if deletion runs first.

    Never called on a failed or rejected analysis: the player may want to retry, and a
    rejected clip is often the one worth keeping to diagnose a capture problem.
    """
    state = target_state(consent_for_training)

    if storage is None:
        return RetentionOutcome(
            state=state,
            note=(
                "no storage configured - retention policy decided but not applied. "
                "Nothing was deleted."
            ),
        )

    if not storage.keypoints_persisted(clip_id):
        return RetentionOutcome(
            state=RetentionState.ORIGINAL,
            note=(
                "keypoints not confirmed in storage; keeping the original. Deleting video "
                "before the keypoints are safe would destroy the only durable artefact."
            ),
        )

    if state is RetentionState.PROXY_ONLY:
        proxy_path = storage.transcode_proxy(
            clip_id, video_path, max_height=PROXY_MAX_HEIGHT, crf=PROXY_CRF
        )
        freed = storage.delete_original(clip_id)
        return RetentionOutcome(state=state, proxy_path=proxy_path, bytes_freed=freed)

    freed = storage.delete_original(clip_id) + storage.delete_proxy(clip_id)
    return RetentionOutcome(state=state, bytes_freed=freed)


class RetentionStorage:
    """Interface the worker's storage layer must satisfy. See :func:`apply` for ordering."""

    def keypoints_persisted(self, clip_id: str) -> bool:
        raise NotImplementedError

    def transcode_proxy(self, clip_id: str, video_path: str, *, max_height: int, crf: int) -> str:
        raise NotImplementedError

    def delete_original(self, clip_id: str) -> int:
        raise NotImplementedError

    def delete_proxy(self, clip_id: str) -> int:
        raise NotImplementedError


def estimate_storage_bytes(
    *, n_clips: int, mean_clip_mb: float = 15.0, consent_rate: float = 0.5
) -> dict:
    """Project storage cost. Implemented, because it informs a real hosting decision.

    Illustrates the point of this module: keypoints are negligible next to video, so the
    consent rate - not the clip count - drives the bill.
    """
    keypoints_mb = n_clips * 0.15
    proxy_mb = n_clips * consent_rate * (mean_clip_mb * 0.25)
    return {
        "keypoints_mb": round(keypoints_mb, 1),
        "proxy_video_mb": round(proxy_mb, 1),
        "total_mb": round(keypoints_mb + proxy_mb, 1),
        "naive_total_mb": round(n_clips * mean_clip_mb, 1),
    }
