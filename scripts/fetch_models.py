"""Download the RTMDet and RTMPose ONNX models at build time.

The weights are ~58 MB and are not committed. Both are Apache 2.0 from OpenMMLab, so they
can be redistributed freely - they are fetched rather than vendored to keep the repository
small and the licence provenance obvious.

    python scripts/fetch_models.py

Idempotent: existing files of the right size are left alone, so a rebuild with a warm cache
costs nothing.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"

MODELS = {
    "rtmdet.onnx": (
        f"{BASE}/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip",
        4_010_667,
    ),
    # 384x288 rather than the 256x192 variant. Measured on real nets footage: overall
    # keypoint confidence 0.659 -> 0.697, and wrists - the joints that gate impact detection -
    # 0.441 -> 0.557, a 26% gain where it matters most. Costs ~2x inference time.
    #
    # Note rtmpose-L at the same resolution scored WORSE (0.673 overall) at 5x the cost:
    # input resolution matters more than model capacity for this footage, so do not assume a
    # bigger model is an upgrade without measuring it.
    "rtmpose.onnx": (
        f"{BASE}/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.zip",
        54_000_000,
    ),
}

TARGET = Path(__file__).resolve().parent.parent / "analysis-service" / "models"


def fetch(name: str, url: str, expected_size: int) -> None:
    out = TARGET / name

    # Size check rather than mere existence: a build interrupted mid-download leaves a
    # truncated file that loads as a corrupt-model error much later and much less clearly.
    if out.exists() and abs(out.stat().st_size - expected_size) < expected_size * 0.02:
        print(f"  {name} already present ({out.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  fetching {name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.endswith("end2end.onnx")]
        if not members:
            raise SystemExit(f"No end2end.onnx inside {url}")
        out.write_bytes(archive.read(members[0]))

    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {TARGET}")
    for name, (url, size) in MODELS.items():
        fetch(name, url, size)
    print("Models ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
