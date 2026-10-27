"""Flask entrypoint for the analysis service.

Mirrors the Meco voice-id service pattern: shared-secret header between the Node API and
this service, no public exposure, CPU-only.

``POST /analyze`` **enqueues and returns immediately**. It does not block on analysis - a
clip takes an estimated 1-2 minutes on a fractional-vCPU instance, and Render free-tier
services spin down after 15 minutes idle, so a synchronous design would lose work.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from typing import Dict

from flask import Flask, jsonify, request

import template_validator
from worker import JobResult, JobStatus, process

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

app = Flask(__name__)

SHARED_SECRET = os.environ.get("ANALYSIS_SHARED_SECRET", "")
SECRET_HEADER = "X-Analysis-Secret"

#: In-process queue. Adequate for a single instance; move to Redis or Appwrite Functions
#: before running more than one worker, or jobs will be lost on restart.
_jobs: "queue.Queue" = queue.Queue()
_results: Dict[str, JobResult] = {}
_results_lock = threading.Lock()


def _require_secret() -> bool:
    if not SHARED_SECRET:
        log.error("ANALYSIS_SHARED_SECRET is unset - refusing all requests")
        return False
    return request.headers.get(SECRET_HEADER) == SHARED_SECRET


def _worker_loop() -> None:
    while True:
        job = _jobs.get()
        try:
            result = process(job)
        except Exception:  # noqa: BLE001 - never let the loop die
            log.exception("Worker loop error on clip %s", getattr(job, "clip_id", "?"))
            result = JobResult(
                clip_id=getattr(job, "clip_id", "unknown"),
                status=JobStatus.FAILED,
                error="internal error",
            )
        with _results_lock:
            _results[result.clip_id] = result
        _jobs.task_done()


@app.route("/health", methods=["GET"])
def health():
    """Liveness plus template validity.

    Reports unhealthy when templates fail to load, because a service scoring against
    unvalidated templates is worse than one that is down.
    """
    try:
        templates = template_validator.load_all()
        return jsonify(
            {
                "status": "ok",
                "templates": sorted(templates),
                "queue_depth": _jobs.qsize(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "unhealthy", "error": str(exc)}), 503


@app.route("/analyze", methods=["POST"])
def analyze():
    """Enqueue a clip. Returns 202 with a job id.

    TODO: build a ``JobRequest`` from the payload, download the video from Appwrite Storage
    to a temp path, and enqueue. Validate ``form_type`` and ``declared_style`` against known
    values before enqueueing so a bad request fails here rather than inside the worker.
    """
    if not _require_secret():
        return jsonify({"error": "unauthorised"}), 401

    payload = request.get_json(silent=True) or {}
    clip_id = payload.get("clip_id") or str(uuid.uuid4())

    return (
        jsonify(
            {
                "error": "not_implemented",
                "detail": "server.analyze does not yet construct JobRequest",
                "clip_id": clip_id,
            }
        ),
        501,
    )


@app.route("/status/<clip_id>", methods=["GET"])
def status(clip_id: str):
    if not _require_secret():
        return jsonify({"error": "unauthorised"}), 401

    with _results_lock:
        result = _results.get(clip_id)

    if result is None:
        return jsonify({"clip_id": clip_id, "status": JobStatus.QUEUED.value})
    return jsonify(result.as_dict())


def create_app() -> Flask:
    """Validate templates and start the worker thread.

    Template validation happens here rather than lazily so a bad template stops deployment
    instead of surfacing as a runtime error on a user's first upload.
    """
    template_validator.load_all()
    threading.Thread(target=_worker_loop, daemon=True, name="analysis-worker").start()
    return app


if __name__ == "__main__":
    create_app().run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
