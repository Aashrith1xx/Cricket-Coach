# Putting the demo online

Everything needed is in the repo. What follows is the part only you can do, because it
needs your accounts.

---

## What you're deploying

The demo web app — upload a clip, get pose overlay, phases and measurements. Not the mobile
app, and not the Node/Appwrite service, neither of which is needed to try this.

## Sizing, measured rather than estimated

| | |
| --- | --- |
| Models + Python runtime | 136 MB |
| Decoded frames, worst case after downscaling | ~210 MB |
| Flask + request overhead | ~40 MB |
| **Peak** | **~390 MB** |

That fits Render's 512 MB free tier, but not with much room. Two constraints follow, both
already encoded in `render.yaml`:

**One gunicorn worker.** Each worker loads its own copy of the ONNX models, so a second
worker adds another 136 MB and OOMs the instance. Concurrency here is bounded by memory, not
CPU — one clip at a time.

**Frames are downscaled to 720px on the long side** (`pose_extraction.DEFAULT_MAX_DIMENSION`).
Without it, a 3-second 1080p clip at 240 fps is 720 frames of uncompressed BGR — about
4.3 GB — which OOMs anything. Keypoint precision depends on how many pixels the *subject*
occupies rather than the frame size, so this is safe given the capture protocol already
requires the player to fill the frame.

---

## Steps

### 1. Push to GitHub

Render deploys from a repository, and this directory isn't one yet.

```bash
cd cricket-form-app && git init && git add -A && git commit -m "Cricket form analysis"
```

Then create an empty repo on GitHub and push to it. `.gitignore` already excludes the model
weights (58 MB) and the venv — they're fetched at build time instead.

### 2. Create the service

On [render.com](https://render.com): **New → Blueprint**, connect the repo, select it.
Render reads `render.yaml` and configures everything — build command, start command, health
check, environment. Click apply.

The first build takes several minutes: it installs the dependencies and downloads the models.

### 3. Check it came up

```bash
curl https://YOUR-SERVICE.onrender.com/health
```

Expect `"status": "ok"` and `"models_present": true`. If models are missing the service
reports itself unhealthy rather than accepting uploads it cannot process.

Then open the URL and click a built-in sample — that exercises the whole pipeline without
needing footage.

---

## What the free tier will feel like

**It sleeps.** After 15 minutes idle the instance spins down, and the next request waits
30–60 seconds for a cold start plus model load. The upload page warns about this. If it
matters, the Starter plan (~$7/month) stays warm.

**One clip at a time.** A second concurrent upload queues behind the first.

**Analysis takes 30–90 seconds** on a shared vCPU. `--timeout 300` in `render.yaml` stops
gunicorn killing the request part-way, which would otherwise look like a crash.

**Uploads are capped at 60 MB** via `MAX_UPLOAD_MB`. The request body is buffered before
decoding, so it's memory spent before any analysis budget is touched. A one-to-two second
clip is all the analysis uses anyway.

**Storage is ephemeral.** Rendered overlays go to `/tmp` and are swept after an hour.
Nothing persists across deploys — fine for a demo, and it means no uploaded footage is
retained.

---

## If something breaks

**Build fails downloading models** — OpenMMLab was unreachable. Re-run the deploy;
`fetch_models.py` is idempotent and skips files already present at the right size.

**Service OOMs mid-analysis** — someone uploaded something long. Lower
`DEFAULT_MAX_FRAMES` in `pose_extraction.py`, or move to a paid instance.

**Every clip is rejected** — most likely working correctly. Check the reason it gives; 30 fps
and off-axis camera angles are the usual causes, and both are real problems with the footage
rather than with the service. `docs/ACCURACY.md` explains why each gate exists.

**Health check fails but the app looks fine** — a reference template stopped validating.
Run `python template_validator.py` locally for the specific error.

---

## Before sharing it widely

The demo has no authentication, no rate limiting, and no upload quota. That's fine for
sending to a few people, and not fine for posting publicly — one person uploading a long
clip repeatedly will saturate a single-worker free instance. Add a rate limit first, or keep
the URL private.

Uploaded video is deleted immediately after analysis and never leaves the instance. Worth
saying explicitly to anyone you ask to try it, since they'll be sending you footage of
themselves.
