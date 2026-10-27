/**
 * Client for the Python analysis service.
 *
 * Shared-secret header, same pattern as the Meco voice-id service. The analysis service is
 * never exposed publicly - only this API talks to it.
 */

const BASE_URL = process.env.ANALYSIS_SERVICE_URL || 'http://localhost:5000';
const SHARED_SECRET = process.env.ANALYSIS_SHARED_SECRET || '';
const SECRET_HEADER = 'X-Analysis-Secret';

/** Generous, because a cold Render instance can take 30-60s to wake after spin-down. */
const REQUEST_TIMEOUT_MS = 90_000;

function assertConfigured() {
  if (!SHARED_SECRET) {
    throw new Error(
      'ANALYSIS_SHARED_SECRET is not set. Refusing to call the analysis service unauthenticated.',
    );
  }
}

async function request(path, options = {}) {
  assertConfigured();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        [SECRET_HEADER]: SHARED_SECRET,
        ...(options.headers || {}),
      },
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(body.detail || body.error || `analysis service ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

export async function analysisServiceHealth() {
  try {
    const body = await request('/health', { method: 'GET' });
    return { ok: body.status === 'ok', ...body };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

/**
 * Enqueue a clip. Returns immediately with a job handle - analysis is asynchronous and
 * takes an estimated 1-2 minutes.
 *
 * TODO: called by routes/clips.mjs after the upload lands in Appwrite Storage.
 */
export async function enqueueAnalysis({
  clipId,
  sessionId,
  userId,
  formType,
  videoFileId,
  frontOnFileId = null,
  captureMetadata,
  declaredStyle,
  handedness,
  playerGroup,
  consentForTraining,
}) {
  return request('/analyze', {
    method: 'POST',
    body: JSON.stringify({
      clip_id: clipId,
      session_id: sessionId,
      user_id: userId,
      form_type: formType,
      video_file_id: videoFileId,
      front_on_file_id: frontOnFileId,
      metadata: captureMetadata,
      declared_style: declaredStyle,
      handedness,
      player_group: playerGroup,
      consent_for_training: consentForTraining,
    }),
  });
}

export async function getAnalysisStatus(clipId) {
  return request(`/status/${encodeURIComponent(clipId)}`, { method: 'GET' });
}
