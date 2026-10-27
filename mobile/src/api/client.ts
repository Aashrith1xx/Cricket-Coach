/**
 * API client.
 *
 * Status: contracts real, network calls TODO.
 *
 * Analysis is asynchronous - a clip takes an estimated 1-2 minutes - so uploads return a
 * clip id and the app polls. `pollClipResult` implements backoff rather than hammering a
 * service that may be cold-starting from Render's 15-minute spin-down.
 */

import type {
  ClipResult,
  FormType,
  MetricTrend,
  RenderPayload,
  Style,
} from '../types';

const BASE_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:3000';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init: RequestInit = {}, token?: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init.headers ?? {}),
    },
  });

  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(body.message ?? `Request failed (${res.status})`, res.status, body.error);
  }
  return body as T;
}

export async function createSession(
  params: { formType: FormType; sessionStyleOverride?: Style },
  token: string,
): Promise<{ sessionId: string }> {
  return request('/api/sessions', { method: 'POST', body: JSON.stringify(params) }, token);
}

/**
 * Upload a clip.
 *
 * `captureMetadata.fps` must be read from the recorded file, not assumed from camera
 * settings. Everything temporal scales with it, and a wrong value would silently corrupt
 * every timing metric while looking entirely plausible.
 *
 * TODO: upload to Appwrite Storage first, then POST the resulting file id here.
 */
export async function uploadClip(
  params: {
    sessionId: string;
    formType: FormType;
    videoUri: string;
    captureMetadata: {
      fps: number;
      width: number;
      height: number;
      durationS: number;
      declaredView: 'side_on' | 'front_on';
      device?: string;
    };
  },
  token: string,
): Promise<{ clipId: string }> {
  throw new Error('uploadClip is not implemented');
}

export async function getClipResult(clipId: string, token: string): Promise<ClipResult> {
  return request(`/api/clips/${encodeURIComponent(clipId)}`, { method: 'GET' }, token);
}

/**
 * Poll until analysis settles.
 *
 * Backs off from 2s toward 15s. `rejected` is a terminal state and resolves normally - it
 * is a successful outcome with guidance for the player, not an error to throw on.
 */
export async function pollClipResult(
  clipId: string,
  token: string,
  opts: { timeoutMs?: number; onUpdate?: (r: ClipResult) => void } = {},
): Promise<ClipResult> {
  const timeoutMs = opts.timeoutMs ?? 300_000;
  const started = Date.now();
  let delay = 2_000;

  for (;;) {
    const result = await getClipResult(clipId, token);
    opts.onUpdate?.(result);

    if (result.status === 'done' || result.status === 'rejected' || result.status === 'failed') {
      return result;
    }
    if (Date.now() - started > timeoutMs) {
      throw new ApiError('Analysis is taking longer than expected.', 504, 'timeout');
    }

    await new Promise((r) => setTimeout(r, delay));
    delay = Math.min(delay * 1.5, 15_000);
  }
}

/** Fetch the render payload and check schema compatibility before using it. */
export async function getRenderPayload(url: string): Promise<RenderPayload> {
  const res = await fetch(url);
  if (!res.ok) throw new ApiError('Could not load clip data', res.status);
  const payload = (await res.json()) as RenderPayload;

  const [major] = payload.schemaVersion.split('.');
  const [expectedMajor] = '1.0.0'.split('.');
  if (major !== expectedMajor) {
    throw new ApiError(
      'This clip was analysed by a newer version of the app. Please update.',
      409,
      'schema_mismatch',
    );
  }
  return payload;
}

export async function getProgress(
  formType: FormType,
  token: string,
): Promise<{ trends: MetricTrend[]; sufficient: boolean; minSessions: number }> {
  return request(`/api/progress/${formType}`, { method: 'GET' }, token);
}

export async function updateStyleProfile(
  params: { declaredStyle: Style; handedness?: 'left' | 'right'; consentForTraining?: boolean },
  token: string,
): Promise<void> {
  throw new Error('updateStyleProfile is not implemented');
}
