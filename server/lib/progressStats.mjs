/**
 * Progress statistics: session aggregation and minimal-detectable-change gating.
 *
 * This is the module most likely to mislead a user if it is wrong, so it is implemented
 * rather than stubbed.
 *
 * The problem: with roughly 8 degrees RMSE on the best-measured angle, a 3 degree
 * "improvement" between sessions is noise. A player training hard wants to see a number
 * move, and a naive progress chart will happily show them movement that is entirely
 * measurement error. That is a worse failure than showing nothing, because it is
 * indistinguishable from real progress and it rewards the wrong things.
 *
 * The response:
 *   - aggregate per session rather than per clip, which shrinks standard error by sqrt(n)
 *   - compute a minimal detectable change and report anything below it as "no change"
 *   - trend over rolling windows rather than consecutive-point deltas
 *   - Tier A only on charts; Tier B as bands; Tier C never
 *
 * Mirrors analysis-service/uncertainty.py. Keep the two in step - if the Z constant or the
 * MDC formula changes there, it changes here.
 */

export const Z_95 = 1.959963984540054;

/** Sessions needed before a trend line is drawn at all. */
export const MIN_SESSIONS_FOR_TREND = 3;

/** Clips per session the app asks for. Below this, the session mean is barely better than
 *  a single clip - see mdc() for the arithmetic. */
export const TARGET_CLIPS_PER_SESSION = 5;
export const MIN_CLIPS_PER_SESSION = 3;

export const ChangeVerdict = Object.freeze({
  IMPROVED: 'improved',
  REGRESSED: 'regressed',
  NO_CHANGE: 'no_change_detected',
  INSUFFICIENT: 'insufficient_data',
});

/**
 * Aggregate per-clip measurements of one metric into a session estimate.
 *
 * Uses the larger of observed between-clip scatter and propagated instrument error.
 * Between-clip variation is genuine signal about consistency, but with 3-5 clips the
 * observed SD is itself unstable, so the instrument floor stops a lucky run of similar
 * clips implying more precision than the camera can support.
 *
 * @param {Array<{value: number, sem: number}>} measurements
 * @returns {{mean: number, sem: number, n: number}|null}
 */
export function aggregateSession(measurements) {
  if (!measurements || measurements.length === 0) return null;

  const n = measurements.length;
  const mean = measurements.reduce((acc, m) => acc + m.value, 0) / n;

  const instrumentSem =
    Math.sqrt(measurements.reduce((acc, m) => acc + m.sem * m.sem, 0)) / n;

  let observedSem = 0;
  if (n >= 2) {
    const variance =
      measurements.reduce((acc, m) => acc + (m.value - mean) ** 2, 0) / (n - 1);
    observedSem = Math.sqrt(variance / n);
  }

  return { mean, sem: Math.max(instrumentSem, observedSem), n };
}

/**
 * Minimal detectable change: `z * sqrt(2) * sem / sqrt(n)`.
 *
 * The sqrt(2) is because a change is a difference of two independent measurements; the
 * sqrt(n) is because we compare session means.
 *
 * Worked example: sem 10 degrees, one clip gives MDC ~28 degrees, which is uselessly large.
 * Five clips brings it to ~12. That arithmetic is the entire reason the app asks for
 * several clips per session instead of accepting one.
 */
export function mdc(sem, n = 1, z = Z_95) {
  if (n < 1) throw new Error('n must be >= 1');
  return (z * Math.SQRT2 * sem) / Math.sqrt(n);
}

/**
 * Compare two sessions for a single metric.
 *
 * `higherIsBetter` is deliberately nullable. Several metrics have an *optimum band* rather
 * than a direction - stride length most notably, where the literature shows experienced
 * batters take a deliberately short-to-medium stride and longer is not better. For those,
 * pass null and the result reports movement without labelling it improvement.
 *
 * @param {{mean: number, sem: number, n: number}} earlier
 * @param {{mean: number, sem: number, n: number}} later
 * @param {boolean|null} higherIsBetter
 */
export function compareSessions(earlier, later, higherIsBetter = null) {
  if (!earlier || !later) {
    return { verdict: ChangeVerdict.INSUFFICIENT, delta: null, threshold: null };
  }

  const delta = later.mean - earlier.mean;
  const threshold = Z_95 * Math.hypot(earlier.sem, later.sem);

  if (Math.abs(delta) <= threshold) {
    return {
      verdict: ChangeVerdict.NO_CHANGE,
      delta,
      threshold,
      explanation:
        'The difference between these sessions is smaller than what we can reliably ' +
        'measure, so we are not calling it a change either way.',
    };
  }

  if (higherIsBetter === null) {
    return {
      verdict: ChangeVerdict.NO_CHANGE,
      delta,
      threshold,
      explanation:
        'This measurement moved, but it has a target range rather than a direction, so ' +
        'more is not automatically better.',
      movedOutsideNoise: true,
    };
  }

  const improved = higherIsBetter ? delta > 0 : delta < 0;
  return {
    verdict: improved ? ChangeVerdict.IMPROVED : ChangeVerdict.REGRESSED,
    delta,
    threshold,
  };
}

/**
 * Rolling-window trend across sessions.
 *
 * Deliberately not consecutive-point deltas: with realistic error bars, consecutive points
 * bounce around enough that a chart of them reads as noise, and users interpret noise as
 * signal. A rolling mean with bands is both more honest and easier to read.
 *
 * @param {Array<{sessionId: string, date: string, mean: number, sem: number, n: number}>} sessions
 * @param {number} windowSize
 */
export function rollingTrend(sessions, windowSize = 3) {
  if (!sessions || sessions.length < MIN_SESSIONS_FOR_TREND) {
    return { points: [], sufficient: false, reason: 'not enough sessions yet' };
  }

  const ordered = [...sessions].sort((a, b) => new Date(a.date) - new Date(b.date));
  const points = [];

  for (let i = windowSize - 1; i < ordered.length; i += 1) {
    const window = ordered.slice(i - windowSize + 1, i + 1);
    const totalN = window.reduce((acc, s) => acc + s.n, 0);
    const mean = window.reduce((acc, s) => acc + s.mean * s.n, 0) / totalN;
    const sem = Math.sqrt(window.reduce((acc, s) => acc + s.sem * s.sem, 0)) / window.length;

    points.push({
      date: ordered[i].date,
      sessionId: ordered[i].sessionId,
      mean,
      ciLow: mean - Z_95 * sem,
      ciHigh: mean + Z_95 * sem,
      windowSessions: window.length,
    });
  }

  return { points, sufficient: true };
}

/**
 * Whether two sessions may be compared at all.
 *
 * Comparing a 240 fps side-on clip against a 120 fps oblique one would attribute a capture
 * difference to the player. Like-for-like only.
 */
export function comparable(a, b) {
  const reasons = [];
  if (a.formType !== b.formType) reasons.push('different shot or delivery type');
  if (a.declaredStyle !== b.declaredStyle) reasons.push('different playstyle profile');
  if (a.templateVersion !== b.templateVersion) {
    reasons.push('reference ranges changed between these sessions');
  }
  if (Math.abs((a.fps ?? 0) - (b.fps ?? 0)) > 60) {
    reasons.push('recorded at noticeably different frame rates');
  }
  return { comparable: reasons.length === 0, reasons };
}

/**
 * Session quality flag for the UI.
 *
 * A session with one clip is not wrong, but its error bars are roughly 2.2x wider than a
 * five-clip session, and the user should know that before reading anything into it.
 */
export function sessionQuality(nClips) {
  if (nClips >= TARGET_CLIPS_PER_SESSION) return { level: 'good', note: null };
  if (nClips >= MIN_CLIPS_PER_SESSION) {
    return {
      level: 'ok',
      note: `Recording ${TARGET_CLIPS_PER_SESSION} clips per session would tighten these numbers.`,
    };
  }
  return {
    level: 'low',
    note:
      `With only ${nClips} clip${nClips === 1 ? '' : 's'}, single-clip variation dominates. ` +
      `Aim for ${TARGET_CLIPS_PER_SESSION} clips before reading much into a change.`,
  };
}
