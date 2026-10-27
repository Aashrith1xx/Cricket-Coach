/**
 * Progress over time.
 *
 * The rules this endpoint enforces, all from `lib/progressStats.mjs`:
 *
 *   - Tier A metrics only. Tier B appears as banded categories, Tier C never appears.
 *   - Changes below the minimal detectable change are reported as 'no_change_detected',
 *     never as a small improvement or regression.
 *   - Only like-for-like sessions are compared: same form type, same declared style, same
 *     template version, similar frame rate.
 *   - Metrics with an optimum band rather than a direction (stride length) report movement
 *     without labelling it improvement.
 *
 * Status: routes and contracts are real; Appwrite reads are TODO. The statistics they
 * depend on are implemented and tested.
 */

import express from 'express';
import { getAuth } from '@clerk/express';

import {
  compareSessions,
  rollingTrend,
  comparable,
  ChangeVerdict,
  MIN_SESSIONS_FOR_TREND,
} from '../lib/progressStats.mjs';

const router = express.Router();

/**
 * Metrics whose direction is meaningful, and those that merely have an optimum.
 *
 * `null` means "moved, but more is not better". Stride length is the important case:
 * experienced batters deliberately take a short-to-medium stride so they can play the ball
 * later, and no significant stride-length difference was found across skill levels. Marking
 * it `true` would reward players for lengthening a stride they should not lengthen.
 *
 * Keep in step with `analysis-service/metric_tiers.py`.
 */
export const METRIC_DIRECTION = Object.freeze({
  'batting.lead_knee_angle_imp': null,
  'batting.lead_knee_extension_se_imp': null,
  'batting.head_vertical_displacement_ds_imp': false, // less head lift is better
  'batting.head_horizontal_displacement_ds_imp': null,
  'batting.stride_duration_ss_se': null,
  'batting.interval_se_imp': null,
  'batting.tempo_bc_ds': null,
  'batting.trigger_timing': null,
  'batting.com_anterior_displacement_min_imp': null,
  'batting.base_length_pct_stature_imp': null,
  'batting.stride_length_norm': null, // optimum band, NOT longer-is-better
  'bowling.front_knee_angle_ffc': null,
  'bowling.front_knee_angle_br': null,
  'bowling.front_knee_change_ffc_br': null,
  'bowling.trunk_lateral_flexion_br': null,
  'bowling.interval_bfc_ffc': null,
  'bowling.interval_ffc_br': null,
  'bowling.runup_final_stride_tempo': null,
});

/**
 * GET /api/progress/:formType
 *
 * Trend across sessions for one form type.
 *
 * TODO: load session aggregates from Appwrite, filter to comparable sessions via
 * `comparable()`, then run `rollingTrend()` per Tier A metric.
 *
 * Returns `sufficient: false` with a reason when there are fewer than
 * MIN_SESSIONS_FOR_TREND sessions - the client should show an encouraging "keep recording"
 * state rather than an empty chart, which reads as a failure.
 */
router.get('/:formType', async (req, res, next) => {
  try {
    const { userId } = getAuth(req);
    return res.status(501).json({
      error: 'not_implemented',
      minSessions: MIN_SESSIONS_FOR_TREND,
    });
  } catch (err) {
    return next(err);
  }
});

/**
 * GET /api/progress/:formType/compare?from=<sessionId>&to=<sessionId>
 *
 * Compare two specific sessions.
 *
 * TODO. Must call `comparable()` first and return 409 with the reasons when the sessions
 * are not like-for-like, rather than comparing anyway. Attributing a capture difference to
 * the player is exactly the kind of confident wrong answer this system is built to avoid.
 */
router.get('/:formType/compare', async (_req, res) => {
  res.status(501).json({
    error: 'not_implemented',
    verdicts: Object.values(ChangeVerdict),
  });
});

export default router;
