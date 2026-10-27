/**
 * Session management.
 *
 * A session groups the 3-5 clips a player records in one go. Sessions, not clips, are the
 * unit of progress: aggregating across clips shrinks the standard error by sqrt(n), which
 * is what makes any change detectable at all given realistic measurement error.
 *
 * Status: routes and contracts are real; Appwrite calls are TODO.
 */

import express from 'express';
import { getAuth } from '@clerk/express';

import { sessionQuality, aggregateSession, TARGET_CLIPS_PER_SESSION } from '../lib/progressStats.mjs';

const router = express.Router();

export const STYLES = Object.freeze(['attacking', 'balanced', 'defensive']);

/**
 * POST /api/sessions
 *
 * `sessionStyleOverride` lets a player say "today I was working on defence" without
 * changing their profile. It selects the reference variant for this session only.
 *
 * TODO: create the Appwrite document.
 */
router.post('/', async (req, res, next) => {
  try {
    const { userId } = getAuth(req);
    const { formType, sessionStyleOverride } = req.body || {};

    if (sessionStyleOverride && !STYLES.includes(sessionStyleOverride)) {
      return res.status(400).json({
        error: 'unknown_style',
        message: `sessionStyleOverride must be one of: ${STYLES.join(', ')}`,
      });
    }

    return res.status(501).json({ error: 'not_implemented' });
  } catch (err) {
    return next(err);
  }
});

/**
 * GET /api/sessions/:sessionId
 *
 * Returns the session with its clips and per-metric aggregates.
 *
 * The response carries a `quality` flag from `sessionQuality()`. A one-clip session is not
 * refused, but its error bars are roughly 2.2x wider than a five-clip session and the
 * player should be told that before reading anything into it.
 *
 * TODO: load clips, aggregate Tier A metrics via aggregateSession().
 */
router.get('/:sessionId', async (_req, res) => {
  res.status(501).json({
    error: 'not_implemented',
    hint: `aggregate over ${TARGET_CLIPS_PER_SESSION} clips where available`,
  });
});

/** GET /api/sessions - list, newest first. TODO. */
router.get('/', async (_req, res) => {
  res.status(501).json({ error: 'not_implemented' });
});

export default router;
