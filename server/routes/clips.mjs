/**
 * Clip upload and analysis status.
 *
 * Status: routes and contracts are real; Appwrite calls are TODO.
 */

import express from 'express';
import { getAuth } from '@clerk/express';

import { enqueueAnalysis, getAnalysisStatus } from '../lib/analysisServiceClient.mjs';

const router = express.Router();

export const FORM_TYPES = Object.freeze([
  'cover_drive',
  'pull_shot',
  'fast_arm_bowling',
  'spin_bowling',
]);

/**
 * POST /api/clips
 *
 * Registers a clip and enqueues analysis.
 *
 * `captureMetadata.fps` is required and must be genuine - read from the file, not declared
 * by the client. Everything temporal depends on it, and a wrong fps would silently scale
 * every timing metric while looking perfectly plausible.
 *
 * TODO: create the Appwrite document, then enqueue. In that order - a job referencing a
 * document that does not exist yet is a race the worker cannot recover from.
 */
router.post('/', async (req, res, next) => {
  try {
    const { userId } = getAuth(req);
    const { sessionId, formType, videoFileId, captureMetadata } = req.body || {};

    if (!FORM_TYPES.includes(formType)) {
      return res.status(400).json({
        error: 'unknown_form_type',
        message: `formType must be one of: ${FORM_TYPES.join(', ')}`,
      });
    }
    if (!videoFileId || !sessionId) {
      return res.status(400).json({ error: 'missing_fields' });
    }
    if (!captureMetadata?.fps) {
      return res.status(400).json({
        error: 'missing_fps',
        message: 'captureMetadata.fps is required and must be read from the video file.',
      });
    }

    return res.status(501).json({
      error: 'not_implemented',
      detail: 'clips.post does not yet write the Appwrite document',
    });
  } catch (err) {
    return next(err);
  }
});

/**
 * GET /api/clips/:clipId
 *
 * Returns analysis state. Three outcomes the client must handle distinctly:
 *
 *   'rejected'  the capture gate refused it. This is a SUCCESSFUL outcome with actionable
 *               guidance, not an error - show the remedy, offer a reshoot.
 *   'done'      results are ready. Note compositeScore may still be null, meaning too
 *               little reliable evidence for an overall grade; render the observations.
 *   'failed'    something broke on our side. Apologise, do not blame the clip.
 */
router.get('/:clipId', async (req, res, next) => {
  try {
    const status = await getAnalysisStatus(req.params.clipId);
    return res.json(status);
  } catch (err) {
    return next(err);
  }
});

/**
 * GET /api/clips/:clipId/render
 *
 * Signed URL for the render payload in Storage.
 *
 * TODO. Must verify the clip belongs to the requesting user before signing - render data
 * contains full per-frame keypoints of a person, which is exactly the kind of thing that
 * should not be enumerable by id.
 */
router.get('/:clipId/render', async (_req, res) => {
  res.status(501).json({ error: 'not_implemented' });
});

export default router;
