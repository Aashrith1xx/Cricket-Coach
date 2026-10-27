/**
 * API server: Express + Clerk auth + Appwrite storage.
 *
 * Follows the Meco deployment pattern - Node API in front, heavy ML in a separate Python
 * service, shared secret between them, neither exposing the other publicly.
 *
 * This layer owns auth, uploads, session grouping and progress statistics. It owns no
 * analysis logic. In particular it must never compute or re-derive a score: the tier rules
 * live in `analysis-service/metric_tiers.py` and duplicating any of them here would create
 * two sources of truth for what may be shown to a user.
 */

import express from 'express';
import { clerkMiddleware, requireAuth, getAuth } from '@clerk/express';

import clipsRouter from './routes/clips.mjs';
import sessionsRouter from './routes/sessions.mjs';
import progressRouter from './routes/progress.mjs';
import { analysisServiceHealth } from './lib/analysisServiceClient.mjs';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: '1mb' }));
app.use(clerkMiddleware());

/**
 * Liveness. Reports degraded when the analysis service is unreachable, because uploads
 * would queue indefinitely and the client should be able to say so rather than spinning.
 */
app.get('/health', async (_req, res) => {
  const analysis = await analysisServiceHealth();
  res.status(analysis.ok ? 200 : 503).json({
    status: analysis.ok ? 'ok' : 'degraded',
    analysisService: analysis,
  });
});

app.use('/api/clips', requireAuth(), clipsRouter);
app.use('/api/sessions', requireAuth(), sessionsRouter);
app.use('/api/progress', requireAuth(), progressRouter);

/**
 * Error handler.
 *
 * Never leaks internals to the client. An analysis failure and a capture rejection are
 * different things and are distinguished upstream - a rejection is a 200 with guidance the
 * player can act on, not an error.
 */
app.use((err, _req, res, _next) => {
  console.error('[api]', err);
  const status = err.status || 500;
  res.status(status).json({
    error: status === 500 ? 'internal_error' : err.code || 'request_failed',
    message: status === 500 ? 'Something went wrong on our side.' : err.message,
  });
});

app.listen(PORT, () => {
  console.log(`API listening on :${PORT}`);
});

export default app;
