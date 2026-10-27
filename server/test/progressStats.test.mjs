import test from 'node:test';
import assert from 'node:assert/strict';

import {
  aggregateSession,
  mdc,
  compareSessions,
  rollingTrend,
  comparable,
  sessionQuality,
  ChangeVerdict,
  MIN_SESSIONS_FOR_TREND,
} from '../lib/progressStats.mjs';

test('aggregateSession returns null for no measurements', () => {
  assert.equal(aggregateSession([]), null);
  assert.equal(aggregateSession(null), null);
});

test('aggregateSession averages values', () => {
  const agg = aggregateSession([
    { value: 150, sem: 8 },
    { value: 152, sem: 8 },
    { value: 148, sem: 8 },
  ]);
  assert.equal(agg.n, 3);
  assert.ok(Math.abs(agg.mean - 150) < 1e-9);
});

test('aggregateSession never reports more precision than the instrument allows', () => {
  // Three near-identical clips would give a tiny observed SD. The instrument floor must win.
  const agg = aggregateSession([
    { value: 150.0, sem: 8 },
    { value: 150.01, sem: 8 },
    { value: 149.99, sem: 8 },
  ]);
  assert.ok(agg.sem >= 8 / 3, `sem ${agg.sem} fell below the instrument floor`);
});

test('mdc shrinks with more clips - the reason sessions want 3-5', () => {
  const single = mdc(10, 1);
  const five = mdc(10, 5);
  assert.ok(single > five);
  assert.ok(Math.abs(single - 27.7) < 0.5, `single-clip MDC was ${single}`);
  assert.ok(Math.abs(five - 12.4) < 0.5, `five-clip MDC was ${five}`);
});

test('a small change is reported as no change, not as progress', () => {
  const earlier = { mean: 150, sem: 4, n: 5 };
  const later = { mean: 153, sem: 4, n: 5 };
  const result = compareSessions(earlier, later, true);
  assert.equal(result.verdict, ChangeVerdict.NO_CHANGE);
  assert.match(result.explanation, /smaller than what we can reliably measure/);
});

test('a large change in the good direction is reported as improvement', () => {
  const result = compareSessions({ mean: 150, sem: 2, n: 5 }, { mean: 172, sem: 2, n: 5 }, true);
  assert.equal(result.verdict, ChangeVerdict.IMPROVED);
});

test('direction is respected when lower is better', () => {
  const result = compareSessions({ mean: 0.1, sem: 0.005, n: 5 }, { mean: 0.02, sem: 0.005, n: 5 }, false);
  assert.equal(result.verdict, ChangeVerdict.IMPROVED);
});

test('optimum-band metrics never report improvement', () => {
  // Stride length moved a long way, but longer is not better.
  const result = compareSessions({ mean: 0.30, sem: 0.005, n: 5 }, { mean: 0.45, sem: 0.005, n: 5 }, null);
  assert.equal(result.verdict, ChangeVerdict.NO_CHANGE);
  assert.equal(result.movedOutsideNoise, true);
  assert.match(result.explanation, /more is not automatically better/);
});

test('missing sessions yield insufficient rather than a guess', () => {
  assert.equal(compareSessions(null, { mean: 1, sem: 1, n: 1 }, true).verdict, ChangeVerdict.INSUFFICIENT);
});

test('rollingTrend refuses to draw a line from too few sessions', () => {
  const sessions = [{ sessionId: 'a', date: '2026-01-01', mean: 150, sem: 4, n: 5 }];
  const trend = rollingTrend(sessions);
  assert.equal(trend.sufficient, false);
  assert.ok(trend.points.length === 0);
});

test('rollingTrend produces banded points once there are enough sessions', () => {
  const sessions = Array.from({ length: 5 }, (_, i) => ({
    sessionId: `s${i}`,
    date: `2026-0${i + 1}-01`,
    mean: 150 + i,
    sem: 4,
    n: 5,
  }));
  const trend = rollingTrend(sessions, 3);
  assert.equal(trend.sufficient, true);
  assert.equal(trend.points.length, 3);
  for (const p of trend.points) {
    assert.ok(p.ciLow < p.mean && p.mean < p.ciHigh);
  }
});

test('sessions with different form types are not comparable', () => {
  const a = { formType: 'cover_drive', declaredStyle: 'balanced', templateVersion: '0.1.0', fps: 240 };
  const b = { formType: 'pull_shot', declaredStyle: 'balanced', templateVersion: '0.1.0', fps: 240 };
  const result = comparable(a, b);
  assert.equal(result.comparable, false);
  assert.ok(result.reasons.some((r) => /shot or delivery type/.test(r)));
});

test('a large frame-rate difference blocks comparison', () => {
  const a = { formType: 'cover_drive', declaredStyle: 'balanced', templateVersion: '0.1.0', fps: 240 };
  const b = { formType: 'cover_drive', declaredStyle: 'balanced', templateVersion: '0.1.0', fps: 120 };
  assert.equal(comparable(a, b).comparable, false);
});

test('a template version change blocks comparison', () => {
  const a = { formType: 'cover_drive', declaredStyle: 'balanced', templateVersion: '0.1.0', fps: 240 };
  const b = { formType: 'cover_drive', declaredStyle: 'balanced', templateVersion: '0.2.0', fps: 240 };
  assert.equal(comparable(a, b).comparable, false);
});

test('identical conditions are comparable', () => {
  const a = { formType: 'cover_drive', declaredStyle: 'balanced', templateVersion: '0.1.0', fps: 240 };
  assert.equal(comparable(a, { ...a }).comparable, true);
});

test('sessionQuality warns on thin sessions', () => {
  assert.equal(sessionQuality(5).level, 'good');
  assert.equal(sessionQuality(3).level, 'ok');
  assert.equal(sessionQuality(1).level, 'low');
  assert.match(sessionQuality(1).note, /single-clip variation dominates/);
});
