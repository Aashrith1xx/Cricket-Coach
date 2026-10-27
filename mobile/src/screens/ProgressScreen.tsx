/**
 * Progress across sessions.
 *
 * The screen most likely to mislead, so it is the most constrained.
 *
 * With ~8 degrees RMSE on the best-measured angle, a 3 degree shift between sessions is
 * noise. A player training hard *wants* the number to move, and a conventional progress
 * chart will happily show them movement that is entirely measurement error. So:
 *
 *   - changes below the minimal detectable change read "no change detected", never a small
 *     improvement or regression
 *   - trends are rolling windows with confidence bands, not consecutive-point deltas
 *   - metrics with an optimum band rather than a direction (stride length) show movement
 *     without calling it progress
 *   - Tier A only; Tier B as bands; Tier C never appears
 *   - only like-for-like sessions are compared
 *
 * Status: layout real; chart rendering and data wiring TODO.
 */

import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import type { MetricTrend, SessionQuality } from '../types';

interface Props {
  trends: MetricTrend[];
  sufficient: boolean;
  minSessions: number;
  sessionCount: number;
  latestQuality?: SessionQuality;
}

export function ProgressScreen({
  trends,
  sufficient,
  minSessions,
  sessionCount,
  latestQuality,
}: Props) {
  if (!sufficient) {
    return (
      <View style={styles.centred}>
        <Text style={styles.title}>Keep recording</Text>
        <Text style={styles.body}>
          You&apos;ve done {sessionCount} session{sessionCount === 1 ? '' : 's'}. We need at
          least {minSessions} before a trend means anything — with fewer, we&apos;d mostly be
          showing you measurement wobble rather than real change.
        </Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Your progress</Text>

      {latestQuality?.note && (
        <View style={styles.qualityNote}>
          <Text style={styles.qualityText}>{latestQuality.note}</Text>
        </View>
      )}

      {trends.map((trend) => (
        <View key={trend.metricKey} style={styles.trendCard}>
          <Text style={styles.trendTitle}>{trend.summary}</Text>

          {trend.higherIsBetter === null && (
            <Text style={styles.optimumNote}>
              This one has a target range rather than a direction — more isn&apos;t better.
            </Text>
          )}

          {/* TODO: line chart of trend.points with ciLow/ciHigh shaded as a band.
              The band must be visually prominent, not a faint decoration: it is the honest
              content, and the line through the middle is the least certain part. */}
          <View style={styles.chartPlaceholder}>
            <Text style={styles.chartText}>
              {trend.points.length} points — chart TODO
            </Text>
          </View>

          {!trend.sufficient && trend.reason && (
            <Text style={styles.insufficient}>{trend.reason}</Text>
          )}
        </View>
      ))}

      <Text style={styles.footnote}>
        We only show a change when it&apos;s bigger than what we can reliably measure. If a
        number looks flat, that genuinely means we couldn&apos;t tell it apart from
        measurement noise — not that nothing happened.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, paddingBottom: 48 },
  centred: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32 },
  title: { fontSize: 26, fontWeight: '700', color: '#1a1f26', marginBottom: 8 },
  body: { fontSize: 15, lineHeight: 22, color: '#5b6470', textAlign: 'center' },
  qualityNote: { backgroundColor: '#fff4e5', borderRadius: 10, padding: 14, marginBottom: 20 },
  qualityText: { fontSize: 13, lineHeight: 19, color: '#8a5300' },
  trendCard: {
    borderWidth: 1,
    borderColor: '#e6e9ed',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  trendTitle: { fontSize: 16, fontWeight: '600', color: '#1a1f26' },
  optimumNote: { fontSize: 12, color: '#7a838d', fontStyle: 'italic', marginTop: 4 },
  chartPlaceholder: {
    height: 140,
    backgroundColor: '#f7f8fa',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 12,
  },
  chartText: { fontSize: 12, color: '#7a838d' },
  insufficient: { fontSize: 12, color: '#7a838d', marginTop: 8 },
  footnote: { fontSize: 12, lineHeight: 18, color: '#7a838d', marginTop: 12 },
});
