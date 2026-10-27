/**
 * Renders one metric.
 *
 * This component is where the tier display rules become visible to the user, so it enforces
 * them rather than trusting callers:
 *
 *   Tier A       may show a number, always with its confidence interval
 *   Tier B       shows a BAND ONLY - never a precise number, because the underlying
 *                measurement does not support one
 *   unclear      shows neither a number nor a verdict, plus a capture fix
 *   Tier C       cannot reach here; it is filtered server-side and absent from the payload
 *
 * The `unclear` case is the important one. It means the measurement's confidence interval
 * straddles the edge of the expected range, so this clip genuinely cannot tell in-range from
 * out-of-range. Rendering that as a marginal pass or fail would turn measurement noise into
 * a verdict, which is the failure mode the whole tier system exists to prevent.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import type { MetricAnnotation } from '../types';
import { ConfidenceBand } from './ConfidenceBand';

interface Props {
  metric: MetricAnnotation;
  showEvidence?: boolean;
}

const VERDICT_STYLE = {
  in_range: { label: 'On track', colour: '#2f7d32', bg: '#e8f5e9' },
  out_of_range: { label: 'Work on this', colour: '#b26a00', bg: '#fff4e5' },
  unclear: { label: "Couldn't measure", colour: '#5b6470', bg: '#eef1f4' },
  not_scored: { label: 'For reference', colour: '#5b6470', bg: '#eef1f4' },
} as const;

/** Band wording for Tier B. Deliberately vague - the measurement does not support better. */
function bandWord(metric: MetricAnnotation): string {
  if (metric.value === null || metric.bandLow === null || metric.bandHigh === null) {
    return 'in the usual range';
  }
  if (metric.value < metric.bandLow) return 'on the low side';
  if (metric.value > metric.bandHigh) return 'on the high side';
  return 'in the usual range';
}

function formatValue(value: number, units: string): string {
  switch (units) {
    case 'deg':
      return `${Math.round(value)}°`;
    case 's':
      return `${value.toFixed(2)}s`;
    case 'pct_stature':
      return `${Math.round(value)}% of your height`;
    case 'frac_stature':
      return `${Math.round(value * 100)}% of your height`;
    case 'ratio':
      return value.toFixed(2);
    default:
      return String(Math.round(value * 100) / 100);
  }
}

export function MetricCard({ metric, showEvidence = false }: Props) {
  const verdict = VERDICT_STYLE[metric.verdict];

  // Guard rather than assume: a Tier B annotation arriving with a value would be an
  // upstream bug, and silently rendering it would leak precision we do not have.
  const mayShowNumber = metric.tier === 'A' && metric.value !== null;

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.summary}>{metric.summary}</Text>
        <View style={[styles.badge, { backgroundColor: verdict.bg }]}>
          <Text style={[styles.badgeText, { color: verdict.colour }]}>{verdict.label}</Text>
        </View>
      </View>

      {metric.verdict === 'unclear' ? (
        <Text style={styles.unclear}>
          The reading from this clip wasn&apos;t precise enough to say either way. A steadier
          camera at the recommended angle, or a higher frame rate, would fix it.
        </Text>
      ) : mayShowNumber ? (
        <>
          <Text style={styles.value}>{formatValue(metric.value as number, metric.units)}</Text>
          <ConfidenceBand
            value={metric.value as number}
            ciLow={metric.ciLow}
            ciHigh={metric.ciHigh}
            bandLow={metric.bandLow}
            bandHigh={metric.bandHigh}
            units={metric.units}
          />
        </>
      ) : (
        <Text style={styles.banded}>{bandWord(metric)}</Text>
      )}

      {metric.tier === 'B' && (
        <Text style={styles.tierNote}>
          Shown as a range rather than a number — this one can&apos;t be measured precisely
          enough from a single camera.
        </Text>
      )}

      {showEvidence && metric.source && (
        <Text style={styles.evidence}>
          Compared against {metric.population?.replace(/_/g, ' ') ?? 'reference'} data
          {metric.evidenceConfidence === 'low' ? ' (limited evidence)' : ''}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e6e9ed',
  },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  summary: { flex: 1, fontSize: 15, fontWeight: '600', color: '#1a1f26' },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999, marginLeft: 8 },
  badgeText: { fontSize: 12, fontWeight: '600' },
  value: { fontSize: 28, fontWeight: '700', color: '#1a1f26', marginTop: 12 },
  banded: { fontSize: 18, fontWeight: '600', color: '#3d4650', marginTop: 12 },
  unclear: { fontSize: 14, lineHeight: 20, color: '#5b6470', marginTop: 10 },
  tierNote: { fontSize: 12, color: '#7a838d', marginTop: 10, fontStyle: 'italic' },
  evidence: { fontSize: 12, color: '#7a838d', marginTop: 8 },
});
