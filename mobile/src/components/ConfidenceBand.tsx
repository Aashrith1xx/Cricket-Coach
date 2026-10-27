/**
 * Visualises a measurement against its expected range.
 *
 * Deliberately draws the confidence interval as a *wide bar*, not a point with small error
 * whiskers. Users read a dot as a fact and whiskers as decoration, which is precisely
 * backwards here: with roughly 8 degrees RMSE on the best-measured angle, the interval is
 * the honest content and the point estimate is the decoration.
 *
 * When the interval overlaps the edge of the expected range, the bar is drawn in the neutral
 * "unclear" colour rather than pass/fail colours, matching what `MetricCard` says in words.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

interface Props {
  value: number;
  ciLow: number | null;
  ciHigh: number | null;
  bandLow: number | null;
  bandHigh: number | null;
  units: string;
}

const COLOURS = {
  inRange: '#2f7d32',
  outOfRange: '#b26a00',
  unclear: '#8a939d',
  bandFill: '#e8f5e9',
  track: '#eef1f4',
};

export function ConfidenceBand({ value, ciLow, ciHigh, bandLow, bandHigh }: Props) {
  if (bandLow === null || bandHigh === null || ciLow === null || ciHigh === null) {
    return null;
  }

  // Pad the axis so the interval is never clipped at the edges.
  const span = Math.max(bandHigh - bandLow, ciHigh - ciLow) * 1.6;
  const centre = (bandLow + bandHigh) / 2;
  const axisMin = Math.min(centre - span / 2, ciLow);
  const axisMax = Math.max(centre + span / 2, ciHigh);
  const axisRange = axisMax - axisMin || 1;

  const pct = (v: number) => ((v - axisMin) / axisRange) * 100;

  const straddles =
    !(ciLow >= bandLow && ciHigh <= bandHigh) && !(ciHigh < bandLow || ciLow > bandHigh);
  const inRange = ciLow >= bandLow && ciHigh <= bandHigh;
  const barColour = straddles
    ? COLOURS.unclear
    : inRange
      ? COLOURS.inRange
      : COLOURS.outOfRange;

  return (
    <View style={styles.wrapper}>
      <View style={styles.track}>
        <View
          style={[
            styles.expectedBand,
            { left: `${pct(bandLow)}%`, width: `${pct(bandHigh) - pct(bandLow)}%` },
          ]}
        />
        <View
          style={[
            styles.interval,
            {
              left: `${pct(ciLow)}%`,
              width: `${Math.max(pct(ciHigh) - pct(ciLow), 2)}%`,
              backgroundColor: barColour,
            },
          ]}
        />
      </View>
      <View style={styles.legend}>
        <Text style={styles.legendText}>Typical range</Text>
        <Text style={styles.legendText}>
          {straddles ? 'Too close to call from this clip' : 'Your reading'}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { marginTop: 12 },
  track: {
    height: 28,
    backgroundColor: COLOURS.track,
    borderRadius: 6,
    position: 'relative',
    overflow: 'hidden',
  },
  expectedBand: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    backgroundColor: COLOURS.bandFill,
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: '#c8e6c9',
  },
  interval: { position: 'absolute', top: 9, height: 10, borderRadius: 5, opacity: 0.9 },
  legend: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 6 },
  legendText: { fontSize: 11, color: '#7a838d' },
});
