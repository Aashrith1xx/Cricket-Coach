/**
 * Technical style markers: backlift direction, trigger movement, stance width.
 *
 * Rendered **neutrally and never as a score**. No tick, no cross, no colour coding, no
 * ordering that implies better or worse.
 *
 * This is not politeness, it is accuracy. The clearest case is the backlift: 77% of
 * successful batsmen and 90% of IPL batsmen use a *lateral* backlift, while 83% of coaches
 * teach the *straight* one. Marking a lateral backlift as a fault would penalise the
 * technique most elite players actually use. Trigger movement is the same story - its
 * prevalence ranges from 47% in ODI cricket to 76% in the IPL, so absence is a style, not
 * an error.
 *
 * If you are adding a visual "correct/incorrect" treatment here, read docs/ACCURACY.md
 * section 6 first.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

interface StyleMarker {
  key: string;
  label: string;
  value: string;
}

interface Props {
  markers: StyleMarker[];
}

export function StyleMarkerCard({ markers }: Props) {
  if (markers.length === 0) return null;

  return (
    <View style={styles.card}>
      <Text style={styles.title}>Your technique</Text>
      <Text style={styles.subtitle}>
        These are things we noticed about how you play. They&apos;re style choices, not
        faults — good players do all of these differently.
      </Text>

      {markers.map((m) => (
        <View key={m.key} style={styles.row}>
          <Text style={styles.label}>{m.label}</Text>
          <Text style={styles.value}>{m.value}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#f7f8fa',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e6e9ed',
  },
  title: { fontSize: 15, fontWeight: '600', color: '#1a1f26' },
  subtitle: { fontSize: 13, lineHeight: 19, color: '#5b6470', marginTop: 6, marginBottom: 12 },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: '#e6e9ed',
  },
  label: { fontSize: 14, color: '#3d4650' },
  // Same weight and colour as the label: nothing here should read as a verdict.
  value: { fontSize: 14, color: '#1a1f26', fontWeight: '500' },
});
