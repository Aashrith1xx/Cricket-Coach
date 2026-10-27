/**
 * Playstyle profile.
 *
 * The player **declares** their style; this selects which reference ranges they are graded
 * against. A pose-based classifier also estimates the intent of each shot, but it never
 * silently changes the reference.
 *
 * Why declared rather than detected: the published classifier reaches 77% accuracy (74% on
 * a held-out player). At roughly one clip in four misclassified, routing grading by
 * classifier output would give a quarter of users the wrong reference band with no signal
 * that anything went wrong — confidently wrong feedback, which is the failure mode this
 * whole system is designed to avoid. It also classifies per-shot *intent* rather than player
 * *style*, and its training labels were derived from runs scored, which conflates intent
 * with outcome.
 *
 * What the classifier is good enough for is noticing divergence and mentioning it. An
 * intent–execution mismatch is useful coaching feedback in its own right.
 *
 * Status: layout real; persistence TODO.
 */

import React, { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import type { Style } from '../types';

const OPTIONS: { value: Style; label: string; blurb: string }[] = [
  {
    value: 'attacking',
    label: 'Attacking',
    blurb: 'You look to score. You back yourself to take the aggressive option.',
  },
  {
    value: 'balanced',
    label: 'Balanced',
    blurb: 'You pick your moments — solid by default, attacking when it’s on.',
  },
  {
    value: 'defensive',
    label: 'Defensive',
    blurb: 'You value your wicket. Occupying the crease comes first.',
  },
];

interface Props {
  declaredStyle: Style | null;
  observedNote: string | null;
  onSelect: (style: Style) => void;
}

export function StyleProfileScreen({ declaredStyle, observedNote, onSelect }: Props) {
  const [selected, setSelected] = useState<Style | null>(declaredStyle);

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>How do you play?</Text>
      <Text style={styles.lede}>
        This changes what we compare you against. There&apos;s no right answer — a good
        defensive player and a good attacking player do genuinely different things, and
        grading them the same way would be wrong.
      </Text>

      {OPTIONS.map((opt) => {
        const active = selected === opt.value;
        return (
          <Pressable
            key={opt.value}
            style={[styles.option, active && styles.optionActive]}
            onPress={() => {
              setSelected(opt.value);
              onSelect(opt.value);
            }}
          >
            <Text style={[styles.optionLabel, active && styles.optionLabelActive]}>
              {opt.label}
            </Text>
            <Text style={styles.optionBlurb}>{opt.blurb}</Text>
          </Pressable>
        );
      })}

      {observedNote && (
        <View style={styles.observed}>
          <Text style={styles.observedTitle}>Something we noticed</Text>
          <Text style={styles.observedBody}>{observedNote}</Text>
          <Text style={styles.observedFootnote}>
            We haven&apos;t changed anything — you decide what profile you&apos;re working to.
          </Text>
        </View>
      )}

      <Text style={styles.footnote}>
        You can change this any time, and you can set a different intent for a single session
        if you&apos;re working on something specific that day.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, paddingBottom: 48 },
  title: { fontSize: 26, fontWeight: '700', color: '#1a1f26' },
  lede: { fontSize: 15, lineHeight: 22, color: '#5b6470', marginTop: 8, marginBottom: 24 },
  option: {
    borderWidth: 1.5,
    borderColor: '#e6e9ed',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  optionActive: { borderColor: '#1a1f26', backgroundColor: '#f7f8fa' },
  optionLabel: { fontSize: 17, fontWeight: '600', color: '#3d4650' },
  optionLabelActive: { color: '#1a1f26' },
  optionBlurb: { fontSize: 14, lineHeight: 20, color: '#5b6470', marginTop: 4 },
  observed: {
    backgroundColor: '#eef1f4',
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
  },
  observedTitle: { fontSize: 15, fontWeight: '600', color: '#1a1f26', marginBottom: 6 },
  observedBody: { fontSize: 14, lineHeight: 20, color: '#3d4650' },
  observedFootnote: { fontSize: 12, color: '#7a838d', marginTop: 8, fontStyle: 'italic' },
  footnote: { fontSize: 12, lineHeight: 18, color: '#7a838d', marginTop: 20 },
});
