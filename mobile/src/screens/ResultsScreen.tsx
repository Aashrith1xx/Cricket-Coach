/**
 * Analysis results for one clip.
 *
 * Three outcomes, all normal, all rendered differently:
 *
 *  1. **Rejected.** The capture gate refused the clip. This is a *successful* outcome with
 *     actionable guidance, not an error, and is presented as "here's what to change" rather
 *     than as a failure.
 *  2. **Done, with a score.** Enough reliable metrics survived to justify an overall grade.
 *  3. **Done, without a score.** `compositeScore === null` means too little reliable
 *     evidence. The screen shows the individual observations and no overall number. It must
 *     never substitute 0, "N/A" styled as a grade, or a placeholder — fabricating an
 *     authoritative-looking score from thin evidence is exactly what the null is there to
 *     prevent.
 *
 * Status: layout real; data wiring TODO.
 */

import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import type { ClipResult } from '../types';
import { MetricCard } from '../components/MetricCard';
import { StyleMarkerCard } from '../components/StyleMarkerCard';

interface Props {
  result: ClipResult;
  styleMarkers?: { key: string; label: string; value: string }[];
}

export function ResultsScreen({ result, styleMarkers = [] }: Props) {
  if (result.status === 'rejected') {
    return <RejectedView result={result} />;
  }

  if (result.status === 'failed') {
    return (
      <View style={styles.centred}>
        <Text style={styles.title}>Something went wrong on our side</Text>
        <Text style={styles.body}>
          Your clip was fine — this one&apos;s on us. Try again in a moment.
        </Text>
      </View>
    );
  }

  const scored = result.metrics.filter((m) => m.verdict !== 'not_scored');
  const unclear = result.metrics.filter((m) => m.verdict === 'unclear');
  const graded = scored.filter((m) => m.verdict !== 'unclear');

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      {result.compositeScore === null ? (
        <View style={styles.noScoreCard}>
          <Text style={styles.noScoreTitle}>No overall score for this one</Text>
          <Text style={styles.noScoreBody}>
            {result.suppressedReason ??
              'Not enough of this clip could be measured reliably to give you an overall ' +
                'mark. Here’s what we could see.'}
          </Text>
        </View>
      ) : (
        <View style={styles.scoreCard}>
          <Text style={styles.scoreValue}>{Math.round(result.compositeScore * 100)}</Text>
          <Text style={styles.scoreOutOf}>out of 100</Text>
          {result.compositeCiLow !== null && result.compositeCiHigh !== null && (
            <Text style={styles.scoreRange}>
              likely between {Math.round(result.compositeCiLow * 100)} and{' '}
              {Math.round(result.compositeCiHigh * 100)}
            </Text>
          )}
        </View>
      )}

      {result.feedback && (
        <>
          <Text style={styles.headline}>{result.feedback.headline}</Text>
          {result.feedback.items
            .filter((i) => i.kind === 'work_on')
            .map((item, i) => (
              <Text key={`w${i}`} style={styles.workOn}>
                • {item.text}
              </Text>
            ))}
          {result.feedback.items
            .filter((i) => i.kind === 'strength')
            .map((item, i) => (
              <Text key={`s${i}`} style={styles.strength}>
                • {item.text}
              </Text>
            ))}
        </>
      )}

      {result.warnings.length > 0 && (
        <View style={styles.warnings}>
          {result.warnings.map((w, i) => (
            <Text key={i} style={styles.warningText}>
              {w}
            </Text>
          ))}
        </View>
      )}

      <StyleMarkerCard markers={styleMarkers} />

      <Text style={styles.sectionTitle}>What we measured</Text>
      {graded.map((m) => (
        <MetricCard key={m.key} metric={m} showEvidence />
      ))}

      {unclear.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Couldn&apos;t measure this time</Text>
          <Text style={styles.sectionNote}>
            These weren&apos;t clear enough in this clip to call either way. Usually a camera
            angle or frame rate fix.
          </Text>
          {unclear.map((m) => (
            <MetricCard key={m.key} metric={m} />
          ))}
        </>
      )}
    </ScrollView>
  );
}

function RejectedView({ result }: { result: ClipResult }) {
  const rejections = result.gate?.rejections ?? [];
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Let&apos;s get a better clip</Text>
      <Text style={styles.body}>
        We&apos;d rather ask you to reshoot than give you numbers we don&apos;t trust.
        Here&apos;s what to change:
      </Text>

      {rejections.map((r, i) => (
        <View key={i} style={styles.rejection}>
          <Text style={styles.rejectionRemedy}>{r.remedy}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, paddingBottom: 48 },
  centred: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32 },
  title: { fontSize: 24, fontWeight: '700', color: '#1a1f26', marginBottom: 8 },
  body: { fontSize: 15, lineHeight: 22, color: '#5b6470', marginBottom: 20 },

  scoreCard: {
    alignItems: 'center',
    backgroundColor: '#f7f8fa',
    borderRadius: 16,
    paddingVertical: 28,
    marginBottom: 20,
  },
  scoreValue: { fontSize: 56, fontWeight: '800', color: '#1a1f26' },
  scoreOutOf: { fontSize: 14, color: '#7a838d', marginTop: -4 },
  scoreRange: { fontSize: 13, color: '#7a838d', marginTop: 8 },

  noScoreCard: {
    backgroundColor: '#eef1f4',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
  },
  noScoreTitle: { fontSize: 17, fontWeight: '700', color: '#1a1f26', marginBottom: 6 },
  noScoreBody: { fontSize: 14, lineHeight: 20, color: '#5b6470' },

  headline: { fontSize: 17, fontWeight: '600', color: '#1a1f26', marginBottom: 12 },
  workOn: { fontSize: 15, lineHeight: 22, color: '#8a5300', marginBottom: 6 },
  strength: { fontSize: 15, lineHeight: 22, color: '#2f7d32', marginBottom: 6 },

  warnings: { backgroundColor: '#fff4e5', borderRadius: 10, padding: 14, marginVertical: 16 },
  warningText: { fontSize: 13, lineHeight: 19, color: '#8a5300' },

  sectionTitle: { fontSize: 18, fontWeight: '700', color: '#1a1f26', marginTop: 24, marginBottom: 4 },
  sectionNote: { fontSize: 13, lineHeight: 19, color: '#7a838d', marginBottom: 12 },

  rejection: { backgroundColor: '#fff4e5', borderRadius: 10, padding: 16, marginBottom: 12 },
  rejectionRemedy: { fontSize: 15, lineHeight: 22, color: '#8a5300' },
});
