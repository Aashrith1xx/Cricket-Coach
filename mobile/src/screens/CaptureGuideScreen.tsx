/**
 * Guided capture setup.
 *
 * The highest-leverage screen in the app. Capture conditions dominate measurement accuracy -
 * more than the pose model, more than any downstream processing - because a joint angle is
 * only measurable when it lies in the camera's image plane. Getting the player to stand the
 * camera in the right place is worth more than any algorithmic improvement available to us.
 *
 * The frame-rate check is not advisory. At 30 fps the whole arm-horizontal-to-release window
 * is 1-3 frames and bat-ball contact falls between frames entirely, so release- and
 * impact-referenced metrics are suppressed rather than estimated below 120 fps.
 *
 * Status: layout and requirements real; camera integration TODO.
 */

import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import type { CaptureRequirements, FormType } from '../types';

export const CAPTURE_REQUIREMENTS: Record<FormType, CaptureRequirements> = {
  cover_drive: {
    formType: 'cover_drive',
    view: 'side_on',
    distanceM: [6, 10],
    heightM: 1.5,
    minFps: 120,
    preferredFps: 240,
    requiresTripod: true,
    optionalSecondAngle: 'front_on',
  },
  pull_shot: {
    formType: 'pull_shot',
    view: 'side_on',
    distanceM: [6, 10],
    heightM: 1.5,
    minFps: 120,
    preferredFps: 240,
    requiresTripod: true,
    optionalSecondAngle: 'front_on',
  },
  fast_arm_bowling: {
    formType: 'fast_arm_bowling',
    view: 'side_on',
    distanceM: [10, 15],
    heightM: 1.5,
    minFps: 120,
    preferredFps: 240,
    requiresTripod: true,
    optionalSecondAngle: null,
  },
  spin_bowling: {
    formType: 'spin_bowling',
    view: 'side_on',
    distanceM: [8, 12],
    heightM: 1.5,
    minFps: 120,
    preferredFps: 240,
    requiresTripod: true,
    optionalSecondAngle: null,
  },
};

interface Props {
  formType: FormType;
  deviceMaxFps: number | null;
  onReady: () => void;
}

export function CaptureGuideScreen({ formType, deviceMaxFps }: Props) {
  const req = CAPTURE_REQUIREMENTS[formType];
  const fpsOk = deviceMaxFps === null || deviceMaxFps >= req.minFps;

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Set up your camera</Text>
      <Text style={styles.lede}>
        Where you put the camera matters more than anything else. Angles we can&apos;t see
        square-on, we can&apos;t measure — so this bit is worth getting right.
      </Text>

      {/* TODO: framing overlay with a silhouette and a live "you're at the right angle" check */}
      <View style={styles.diagramPlaceholder}>
        <Text style={styles.diagramText}>Camera position diagram — TODO</Text>
      </View>

      <Step
        n={1}
        title={req.view === 'side_on' ? 'Stand square to the side' : 'Stand in front'}
        body={
          req.view === 'side_on'
            ? 'Put the camera directly to the side, square on. Not at an angle, not behind. If it’s off by more than about 20°, we’ll ask you to reshoot.'
            : 'Put the camera directly in front, facing the player.'
        }
      />
      <Step
        n={2}
        title={`${req.distanceM[0]}–${req.distanceM[1]} metres away`}
        body={`About chest height (${req.heightM} m). You should fill most of the frame — if you look small, move closer.`}
      />
      <Step
        n={3}
        title="Rest the phone on something solid"
        body="A tripod, a bag, a fence post — anything that doesn’t move. Handheld footage gets rejected, because camera movement looks exactly like player movement to the analysis."
      />
      <Step
        n={4}
        title="Turn on slow motion"
        body={`Record at ${req.preferredFps} fps if your phone offers it, ${req.minFps} fps at minimum. At normal speed the moment of release or contact happens between frames, so we simply can’t see it.`}
        warning={!fpsOk}
      />

      {!fpsOk && (
        <View style={styles.warning}>
          <Text style={styles.warningText}>
            This device seems to top out at {deviceMaxFps} fps. You can still record, and
            timing and posture feedback will work — but anything measured at the exact moment
            of contact or release will be skipped rather than guessed at.
          </Text>
        </View>
      )}

      {req.optionalSecondAngle && (
        <View style={styles.optional}>
          <Text style={styles.optionalTitle}>Optional: a second angle</Text>
          <Text style={styles.optionalBody}>
            A front-on clip as well lets us look at how your hips and shoulders separate, and
            which way your backlift goes. Neither can be seen from the side.
          </Text>
        </View>
      )}

      <Text style={styles.footnote}>
        We check the clip before analysing it. If something&apos;s off, we&apos;ll tell you
        exactly what to change rather than giving you numbers we don&apos;t trust.
      </Text>
    </ScrollView>
  );
}

function Step({
  n,
  title,
  body,
  warning,
}: {
  n: number;
  title: string;
  body: string;
  warning?: boolean;
}) {
  return (
    <View style={[styles.step, warning && styles.stepWarning]}>
      <View style={styles.stepNumber}>
        <Text style={styles.stepNumberText}>{n}</Text>
      </View>
      <View style={styles.stepBody}>
        <Text style={styles.stepTitle}>{title}</Text>
        <Text style={styles.stepText}>{body}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, paddingBottom: 48 },
  title: { fontSize: 26, fontWeight: '700', color: '#1a1f26' },
  lede: { fontSize: 15, lineHeight: 22, color: '#5b6470', marginTop: 8, marginBottom: 20 },
  diagramPlaceholder: {
    height: 180,
    backgroundColor: '#f7f8fa',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#e6e9ed',
    borderStyle: 'dashed',
    marginBottom: 24,
  },
  diagramText: { color: '#7a838d', fontSize: 13 },
  step: { flexDirection: 'row', marginBottom: 20 },
  stepWarning: { opacity: 1 },
  stepNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#1a1f26',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  stepNumberText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  stepBody: { flex: 1 },
  stepTitle: { fontSize: 16, fontWeight: '600', color: '#1a1f26', marginBottom: 4 },
  stepText: { fontSize: 14, lineHeight: 20, color: '#5b6470' },
  warning: {
    backgroundColor: '#fff4e5',
    borderRadius: 10,
    padding: 14,
    marginBottom: 20,
  },
  warningText: { fontSize: 13, lineHeight: 19, color: '#8a5300' },
  optional: {
    backgroundColor: '#f7f8fa',
    borderRadius: 10,
    padding: 14,
    marginBottom: 20,
  },
  optionalTitle: { fontSize: 14, fontWeight: '600', color: '#1a1f26', marginBottom: 4 },
  optionalBody: { fontSize: 13, lineHeight: 19, color: '#5b6470' },
  footnote: { fontSize: 12, lineHeight: 18, color: '#7a838d', marginTop: 8 },
});
