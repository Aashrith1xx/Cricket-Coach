/**
 * Traces a keypoint's path across the clip - most usefully the head, and the lead wrist.
 *
 * The head trace is the one worth showing. Head-ball coupling is the strongest measurable
 * discriminator in batting (elite batters hold a mean head-ball angle of 2.4 deg against
 * 3.7 deg for club players, with maxima of 10.7 vs 20.6 deg), and the head is also the
 * best-tracked region in pose estimation - large, high contrast, rarely self-occluded. Skill
 * signal and measurability coinciding is rare, so the head trace is the single most
 * informative thing this overlay can draw.
 *
 * It stops at the head, though: gaze is not inferable from head pose, and the app makes no
 * claim about where the player was looking.
 *
 * Status: contract real; drawing TODO.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import type { RenderPayload } from '../types';

interface Props {
  payload: RenderPayload;
  keypointName: string;
  /** Restrict to a phase span, e.g. DS to IMP for the head trace. */
  fromEvent?: string;
  toEvent?: string;
  width: number;
  height: number;
}

export function TrajectoryOverlay({
  payload,
  keypointName,
  fromEvent,
  toEvent,
  width,
  height,
}: Props) {
  const jointIndex = payload.keypointNames.indexOf(keypointName);
  if (jointIndex < 0) return null;

  const startFrame = fromEvent
    ? (payload.events.find((e) => e.name === fromEvent)?.frame ?? 0)
    : 0;
  const endFrame = toEvent
    ? (payload.events.find((e) => e.name === toEvent)?.frame ?? payload.nFrames)
    : payload.nFrames;

  // TODO: render with react-native-svg.
  //
  //   - build a polyline from frame.xy[jointIndex * 2], [jointIndex * 2 + 1] over the span
  //   - segments where confidence < payload.lowConfidenceThreshold render dashed, so an
  //     uncertain stretch of the path is visibly uncertain rather than smoothed into a
  //     confident-looking curve
  //   - do NOT interpolate across missing frames; break the line instead
  return (
    <View style={[styles.container, { width, height }]}>
      <Text style={styles.placeholder}>
        {keypointName} path, frames {startFrame}–{endFrame} — TODO
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { position: 'absolute', top: 0, left: 0, alignItems: 'center', justifyContent: 'center' },
  placeholder: { fontSize: 11, color: '#7a838d' },
});
