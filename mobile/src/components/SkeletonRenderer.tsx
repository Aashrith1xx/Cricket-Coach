/**
 * Scrubable skeleton overlay.
 *
 * Written against the `RenderPayload` contract defined in
 * `analysis-service/render_output.py`. Both sides agree on the shape before either is
 * finished, which is why the contract lives in one place and is versioned.
 *
 * Two rendering rules carry meaning rather than being cosmetic:
 *
 *  1. **Low-confidence joints are drawn dimmed.** The player should be able to see where the
 *     analysis was unsure. A uniformly crisp skeleton drawn over guessed joints implies a
 *     certainty the system does not have.
 *  2. **Event markers are drawn as bands, not lines.** Each event carries `frameSem`;
 *     impact in particular is inferred from peak wrist speed rather than observed, so a hard
 *     line on the scrubber would overstate it.
 *
 * Status: contract real; SVG drawing TODO.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import type { EventMarker, FramePose, RenderPayload } from '../types';

interface Props {
  payload: RenderPayload;
  frameIndex: number;
  /** Overlay a reference skeleton for comparison. Off by default - it invites a
   *  "match this shape" reading, which is wrong for a metric set built on bands. */
  showReference?: boolean;
  width: number;
  height: number;
}

export function SkeletonRenderer({ payload, frameIndex, showReference = false, width, height }: Props) {
  const frame: FramePose | undefined = payload.frames[frameIndex];

  if (!frame) {
    return (
      <View style={[styles.container, { width, height }]}>
        <Text style={styles.placeholder}>No pose data for this frame</Text>
      </View>
    );
  }

  // TODO: render with react-native-svg.
  //
  //   - coordinates in `frame.xy` are normalised to [0, 1]; multiply by width/height
  //   - draw `payload.skeletonEdges` as lines, `frame.xy` as circles
  //   - joints with confidence < payload.lowConfidenceThreshold render at reduced opacity
  //     AND with a dashed stroke, so the distinction survives greyscale and colour-blindness
  //   - when showReference, draw payload.referenceFrames beneath at low opacity
  return (
    <View style={[styles.container, { width, height }]}>
      <Text style={styles.placeholder}>
        Skeleton overlay — TODO ({payload.keypointNames.length} joints,{' '}
        {payload.skeletonEdges.length} edges)
      </Text>
    </View>
  );
}

/**
 * Scrubber tick for one phase event.
 *
 * Renders `frameSem` as a band around the marker. Exported separately so the scrubber can
 * lay these out without the skeleton renderer owning timeline layout.
 */
export function EventTick({
  event,
  totalFrames,
  trackWidth,
}: {
  event: EventMarker;
  totalFrames: number;
  trackWidth: number;
}) {
  if (event.frame === null) return null;

  const centre = (event.frame / totalFrames) * trackWidth;
  const bandWidth = Math.max((event.frameSem / totalFrames) * trackWidth * 2, 3);

  return (
    <View style={[styles.eventBand, { left: centre - bandWidth / 2, width: bandWidth }]}>
      <Text style={styles.eventLabel}>{event.name}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#f7f8fa',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholder: { fontSize: 12, color: '#7a838d', textAlign: 'center', paddingHorizontal: 16 },
  eventBand: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    backgroundColor: 'rgba(26, 31, 38, 0.18)',
    borderRadius: 2,
  },
  eventLabel: { fontSize: 9, color: '#5b6470', position: 'absolute', top: -14 },
});
