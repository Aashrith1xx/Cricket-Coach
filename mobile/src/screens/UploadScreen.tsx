/**
 * Record or pick a clip, then upload.
 *
 * The app asks for 3-5 clips per session rather than one. That is not a nicety: a session
 * mean shrinks the standard error by sqrt(n), and with realistic measurement error a
 * single-clip minimal detectable change is roughly 28 degrees on a knee angle, versus about
 * 12 with five clips. One clip cannot support a claim about progress.
 *
 * Status: layout real; camera and upload TODO.
 */

import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import type { FormType } from '../types';

const TARGET_CLIPS = 5;
const MIN_CLIPS = 3;

interface Props {
  formType: FormType;
  clipsRecorded: number;
  uploading: boolean;
  onRecord: () => void;
  onPickExisting: () => void;
  onFinishSession: () => void;
}

export function UploadScreen({
  clipsRecorded,
  uploading,
  onRecord,
  onPickExisting,
  onFinishSession,
}: Props) {
  const enough = clipsRecorded >= MIN_CLIPS;

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>This session</Text>

      <View style={styles.counter}>
        <Text style={styles.counterValue}>
          {clipsRecorded}
          <Text style={styles.counterTarget}> / {TARGET_CLIPS}</Text>
        </Text>
        <Text style={styles.counterLabel}>clips recorded</Text>
      </View>

      <Text style={styles.why}>
        {clipsRecorded < MIN_CLIPS
          ? `Record at least ${MIN_CLIPS}. Any single clip has enough natural variation that we can't tell a real change from measurement wobble — averaging across a few fixes that.`
          : clipsRecorded < TARGET_CLIPS
            ? `Enough to work with. ${TARGET_CLIPS} would tighten the numbers further.`
            : 'Good sample size — your numbers this session will be about as tight as we can make them.'}
      </Text>

      <Pressable style={styles.primary} onPress={onRecord} disabled={uploading}>
        <Text style={styles.primaryText}>Record a clip</Text>
      </Pressable>

      <Pressable style={styles.secondary} onPress={onPickExisting} disabled={uploading}>
        <Text style={styles.secondaryText}>Choose from camera roll</Text>
      </Pressable>

      {enough && (
        <Pressable style={styles.finish} onPress={onFinishSession} disabled={uploading}>
          <Text style={styles.finishText}>Finish session</Text>
        </Pressable>
      )}

      <Text style={styles.footnote}>
        Analysis takes a minute or two per clip. You can close the app — we&apos;ll have
        results waiting.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, paddingBottom: 48 },
  title: { fontSize: 26, fontWeight: '700', color: '#1a1f26', marginBottom: 20 },
  counter: { alignItems: 'center', paddingVertical: 24 },
  counterValue: { fontSize: 48, fontWeight: '800', color: '#1a1f26' },
  counterTarget: { fontSize: 28, fontWeight: '600', color: '#a4acb5' },
  counterLabel: { fontSize: 14, color: '#7a838d', marginTop: 2 },
  why: { fontSize: 14, lineHeight: 21, color: '#5b6470', textAlign: 'center', marginBottom: 28 },
  primary: {
    backgroundColor: '#1a1f26',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 12,
  },
  primaryText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  secondary: {
    borderWidth: 1.5,
    borderColor: '#e6e9ed',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 12,
  },
  secondaryText: { color: '#3d4650', fontSize: 16, fontWeight: '500' },
  finish: { paddingVertical: 16, alignItems: 'center' },
  finishText: { color: '#2f7d32', fontSize: 16, fontWeight: '600' },
  footnote: { fontSize: 12, lineHeight: 18, color: '#7a838d', textAlign: 'center', marginTop: 16 },
});
