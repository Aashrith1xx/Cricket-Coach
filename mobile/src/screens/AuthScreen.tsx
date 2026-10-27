/**
 * Sign in, and the training-data consent prompt.
 *
 * Consent is an explicit opt-in, defaulted OFF, and is **not implied by uploading a clip**.
 * The app retains analysed clips as future model-training data, and that is a materially
 * different thing from processing a video and discarding it, so it is asked plainly rather
 * than buried in terms.
 *
 * The wording is deliberately concrete about what is kept, because "we may use your data to
 * improve our services" is not informed consent for retaining video of a named person.
 *
 * Declining costs the player nothing: analysis is identical either way. Only retention
 * differs — keypoints are kept regardless (they are anonymous coordinate data and are what
 * makes progress tracking possible), while the video itself is kept only with consent.
 *
 * Status: layout real; Clerk wiring TODO.
 */

import React, { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';

interface Props {
  onSignIn: () => void;
  onConsentChange: (consent: boolean) => void;
}

export function AuthScreen({ onSignIn, onConsentChange }: Props) {
  const [consent, setConsent] = useState(false);

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Cricket Form</Text>
      <Text style={styles.lede}>
        Upload a clip of a shot or a delivery, and get feedback on your technique that
        you can track over time.
      </Text>

      {/* TODO: Clerk sign-in component */}
      <Pressable style={styles.primary} onPress={onSignIn}>
        <Text style={styles.primaryText}>Sign in</Text>
      </Pressable>

      <View style={styles.consentCard}>
        <View style={styles.consentHeader}>
          <Text style={styles.consentTitle}>Help improve the analysis</Text>
          <Switch
            value={consent}
            onValueChange={(v) => {
              setConsent(v);
              onConsentChange(v);
            }}
          />
        </View>
        <Text style={styles.consentBody}>
          If you turn this on, we keep your clips so we can improve how the app reads
          technique. If you leave it off, your video is deleted once it&apos;s been analysed.
        </Text>
        <Text style={styles.consentBody}>
          Either way we keep the movement data we extract — that&apos;s just coordinates, no
          footage — because that&apos;s what lets you see your progress over time.
        </Text>
        <Text style={styles.consentFootnote}>
          You can change this whenever you like, and you get exactly the same feedback either
          way.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 24, paddingTop: 80, paddingBottom: 48 },
  title: { fontSize: 32, fontWeight: '800', color: '#1a1f26' },
  lede: { fontSize: 16, lineHeight: 24, color: '#5b6470', marginTop: 10, marginBottom: 32 },
  primary: {
    backgroundColor: '#1a1f26',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 32,
  },
  primaryText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  consentCard: {
    backgroundColor: '#f7f8fa',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e6e9ed',
  },
  consentHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  consentTitle: { fontSize: 15, fontWeight: '600', color: '#1a1f26', flex: 1 },
  consentBody: { fontSize: 13, lineHeight: 19, color: '#5b6470', marginBottom: 8 },
  consentFootnote: { fontSize: 12, lineHeight: 18, color: '#7a838d', fontStyle: 'italic' },
});
