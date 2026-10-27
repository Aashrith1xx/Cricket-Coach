/**
 * Appwrite client, collection ids, and the document shapes this app stores.
 *
 * Status: client wiring is TODO; the schema below is the real contract and is what the
 * mobile app and analysis service are written against.
 */

import { Client, Databases, Storage, ID, Query } from 'node-appwrite';

const endpoint = process.env.APPWRITE_ENDPOINT;
const projectId = process.env.APPWRITE_PROJECT_ID;
const apiKey = process.env.APPWRITE_API_KEY;

export const DATABASE_ID = process.env.APPWRITE_DATABASE_ID || 'cricket_form';

export const COLLECTIONS = Object.freeze({
  USERS: 'users',
  SESSIONS: 'sessions',
  CLIPS: 'clips',
  ANALYSIS_RESULTS: 'analysisResults',
});

export const BUCKETS = Object.freeze({
  CLIPS: 'clips',
  RENDER_DATA: 'renderData',
});

/**
 * Document shapes.
 *
 * users
 *   clerkUserId          string, indexed
 *   declaredStyle        'attacking' | 'balanced' | 'defensive'
 *                        Selects reference template variants. DECLARED by the player -
 *                        the pose-based intent classifier only corroborates it, because at
 *                        ~74-77% accuracy it would mis-route roughly one user in four.
 *   observedStyleHistory JSON, rolling intent distribution. Advisory only.
 *   handedness           'left' | 'right'
 *   playerGroup          'senior' | 'junior_u19' | 'junior_u16' | 'unknown'
 *                        Drives the population caveat - elite adult male reference ranges
 *                        must not be applied silently to a junior.
 *   consentForTraining   boolean, explicit opt-in, editable. Not implied by uploading.
 *
 * sessions
 *   userId, formType, sessionStyleOverride (nullable), createdAt, clipCount
 *
 * clips
 *   sessionId, userId, formType, videoFileId, frontOnFileId (nullable)
 *   captureMetadata      JSON: fps, width, height, duration, device, declaredView
 *   gateResult           JSON: passed, rejections[], warnings[], suppressedMetricKeys[]
 *   status               'queued' | 'processing' | 'done' | 'rejected' | 'failed'
 *   retentionState       'original' | 'proxy_only' | 'keypoints_only'
 *
 * analysisResults
 *   clipId, sessionId, userId
 *   metrics              JSON, each {value, ciLow, ciHigh, tier, convention, confidence}
 *   phaseEvents          JSON, each event with frame uncertainty
 *   observedIntent       JSON {label, confidence} - cross-check only, never gates grading
 *   styleMarkers         JSON - backlift direction, trigger, stance width. Descriptive,
 *                        unscored. See docs/ACCURACY.md section 6 for why.
 *   compositeScore       number, NULLABLE. Null means too little reliable evidence, and the
 *                        client must render observations without an overall grade rather
 *                        than substituting a default.
 *   compositeCiLow/High  number, nullable
 *   feedbackText         JSON, the FeedbackBundle
 *   templateId, templateVersion, styleVariant
 *   renderDataUrl        string - payload lives in Storage, not inline; per-frame keypoints
 *                        are far too large for a document field
 */

let client = null;

export function getClient() {
  if (!endpoint || !projectId || !apiKey) {
    throw new Error(
      'Appwrite is not configured. Set APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID and APPWRITE_API_KEY.',
    );
  }
  if (!client) {
    client = new Client().setEndpoint(endpoint).setProject(projectId).setKey(apiKey);
  }
  return client;
}

export function databases() {
  return new Databases(getClient());
}

export function storage() {
  return new Storage(getClient());
}

export { ID, Query };

/**
 * TODO: collection creation / migration helper.
 *
 * Worth writing as code rather than clicking through the console, so the schema above and
 * the deployed reality cannot drift apart.
 */
export async function ensureSchema() {
  throw new Error('appwrite.ensureSchema is not implemented');
}
