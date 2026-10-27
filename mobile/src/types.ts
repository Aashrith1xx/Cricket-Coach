/**
 * Shared types: the contract between the analysis service and this app.
 *
 * Mirrors `analysis-service/render_output.py` and `analysis-service/scoring.py`. If the
 * payload shape changes there, `RENDER_SCHEMA_VERSION` is bumped and this file must follow.
 *
 * The tier system is expressed in the types themselves so that the display rules are hard
 * to violate by accident:
 *   - Tier C never appears in a payload at all, so there is no type for it here.
 *   - Tier B annotations carry `value: null` - the band is what gets rendered.
 *   - `compositeScore` is nullable, meaning "too little reliable evidence for a grade".
 */

export const RENDER_SCHEMA_VERSION = '1.0.0';

export type Tier = 'A' | 'B' | 'DESCRIPTOR';

export type Verdict = 'in_range' | 'out_of_range' | 'unclear' | 'not_scored';

export type FormType = 'cover_drive' | 'pull_shot' | 'fast_arm_bowling' | 'spin_bowling';

export type Style = 'attacking' | 'balanced' | 'defensive';

export type EvidenceConfidence = 'high' | 'medium' | 'low';

export type Population =
  | 'elite_male'
  | 'elite_female'
  | 'elite_mixed'
  | 'club'
  | 'junior_u19'
  | 'junior_u16'
  | 'mixed_standard'
  | 'unknown';

/**
 * One metric as shown to the player.
 *
 * `value`, `ciLow` and `ciHigh` are non-null only for Tier A. Tier B is banded by
 * definition and must never be rendered as a precise number - see `MetricCard`.
 */
export interface MetricAnnotation {
  key: string;
  summary: string;
  verdict: Verdict;
  tier: Tier;
  value: number | null;
  ciLow: number | null;
  ciHigh: number | null;
  bandLow: number | null;
  bandHigh: number | null;
  units: string;
  event: string;
  source: string | null;
  population: Population | null;
  evidenceConfidence: EvidenceConfidence | null;
}

/**
 * A detected phase event.
 *
 * `frameSem` is the frame uncertainty. The scrubber renders it as a band rather than a hard
 * line, because presenting an estimated impact frame as exact would overstate what the
 * system knows.
 */
export interface EventMarker {
  name: string;
  frame: number | null;
  timeS: number | null;
  confidence: number;
  frameSem: number;
}

export interface FramePose {
  frame: number;
  timeS: number;
  /** Flattened [x0, y0, x1, y1, ...], normalised to [0, 1]. */
  xy: number[];
  confidence: number[];
}

export interface RenderPayload {
  schemaVersion: string;
  clipId: string;
  fps: number;
  nFrames: number;
  keypointNames: string[];
  skeletonEdges: number[][];
  frames: FramePose[];
  events: EventMarker[];
  metrics: MetricAnnotation[];
  lowConfidenceThreshold: number;
  referenceFrames: FramePose[];
}

export type FeedbackKind = 'strength' | 'work_on' | 'unmeasured' | 'descriptor' | 'meta';

export interface FeedbackItem {
  kind: FeedbackKind;
  metricKey: string | null;
  text: string;
}

export interface FeedbackBundle {
  headline: string;
  items: FeedbackItem[];
}

export type ClipStatus = 'queued' | 'processing' | 'done' | 'rejected' | 'failed';

export interface Rejection {
  code: string;
  detail: string;
  /** What the player should do differently. Always present - a bare rejection is useless. */
  remedy: string;
}

export interface GateResult {
  passed: boolean;
  rejections: Rejection[];
  warnings: string[];
  suppressedMetricKeys: string[];
  effectiveFps: number | null;
}

/**
 * Analysis result for one clip.
 *
 * `compositeScore === null` is a normal, expected outcome, not an error: it means too few
 * metrics survived to justify an overall grade. The UI shows observations instead. Do not
 * substitute 0 or a placeholder.
 */
export interface ClipResult {
  clipId: string;
  status: ClipStatus;
  gate: GateResult | null;
  compositeScore: number | null;
  compositeCiLow: number | null;
  compositeCiHigh: number | null;
  suppressedReason: string | null;
  feedback: FeedbackBundle | null;
  metrics: MetricAnnotation[];
  renderDataUrl: string | null;
  templateId: string | null;
  templateVersion: string | null;
  styleVariant: string | null;
  warnings: string[];
  error: string | null;
}

export type ChangeVerdict =
  | 'improved'
  | 'regressed'
  | 'no_change_detected'
  | 'insufficient_data';

export interface TrendPoint {
  date: string;
  sessionId: string;
  mean: number;
  ciLow: number;
  ciHigh: number;
  windowSessions: number;
}

export interface MetricTrend {
  metricKey: string;
  summary: string;
  units: string;
  points: TrendPoint[];
  sufficient: boolean;
  reason?: string;
  /** null means the metric has an optimum band rather than a direction. */
  higherIsBetter: boolean | null;
}

export interface SessionQuality {
  level: 'good' | 'ok' | 'low';
  note: string | null;
}

export interface CaptureRequirements {
  formType: FormType;
  view: 'side_on' | 'front_on';
  distanceM: [number, number];
  heightM: number;
  minFps: number;
  preferredFps: number;
  requiresTripod: boolean;
  optionalSecondAngle: 'front_on' | null;
}
