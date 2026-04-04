export type InferenceMode = 'demo' | 'remote';

function toInt(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

const rawMode = (import.meta.env.VITE_INFERENCE_MODE ?? 'demo') as string;

export const appConfig = {
  inferenceMode: (rawMode === 'remote' ? 'remote' : 'demo') as InferenceMode,
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000/api/pinn',
  requestTimeoutMs: toInt(import.meta.env.VITE_REQUEST_TIMEOUT_MS, 15000),
  maxRetries: Math.max(0, Math.min(toInt(import.meta.env.VITE_MAX_RETRIES, 2), 5)),
  localPreviewCacheTtlMs: Math.max(60_000, Math.min(toInt(import.meta.env.VITE_LOCAL_PREVIEW_CACHE_TTL_MS, 1_800_000), 86_400_000))
};
