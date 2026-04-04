import { appConfig } from './config';
import { hashScenario } from './utils';
import type { ScenarioInput, ScenarioResult } from '../types/pinn';

const CACHE_PREFIX = 'pinn-flow-visual-demo-v4';
const CACHE_VERSION = 3;

interface CacheEntry {
  version: number;
  savedAt: number;
  result: ScenarioResult;
}

function getStorage(): Storage | null {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function buildKey(input: ScenarioInput): string {
  return `${CACHE_PREFIX}:preview:${hashScenario(input)}`;
}

export function readPreviewCache(input: ScenarioInput): ScenarioResult | null {
  const storage = getStorage();
  if (!storage) {
    return null;
  }

  try {
    const raw = storage.getItem(buildKey(input));
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as CacheEntry;
    if (parsed.version !== CACHE_VERSION) {
      storage.removeItem(buildKey(input));
      return null;
    }
    if (Date.now() - parsed.savedAt > appConfig.localPreviewCacheTtlMs) {
      storage.removeItem(buildKey(input));
      return null;
    }
    return parsed.result;
  } catch {
    storage.removeItem(buildKey(input));
    return null;
  }
}

export function writePreviewCache(input: ScenarioInput, result: ScenarioResult): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }

  try {
    const entry: CacheEntry = {
      version: CACHE_VERSION,
      savedAt: Date.now(),
      result
    };
    storage.setItem(buildKey(input), JSON.stringify(entry));
  } catch {
    // ignore quota / serialization failures
  }
}
