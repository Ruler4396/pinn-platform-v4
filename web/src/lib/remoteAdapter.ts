import type {
  FieldPoint,
  InferenceAdapter,
  ScenarioProbes,
  ScenarioInput,
  ScenarioMetrics,
  ScenarioResult,
  SimulateOptions,
  SweepVariable
} from '../types/pinn';
import { appConfig } from './config';
import { getBendCoordinateTransform } from './geometry';

interface ProgressEvent {
  label: string;
  state: 'running' | 'retrying' | 'success' | 'error';
  attempt: number;
  message: string;
}

interface CreateRemoteAdapterOptions {
  onProgress?: (event: ProgressEvent) => void;
}

function mapBendPointToDisplay(input: ScenarioInput, point: FieldPoint): FieldPoint {
  if (input.geometry.type !== 'bend') {
    return point;
  }
  const transform = getBendCoordinateTransform(input);
  const mapped = transform.toDisplay({ x: point.x, y: point.y });
  return {
    ...point,
    x: mapped.x,
    y: mapped.y
  };
}

function mapBendPolylineToDisplay(
  input: ScenarioInput,
  line: Array<{ x: number; y: number }>
): Array<{ x: number; y: number }> {
  if (input.geometry.type !== 'bend') {
    return line;
  }
  const transform = getBendCoordinateTransform(input);
  return line.map((point) => transform.toDisplay(point));
}

function mapScenarioResultToDisplay(input: ScenarioInput, result: ScenarioResult): ScenarioResult {
  if (input.geometry.type !== 'bend') {
    return result;
  }
  return {
    ...result,
    field: result.field.map((point) => mapBendPointToDisplay(input, point)),
    streamlines: result.streamlines?.map((line) => mapBendPolylineToDisplay(input, line)),
    sparsePoints: result.sparsePoints?.map((point) => mapBendPointToDisplay(input, point)),
    reconstruction: result.reconstruction?.map((point) => mapBendPointToDisplay(input, point))
  };
}

function mapDisplayPointToModel(input: ScenarioInput, point: { x: number; y: number }): { x: number; y: number } {
  if (input.geometry.type !== 'bend') {
    return point;
  }
  const transform = getBendCoordinateTransform(input);
  return transform.toModel(point);
}

async function postJson<TResponse>(
  path: string,
  payload: unknown,
  onProgress?: (event: ProgressEvent) => void,
  label = path
): Promise<TResponse> {
  const attempts = Math.max(1, appConfig.maxRetries + 1);
  let lastError: unknown;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), appConfig.requestTimeoutMs);
    onProgress?.({
      label,
      state: attempt === 1 ? 'running' : 'retrying',
      attempt,
      message:
        attempt === 1
          ? `远端推理请求已发出，单次超时 ${appConfig.requestTimeoutMs}ms。`
          : `第 ${attempt} 次尝试中，最多重试 ${appConfig.maxRetries} 次。`
    });

    try {
      const response = await fetch(`${appConfig.apiBaseUrl}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = (await response.json()) as TResponse;
      onProgress?.({
        label,
        state: 'success',
        attempt,
        message: '远端推理已完成。'
      });
      return data;
    } catch (error) {
      lastError = error;
      const state = attempt >= attempts ? 'error' : 'retrying';
      onProgress?.({
        label,
        state,
        attempt,
        message:
          attempt >= attempts
            ? `远端推理失败：${error instanceof Error ? error.message : '未知错误'}`
            : `远端推理异常，准备重试：${error instanceof Error ? error.message : '未知错误'}`
      });
    } finally {
      clearTimeout(timeout);
    }
  }

  throw lastError instanceof Error ? lastError : new Error('远端推理失败');
}

export function createRemoteAdapter(options: CreateRemoteAdapterOptions = {}): InferenceAdapter {
  const { onProgress } = options;
  return {
    simulate(input: ScenarioInput, simulateOptions: SimulateOptions = {}): Promise<ScenarioResult> {
      return postJson<ScenarioResult>('/simulate', { input, options: simulateOptions }, onProgress, 'simulate').then(
        (response) => mapScenarioResultToDisplay(input, response)
      );
    },
    queryPoint(input: ScenarioInput, point: { x: number; y: number }): Promise<FieldPoint | null> {
      return postJson<FieldPoint | null>(
        '/query-point',
        { input, point: mapDisplayPointToModel(input, point) },
        onProgress,
        'query'
      ).then((response) => (response ? mapBendPointToDisplay(input, response) : null));
    },
    reconstruct(input: ScenarioInput, simulateOptions: SimulateOptions = {}): Promise<ScenarioResult> {
      return postJson<ScenarioResult>('/reconstruct', { input, options: simulateOptions }, onProgress, 'reconstruct').then(
        (response) => mapScenarioResultToDisplay(input, response)
      );
    },
    loadStreamlines(
      input: ScenarioInput,
      simulateOptions: Pick<SimulateOptions, 'resolution'> = {}
    ): Promise<NonNullable<ScenarioResult['streamlines']>> {
      return postJson<{ streamlines: NonNullable<ScenarioResult['streamlines']> }>(
        '/streamlines',
        { input, options: simulateOptions },
        onProgress,
        'streamlines'
      ).then((response) => response.streamlines.map((line) => mapBendPolylineToDisplay(input, line)));
    },
    loadProbes(input: ScenarioInput): Promise<ScenarioProbes> {
      return postJson<{ probes: ScenarioProbes }>('/probes', { input }, onProgress, 'probes').then(
        (response) => response.probes
      );
    },
    calibrateViscosity(
      input: ScenarioInput,
      targetPoints: FieldPoint[]
    ): Promise<{ bestViscosity: number; error: number }> {
      return postJson<{ bestViscosity: number; error: number }>(
        '/calibrate-viscosity',
        { input, targetPoints },
        onProgress,
        'calibrate'
      );
    },
    sweep(
      input: ScenarioInput,
      variable: SweepVariable,
      values: number[]
    ): Promise<Array<{ value: number; metrics: ScenarioMetrics }>> {
      return postJson<Array<{ value: number; metrics: ScenarioMetrics }>>(
        '/sweep',
        { input, variable, values },
        onProgress,
        'sweep'
      );
    }
  };
}
