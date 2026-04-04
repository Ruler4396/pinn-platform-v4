import { useEffect, useMemo, useState } from 'react';
import { FieldCanvas } from './components/FieldCanvas';
import { appConfig } from './lib/config';
import { createInferenceAdapter } from './lib/inferenceAdapter';
import { readPreviewCache, writePreviewCache } from './lib/localResultCache';
import { defaultScenario, demoPresets } from './lib/presets';
import { hashScenario } from './lib/utils';
import type { ActionStatus, DemoPreset, FieldLayer, FieldPoint, ScenarioInput, ScenarioResult } from './types/pinn';

const emptyStatus = (label: string): ActionStatus => ({
  label,
  state: 'idle',
  detail: '待执行',
  updatedAt: undefined
});

const fluidPresets = {
  water: { density: 997.05, viscosity: 8.902e-4 },
  glycerol10: { density: 1035, viscosity: 0.00135 },
  glycerol30: { density: 1100, viscosity: 0.0024 }
} as const;

const customPresetIds = {
  contraction: 'custom-contraction',
  bend: 'custom-bend'
} as const;

function formatVelocity(value: number): string {
  return `${(value * 1000).toFixed(3)} mm/s`;
}

function formatPressure(value: number): string {
  return `${value.toFixed(3)} Pa`;
}

function formatGeometryType(type: ScenarioInput['geometry']['type']): string {
  return type === 'contraction' ? '收缩流道' : '弯曲流道';
}

function geometryFamilyLabel(type: ScenarioInput['geometry']['type']): string {
  return type === 'contraction' ? 'contraction_2d' : 'bend_2d';
}

function presetIdForScenario(input: ScenarioInput): string {
  const target = hashScenario(input);
  const matched = demoPresets.find((preset) => hashScenario(preset.input) === target);
  return matched?.id ?? customPresetIds[input.geometry.type];
}

function getPresetById(id: string): DemoPreset | undefined {
  return demoPresets.find((preset) => preset.id === id);
}

function buildCustomScenario(type: ScenarioInput['geometry']['type'], current: ScenarioInput): ScenarioInput {
  const base = demoPresets.find((preset) => preset.input.geometry.type === type)?.input ?? defaultScenario;
  return {
    geometry: {
      ...base.geometry,
      type,
      wUm: current.geometry.wUm,
      lInOverW: current.geometry.lInOverW,
      lOutOverW: current.geometry.lOutOverW,
      beta: type === 'contraction' && current.geometry.type === 'contraction' ? current.geometry.beta : base.geometry.beta,
      lCOverW:
        type === 'contraction' && current.geometry.type === 'contraction' ? current.geometry.lCOverW : base.geometry.lCOverW,
      rcOverW: type === 'bend' && current.geometry.type === 'bend' ? current.geometry.rcOverW : base.geometry.rcOverW,
      thetaDeg:
        type === 'bend' && current.geometry.type === 'bend' ? current.geometry.thetaDeg : base.geometry.thetaDeg,
      inletProfile:
        type === 'bend' && current.geometry.type === 'bend' ? current.geometry.inletProfile : base.geometry.inletProfile
    },
    fluid: { ...current.fluid },
    flow: { ...current.flow },
    sparse: { ...current.sparse }
  };
}

function normalizeScenario(input: ScenarioInput): ScenarioInput {
  const type = input.geometry.type;
  return {
    ...input,
    geometry: {
      ...input.geometry,
      wUm: Math.min(Math.max(input.geometry.wUm, 120), 260),
      lInOverW: Math.min(Math.max(input.geometry.lInOverW, 2), 8),
      lOutOverW: Math.min(Math.max(input.geometry.lOutOverW, 4), 10),
      beta: type === 'contraction' ? Math.min(Math.max(input.geometry.beta, 0.35), 0.9) : input.geometry.beta,
      lCOverW: type === 'contraction' ? Math.min(Math.max(input.geometry.lCOverW, 2), 8) : input.geometry.lCOverW,
      rcOverW: type === 'bend' ? Math.min(Math.max(input.geometry.rcOverW, 2.5), 8) : input.geometry.rcOverW,
      thetaDeg: type === 'bend' ? Math.min(Math.max(input.geometry.thetaDeg, 45), 120) : input.geometry.thetaDeg
    },
    fluid: {
      ...input.fluid,
      density: Math.min(Math.max(input.fluid.density, 900), 1300),
      viscosity: Math.min(Math.max(input.fluid.viscosity, 0.0004), 0.008)
    },
    flow: {
      ...input.flow,
      meanVelocity: Math.min(Math.max(input.flow.meanVelocity, 0.00001), 0.0012),
      outletPressure: Math.min(Math.max(input.flow.outletPressure, -20), 20)
    },
    sparse: {
      ...input.sparse,
      sampleRatePct: Math.min(Math.max(input.sparse.sampleRatePct, 5), 15),
      noisePct: Math.min(Math.max(input.sparse.noisePct, 0), 10)
    }
  };
}

export default function App() {
  const [scenario, setScenario] = useState<ScenarioInput>(defaultScenario);
  const [solvedScenario, setSolvedScenario] = useState<ScenarioInput>(defaultScenario);
  const [selectedPresetId, setSelectedPresetId] = useState<string>(() => presetIdForScenario(defaultScenario));
  const [fieldLayer, setFieldLayer] = useState<FieldLayer>('speed');
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [reconstructionResult, setReconstructionResult] = useState<ScenarioResult | null>(null);
  const [showReconstruction, setShowReconstruction] = useState(false);
  const [previewCacheState, setPreviewCacheState] = useState<'idle' | 'hit' | 'miss'>('idle');
  const [streamlineLoadState, setStreamlineLoadState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [probeInput, setProbeInput] = useState({ x: 0, y: 0 });
  const [probeResult, setProbeResult] = useState<FieldPoint | null>(null);
  const [, setStatuses] = useState<Record<string, ActionStatus>>({
    simulate: emptyStatus('simulate'),
    reconstruct: emptyStatus('reconstruct'),
    probe: emptyStatus('probe'),
    streamlines: emptyStatus('streamlines')
  });

  const adapter = useMemo(
    () =>
      createInferenceAdapter({
        updateStatus(label, status) {
          setStatuses((current) => ({
            ...current,
            [String(label)]: status
          }));
        }
      }),
    []
  );

  useEffect(() => {
    const normalizedDefault = normalizeScenario(defaultScenario);
    const cached = readPreviewCache(normalizedDefault);
    if (cached) {
      setScenario(normalizedDefault);
      setSolvedScenario(normalizedDefault);
      setSelectedPresetId(presetIdForScenario(normalizedDefault));
      setResult(cached);
      setPreviewCacheState('hit');
      return;
    }
    void handleSimulate(defaultScenario);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scenarioDirty = useMemo(
    () => hashScenario(normalizeScenario(scenario)) !== hashScenario(solvedScenario),
    [scenario, solvedScenario]
  );

  const activePresetId = selectedPresetId;

  const updateScenario = <K extends keyof ScenarioInput>(group: K, value: ScenarioInput[K]) => {
    setScenario((current) => ({
      ...current,
      [group]: value
    }));
    setSelectedPresetId(customPresetIds[scenario.geometry.type]);
  };

  const updateGeometry = <K extends keyof ScenarioInput['geometry']>(
    key: K,
    value: ScenarioInput['geometry'][K]
  ) => {
    setScenario((current) => ({
      ...current,
      geometry: {
        ...current.geometry,
        [key]: value
      }
    }));
    setSelectedPresetId(customPresetIds[scenario.geometry.type]);
  };

  const updateFlow = <K extends keyof ScenarioInput['flow']>(
    key: K,
    value: ScenarioInput['flow'][K]
  ) => {
    setScenario((current) => ({
      ...current,
      flow: {
        ...current.flow,
        [key]: value
      }
    }));
    setSelectedPresetId(customPresetIds[scenario.geometry.type]);
  };

  const updateFluid = <K extends keyof ScenarioInput['fluid']>(
    key: K,
    value: ScenarioInput['fluid'][K]
  ) => {
    setScenario((current) => ({
      ...current,
      fluid: {
        ...current.fluid,
        [key]: value
      }
    }));
    setSelectedPresetId(customPresetIds[scenario.geometry.type]);
  };

  const updateSparse = <K extends keyof ScenarioInput['sparse']>(
    key: K,
    value: ScenarioInput['sparse'][K]
  ) => {
    setScenario((current) => ({
      ...current,
      sparse: {
        ...current.sparse,
        [key]: value
      }
    }));
    setSelectedPresetId(customPresetIds[scenario.geometry.type]);
  };

  const handleApplyPreset = (presetId: string) => {
    if (presetId === customPresetIds.contraction) {
      setSelectedPresetId(customPresetIds.contraction);
      setScenario((current) => normalizeScenario(buildCustomScenario('contraction', current)));
      return;
    }
    if (presetId === customPresetIds.bend) {
      setSelectedPresetId(customPresetIds.bend);
      setScenario((current) => normalizeScenario(buildCustomScenario('bend', current)));
      return;
    }
    const preset = getPresetById(presetId);
    if (!preset) {
      return;
    }
    setSelectedPresetId(preset.id);
    setScenario(normalizeScenario(preset.input));
  };

  const handleSimulate = async (input = scenario) => {
    const normalizedInput = normalizeScenario(input);
    if (hashScenario(normalizedInput) !== hashScenario(input)) {
      setScenario(normalizedInput);
    }
    try {
      const next = await adapter.simulate(normalizedInput, {
        resolution: 'preview',
        includeStreamlines: false,
        includeProbes: false,
        includeSparsePoints: false,
        includeReconstruction: false
      });
      writePreviewCache(normalizedInput, next);
      setSolvedScenario(normalizedInput);
      setResult(next);
      setReconstructionResult(null);
      setShowReconstruction(false);
      setPreviewCacheState('miss');
      setStreamlineLoadState('idle');
      setProbeResult(null);
      if (fieldLayer === 'streamline') {
        void handleLoadStreamlines(normalizedInput);
      }
    } catch (error) {
      setStatuses((current) => ({
        ...current,
        simulate: {
          label: 'simulate',
          state: 'error',
          detail: error instanceof Error ? error.message : '计算失败'
        }
      }));
    }
  };

  const handleLoadStreamlines = async (input = solvedScenario) => {
    if (streamlineLoadState === 'loading') {
      return;
    }
    setStreamlineLoadState('loading');
    try {
      const streamlines = await adapter.loadStreamlines(input, { resolution: 'preview' });
      setResult((current) =>
        current
          ? {
              ...current,
              streamlines
            }
          : current
      );
      setStreamlineLoadState('ready');
    } catch {
      setStreamlineLoadState('error');
    }
  };

  const handleQuery = async (point = probeInput) => {
    setStatuses((current) => ({
      ...current,
      probe: {
        label: 'probe',
        state: 'running',
        detail: '查询中'
      }
    }));
    try {
      const value = await adapter.queryPoint(solvedScenario, point);
      setProbeResult(value);
      setStatuses((current) => ({
        ...current,
        probe: {
          label: 'probe',
          state: 'success',
          detail: value ? '已返回点位参数' : '该点位于流道外'
        }
      }));
    } catch (error) {
      setStatuses((current) => ({
        ...current,
        probe: {
          label: 'probe',
          state: 'error',
          detail: error instanceof Error ? error.message : '查询失败'
        }
      }));
    }
  };

  const handleCanvasQuery = async (point: { x: number; y: number }) => {
    setProbeInput(point);
    await handleQuery(point);
  };

  const handleReconstruct = async () => {
    try {
      const normalizedScenario = normalizeScenario(scenario);
      if (hashScenario(normalizedScenario) !== hashScenario(scenario)) {
        setScenario(normalizedScenario);
      }
      const next = await adapter.reconstruct(normalizedScenario, {
        resolution: 'preview',
        includeStreamlines: false,
        includeProbes: false
      });
      setSolvedScenario(normalizedScenario);
      setResult({
        field: next.field,
        metrics: next.metrics
      });
      setReconstructionResult(next);
      setShowReconstruction(true);
      setStreamlineLoadState('idle');
      setProbeResult(null);
      if (fieldLayer === 'streamline') {
        void handleLoadStreamlines(normalizedScenario);
      }
    } catch (error) {
      setStatuses((current) => ({
        ...current,
        reconstruct: {
          label: 'reconstruct',
          state: 'error',
          detail: error instanceof Error ? error.message : '重建失败'
        }
      }));
    }
  };

  const activeResult = showReconstruction && reconstructionResult ? reconstructionResult : result;
  const displayStreamlines = activeResult?.streamlines ?? result?.streamlines;
  const activeSparsePoints =
    showReconstruction && reconstructionResult?.sparsePoints
      ? reconstructionResult.sparsePoints
      : result?.sparsePoints;
  const activeReconstruction =
    showReconstruction && reconstructionResult?.reconstruction
      ? reconstructionResult.reconstruction
      : undefined;
  const currentMetrics = activeResult?.metrics;

  const handleLayerChange = (nextLayer: FieldLayer) => {
    setFieldLayer(nextLayer);
    if (nextLayer === 'streamline' && !displayStreamlines?.length) {
      void handleLoadStreamlines();
    }
  };

  return (
    <div className="app-shell">
      <header className="page-hero">
        <div className="hero-copy">
          <span className="hero-kicker">PINN FLOW VISUAL DEMO V4</span>
          <h1>收缩 / 弯曲流道交互工作台</h1>
          <div className="hero-meta">
            <span className="mode-chip">
              {appConfig.inferenceMode === 'demo' ? 'demo numerical layer' : 'remote inference'}
            </span>
            <span className={scenarioDirty ? 'state-chip warning' : 'state-chip ok'}>
              {scenarioDirty ? '参数已修改，待重算' : '当前结果已同步'}
            </span>
            {previewCacheState === 'hit' ? <span className="state-chip ok">首屏缓存命中</span> : null}
            <span className="state-chip subtle">{formatGeometryType(solvedScenario.geometry.type)}</span>
          </div>
        </div>
        <div className="hero-actions">
          <div className="hero-select-field">
            <select aria-label="案例预设" value={activePresetId} onChange={(event) => handleApplyPreset(event.target.value)}>
              {demoPresets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
              <option value={customPresetIds.contraction}>自定义收缩流道</option>
              <option value={customPresetIds.bend}>自定义弯曲流道</option>
            </select>
          </div>
          <button type="button" className="primary" onClick={() => void handleSimulate()}>
            {scenarioDirty ? '更新流场' : '重新计算'}
          </button>
          <button type="button" className="secondary" onClick={() => void handleReconstruct()}>
            稀疏重建
          </button>
          <button
            type="button"
            className={showReconstruction ? 'secondary active' : 'secondary'}
            onClick={() => setShowReconstruction((value) => !value)}
            disabled={!reconstructionResult}
          >
            {showReconstruction ? '显示基准场' : '显示重建场'}
          </button>
        </div>
      </header>

      <div className="workspace-grid">
        <main className="stage-column">
          <section className="visual-stage-card stage-focus-card">
            <div className="stage-topbar">
              <div className="layer-switches">
                <button
                  type="button"
                  className={fieldLayer === 'speed' ? 'mini-toggle active' : 'mini-toggle'}
                  onClick={() => handleLayerChange('speed')}
                >
                  速度场
                </button>
                <button
                  type="button"
                  className={fieldLayer === 'pressure' ? 'mini-toggle active' : 'mini-toggle'}
                  onClick={() => handleLayerChange('pressure')}
                >
                  压力场
                </button>
                <button
                  type="button"
                  className={fieldLayer === 'streamline' ? 'mini-toggle active' : 'mini-toggle'}
                  onClick={() => handleLayerChange('streamline')}
                >
                  流线
                </button>
              </div>

              <div className="stage-stats">
                <div>
                  <span>峰值速度</span>
                  <strong>{currentMetrics ? formatVelocity(currentMetrics.maxSpeed) : '—'}</strong>
                </div>
                <div>
                  <span>平均压降</span>
                  <strong>{currentMetrics ? formatPressure(currentMetrics.avgPressureDrop) : '—'}</strong>
                </div>
                <div>
                  <span>显示模式</span>
                  <strong>{showReconstruction ? '重建场' : '基准场'}</strong>
                </div>
                <div>
                  <span>Re</span>
                  <strong>{currentMetrics ? currentMetrics.reynolds.toFixed(3) : '—'}</strong>
                </div>
                <div>
                  <span>壁面代理</span>
                  <strong>{currentMetrics ? currentMetrics.wallShearProxy.toExponential(2) : '—'}</strong>
                </div>
                <div>
                  <span>曲率代理</span>
                  <strong>{currentMetrics ? currentMetrics.streamlineCurvatureProxy.toFixed(3) : '—'}</strong>
                </div>
              </div>
            </div>

            <FieldCanvas
              input={solvedScenario}
              result={activeResult}
              streamlines={displayStreamlines}
              layer={fieldLayer}
              reconstruction={activeReconstruction}
              sparsePoints={activeSparsePoints}
              probe={probeResult}
              onQuery={handleCanvasQuery}
            />
          </section>
        </main>

        <section className="support-grid" aria-label="参数与分析面板">
          <section className="panel-card support-panel-card">
            <div className="panel-head compact">
              <div>
                <span className="card-kicker">Geometry</span>
                <h2>流道参数</h2>
              </div>
              <span className="panel-note">{geometryFamilyLabel(scenario.geometry.type)}</span>
            </div>

            <div className="field-grid two-col">
              <label>
                通道宽度 W (μm)
                <input
                  type="number"
                  min="120"
                  max="260"
                  value={scenario.geometry.wUm}
                  onChange={(event) => updateGeometry('wUm', Number(event.target.value))}
                />
              </label>

              <label>
                入口直段 Lin / W
                <input
                  type="number"
                  min="2"
                  max="8"
                  step="0.5"
                  value={scenario.geometry.lInOverW}
                  onChange={(event) => updateGeometry('lInOverW', Number(event.target.value))}
                />
              </label>

              <label>
                出口直段 Lout / W
                <input
                  type="number"
                  min="4"
                  max="10"
                  step="0.5"
                  value={scenario.geometry.lOutOverW}
                  onChange={(event) => updateGeometry('lOutOverW', Number(event.target.value))}
                />
              </label>

              {scenario.geometry.type === 'contraction' ? (
                <>
                  <label>
                    收缩比 β
                    <input
                      type="number"
                      min="0.35"
                      max="0.9"
                      step="0.05"
                      value={scenario.geometry.beta}
                      onChange={(event) => updateGeometry('beta', Number(event.target.value))}
                    />
                  </label>
                  <label className="span-2">
                    收缩段长度 Lc / W
                    <input
                      type="number"
                      min="2"
                      max="8"
                      step="0.5"
                      value={scenario.geometry.lCOverW}
                      onChange={(event) => updateGeometry('lCOverW', Number(event.target.value))}
                    />
                  </label>
                </>
              ) : (
                <>
                  <label>
                    曲率半径 Rc / W
                    <input
                      type="number"
                      min="2.5"
                      max="8"
                      step="0.5"
                      value={scenario.geometry.rcOverW}
                      onChange={(event) => updateGeometry('rcOverW', Number(event.target.value))}
                    />
                  </label>
                  <label>
                    弯角 θ (°)
                    <input
                      type="number"
                      min="45"
                      max="120"
                      step="5"
                      value={scenario.geometry.thetaDeg}
                      onChange={(event) => updateGeometry('thetaDeg', Number(event.target.value))}
                    />
                  </label>
                  <label className="span-2">
                    入口剖面
                    <select
                      value={scenario.geometry.inletProfile}
                      onChange={(event) =>
                        updateGeometry('inletProfile', event.target.value as ScenarioInput['geometry']['inletProfile'])
                      }
                    >
                      <option value="parabolic">parabolic</option>
                      <option value="blunted">blunted</option>
                      <option value="skewed_top">skewed_top</option>
                      <option value="skewed_bottom">skewed_bottom</option>
                    </select>
                  </label>
                </>
              )}
            </div>
          </section>

          <section className="panel-card support-panel-card">
            <div className="panel-head compact">
              <div>
                <span className="card-kicker">Physics</span>
                <h2>流体与采样</h2>
              </div>
              <span className="panel-note">核心控制</span>
            </div>

            <div className="field-grid two-col">
              <label>
                流体预设
                <select
                  value={scenario.fluid.preset}
                  onChange={(event) => {
                    const preset = event.target.value as ScenarioInput['fluid']['preset'];
                    if (preset === 'custom') {
                      updateFluid('preset', preset);
                      return;
                    }
                    updateScenario('fluid', {
                      preset,
                      density: fluidPresets[preset].density,
                      viscosity: fluidPresets[preset].viscosity
                    });
                  }}
                >
                  <option value="water">25℃ 去离子水</option>
                  <option value="glycerol10">10% 甘油</option>
                  <option value="glycerol30">30% 甘油</option>
                  <option value="custom">自定义</option>
                </select>
              </label>

              <label>
                采样策略
                <select
                  value={scenario.sparse.strategy}
                  onChange={(event) =>
                    updateSparse('strategy', event.target.value as ScenarioInput['sparse']['strategy'])
                  }
                >
                  <option value="region_aware">region_aware</option>
                  <option value="uniform">uniform</option>
                </select>
              </label>

              <label>
                密度 ρ (kg/m³)
                <input
                  type="number"
                  min="900"
                  max="1300"
                  value={scenario.fluid.density}
                  onChange={(event) => updateFluid('density', Number(event.target.value))}
                />
              </label>

              <label>
                黏度 μ (Pa·s)
                <input
                  type="number"
                  step="0.0001"
                  min="0.0004"
                  max="0.008"
                  value={scenario.fluid.viscosity}
                  onChange={(event) => {
                    updateFluid('preset', 'custom');
                    updateFluid('viscosity', Number(event.target.value));
                  }}
                />
              </label>

              <label>
                平均入口速度 (mm/s)
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="1.2"
                  value={Number((scenario.flow.meanVelocity * 1000).toFixed(3))}
                  onChange={(event) => updateFlow('meanVelocity', Number(event.target.value) / 1000)}
                />
              </label>

              <label>
                出口压力 (Pa)
                <input
                  type="number"
                  step="0.1"
                  min="-20"
                  max="20"
                  value={scenario.flow.outletPressure}
                  onChange={(event) => updateFlow('outletPressure', Number(event.target.value))}
                />
              </label>
            </div>

            <div className="field-grid two-col range-grid">
              <label>
                采样率 {scenario.sparse.sampleRatePct}%
                <input
                  type="range"
                  min="5"
                  max="15"
                  step="5"
                  value={scenario.sparse.sampleRatePct}
                  onChange={(event) => updateSparse('sampleRatePct', Number(event.target.value))}
                />
              </label>
              <label>
                噪声 {scenario.sparse.noisePct}%
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="1"
                  value={scenario.sparse.noisePct}
                  onChange={(event) => updateSparse('noisePct', Number(event.target.value))}
                />
              </label>
            </div>
          </section>
          <section className="panel-card support-panel-card probe-panel-card">
            <div className="panel-head compact">
              <div>
                <span className="card-kicker">Probe</span>
                <h2>点位查询</h2>
              </div>
              <span className="panel-note">μm</span>
            </div>

            <div className="field-grid two-col">
              <label>
                x
                <input
                  type="number"
                  value={Number(probeInput.x.toFixed(2))}
                  onChange={(event) => setProbeInput((current) => ({ ...current, x: Number(event.target.value) }))}
                />
              </label>
              <label>
                y
                <input
                  type="number"
                  value={Number(probeInput.y.toFixed(2))}
                  onChange={(event) => setProbeInput((current) => ({ ...current, y: Number(event.target.value) }))}
                />
              </label>
            </div>

            <button type="button" className="primary full-width" onClick={() => void handleQuery()}>
              查询点位
            </button>

            {probeResult ? (
              <dl className="data-list compact-data-list inline-metric-list">
                <div>
                  <dt>speed</dt>
                  <dd>{formatVelocity(probeResult.speed)}</dd>
                </div>
                <div>
                  <dt>ux</dt>
                  <dd>{formatVelocity(probeResult.ux)}</dd>
                </div>
                <div>
                  <dt>uy</dt>
                  <dd>{formatVelocity(probeResult.uy)}</dd>
                </div>
                <div>
                  <dt>pressure</dt>
                  <dd>{formatPressure(probeResult.p)}</dd>
                </div>
              </dl>
            ) : (
              <div className="empty-state compact-empty">点击流道或输入坐标。</div>
            )}
          </section>
        </section>
      </div>
    </div>
  );
}
