export type GeometryType = 'contraction' | 'bend';
export type InletProfile = 'parabolic' | 'blunted' | 'skewed_top' | 'skewed_bottom';
export type FluidPreset = 'water' | 'glycerol10' | 'glycerol30' | 'custom';
export type SparseStrategy = 'region_aware' | 'uniform';
export type SweepVariable = 'meanVelocity' | 'viscosity';
export type FieldLayer = 'speed' | 'pressure' | 'streamline';
export type FieldResolution = 'preview' | 'full';
export type SectionKey =
  | 'overview'
  | 'geometry'
  | 'visualization'
  | 'reconstruction'
  | 'analysis'
  | 'calibration'
  | 'methods';

export interface ScenarioInput {
  geometry: {
    type: GeometryType;
    wUm: number;
    lInOverW: number;
    lOutOverW: number;
    beta: number;
    lCOverW: number;
    rcOverW: number;
    thetaDeg: number;
    inletProfile: InletProfile;
  };
  fluid: {
    preset: FluidPreset;
    density: number;
    viscosity: number;
  };
  flow: {
    meanVelocity: number;
    outletPressure: number;
  };
  sparse: {
    sampleRatePct: number;
    noisePct: number;
    strategy: SparseStrategy;
  };
}

export interface FieldPoint {
  x: number;
  y: number;
  ux: number;
  uy: number;
  p: number;
  speed: number;
}

export interface ScenarioMetrics {
  reynolds: number;
  maxSpeed: number;
  avgPressureDrop: number;
  flowSplitRatio?: number;
  wallShearProxy: number;
  streamlineCurvatureProxy: number;
  centerlinePressureGradient: number;
}

export interface CurvePoint {
  s: number;
  speed: number;
  p: number;
}

export interface ScenarioProbes {
  mainCenterline: CurvePoint[];
  branchCenterline?: CurvePoint[];
}

export interface ScenarioResult {
  field: FieldPoint[];
  streamlines?: Array<Array<{ x: number; y: number }>>;
  sparsePoints?: FieldPoint[];
  reconstruction?: FieldPoint[];
  metrics: ScenarioMetrics;
  probes?: ScenarioProbes;
}

export interface SimulateOptions {
  resolution?: FieldResolution;
  includeStreamlines?: boolean;
  includeProbes?: boolean;
  includeSparsePoints?: boolean;
  includeReconstruction?: boolean;
}

export interface InferenceAdapter {
  simulate(input: ScenarioInput, options?: SimulateOptions): Promise<ScenarioResult>;
  queryPoint(
    input: ScenarioInput,
    point: { x: number; y: number }
  ): Promise<FieldPoint | null>;
  reconstruct(input: ScenarioInput, options?: SimulateOptions): Promise<ScenarioResult>;
  loadStreamlines(
    input: ScenarioInput,
    options?: Pick<SimulateOptions, 'resolution'>
  ): Promise<NonNullable<ScenarioResult['streamlines']>>;
  loadProbes(input: ScenarioInput): Promise<ScenarioProbes>;
  calibrateViscosity(
    input: ScenarioInput,
    targetPoints: FieldPoint[]
  ): Promise<{ bestViscosity: number; error: number }>;
  sweep(
    input: ScenarioInput,
    variable: SweepVariable,
    values: number[]
  ): Promise<Array<{ value: number; metrics: ScenarioMetrics }>>;
}

export interface ActionStatus {
  label: string;
  state: 'idle' | 'running' | 'retrying' | 'success' | 'error';
  detail: string;
  updatedAt?: string;
}

export interface DemoPreset {
  id: string;
  name: string;
  subtitle: string;
  description: string;
  input: ScenarioInput;
}
