import type { DemoPreset, ScenarioInput } from '../types/pinn';

const baseFluid = {
  preset: 'water' as const,
  density: 997.05,
  viscosity: 8.902e-4
};

const baseFlow = {
  meanVelocity: 0.0001,
  outletPressure: 0
};

const baseSparse = {
  sampleRatePct: 10,
  noisePct: 2,
  strategy: 'region_aware' as const
};

export const defaultScenario: ScenarioInput = {
  geometry: {
    type: 'contraction',
    wUm: 200,
    lInOverW: 4,
    lOutOverW: 8,
    beta: 0.7,
    lCOverW: 4,
    rcOverW: 6,
    thetaDeg: 90,
    inletProfile: 'parabolic'
  },
  fluid: baseFluid,
  flow: baseFlow,
  sparse: baseSparse
};

export const demoPresets: DemoPreset[] = [
  {
    id: 'c-base',
    name: '收缩流道 / C-base',
    subtitle: 'β = 0.70 · Lc/W = 4 · Lin/W = 4 · Lout/W = 8',
    description: '对应 pinn_v4 当前 contraction_2d 单工况基线，用于展示喉部加速与压降变化。',
    input: defaultScenario
  },
  {
    id: 'c-test-2',
    name: '收缩流道 / C-test-2',
    subtitle: 'β = 0.40 · Lc/W = 6',
    description: '更强收缩、更长过渡段的外推挑战工况，便于展示几何变化对场分布的影响。',
    input: {
      geometry: {
        type: 'contraction',
        wUm: 200,
        lInOverW: 4,
        lOutOverW: 8,
        beta: 0.4,
        lCOverW: 6,
        rcOverW: 6,
        thetaDeg: 90,
        inletProfile: 'parabolic'
      },
      fluid: baseFluid,
      flow: baseFlow,
      sparse: {
        ...baseSparse,
        sampleRatePct: 15,
        noisePct: 3
      }
    }
  },
  {
    id: 'b-base',
    name: '弯曲流道 / B-base',
    subtitle: 'Rc/W = 6 · θ = 90° · parabolic',
    description: '对应 pinn_v4 当前 bend_2d 单工况基线，用于展示弯道曲率和压力分布。',
    input: {
      geometry: {
        type: 'bend',
        wUm: 200,
        lInOverW: 4,
        lOutOverW: 6,
        beta: 0.7,
        lCOverW: 4,
        rcOverW: 6,
        thetaDeg: 90,
        inletProfile: 'parabolic'
      },
      fluid: baseFluid,
      flow: baseFlow,
      sparse: baseSparse
    }
  },
  {
    id: 'b-test-1-blunted',
    name: '弯曲流道 / B-test-1',
    subtitle: 'Rc/W = 3 · θ = 90° · blunted',
    description: '高曲率挑战工况，并叠加 blunted 入口剖面，突出弯道中的入口条件影响。',
    input: {
      geometry: {
        type: 'bend',
        wUm: 200,
        lInOverW: 4,
        lOutOverW: 6,
        beta: 0.7,
        lCOverW: 4,
        rcOverW: 3,
        thetaDeg: 90,
        inletProfile: 'blunted'
      },
      fluid: baseFluid,
      flow: baseFlow,
      sparse: {
        ...baseSparse,
        sampleRatePct: 5
      }
    }
  }
];
