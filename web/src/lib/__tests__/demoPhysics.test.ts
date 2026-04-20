import { describe, expect, it } from 'vitest';
import { createDemoAdapter, computeReynolds } from '../demoPhysics';
import { defaultScenario, demoPresets } from '../presets';

const bendScenario = demoPresets.find((preset) => preset.id === 'b-base')!.input;
const strongContraction = demoPresets.find((preset) => preset.id === 'c-test-2')!.input;


describe('demo adapter', () => {
  const adapter = createDemoAdapter();

  it('generates a stable contraction scenario result', async () => {
    const result = await adapter.simulate(defaultScenario);
    expect(result.field.length).toBeGreaterThan(800);
    expect(result.metrics.maxSpeed).toBeGreaterThan(0);
    expect(result.probes?.mainCenterline.length ?? 0).toBeGreaterThan(10);
  });

  it('can also simulate bend_2d scenarios', async () => {
    const result = await adapter.simulate(bendScenario);
    expect(result.field.length).toBeGreaterThan(700);
    expect(result.streamlines?.length ?? 0).toBeGreaterThan(5);
    expect(result.probes?.branchCenterline?.length ?? 0).toBeGreaterThan(5);
  });

  it('increases peak speed for a stronger contraction', async () => {
    const base = await adapter.simulate(defaultScenario);
    const stronger = await adapter.simulate(strongContraction);
    expect(stronger.metrics.maxSpeed).toBeGreaterThan(base.metrics.maxSpeed);
  });

  it('can reconstruct from sparse points', async () => {
    const result = await adapter.reconstruct(defaultScenario);
    expect(result.sparsePoints?.length).toBeGreaterThan(10);
    expect(result.reconstruction?.length).toBe(result.field.length);
    const changed = result.reconstruction?.some((point, index) => {
      const baseline = result.field[index];
      return Math.abs(point.speed - baseline.speed) > 1e-9 || Math.abs(point.p - baseline.p) > 1e-9;
    });
    expect(changed).toBe(true);
  });

  it('supports viscosity calibration and parameter sweep', async () => {
    const base = await adapter.simulate(defaultScenario);
    const targets = base.field.slice(0, 12);
    const calibration = await adapter.calibrateViscosity(defaultScenario, targets);
    expect(calibration.bestViscosity).toBeGreaterThan(0);

    const sweep = await adapter.sweep(defaultScenario, 'viscosity', [
      defaultScenario.fluid.viscosity * 0.8,
      defaultScenario.fluid.viscosity,
      defaultScenario.fluid.viscosity * 1.2
    ]);
    expect(sweep).toHaveLength(3);
  });

  it('computes low Reynolds numbers for the default case', () => {
    expect(computeReynolds(defaultScenario)).toBeLessThan(10);
  });
});
