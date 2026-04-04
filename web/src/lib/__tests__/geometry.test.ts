import { describe, expect, it } from 'vitest';
import { buildGeometry, isPointInsideGeometry, sampleContractionWidthAt } from '../geometry';
import { defaultScenario, demoPresets } from '../presets';

const bendScenario = demoPresets.find((preset) => preset.id === 'b-base')!.input;

describe('geometry mask', () => {
  it('uses contraction_2d as the default geometry family', () => {
    expect(defaultScenario.geometry.type).toBe('contraction');
    const geometry = buildGeometry(defaultScenario);
    expect(geometry.meta.familyLabel).toBe('contraction_2d');
    expect(isPointInsideGeometry(defaultScenario, { x: 200, y: 0 })).toBe(true);
  });

  it('narrows the channel through the contraction section', () => {
    const startWidth = sampleContractionWidthAt(defaultScenario, 100);
    const throatWidth = sampleContractionWidthAt(defaultScenario, 1600);
    expect(startWidth).toBeGreaterThan(throatWidth);
    expect(throatWidth).toBeCloseTo(defaultScenario.geometry.wUm * defaultScenario.geometry.beta, 0);
  });

  it('excludes far outside points for contraction geometry', () => {
    expect(isPointInsideGeometry(defaultScenario, { x: 3000, y: 3000 })).toBe(false);
  });

  it('builds a bend_2d polygon and keeps the bend centerline inside the channel', () => {
    const geometry = buildGeometry(bendScenario);
    expect(geometry.meta.familyLabel).toBe('bend_2d');
    expect(geometry.guideSegments.length).toBeGreaterThan(4);
    expect(isPointInsideGeometry(bendScenario, geometry.junction)).toBe(true);
  });

  it('extends the bend outlet toward the rotated downstream direction', () => {
    const geometry = buildGeometry(bendScenario);
    expect((geometry.centerlines.right?.end.y ?? 0)).toBeGreaterThan((geometry.centerlines.right?.start.y ?? 0));
    const width = geometry.bounds.xMax - geometry.bounds.xMin;
    const height = geometry.bounds.yMax - geometry.bounds.yMin;
    expect(width).toBeGreaterThan(height);
  });

  it('supports tighter bend cases from the preset library', () => {
    const tightBend = demoPresets.find((preset) => preset.id === 'b-test-1-blunted')!.input;
    const geometry = buildGeometry(tightBend);
    expect(geometry.meta.rcUm).toBeCloseTo(tightBend.geometry.rcOverW * tightBend.geometry.wUm, 6);
    expect(isPointInsideGeometry(tightBend, geometry.centerlines.stem.start)).toBe(true);
  });
});
