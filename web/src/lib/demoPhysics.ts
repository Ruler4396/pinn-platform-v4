import type {
  FieldResolution,
  FieldPoint,
  InferenceAdapter,
  ScenarioProbes,
  ScenarioInput,
  ScenarioMetrics,
  ScenarioResult,
  SimulateOptions,
  SweepVariable
} from '../types/pinn';
import {
  buildGeometry,
  pointAlongSegment,
  pointInPolygon,
  projectToSegment,
  sampleContractionWidthAt,
  type GeometryModel,
  type Point2D,
  type StationedSegment
} from './geometry';
import { clamp, distance, hashScenario, lerp, mean, mulberry32, smoothstep } from './utils';

function parabolicProfile(offset: number, width: number): number {
  const radius = Math.max(width / 2, 1e-6);
  return clamp(1 - (offset / radius) ** 2, 0, 1);
}

function normalizeVector(vector: Point2D): Point2D {
  const length = Math.hypot(vector.x, vector.y) || 1;
  return {
    x: vector.x / length,
    y: vector.y / length
  };
}

function applyDirection(direction: Point2D, magnitude: number): Pick<FieldPoint, 'ux' | 'uy'> {
  return {
    ux: direction.x * magnitude,
    uy: direction.y * magnitude
  };
}

function estimatePressureDrop(viscosity: number, velocity: number, lengthUm: number, widthUm: number): number {
  const lengthM = lengthUm * 1e-6;
  const widthM = widthUm * 1e-6;
  if (widthM <= 0 || lengthM <= 0) {
    return 0;
  }
  return (12 * viscosity * velocity * lengthM) / (widthM * widthM);
}

function getTotalGuideLength(geometry: GeometryModel): number {
  return geometry.guideStations[geometry.guideStations.length - 1]?.end ?? geometry.centerlines.stem.length;
}

function getNearestGuideStation(point: Point2D, geometry: GeometryModel) {
  const projections = geometry.guideStations.map((station) => ({
    station,
    projection: projectToSegment(point, station.segment)
  }));

  projections.sort((left, right) => {
    const leftPenalty = Math.abs(left.projection.t) + left.projection.distance * 0.1;
    const rightPenalty = Math.abs(right.projection.t) + right.projection.distance * 0.1;
    return leftPenalty - rightPenalty;
  });

  return projections[0];
}

function bendProfile(offset: number, width: number, inletProfile: ScenarioInput['geometry']['inletProfile'], blend: number): number {
  const radius = Math.max(width / 2, 1e-6);
  const eta = clamp(offset / radius, -1, 1);
  const base = parabolicProfile(offset, width);

  let profile = base;
  switch (inletProfile) {
    case 'blunted':
      profile = clamp(1.12 - 0.72 * Math.abs(eta) ** 4, 0, 1.22);
      break;
    case 'skewed_top':
      profile = clamp(base * (1 + 0.42 * eta), 0, 1.28);
      break;
    case 'skewed_bottom':
      profile = clamp(base * (1 - 0.42 * eta), 0, 1.28);
      break;
    case 'parabolic':
    default:
      profile = base;
      break;
  }

  return clamp(lerp(profile, base, blend), 0, 1.3);
}

export function computeReynolds(input: ScenarioInput): number {
  const hydraulicLength = input.geometry.wUm * 1e-6;
  return (input.fluid.density * input.flow.meanVelocity * hydraulicLength) / input.fluid.viscosity;
}

function evaluateContractionPoint(
  input: ScenarioInput,
  geometry: GeometryModel,
  point: Point2D
): FieldPoint | null {
  const inside = geometry.polygons.some((polygon) => pointInPolygon(point, polygon));
  if (!inside) {
    return null;
  }

  const w = input.geometry.wUm;
  const throatWidth = w * input.geometry.beta;
  const lIn = input.geometry.lInOverW * w;
  const lC = input.geometry.lCOverW * w;
  const lOut = input.geometry.lOutOverW * w;
  const halfWidth = sampleContractionWidthAt(input, point.x) / 2;
  const localWidth = halfWidth * 2;
  const localMeanVelocity = input.flow.meanVelocity * (w / Math.max(localWidth, 1));
  const profile = parabolicProfile(point.y, localWidth);
  const contractionProgress = lC <= 1e-6 ? 0 : clamp((point.x - lIn) / lC, 0, 1);
  const transitionBlend = smoothstep(0, 1, contractionProgress);
  const centerPull = (1 - input.geometry.beta) * transitionBlend * (1 - Math.abs(point.y) / Math.max(halfWidth, 1));
  const axialBoost = 1 + 0.08 * transitionBlend + 0.06 * smoothstep(lIn + lC * 0.35, lIn + lC, point.x);
  const speedMagnitude = 1.5 * localMeanVelocity * profile * axialBoost;
  const ux = speedMagnitude;
  const uy = -Math.sign(point.y || 1) * input.flow.meanVelocity * centerPull * 0.28;

  const transitionWidth = (w + throatWidth) / 2;
  const dropInlet = estimatePressureDrop(input.fluid.viscosity, input.flow.meanVelocity, lIn, w);
  const dropTransition = estimatePressureDrop(
    input.fluid.viscosity,
    input.flow.meanVelocity * (w / Math.max(transitionWidth, 1)),
    lC,
    transitionWidth
  );
  const dropOutlet = estimatePressureDrop(
    input.fluid.viscosity,
    input.flow.meanVelocity * (w / Math.max(throatWidth, 1)),
    lOut,
    throatWidth
  );

  let remainingPressure = 0;
  if (point.x <= lIn) {
    remainingPressure = dropInlet * (1 - point.x / Math.max(lIn, 1)) + dropTransition + dropOutlet;
  } else if (point.x <= lIn + lC) {
    remainingPressure = dropTransition * (1 - (point.x - lIn) / Math.max(lC, 1)) + dropOutlet;
  } else {
    remainingPressure = dropOutlet * (1 - (point.x - lIn - lC) / Math.max(lOut, 1));
  }

  const throatPenalty = (1 - input.geometry.beta) * smoothstep(lIn + lC * 0.4, lIn + lC, point.x) * 0.12;
  const pressure =
    input.flow.outletPressure +
    remainingPressure +
    throatPenalty +
    (Math.abs(point.y) / Math.max(halfWidth, 1)) * remainingPressure * 0.02;

  return {
    x: point.x,
    y: point.y,
    ux,
    uy,
    p: pressure,
    speed: Math.hypot(ux, uy)
  };
}

function evaluateBendPoint(
  input: ScenarioInput,
  geometry: GeometryModel,
  point: Point2D
): FieldPoint | null {
  const inside = geometry.polygons.some((polygon) => pointInPolygon(point, polygon));
  if (!inside) {
    return null;
  }

  const nearest = getNearestGuideStation(point, geometry);
  const { station, projection } = nearest;
  const totalGuideLength = Math.max(getTotalGuideLength(geometry), 1);
  const pathS = station.start + projection.s * station.segment.length;
  const pathRatio = clamp(pathS / totalGuideLength, 0, 1);
  const arcStart = geometry.meta.arcStart ?? totalGuideLength * 0.25;
  const arcEnd = geometry.meta.arcEnd ?? totalGuideLength * 0.7;
  const inArc = smoothstep(arcStart, arcStart + geometry.meta.wUm * 0.5, pathS) *
    (1 - smoothstep(arcEnd - geometry.meta.wUm * 0.5, arcEnd + geometry.meta.wUm * 0.4, pathS));

  const baseProfile = bendProfile(
    projection.t,
    geometry.meta.wUm,
    input.geometry.inletProfile,
    smoothstep(0.3, 0.9, pathRatio)
  );
  const eta = clamp(projection.t / Math.max(geometry.meta.wUm / 2, 1), -1, 1);
  const curvatureRatio = geometry.meta.wUm / Math.max(geometry.meta.rcUm ?? geometry.meta.wUm * 2, geometry.meta.wUm);
  const curvatureBias = -eta * curvatureRatio * inArc * 0.18;
  const localMeanVelocity = input.flow.meanVelocity * (1 + curvatureBias);
  const speedMagnitude = 1.5 * localMeanVelocity * baseProfile * (1 + inArc * 0.05);
  const directional = applyDirection(station.segment.dir, speedMagnitude);
  const radialCorrection = station.segment.normal.y * 0; // kept explicit for readability / later remote alignment
  const ux = directional.ux + station.segment.normal.x * (-eta * inArc * input.flow.meanVelocity * 0.05);
  const uy = directional.uy + station.segment.normal.y * (-eta * inArc * input.flow.meanVelocity * 0.05 + radialCorrection);

  const totalDropBase = estimatePressureDrop(
    input.fluid.viscosity,
    input.flow.meanVelocity,
    totalGuideLength,
    geometry.meta.wUm
  );
  const bendPenaltyFactor = 1 + curvatureRatio * (input.geometry.thetaDeg / 90) * 0.28;
  const pressure =
    input.flow.outletPressure +
    totalDropBase * bendPenaltyFactor * (1 - pathRatio) +
    inArc * curvatureRatio * 0.18 +
    eta * curvatureRatio * 0.06;

  return {
    x: point.x,
    y: point.y,
    ux,
    uy,
    p: pressure,
    speed: Math.hypot(ux, uy)
  };
}

function evaluatePointOnGeometry(
  input: ScenarioInput,
  geometry: GeometryModel,
  point: Point2D
): FieldPoint | null {
  if (geometry.type === 'bend') {
    return evaluateBendPoint(input, geometry, point);
  }
  return evaluateContractionPoint(input, geometry, point);
}

function evaluatePoint(input: ScenarioInput, point: Point2D): FieldPoint | null {
  return evaluatePointOnGeometry(input, buildGeometry(input), point);
}

function createInteriorPoints(
  geometry: GeometryModel,
  resolution: FieldResolution = 'full'
): Point2D[] {
  const bounds = geometry.bounds;
  const spanX = bounds.xMax - bounds.xMin;
  const spanY = bounds.yMax - bounds.yMin;
  const maxSamples = resolution === 'preview' ? 112 : 168;
  const minSamples = resolution === 'preview' ? 72 : 104;
  const aspect = spanX / Math.max(spanY, 1);
  const samplesX = clamp(Math.round(aspect >= 1 ? maxSamples : maxSamples * aspect), minSamples, maxSamples);
  const samplesY = clamp(Math.round(aspect >= 1 ? maxSamples / aspect : maxSamples), minSamples, maxSamples);

  const points: Point2D[] = [];
  for (let iy = 0; iy < samplesY; iy += 1) {
    const y = lerp(bounds.yMin, bounds.yMax, iy / Math.max(samplesY - 1, 1));
    for (let ix = 0; ix < samplesX; ix += 1) {
      const x = lerp(bounds.xMin, bounds.xMax, ix / Math.max(samplesX - 1, 1));
      if (geometry.polygons.some((polygon) => pointInPolygon({ x, y }, polygon))) {
        points.push({ x, y });
      }
    }
  }
  return points;
}

function createGrid(input: ScenarioInput, geometry: GeometryModel, resolution: FieldResolution = 'full'): FieldPoint[] {
  return createInteriorPoints(geometry, resolution)
    .map((point) => evaluatePointOnGeometry(input, geometry, point))
    .filter((point): point is FieldPoint => point !== null);
}

function buildStreamlines(input: ScenarioInput, geometry: GeometryModel): Array<Array<{ x: number; y: number }>> {
  const seedCount = geometry.type === 'bend' ? 10 : 11;
  const stepLength = Math.max(Math.min(geometry.meta.wUm * 0.22, 40), 20);
  const inletSegment = geometry.centerlines.stem;
  const seedOffsets = Array.from({ length: seedCount }, (_, index) =>
    lerp(-inletSegment.width / 2 + 12, inletSegment.width / 2 - 12, index / Math.max(seedCount - 1, 1))
  );

  const seeds = seedOffsets.map((offset) => ({
    x: inletSegment.start.x + inletSegment.dir.x * Math.min(inletSegment.length * 0.08, geometry.meta.wUm * 0.6) + inletSegment.normal.x * offset,
    y: inletSegment.start.y + inletSegment.dir.y * Math.min(inletSegment.length * 0.08, geometry.meta.wUm * 0.6) + inletSegment.normal.y * offset
  }));

  return seeds.map((seed) => {
    const path: Array<{ x: number; y: number }> = [{ x: seed.x, y: seed.y }];
    let current = seed;

    for (let step = 0; step < 180; step += 1) {
      const sample = evaluatePointOnGeometry(input, geometry, current);
      if (!sample) {
        break;
      }

      const direction = normalizeVector({ x: sample.ux, y: sample.uy });
      const next = {
        x: current.x + direction.x * stepLength,
        y: current.y + direction.y * stepLength
      };

      const nextSample = evaluatePointOnGeometry(input, geometry, next);
      if (!nextSample) {
        break;
      }

      path.push(next);
      current = next;
    }

    return path;
  });
}

function sampleStations(
  input: ScenarioInput,
  geometry: GeometryModel,
  stations: StationedSegment[]
): Array<{ s: number; speed: number; p: number }> {
  const totalLength = Math.max(stations[stations.length - 1]?.end ?? 1, 1);
  const points: Array<{ s: number; speed: number; p: number }> = [];

  stations.forEach((station) => {
    const steps = Math.max(10, Math.round(station.segment.length / Math.max(geometry.meta.wUm * 0.18, 28)));
    for (let index = 0; index < steps; index += 1) {
      const localS = index / Math.max(steps - 1, 1);
      const point = pointAlongSegment(station.segment, localS);
      const value = evaluatePointOnGeometry(input, geometry, point);
      if (!value) {
        continue;
      }
      const globalS = (station.start + localS * station.segment.length) / totalLength;
      const prev = points[points.length - 1];
      if (prev && Math.abs(prev.s - globalS) < 1e-6) {
        continue;
      }
      points.push({
        s: globalS,
        speed: value.speed,
        p: value.p
      });
    }
  });

  return points;
}

function computeMetrics(
  input: ScenarioInput,
  geometry: GeometryModel,
  field: FieldPoint[],
  streamlines: ScenarioResult['streamlines']
): ScenarioMetrics {
  const pressures = field.map((point) => point.p);
  const speeds = field.map((point) => point.speed);
  const maxSpeed = Math.max(...speeds, 0);
  const curvatureSamples = (streamlines ?? []).flatMap((line) => {
    const samples: number[] = [];
    for (let index = 1; index < line.length - 1; index += 1) {
      const prev = line[index - 1];
      const current = line[index];
      const next = line[index + 1];
      const angleA = Math.atan2(current.y - prev.y, current.x - prev.x);
      const angleB = Math.atan2(next.y - current.y, next.x - current.x);
      samples.push(Math.abs(angleB - angleA));
    }
    return samples;
  });

  const totalLength = Math.max(getTotalGuideLength(geometry), 1) * 1e-6;
  const characteristicWidth =
    geometry.type === 'contraction'
      ? Math.max(geometry.meta.throatWidthUm ?? geometry.meta.wUm, 1)
      : geometry.meta.wUm;

  return {
    reynolds: computeReynolds(input),
    maxSpeed,
    avgPressureDrop: Math.max(...pressures, 0) - Math.min(...pressures, 0),
    wallShearProxy:
      (4 * input.fluid.viscosity * Math.max(input.flow.meanVelocity, maxSpeed * 0.72)) /
      Math.max(characteristicWidth * 1e-6, 1e-9),
    streamlineCurvatureProxy: mean(curvatureSamples),
    centerlinePressureGradient:
      (Math.max(...pressures, 0) - Math.min(...pressures, 0)) / Math.max(totalLength, 1e-9)
  };
}

function nearestProjection(point: FieldPoint, geometry: GeometryModel) {
  return getNearestGuideStation(point, geometry).projection;
}

function pointWallBlend(point: Point2D, geometry: GeometryModel): number {
  const nearest = getNearestGuideStation(point, geometry);
  const halfWidth = Math.max(nearest.station.segment.width / 2, 1);
  const wallDistance = Math.max(halfWidth - Math.abs(nearest.projection.t), 0);
  return clamp(wallDistance / Math.max(geometry.meta.wUm * 0.08, 4), 0, 1);
}

function pickSparseObservationPoints(
  input: ScenarioInput,
  geometry: GeometryModel,
  targetPointCount: number,
  candidates: Point2D[]
): Point2D[] {
  const count = Math.max(12, Math.floor(targetPointCount * (input.sparse.sampleRatePct / 100)));
  const rng = mulberry32(hashScenario(input) ^ 0x8b34150d);
  const focusScale = Math.max(geometry.meta.wUm, 1);

  const weighted = candidates.flatMap((point) => {
    const nearest = getNearestGuideStation(point, geometry);
    const halfWidth = Math.max(nearest.station.segment.width / 2, 1);
    const wallProximity = clamp(Math.abs(nearest.projection.t) / halfWidth, 0, 1);
    const inwardMargin = 1 - wallProximity;

    if (inwardMargin < 0.08) {
      return [];
    }

    const featureAnchor = geometry.type === 'contraction' ? geometry.junction : geometry.centerlines.left?.end ?? geometry.junction;
    const focusProximity = 1 / (1 + distance(point.x, point.y, featureAnchor.x, featureAnchor.y) / focusScale);
    const featureBonus =
      input.sparse.strategy === 'region_aware'
        ? focusProximity * 0.58 + wallProximity * 0.14 + inwardMargin * 0.28
        : 0.42;

    return [
      {
        point,
        score: rng() * 0.62 + featureBonus
      }
    ];
  });

  return weighted
    .sort((left, right) => right.score - left.score)
    .slice(0, count)
    .map(({ point }) => point);
}

function buildSparseObservations(
  input: ScenarioInput,
  geometry: GeometryModel,
  targetPointCount: number,
  resolution: FieldResolution
): FieldPoint[] {
  const candidateResolution = resolution === 'preview' ? 'full' : resolution;
  const candidates = createInteriorPoints(geometry, candidateResolution);
  const selected = pickSparseObservationPoints(input, geometry, targetPointCount, candidates);
  const rng = mulberry32(hashScenario(input) ^ 0x13579bdf);
  const noise = input.sparse.noisePct / 100;

  return selected
    .map((point) => evaluatePointOnGeometry(input, geometry, point))
    .filter((point): point is FieldPoint => point !== null)
    .map((point) => {
      const speedScale = 1 + (rng() - 0.5) * noise;
      const pressureScale = 1 + (rng() - 0.5) * noise * 0.6;
      const ux = point.ux * speedScale;
      const uy = point.uy * speedScale;
      return {
        ...point,
        ux,
        uy,
        p: point.p * pressureScale,
        speed: Math.hypot(ux, uy)
      };
    });
}

function pickSparsePoints(input: ScenarioInput, geometry: GeometryModel, field: FieldPoint[]): FieldPoint[] {
  const count = Math.max(12, Math.floor(field.length * (input.sparse.sampleRatePct / 100)));
  const rng = mulberry32(hashScenario(input));
  const focusScale = Math.max(geometry.meta.wUm, 1);

  const weighted = field.flatMap((point) => {
    const nearest = nearestProjection(point, geometry);
    const halfWidth = Math.max(nearest.segment.width / 2, 1);
    const wallProximity = clamp(Math.abs(nearest.t) / halfWidth, 0, 1);
    const inwardMargin = 1 - wallProximity;

    if (inwardMargin < 0.08) {
      return [];
    }

    const featureAnchor = geometry.type === 'contraction' ? geometry.junction : geometry.centerlines.left?.end ?? geometry.junction;
    const focusProximity = 1 / (1 + distance(point.x, point.y, featureAnchor.x, featureAnchor.y) / focusScale);
    const speedBonus = clamp(point.speed / Math.max(input.flow.meanVelocity * 1.8, 1e-6), 0, 1);
    const featureBonus =
      input.sparse.strategy === 'region_aware'
        ? focusProximity * 0.48 + wallProximity * 0.14 + speedBonus * 0.2 + inwardMargin * 0.18
        : 0.42;
    return [
      {
        point,
        score: rng() * 0.62 + featureBonus
      }
    ];
  });

  return weighted
    .sort((left, right) => right.score - left.score)
    .slice(0, count)
    .map(({ point }) => {
      const noise = input.sparse.noisePct / 100;
      const speedScale = 1 + (rng() - 0.5) * noise;
      const pressureScale = 1 + (rng() - 0.5) * noise * 0.6;
      const ux = point.ux * speedScale;
      const uy = point.uy * speedScale;
      return {
        ...point,
        ux,
        uy,
        p: point.p * pressureScale,
        speed: Math.hypot(ux, uy)
      };
    });
}

function fieldPointKey(point: Pick<FieldPoint, 'x' | 'y'>): string {
  return `${point.x.toFixed(6)}:${point.y.toFixed(6)}`;
}

function buildBaselineLookup(field: FieldPoint[]): Map<string, FieldPoint> {
  return new Map(field.map((point) => [fieldPointKey(point), point]));
}

function selectAnchorPoints(sparsePoints: FieldPoint[], maxAnchors = 48): FieldPoint[] {
  if (sparsePoints.length <= maxAnchors) {
    return sparsePoints;
  }
  const coords = sparsePoints.map((point) => [point.x, point.y] as const);
  const centroid = coords.reduce(
    (acc, [x, y]) => ({ x: acc.x + x / coords.length, y: acc.y + y / coords.length }),
    { x: 0, y: 0 }
  );
  let first = 0;
  let bestDist = -1;
  for (let index = 0; index < coords.length; index += 1) {
    const dist2 = (coords[index][0] - centroid.x) ** 2 + (coords[index][1] - centroid.y) ** 2;
    if (dist2 > bestDist) {
      bestDist = dist2;
      first = index;
    }
  }
  const selected = [first];
  const minDist2 = coords.map(([x, y]) => (x - coords[first][0]) ** 2 + (y - coords[first][1]) ** 2);
  while (selected.length < maxAnchors) {
    let next = 0;
    let farthest = -1;
    for (let index = 0; index < minDist2.length; index += 1) {
      if (minDist2[index] > farthest) {
        farthest = minDist2[index];
        next = index;
      }
    }
    selected.push(next);
    for (let index = 0; index < coords.length; index += 1) {
      const dist2 = (coords[index][0] - coords[next][0]) ** 2 + (coords[index][1] - coords[next][1]) ** 2;
      minDist2[index] = Math.min(minDist2[index], dist2);
    }
  }
  return selected.map((index) => sparsePoints[index]);
}

function rbfKernel(
  queryPoints: Array<Pick<FieldPoint, 'x' | 'y'>>,
  anchorPoints: Array<Pick<FieldPoint, 'x' | 'y'>>,
  radius: number
): number[][] {
  const radius2 = Math.max(radius * radius, 1e-12);
  return queryPoints.map((query) =>
    anchorPoints.map((anchor) => {
      const gap = distance(query.x, query.y, anchor.x, anchor.y);
      return Math.exp(-(gap * gap) / radius2);
    })
  );
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const a = matrix.map((row) => [...row]);
  const b = [...rhs];
  for (let pivot = 0; pivot < n; pivot += 1) {
    let maxRow = pivot;
    for (let row = pivot + 1; row < n; row += 1) {
      if (Math.abs(a[row][pivot]) > Math.abs(a[maxRow][pivot])) {
        maxRow = row;
      }
    }
    if (maxRow !== pivot) {
      [a[pivot], a[maxRow]] = [a[maxRow], a[pivot]];
      [b[pivot], b[maxRow]] = [b[maxRow], b[pivot]];
    }
    const diag = Math.abs(a[pivot][pivot]) > 1e-12 ? a[pivot][pivot] : 1e-12;
    for (let row = pivot + 1; row < n; row += 1) {
      const factor = a[row][pivot] / diag;
      if (Math.abs(factor) < 1e-12) {
        continue;
      }
      for (let col = pivot; col < n; col += 1) {
        a[row][col] -= factor * a[pivot][col];
      }
      b[row] -= factor * b[pivot];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let acc = b[row];
    for (let col = row + 1; col < n; col += 1) {
      acc -= a[row][col] * x[col];
    }
    x[row] = acc / (Math.abs(a[row][row]) > 1e-12 ? a[row][row] : 1e-12);
  }
  return x;
}

function solveRidgeWeights(kernelObs: number[][], residual: number[], ridge: number): number[] {
  const m = kernelObs[0]?.length ?? 0;
  const lhs = Array.from({ length: m }, (_, row) =>
    Array.from({ length: m }, (_, col) => {
      let value = row === col ? ridge : 0;
      for (let index = 0; index < kernelObs.length; index += 1) {
        value += kernelObs[index][row] * kernelObs[index][col];
      }
      return value;
    })
  );
  const rhs = Array.from({ length: m }, (_, row) => {
    let value = 0;
    for (let index = 0; index < kernelObs.length; index += 1) {
      value += kernelObs[index][row] * residual[index];
    }
    return value;
  });
  return solveLinearSystem(lhs, rhs);
}

function applyKernel(weightsByPoint: number[][], coeffs: number[]): number[] {
  return weightsByPoint.map((row) => row.reduce((sum, value, index) => sum + value * coeffs[index], 0));
}

function fitAffineDelta(
  prior: number[],
  observed: number[],
  scaleRidge: number,
  biasRidge: number
): { scale: number; bias: number } {
  if (!prior.length || !observed.length) {
    return { scale: 1, bias: 0 };
  }
  let xx = scaleRidge;
  let x1 = 0;
  let one = biasRidge;
  let xb = 0;
  let b = 0;
  for (let index = 0; index < prior.length; index += 1) {
    const x = prior[index];
    const delta = observed[index] - prior[index];
    xx += x * x;
    x1 += x;
    one += 1;
    xb += x * delta;
    b += delta;
  }
  const [deltaScale, bias] = solveLinearSystem(
    [
      [xx, x1],
      [x1, one]
    ],
    [xb, b]
  );
  return {
    scale: 1 + deltaScale,
    bias
  };
}

function inverseReconstructField(
  input: ScenarioInput,
  geometry: GeometryModel,
  resolution: FieldResolution
): { field: FieldPoint[]; sparsePoints: FieldPoint[]; reconstruction: FieldPoint[]; baselineMetrics: ScenarioMetrics } {
  const targetPoints = createInteriorPoints(geometry, resolution);
  const sparsePoints = buildSparseObservations(input, geometry, targetPoints.length, resolution);
  const priorSparse = sparsePoints
    .map((point) => evaluatePointOnGeometry(input, geometry, { x: point.x, y: point.y }))
    .filter((point): point is FieldPoint => point !== null);

  const uxAffine = fitAffineDelta(
    priorSparse.map((point) => point.ux),
    sparsePoints.map((point) => point.ux),
    2.5e-2,
    1.0e-8
  );
  const uyAffine = fitAffineDelta(
    priorSparse.map((point) => point.uy),
    sparsePoints.map((point) => point.uy),
    2.5e-2,
    1.0e-8
  );
  const pAffine = fitAffineDelta(
    priorSparse.map((point) => point.p),
    sparsePoints.map((point) => point.p),
    3.0e-2,
    1.0e-8
  );

  const anchorPoints = selectAnchorPoints(sparsePoints, 48);
  const velocityRadius = Math.max(input.geometry.wUm * 1.05, 30);
  const pressureRadius = Math.max(input.geometry.wUm * 1.55, 42);
  const velocityRidge = 1.8e-2;
  const pressureRidge = 3.0e-2;

  const residuals = sparsePoints.map((point, index) => ({
    x: point.x,
    y: point.y,
    dux: point.ux - (priorSparse[index]?.ux ?? 0) * uxAffine.scale - uxAffine.bias,
    duy: point.uy - (priorSparse[index]?.uy ?? 0) * uyAffine.scale - uyAffine.bias,
    dp: point.p - (priorSparse[index]?.p ?? 0) * pAffine.scale - pAffine.bias
  }));

  const field = targetPoints
    .map((point) => evaluatePointOnGeometry(input, geometry, point))
    .filter((point): point is FieldPoint => point !== null);

  const sparseLookup = buildBaselineLookup(sparsePoints);
  if (!anchorPoints.length || !residuals.length) {
    return {
      field,
      sparsePoints,
      reconstruction: field.map((point) => ({ ...point })),
      baselineMetrics: computeMetrics(input, geometry, field, [])
    };
  }

  const sparseResidualPoints = residuals.map((item) => ({ x: item.x, y: item.y }));
  const anchorResidualPoints = anchorPoints.map((item) => ({ x: item.x, y: item.y }));
  const fieldPoints = field.map((item) => ({ x: item.x, y: item.y }));
  const velKernelObs = rbfKernel(sparseResidualPoints, anchorResidualPoints, velocityRadius);
  const velKernelField = rbfKernel(fieldPoints, anchorResidualPoints, velocityRadius);
  const pressureKernelObs = rbfKernel(sparseResidualPoints, anchorResidualPoints, pressureRadius);
  const pressureKernelField = rbfKernel(fieldPoints, anchorResidualPoints, pressureRadius);
  const duxCoeffs = solveRidgeWeights(
    velKernelObs,
    residuals.map((item) => item.dux),
    velocityRidge
  );
  const duyCoeffs = solveRidgeWeights(
    velKernelObs,
    residuals.map((item) => item.duy),
    velocityRidge
  );
  const dpCoeffs = solveRidgeWeights(
    pressureKernelObs,
    residuals.map((item) => item.dp),
    pressureRidge
  );
  const duxField = applyKernel(velKernelField, duxCoeffs);
  const duyField = applyKernel(velKernelField, duyCoeffs);
  const dpField = applyKernel(pressureKernelField, dpCoeffs);
  const velConfidence = velKernelField.map((row) => clamp(row.reduce((sum, value) => sum + value, 0) / 3.4, 0, 1));
  const pressureConfidence = pressureKernelField.map((row) => clamp(row.reduce((sum, value) => sum + value, 0) / 3.9, 0, 1));

  const reconstruction = field.map((target, index) => {
    const exactMatch = sparseLookup.get(fieldPointKey(target));
    if (exactMatch) {
      return { ...exactMatch };
    }
    const wallBlend = pointWallBlend(target, geometry);
    const ux = target.ux * uxAffine.scale + uxAffine.bias * wallBlend + duxField[index] * velConfidence[index] * wallBlend;
    const uy = target.uy * uyAffine.scale + uyAffine.bias * wallBlend + duyField[index] * velConfidence[index] * wallBlend;
    const p = target.p * pAffine.scale + pAffine.bias + dpField[index] * pressureConfidence[index];

    return {
      ...target,
      ux,
      uy,
      p,
      speed: Math.hypot(ux, uy)
    };
  });

  return {
    field,
    sparsePoints,
    reconstruction,
    baselineMetrics: computeMetrics(input, geometry, field, [])
  };
}

function reconstructField(
  input: ScenarioInput,
  field: FieldPoint[],
  sparsePoints: FieldPoint[]
): FieldPoint[] {
  if (!sparsePoints.length) {
    return field;
  }

  const baselineLookup = buildBaselineLookup(field);
  const sparseLookup = buildBaselineLookup(sparsePoints);
  const anchorPoints = selectAnchorPoints(sparsePoints, 48);
  if (!anchorPoints.length) {
    return field.map((point) => ({ ...point }));
  }
  const velocityRadius = Math.max(input.geometry.wUm * 1.15, 36);
  const pressureRadius = Math.max(input.geometry.wUm * 1.65, 48);
  const velocityRidge = 2.0e-2;
  const pressureRidge = 3.5e-2;

  const residuals = sparsePoints
    .map((point) => {
      const baseline = baselineLookup.get(fieldPointKey(point));
      if (!baseline) {
        return null;
      }
      return {
        x: point.x,
        y: point.y,
        dux: point.ux - baseline.ux,
        duy: point.uy - baseline.uy,
        dp: point.p - baseline.p
      };
    })
    .filter((value): value is NonNullable<typeof value> => value !== null);

  if (!residuals.length) {
    return field.map((point) => ({ ...point }));
  }

  const sparseResidualPoints = residuals.map((item) => ({ x: item.x, y: item.y }));
  const anchorResidualPoints = anchorPoints.map((item) => ({ x: item.x, y: item.y }));
  const fieldPoints = field.map((item) => ({ x: item.x, y: item.y }));
  const velKernelObs = rbfKernel(sparseResidualPoints, anchorResidualPoints, velocityRadius);
  const velKernelField = rbfKernel(fieldPoints, anchorResidualPoints, velocityRadius);
  const pressureKernelObs = rbfKernel(sparseResidualPoints, anchorResidualPoints, pressureRadius);
  const pressureKernelField = rbfKernel(fieldPoints, anchorResidualPoints, pressureRadius);
  const duxCoeffs = solveRidgeWeights(
    velKernelObs,
    residuals.map((item) => item.dux),
    velocityRidge
  );
  const duyCoeffs = solveRidgeWeights(
    velKernelObs,
    residuals.map((item) => item.duy),
    velocityRidge
  );
  const dpCoeffs = solveRidgeWeights(
    pressureKernelObs,
    residuals.map((item) => item.dp),
    pressureRidge
  );
  const duxField = applyKernel(velKernelField, duxCoeffs);
  const duyField = applyKernel(velKernelField, duyCoeffs);
  const dpField = applyKernel(pressureKernelField, dpCoeffs);
  const velConfidence = velKernelField.map((row) => clamp(row.reduce((sum, value) => sum + value, 0) / 3.2, 0, 1));
  const pressureConfidence = pressureKernelField.map((row) => clamp(row.reduce((sum, value) => sum + value, 0) / 3.8, 0, 1));

  return field.map((target, index) => {
    const exactMatch = sparseLookup.get(fieldPointKey(target));
    if (exactMatch) {
      return { ...exactMatch };
    }
    const ux = target.ux + duxField[index] * velConfidence[index];
    const uy = target.uy + duyField[index] * velConfidence[index];
    const p = target.p + dpField[index] * pressureConfidence[index];

    return {
      ...target,
      ux,
      uy,
      p,
      speed: Math.hypot(ux, uy)
    };
  });
}

function buildScenarioResult(input: ScenarioInput, options: SimulateOptions = {}): ScenarioResult {
  const {
    resolution = 'full',
    includeStreamlines = true,
    includeProbes = true,
    includeSparsePoints = true,
    includeReconstruction = false
  } = options;
  const geometry = buildGeometry(input);
  const inverseBundle = includeReconstruction ? inverseReconstructField(input, geometry, resolution) : null;
  const field = inverseBundle?.field ?? createGrid(input, geometry, resolution);
  const streamlines = includeStreamlines ? buildStreamlines(input, geometry) : undefined;
  const sparsePoints =
    inverseBundle?.sparsePoints ?? (includeSparsePoints ? pickSparsePoints(input, geometry, field) : undefined);
  const reconstruction =
    inverseBundle?.reconstruction ??
    (includeReconstruction && sparsePoints ? reconstructField(input, field, sparsePoints) : undefined);
  const guideStations = geometry.guideStations;
  const mainStations = guideStations;
  const branchStations = geometry.centerlines.representative
    ? guideStations.filter((station) => station.segment === geometry.centerlines.representative)
    : [];
  const probes: ScenarioProbes | undefined = includeProbes
    ? {
        mainCenterline: sampleStations(input, geometry, mainStations),
        branchCenterline: branchStations.length > 0 ? sampleStations(input, geometry, branchStations) : undefined
      }
    : undefined;

  return {
    field,
    streamlines,
    sparsePoints,
    reconstruction,
    metrics: computeMetrics(input, geometry, reconstruction ?? field, streamlines ?? []),
    baselineMetrics: inverseBundle?.baselineMetrics,
    probes
  };
}

function mutateScenario(input: ScenarioInput, variable: SweepVariable, value: number): ScenarioInput {
  if (variable === 'meanVelocity') {
    return {
      ...input,
      flow: {
        ...input.flow,
        meanVelocity: value
      }
    };
  }

  return {
    ...input,
    fluid: {
      ...input.fluid,
      viscosity: value
    }
  };
}

export function createDemoAdapter(
  onProgress?: (label: string, state: 'running' | 'success') => void
): InferenceAdapter {
  return {
    async simulate(input: ScenarioInput, options: SimulateOptions = {}): Promise<ScenarioResult> {
      onProgress?.('simulate', 'running');
      const result = buildScenarioResult(input, {
        resolution: options.resolution ?? 'full',
        includeStreamlines: options.includeStreamlines ?? true,
        includeProbes: options.includeProbes ?? true,
        includeSparsePoints: options.includeSparsePoints ?? false,
        includeReconstruction: options.includeReconstruction ?? false
      });
      onProgress?.('simulate', 'success');
      return result;
    },
    async queryPoint(input: ScenarioInput, point: Point2D): Promise<FieldPoint | null> {
      return evaluatePoint(input, point);
    },
    async reconstruct(input: ScenarioInput, options: SimulateOptions = {}): Promise<ScenarioResult> {
      onProgress?.('reconstruct', 'running');
      const result = buildScenarioResult(input, {
        resolution: options.resolution ?? 'full',
        includeStreamlines: options.includeStreamlines ?? true,
        includeProbes: options.includeProbes ?? true,
        includeSparsePoints: true,
        includeReconstruction: true
      });
      onProgress?.('reconstruct', 'success');
      return result;
    },
    async loadStreamlines(
      input: ScenarioInput,
      options: Pick<SimulateOptions, 'resolution'> = {}
    ): Promise<NonNullable<ScenarioResult['streamlines']>> {
      onProgress?.('streamlines', 'running');
      const result = buildScenarioResult(input, {
        resolution: options.resolution ?? 'preview',
        includeStreamlines: true,
        includeProbes: false,
        includeSparsePoints: false,
        includeReconstruction: false
      });
      onProgress?.('streamlines', 'success');
      return result.streamlines ?? [];
    },
    async loadProbes(input: ScenarioInput): Promise<ScenarioProbes> {
      onProgress?.('probes', 'running');
      const result = buildScenarioResult(input, {
        resolution: 'preview',
        includeStreamlines: false,
        includeProbes: true,
        includeSparsePoints: false,
        includeReconstruction: false
      });
      onProgress?.('probes', 'success');
      return result.probes ?? { mainCenterline: [] };
    },
    async calibrateViscosity(
      input: ScenarioInput,
      targetPoints: FieldPoint[]
    ): Promise<{ bestViscosity: number; error: number }> {
      onProgress?.('calibrate', 'running');
      const candidates = [
        input.fluid.viscosity * 0.5,
        input.fluid.viscosity * 0.7,
        input.fluid.viscosity * 0.85,
        input.fluid.viscosity,
        input.fluid.viscosity * 1.15,
        input.fluid.viscosity * 1.35,
        input.fluid.viscosity * 1.6
      ].map((value) => clamp(value, 0.0004, 0.008));

      let best = { bestViscosity: input.fluid.viscosity, error: Number.POSITIVE_INFINITY };
      for (const candidate of candidates) {
        const scenario = {
          ...input,
          fluid: {
            ...input.fluid,
            viscosity: candidate
          }
        };
        const errors = targetPoints.map((point) => {
          const candidatePoint = evaluatePoint(scenario, point);
          if (!candidatePoint) {
            return 1e6;
          }
          return (candidatePoint.speed - point.speed) ** 2 + ((candidatePoint.p - point.p) / 20) ** 2;
        });
        const error = mean(errors);
        if (error < best.error) {
          best = { bestViscosity: candidate, error };
        }
      }
      onProgress?.('calibrate', 'success');
      return best;
    },
    async sweep(
      input: ScenarioInput,
      variable: SweepVariable,
      values: number[]
    ): Promise<Array<{ value: number; metrics: ScenarioMetrics }>> {
      onProgress?.('sweep', 'running');
      const result = values.map((value) => {
        const scenario = mutateScenario(input, variable, value);
        const simulation = buildScenarioResult(scenario, {
          resolution: 'preview',
          includeStreamlines: false,
          includeProbes: false,
          includeSparsePoints: false,
          includeReconstruction: false
        });
        return {
          value,
          metrics: simulation.metrics
        };
      });
      onProgress?.('sweep', 'success');
      return result;
    }
  };
}
