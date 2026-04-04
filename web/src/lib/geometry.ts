import type { GeometryType, InletProfile, ScenarioInput } from '../types/pinn';
import { clamp, distance, lerp, smoothstep } from './utils';

export interface Point2D {
  x: number;
  y: number;
}

export interface Bounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export interface SegmentInfo {
  name: 'stem' | 'left' | 'right';
  start: Point2D;
  end: Point2D;
  width: number;
  length: number;
  dir: Point2D;
  normal: Point2D;
}

export interface CorridorProjection {
  segment: SegmentInfo;
  s: number;
  rawS: number;
  t: number;
  inside: boolean;
  distance: number;
  point: Point2D;
}

export interface StationedSegment {
  segment: SegmentInfo;
  start: number;
  end: number;
}

interface GeometryMeta {
  familyLabel: string;
  wUm: number;
  throatWidthUm?: number;
  lInUm?: number;
  lCUm?: number;
  lOutUm?: number;
  totalLengthUm?: number;
  rcUm?: number;
  thetaDeg?: number;
  inletProfile?: InletProfile;
  arcStart?: number;
  arcEnd?: number;
}

export interface GeometryModel {
  type: GeometryType;
  polygon: Point2D[];
  polygons: Point2D[][];
  bounds: Bounds;
  junction: Point2D;
  centerlines: {
    stem: SegmentInfo;
    left?: SegmentInfo;
    right?: SegmentInfo;
    representative?: SegmentInfo;
  };
  guideSegments: SegmentInfo[];
  guideStations: StationedSegment[];
  meta: GeometryMeta;
}

function normalize(vector: Point2D): Point2D {
  const length = Math.hypot(vector.x, vector.y) || 1;
  return {
    x: vector.x / length,
    y: vector.y / length
  };
}

function createSegment(
  name: SegmentInfo['name'],
  start: Point2D,
  end: Point2D,
  width: number
): SegmentInfo {
  const dir = normalize({
    x: end.x - start.x,
    y: end.y - start.y
  });

  return {
    name,
    start,
    end,
    width,
    length: distance(start.x, start.y, end.x, end.y),
    dir,
    normal: {
      x: -dir.y,
      y: dir.x
    }
  };
}

function buildStations(segments: SegmentInfo[]): StationedSegment[] {
  let cursor = 0;
  return segments.map((segment) => {
    const station = {
      segment,
      start: cursor,
      end: cursor + segment.length
    };
    cursor += segment.length;
    return station;
  });
}

function padBounds(polygon: Point2D[], pad: number): Bounds {
  const xValues = polygon.map((point) => point.x);
  const yValues = polygon.map((point) => point.y);
  return {
    xMin: Math.min(...xValues) - pad,
    xMax: Math.max(...xValues) + pad,
    yMin: Math.min(...yValues) - pad,
    yMax: Math.max(...yValues) + pad
  };
}

function sampleContractionHalfWidth(input: ScenarioInput, x: number): number {
  const w = input.geometry.wUm;
  const throat = w * input.geometry.beta;
  const lIn = input.geometry.lInOverW * w;
  const lC = input.geometry.lCOverW * w;
  const xClamped = clamp(x, 0, lIn + lC + input.geometry.lOutOverW * w);

  if (xClamped <= lIn || lC <= 1e-6) {
    return w / 2;
  }
  if (xClamped >= lIn + lC) {
    return throat / 2;
  }

  const t = (xClamped - lIn) / lC;
  const eased = smoothstep(0, 1, t);
  return lerp(w / 2, throat / 2, eased);
}

function buildContractionGeometry(input: ScenarioInput): GeometryModel {
  const w = input.geometry.wUm;
  const throatWidth = w * input.geometry.beta;
  const lIn = input.geometry.lInOverW * w;
  const lC = input.geometry.lCOverW * w;
  const lOut = input.geometry.lOutOverW * w;
  const total = lIn + lC + lOut;

  const topBoundary: Point2D[] = [{ x: 0, y: w / 2 }, { x: lIn, y: w / 2 }];
  const transitionSteps = 24;
  for (let index = 1; index <= transitionSteps; index += 1) {
    const t = index / transitionSteps;
    const x = lIn + lC * t;
    topBoundary.push({ x, y: sampleContractionHalfWidth(input, x) });
  }
  topBoundary.push({ x: total, y: throatWidth / 2 });

  const bottomBoundary = [...topBoundary]
    .reverse()
    .map((point) => ({ x: point.x, y: -point.y }));
  const polygon = [...topBoundary, ...bottomBoundary];

  const inlet = createSegment('stem', { x: 0, y: 0 }, { x: lIn, y: 0 }, w);
  const throat = createSegment('left', { x: lIn, y: 0 }, { x: lIn + lC, y: 0 }, (w + throatWidth) / 2);
  const outlet = createSegment('right', { x: lIn + lC, y: 0 }, { x: total, y: 0 }, throatWidth);
  const guideSegments = [inlet, throat, outlet];

  return {
    type: 'contraction',
    polygon,
    polygons: [polygon],
    bounds: padBounds(polygon, Math.max(w * 0.8, 120)),
    junction: { x: lIn + lC * 0.72, y: 0 },
    centerlines: {
      stem: inlet,
      left: throat,
      right: outlet,
      representative: outlet
    },
    guideSegments,
    guideStations: buildStations(guideSegments),
    meta: {
      familyLabel: 'contraction_2d',
      wUm: w,
      throatWidthUm: throatWidth,
      lInUm: lIn,
      lCUm: lC,
      lOutUm: lOut,
      totalLengthUm: total
    }
  };
}

function buildArcPoints(center: Point2D, radius: number, startDeg: number, endDeg: number, steps: number): Point2D[] {
  return Array.from({ length: steps + 1 }, (_, index) => {
    const angleDeg = lerp(startDeg, endDeg, index / steps);
    const angle = (angleDeg * Math.PI) / 180;
    return {
      x: center.x + radius * Math.cos(angle),
      y: center.y + radius * Math.sin(angle)
    };
  });
}

function rotatePoint(point: Point2D, angleDeg: number): Point2D {
  const angle = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return {
    x: point.x * cos - point.y * sin,
    y: point.x * sin + point.y * cos
  };
}

interface BendCoordinateTransform {
  toDisplay: (point: Point2D) => Point2D;
  toModel: (point: Point2D) => Point2D;
}

export function getBendCoordinateTransform(input: ScenarioInput): BendCoordinateTransform {
  const w = input.geometry.wUm;
  const half = w / 2;
  const lIn = input.geometry.lInOverW * w;
  const lOut = input.geometry.lOutOverW * w;
  const rc = input.geometry.rcOverW * w;
  const thetaDeg = clamp(input.geometry.thetaDeg, 30, 135);
  const startDeg = -90;
  const endDeg = startDeg + thetaDeg;
  const center = { x: lIn, y: rc };
  const centerlineArcEnd = {
    x: center.x + rc * Math.cos((endDeg * Math.PI) / 180),
    y: center.y + rc * Math.sin((endDeg * Math.PI) / 180)
  };
  const outletDir = normalize({
    x: Math.cos((thetaDeg * Math.PI) / 180),
    y: Math.sin((thetaDeg * Math.PI) / 180)
  });
  const outwardNormal = { x: outletDir.y, y: -outletDir.x };
  const outerEnd = {
    x: centerlineArcEnd.x + outwardNormal.x * half + outletDir.x * lOut,
    y: centerlineArcEnd.y + outwardNormal.y * half + outletDir.y * lOut
  };
  const innerEnd = {
    x: centerlineArcEnd.x - outwardNormal.x * half + outletDir.x * lOut,
    y: centerlineArcEnd.y - outwardNormal.y * half + outletDir.y * lOut
  };

  const arcSteps = Math.max(12, Math.round(thetaDeg / 8));
  const outerArc = buildArcPoints(center, rc + half, startDeg, endDeg, arcSteps);
  const innerArc = buildArcPoints(center, Math.max(rc - half, half * 0.35), endDeg, startDeg, arcSteps);
  const polygon = [
    { x: 0, y: -half },
    { x: lIn, y: -half },
    ...outerArc.slice(1),
    outerEnd,
    innerEnd,
    ...innerArc.slice(1),
    { x: 0, y: half }
  ];

  const rotationDeg = -thetaDeg / 2;
  const rotatedPolygon = polygon.map((point) => rotatePoint(point, rotationDeg));
  const xValues = rotatedPolygon.map((point) => point.x);
  const yValues = rotatedPolygon.map((point) => point.y);
  const offset = {
    x: -Math.min(...xValues),
    y: -((Math.min(...yValues) + Math.max(...yValues)) / 2)
  };

  return {
    toDisplay(point: Point2D): Point2D {
      const rotated = rotatePoint(point, rotationDeg);
      return {
        x: rotated.x + offset.x,
        y: rotated.y + offset.y
      };
    },
    toModel(point: Point2D): Point2D {
      const translated = {
        x: point.x - offset.x,
        y: point.y - offset.y
      };
      return rotatePoint(translated, -rotationDeg);
    }
  };
}

function buildBendGeometry(input: ScenarioInput): GeometryModel {
  const w = input.geometry.wUm;
  const half = w / 2;
  const lIn = input.geometry.lInOverW * w;
  const lOut = input.geometry.lOutOverW * w;
  const rc = input.geometry.rcOverW * w;
  const thetaDeg = clamp(input.geometry.thetaDeg, 30, 135);
  const startDeg = -90;
  const endDeg = startDeg + thetaDeg;
  const center = { x: lIn, y: rc };
  const centerlineStart = { x: 0, y: 0 };
  const centerlineArcStart = { x: lIn, y: 0 };
  const centerlineArcEnd = {
    x: center.x + rc * Math.cos((endDeg * Math.PI) / 180),
    y: center.y + rc * Math.sin((endDeg * Math.PI) / 180)
  };
  const outletDir = normalize({
    x: Math.cos((thetaDeg * Math.PI) / 180),
    y: Math.sin((thetaDeg * Math.PI) / 180)
  });
  const outwardNormal = { x: outletDir.y, y: -outletDir.x };
  const outerEnd = {
    x: centerlineArcEnd.x + outwardNormal.x * half + outletDir.x * lOut,
    y: centerlineArcEnd.y + outwardNormal.y * half + outletDir.y * lOut
  };
  const innerEnd = {
    x: centerlineArcEnd.x - outwardNormal.x * half + outletDir.x * lOut,
    y: centerlineArcEnd.y - outwardNormal.y * half + outletDir.y * lOut
  };

  const arcSteps = Math.max(12, Math.round(thetaDeg / 8));
  const outerArc = buildArcPoints(center, rc + half, startDeg, endDeg, arcSteps);
  const innerArc = buildArcPoints(center, Math.max(rc - half, half * 0.35), endDeg, startDeg, arcSteps);
  const polygon = [
    { x: 0, y: -half },
    { x: lIn, y: -half },
    ...outerArc.slice(1),
    outerEnd,
    innerEnd,
    ...innerArc.slice(1),
    { x: 0, y: half }
  ];

  const outletEnd = {
    x: centerlineArcEnd.x + outletDir.x * lOut,
    y: centerlineArcEnd.y + outletDir.y * lOut
  };
  const arcGuidePoints = buildArcPoints(center, rc, startDeg, endDeg, arcSteps);
  const rotationDeg = -thetaDeg / 2;
  const rotatedPolygon = polygon.map((point) => rotatePoint(point, rotationDeg));
  const rotatedArcGuidePoints = arcGuidePoints.map((point) => rotatePoint(point, rotationDeg));
  const rotatedCenterlineStart = rotatePoint(centerlineStart, rotationDeg);
  const rotatedCenterlineArcStart = rotatePoint(centerlineArcStart, rotationDeg);
  const rotatedCenterlineArcEnd = rotatePoint(centerlineArcEnd, rotationDeg);
  const rotatedOutletEnd = rotatePoint(outletEnd, rotationDeg);
  const rotatedJunction = rotatePoint(buildArcPoints(center, rc, startDeg, endDeg, 2)[1], rotationDeg);

  const xValues = rotatedPolygon.map((point) => point.x);
  const yValues = rotatedPolygon.map((point) => point.y);
  const offset = {
    x: -Math.min(...xValues),
    y: -((Math.min(...yValues) + Math.max(...yValues)) / 2)
  };
  const translatePoint = (point: Point2D): Point2D => ({
    x: point.x + offset.x,
    y: point.y + offset.y
  });

  const transformedPolygon = rotatedPolygon.map(translatePoint);
  const transformedArcGuidePoints = rotatedArcGuidePoints.map(translatePoint);
  const transformedCenterlineStart = translatePoint(rotatedCenterlineStart);
  const transformedCenterlineArcStart = translatePoint(rotatedCenterlineArcStart);
  const transformedCenterlineArcEnd = translatePoint(rotatedCenterlineArcEnd);
  const transformedOutletEnd = translatePoint(rotatedOutletEnd);

  const inlet = createSegment('stem', transformedCenterlineStart, transformedCenterlineArcStart, w);
  const arcSegments = transformedArcGuidePoints.slice(0, -1).map((point, index) =>
    createSegment('left', point, transformedArcGuidePoints[index + 1], w)
  );
  const outlet = createSegment('right', transformedCenterlineArcEnd, transformedOutletEnd, w);
  const guideSegments = [inlet, ...arcSegments, outlet];
  const guideStations = buildStations(guideSegments);
  const arcStart = guideStations[1]?.start ?? inlet.length;
  const arcEnd = guideStations[guideStations.length - 2]?.end ?? inlet.length;

  return {
    type: 'bend',
    polygon: transformedPolygon,
    polygons: [transformedPolygon],
    bounds: padBounds(transformedPolygon, Math.max(w * 0.85, 150)),
    junction: translatePoint(rotatedJunction),
    centerlines: {
      stem: inlet,
      left: arcSegments[Math.floor(arcSegments.length / 2)],
      right: outlet,
      representative: outlet
    },
    guideSegments,
    guideStations,
    meta: {
      familyLabel: 'bend_2d',
      wUm: w,
      lInUm: lIn,
      lOutUm: lOut,
      totalLengthUm: guideStations[guideStations.length - 1]?.end ?? inlet.length + outlet.length,
      rcUm: rc,
      thetaDeg,
      inletProfile: input.geometry.inletProfile,
      arcStart,
      arcEnd
    }
  };
}

export function buildGeometry(input: ScenarioInput): GeometryModel {
  switch (input.geometry.type) {
    case 'bend':
      return buildBendGeometry(input);
    case 'contraction':
    default:
      return buildContractionGeometry(input);
  }
}

export function pointOnSegment(point: Point2D, start: Point2D, end: Point2D, epsilon = 0.8): boolean {
  const segmentLength = distance(start.x, start.y, end.x, end.y);
  const delta =
    distance(point.x, point.y, start.x, start.y) + distance(point.x, point.y, end.x, end.y) - segmentLength;
  return Math.abs(delta) <= epsilon;
}

export function pointInPolygon(point: Point2D, polygon: Point2D[]): boolean {
  let inside = false;

  for (let index = 0, prev = polygon.length - 1; index < polygon.length; prev = index, index += 1) {
    const a = polygon[index];
    const b = polygon[prev];

    if (pointOnSegment(point, a, b)) {
      return true;
    }

    const intersects = (a.y > point.y) !== (b.y > point.y);
    if (!intersects) {
      continue;
    }

    const xIntersect = ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y + 1e-12) + a.x;
    if (point.x < xIntersect) {
      inside = !inside;
    }
  }

  return inside;
}

export function projectToSegment(point: Point2D, segment: SegmentInfo): CorridorProjection {
  const dx = segment.end.x - segment.start.x;
  const dy = segment.end.y - segment.start.y;
  const lengthSquared = dx * dx + dy * dy;
  const px = point.x - segment.start.x;
  const py = point.y - segment.start.y;
  const rawS = lengthSquared === 0 ? 0 : (px * dx + py * dy) / lengthSquared;
  const s = clamp(rawS, 0, 1);
  const projectionPoint = {
    x: segment.start.x + dx * s,
    y: segment.start.y + dy * s
  };
  const t = px * segment.normal.x + py * segment.normal.y;

  return {
    segment,
    s,
    rawS,
    t,
    inside: rawS >= 0 && rawS <= 1 && Math.abs(t) <= segment.width / 2,
    distance: distance(point.x, point.y, projectionPoint.x, projectionPoint.y),
    point: projectionPoint
  };
}

export function pointAlongSegment(segment: SegmentInfo, s: number): Point2D {
  const t = clamp(s, 0, 1);
  return {
    x: segment.start.x + (segment.end.x - segment.start.x) * t,
    y: segment.start.y + (segment.end.y - segment.start.y) * t
  };
}

export function getGeometryPolygons(input: ScenarioInput): Point2D[][] {
  return buildGeometry(input).polygons;
}

export function getBounds(input: ScenarioInput): Bounds {
  return buildGeometry(input).bounds;
}

export function isPointInsideGeometry(input: ScenarioInput, point: Point2D): boolean {
  const geometry = buildGeometry(input);
  return geometry.polygons.some((polygon) => pointInPolygon(point, polygon));
}

export function getJunctionPoint(input?: ScenarioInput): Point2D {
  return input ? buildGeometry(input).junction : { x: 0, y: 0 };
}

export function getRepresentativeBranch(input: ScenarioInput): SegmentInfo | null {
  return buildGeometry(input).centerlines.representative ?? null;
}

export function sampleContractionWidthAt(input: ScenarioInput, x: number): number {
  return sampleContractionHalfWidth(input, x) * 2;
}
