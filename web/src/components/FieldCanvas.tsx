import { useEffect, useId, useMemo, useRef } from 'react';
import type { FieldLayer, FieldPoint, ScenarioInput, ScenarioResult } from '../types/pinn';
import { buildGeometry, type Point2D } from '../lib/geometry';
import { clamp, lerp } from '../lib/utils';

interface FieldCanvasProps {
  input: ScenarioInput;
  result: ScenarioResult | null;
  streamlines?: ScenarioResult['streamlines'];
  layer: FieldLayer;
  reconstruction?: FieldPoint[];
  sparsePoints?: FieldPoint[];
  probe?: FieldPoint | null;
  onQuery?: (point: { x: number; y: number }) => void;
}

interface ColorStop {
  at: number;
  rgb: [number, number, number];
}

const scientificStops: ColorStop[] = [
  { at: 0, rgb: [22, 51, 112] },
  { at: 0.18, rgb: [35, 103, 206] },
  { at: 0.38, rgb: [67, 176, 223] },
  { at: 0.56, rgb: [167, 220, 208] },
  { at: 0.72, rgb: [245, 234, 169] },
  { at: 0.86, rgb: [236, 151, 79] },
  { at: 1, rgb: [184, 45, 38] }
];

function colorAt(t: number): [number, number, number] {
  const clamped = clamp(t, 0, 1);
  const upperIndex = scientificStops.findIndex((stop) => stop.at >= clamped);

  if (upperIndex <= 0) {
    return scientificStops[0].rgb;
  }
  if (upperIndex === -1) {
    return scientificStops[scientificStops.length - 1].rgb;
  }

  const left = scientificStops[upperIndex - 1];
  const right = scientificStops[upperIndex];
  const localT = (clamped - left.at) / Math.max(right.at - left.at, 1e-6);

  return [0, 1, 2].map((index) => {
    const mixed = Math.round(lerp(left.rgb[index], right.rgb[index], localT));
    return Math.round(lerp(mixed, 255, 0.06));
  }) as [number, number, number];
}

function buildGradientString(): string {
  return `linear-gradient(90deg, ${scientificStops
    .map((stop) => {
      const [r, g, b] = stop.rgb;
      return `rgb(${r}, ${g}, ${b}) ${stop.at * 100}%`;
    })
    .join(', ')})`;
}

function rasterize(dataset: FieldPoint[], layer: FieldLayer) {
  const scalarLayer = layer === 'pressure' ? 'pressure' : 'speed';
  const values = dataset.map((point) => (scalarLayer === 'pressure' ? point.p : point.speed));
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;

  const xs = Array.from(new Set(dataset.map((point) => point.x))).sort((left, right) => left - right);
  const ys = Array.from(new Set(dataset.map((point) => point.y))).sort((left, right) => left - right);
  const xIndex = new Map(xs.map((value, index) => [value, index]));
  const yIndex = new Map(ys.map((value, index) => [value, index]));
  const grid = Array.from({ length: ys.length }, () => Array<number | undefined>(xs.length).fill(undefined));

  dataset.forEach((point) => {
    const value = scalarLayer === 'pressure' ? point.p : point.speed;
    const row = yIndex.get(point.y);
    const col = xIndex.get(point.x);
    if (row !== undefined && col !== undefined) {
      grid[row][col] = value;
    }
  });

  const occupancy = xs.length > 0 && ys.length > 0 ? dataset.length / Math.max(xs.length * ys.length, 1) : 0;
  const points = dataset.map((point) => ({
    x: point.x,
    y: point.y,
    value: scalarLayer === 'pressure' ? point.p : point.speed
  }));

  return {
    mode: occupancy >= 0.42 ? 'grid' : 'scatter',
    xs,
    ys,
    grid,
    points,
    occupancy,
    minValue,
    maxValue
  };
}

function interpolatePoint(level: number, a: Point2D, b: Point2D, va: number, vb: number): Point2D {
  const ratio = Math.abs(vb - va) < 1e-6 ? 0.5 : clamp((level - va) / (vb - va), 0, 1);
  return {
    x: lerp(a.x, b.x, ratio),
    y: lerp(a.y, b.y, ratio)
  };
}

const contourCases: Record<number, Array<[number, number]>> = {
  0: [],
  1: [[3, 0]],
  2: [[0, 1]],
  3: [[3, 1]],
  4: [[1, 2]],
  5: [
    [3, 2],
    [0, 1]
  ],
  6: [[0, 2]],
  7: [[3, 2]],
  8: [[2, 3]],
  9: [[0, 2]],
  10: [
    [0, 3],
    [1, 2]
  ],
  11: [[1, 2]],
  12: [[1, 3]],
  13: [[0, 1]],
  14: [[3, 0]],
  15: []
};

function FieldCanvas({
  input,
  result,
  streamlines,
  layer,
  reconstruction,
  sparsePoints,
  probe,
  onQuery
}: FieldCanvasProps) {
  const clipId = useId();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const geometry = useMemo(() => buildGeometry(input), [input]);
  const dataset = reconstruction ?? result?.field ?? [];
  const renderScale = input.geometry.type === 'bend' ? 0.8 : 1;
  const bounds = geometry.bounds;
  const renderCenter = useMemo(
    () => ({
      x: (bounds.xMin + bounds.xMax) / 2,
      y: (bounds.yMin + bounds.yMax) / 2
    }),
    [bounds.xMax, bounds.xMin, bounds.yMax, bounds.yMin]
  );
  const scalePoint = useMemo(
    () => (point: Point2D, factor: number): Point2D => ({
      x: renderCenter.x + (point.x - renderCenter.x) * factor,
      y: renderCenter.y + (point.y - renderCenter.y) * factor
    }),
    [renderCenter.x, renderCenter.y]
  );
  const datasetForRender = useMemo(
    () =>
      dataset.map((point) => {
        const scaled = scalePoint(point, renderScale);
        return {
          ...point,
          x: scaled.x,
          y: scaled.y
        };
      }),
    [dataset, renderScale, scalePoint]
  );
  const polygons = useMemo(
    () => geometry.polygons.map((polygon) => polygon.map((point) => scalePoint(point, renderScale))),
    [geometry.polygons, renderScale, scalePoint]
  );
  const renderStreamlines = useMemo(
    () =>
      streamlines?.map((line) => line.map((point) => scalePoint(point, renderScale))) ?? [],
    [renderScale, scalePoint, streamlines]
  );
  const renderSparsePoints = useMemo(
    () =>
      sparsePoints?.map((point) => {
        const scaled = scalePoint(point, renderScale);
        return {
          ...point,
          x: scaled.x,
          y: scaled.y
        };
      }),
    [renderScale, scalePoint, sparsePoints]
  );
  const renderProbe = useMemo(() => {
    if (!probe) {
      return probe;
    }
    const scaled = scalePoint(probe, renderScale);
    return {
      ...probe,
      x: scaled.x,
      y: scaled.y
    };
  }, [probe, renderScale, scalePoint]);
  const width = bounds.xMax - bounds.xMin;
  const height = bounds.yMax - bounds.yMin;
  const viewBox = `${bounds.xMin} ${-bounds.yMax} ${width} ${height}`;
  const raster = useMemo(() => rasterize(datasetForRender, layer), [datasetForRender, layer]);
  const pixelWidth = 1440;
  const pixelHeight = Math.round((pixelWidth * height) / Math.max(width, 1));
  const aspectRatio = `${width} / ${height}`;
  const scalarLabel = layer === 'pressure' ? '压力' : '速度';
  const legendUnit = layer === 'pressure' ? 'Pa' : 'm/s';
  const legendGradient = useMemo(() => buildGradientString(), []);
  const streamlineStrokeWidth = Math.max(input.geometry.wUm * 0.018, 4);
  const markerRadius = Math.max(input.geometry.wUm * 0.042, 9);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const toCanvasX = (value: number) => ((value - bounds.xMin) / Math.max(width, 1)) * pixelWidth;
    const toCanvasY = (value: number) => pixelHeight - ((value - bounds.yMin) / Math.max(height, 1)) * pixelHeight;
    const toCanvasPoint = (point: Point2D) => ({ x: toCanvasX(point.x), y: toCanvasY(point.y) });

    const drawPolygonPath = () => {
      polygons.forEach((polygon) => {
        polygon.forEach((point, index) => {
          const mapped = toCanvasPoint(point);
          if (index === 0) {
            ctx.moveTo(mapped.x, mapped.y);
          } else {
            ctx.lineTo(mapped.x, mapped.y);
          }
        });
        ctx.closePath();
      });
    };

    ctx.clearRect(0, 0, pixelWidth, pixelHeight);
    ctx.fillStyle = '#f8f8f5';
    ctx.fillRect(0, 0, pixelWidth, pixelHeight);

    ctx.save();
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.05)';
    ctx.lineWidth = 1;
    for (let index = 1; index < 12; index += 1) {
      const x = (pixelWidth * index) / 12;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, pixelHeight);
      ctx.stroke();
    }
    for (let index = 1; index < 8; index += 1) {
      const y = (pixelHeight * index) / 8;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(pixelWidth, y);
      ctx.stroke();
    }
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    drawPolygonPath();
    ctx.clip();

    ctx.fillStyle = '#eef1ec';
    ctx.fillRect(0, 0, pixelWidth, pixelHeight);

    if (raster.mode === 'grid' && raster.xs.length > 1 && raster.ys.length > 1) {
      const source = document.createElement('canvas');
      source.width = raster.xs.length;
      source.height = raster.ys.length;
      const sourceCtx = source.getContext('2d');

      if (sourceCtx) {
        const image = sourceCtx.createImageData(source.width, source.height);
        for (let row = 0; row < raster.ys.length; row += 1) {
          for (let col = 0; col < raster.xs.length; col += 1) {
            const value = raster.grid[row][col];
            const pixelRow = raster.ys.length - 1 - row;
            const pixelIndex = (pixelRow * raster.xs.length + col) * 4;
            if (value === undefined) {
              image.data[pixelIndex + 3] = 0;
              continue;
            }
            const t = clamp((value - raster.minValue) / Math.max(raster.maxValue - raster.minValue, 1e-6), 0, 1);
            const [r, g, b] = colorAt(t);
            image.data[pixelIndex] = r;
            image.data[pixelIndex + 1] = g;
            image.data[pixelIndex + 2] = b;
            image.data[pixelIndex + 3] = layer === 'streamline' ? 164 : 234;
          }
        }
        sourceCtx.putImageData(image, 0, 0);

        ctx.imageSmoothingEnabled = true;
        ctx.filter = 'blur(9px) saturate(0.96)';
        ctx.drawImage(source, 0, 0, pixelWidth, pixelHeight);
        ctx.filter = 'none';
        ctx.globalAlpha = layer === 'streamline' ? 0.48 : 0.92;
        ctx.drawImage(source, 0, 0, pixelWidth, pixelHeight);
        ctx.globalAlpha = 1;

        if (layer !== 'streamline' && raster.maxValue - raster.minValue > 1e-6) {
          const levels = Array.from({ length: 6 }, (_, index) => raster.minValue + ((index + 1) * (raster.maxValue - raster.minValue)) / 7);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.42)';
          ctx.lineWidth = 1.2;

          levels.forEach((level) => {
            ctx.beginPath();
            for (let row = 0; row < raster.ys.length - 1; row += 1) {
              for (let col = 0; col < raster.xs.length - 1; col += 1) {
                const bottomLeft = raster.grid[row][col];
                const bottomRight = raster.grid[row][col + 1];
                const topRight = raster.grid[row + 1][col + 1];
                const topLeft = raster.grid[row + 1][col];
                if (
                  bottomLeft === undefined ||
                  bottomRight === undefined ||
                  topRight === undefined ||
                  topLeft === undefined
                ) {
                  continue;
                }

                const mask =
                  (bottomLeft >= level ? 1 : 0) |
                  (bottomRight >= level ? 2 : 0) |
                  (topRight >= level ? 4 : 0) |
                  (topLeft >= level ? 8 : 0);

                const segments = contourCases[mask];
                if (!segments || segments.length === 0) {
                  continue;
                }

                const pBottomLeft = { x: raster.xs[col], y: raster.ys[row] };
                const pBottomRight = { x: raster.xs[col + 1], y: raster.ys[row] };
                const pTopRight = { x: raster.xs[col + 1], y: raster.ys[row + 1] };
                const pTopLeft = { x: raster.xs[col], y: raster.ys[row + 1] };
                const edges = [
                  interpolatePoint(level, pBottomLeft, pBottomRight, bottomLeft, bottomRight),
                  interpolatePoint(level, pBottomRight, pTopRight, bottomRight, topRight),
                  interpolatePoint(level, pTopRight, pTopLeft, topRight, topLeft),
                  interpolatePoint(level, pTopLeft, pBottomLeft, topLeft, bottomLeft)
                ];

                segments.forEach(([startIndex, endIndex]) => {
                  const start = toCanvasPoint(edges[startIndex]);
                  const end = toCanvasPoint(edges[endIndex]);
                  ctx.moveTo(start.x, start.y);
                  ctx.lineTo(end.x, end.y);
                });
              }
            }
            ctx.stroke();
          });
        }
      }
    }

    if (raster.mode === 'scatter' && raster.points.length > 0) {
      const source = document.createElement('canvas');
      source.width = pixelWidth;
      source.height = pixelHeight;
      const sourceCtx = source.getContext('2d');

      if (sourceCtx) {
        const scatterRadius = Math.max(
          14,
          Math.min(34, Math.round(Math.sqrt((pixelWidth * pixelHeight) / Math.max(raster.points.length, 1)) * 0.75))
        );

        raster.points.forEach((point) => {
          const mapped = toCanvasPoint(point);
          const t = clamp((point.value - raster.minValue) / Math.max(raster.maxValue - raster.minValue, 1e-6), 0, 1);
          const [r, g, b] = colorAt(t);
          const gradient = sourceCtx.createRadialGradient(mapped.x, mapped.y, 0, mapped.x, mapped.y, scatterRadius);
          gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${layer === 'streamline' ? 0.16 : 0.34})`);
          gradient.addColorStop(0.58, `rgba(${r}, ${g}, ${b}, ${layer === 'streamline' ? 0.08 : 0.16})`);
          gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
          sourceCtx.fillStyle = gradient;
          sourceCtx.fillRect(
            mapped.x - scatterRadius,
            mapped.y - scatterRadius,
            scatterRadius * 2,
            scatterRadius * 2
          );
        });

        ctx.imageSmoothingEnabled = true;
        ctx.filter = layer === 'streamline' ? 'blur(10px) saturate(0.9)' : 'blur(14px) saturate(1.04)';
        ctx.drawImage(source, 0, 0, pixelWidth, pixelHeight);
        ctx.filter = 'none';
        ctx.globalAlpha = layer === 'streamline' ? 0.44 : 0.96;
        ctx.drawImage(source, 0, 0, pixelWidth, pixelHeight);
        ctx.globalAlpha = 1;
      }
    }

    ctx.restore();
  }, [
    bounds.xMin,
    bounds.yMin,
    height,
    layer,
    pixelHeight,
    pixelWidth,
    polygons,
    raster.grid,
    raster.maxValue,
    raster.minValue,
    raster.mode,
    raster.points,
    raster.xs,
    raster.ys,
    width
  ]);

  const handleClick = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!onQuery) {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    const x = bounds.xMin + ((event.clientX - rect.left) / rect.width) * width;
    const y = bounds.yMax - ((event.clientY - rect.top) / rect.height) * height;
    const modelPoint = scalePoint({ x, y }, 1 / renderScale);
    onQuery(modelPoint);
  };

  const showStreamlines = Boolean(renderStreamlines.length);

  return (
    <section className="canvas-card">
      <div className="field-canvas-shell" style={{ aspectRatio }}>
        <canvas ref={canvasRef} width={pixelWidth} height={pixelHeight} className="field-canvas-bitmap" />
        <svg
          className="field-canvas-overlay"
          viewBox={viewBox}
          preserveAspectRatio="xMidYMid meet"
          onClick={handleClick}
          role={onQuery ? 'button' : 'img'}
          aria-label="PINN 收缩/弯曲流道二维画布"
        >
          <defs>
            <clipPath id={`clip-${clipId}`} clipPathUnits="userSpaceOnUse">
              {polygons.map((polygon, index) => (
                <polygon
                  key={`clip-${index}`}
                  points={polygon.map((point) => `${point.x},${-point.y}`).join(' ')}
                />
              ))}
            </clipPath>
          </defs>

          {showStreamlines && (
            <g clipPath={`url(#clip-${clipId})`}>
              {renderStreamlines.map((line, index) => (
                <polyline
                  key={`line-${index}`}
                  points={line.map((point) => `${point.x},${-point.y}`).join(' ')}
                  fill="none"
                  stroke={layer === 'streamline' ? 'rgba(17, 24, 39, 0.34)' : 'rgba(17, 24, 39, 0.16)'}
                  strokeWidth={layer === 'streamline' ? streamlineStrokeWidth : streamlineStrokeWidth * 0.72}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))}
            </g>
          )}

          {polygons.map((polygon, index) => (
            <polygon
              key={index}
              points={polygon.map((point) => `${point.x},${-point.y}`).join(' ')}
              fill="rgba(255,255,255,0.02)"
              stroke="rgba(29, 35, 42, 0.76)"
              strokeWidth="4"
              strokeLinejoin="round"
            />
          ))}

          {polygons.map((polygon, index) => (
            <polyline
              key={`inner-${index}`}
              points={polygon.map((point) => `${point.x},${-point.y}`).join(' ')}
              fill="none"
              stroke="rgba(255,255,255,0.54)"
              strokeWidth="1.6"
              strokeLinejoin="round"
            />
          ))}

          <g clipPath={`url(#clip-${clipId})`}>
            {renderSparsePoints?.map((point, index) => (
              <g key={`sparse-${index}`}>
                <circle
                  cx={point.x}
                  cy={-point.y}
                  r={markerRadius * 0.42}
                  fill="rgba(255,255,255,0.92)"
                  stroke="rgba(191, 122, 40, 0.9)"
                  strokeWidth="2.6"
                />
                <circle
                  cx={point.x}
                  cy={-point.y}
                  r={markerRadius * 0.82}
                  fill="none"
                  stroke="rgba(191, 122, 40, 0.18)"
                  strokeWidth="1.8"
                />
              </g>
            ))}

            {renderProbe && (
              <g className="probe-marker">
                <circle cx={renderProbe.x} cy={-renderProbe.y} r={markerRadius * 1.08} fill="rgba(14, 116, 144, 0.12)" />
                <circle
                  cx={renderProbe.x}
                  cy={-renderProbe.y}
                  r={markerRadius * 0.42}
                  fill="rgba(20, 113, 186, 0.96)"
                  stroke="white"
                  strokeWidth="2.8"
                />
              </g>
            )}
          </g>
        </svg>
      </div>

      <div className="canvas-footer">
        <div className="legend-block">
          <span className="legend-caption">{layer === 'streamline' ? '速度底图' : `${scalarLabel}热力图`}</span>
          <div className="legend-scale">
            <span>{raster.minValue.toFixed(layer === 'pressure' ? 3 : 6)}</span>
            <div className="legend-gradient" style={{ background: legendGradient }} />
            <span>{raster.maxValue.toFixed(layer === 'pressure' ? 3 : 6)}</span>
            <small>{legendUnit}</small>
          </div>
        </div>
        <span className="canvas-hint">点击流道查询点位</span>
      </div>
    </section>
  );
}

export { FieldCanvas };
