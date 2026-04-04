interface SeriesConfig {
  key: string;
  label: string;
  color: string;
  axis?: 'left' | 'right';
}

interface LineChartProps<T extends Record<string, number>> {
  title: string;
  data: T[];
  xKey: keyof T;
  series: SeriesConfig[];
  formatX?: (value: number) => string;
  formatLeftY?: (value: number) => string;
  formatRightY?: (value: number) => string;
}

function range(values: number[]): { min: number; max: number } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return { min: min - 1, max: max + 1 };
  }
  return { min, max };
}

function toX(
  value: number,
  min: number,
  max: number,
  width: number,
  paddingLeft: number,
  paddingRight: number
): number {
  const span = Math.max(max - min, 1e-6);
  return paddingLeft + ((value - min) / span) * (width - paddingLeft - paddingRight);
}

function toY(value: number, min: number, max: number, height: number, paddingTop: number, paddingBottom: number): number {
  const span = Math.max(max - min, 1e-6);
  return height - paddingBottom - ((value - min) / span) * (height - paddingTop - paddingBottom);
}

function buildPath<T extends Record<string, number>>(
  data: T[],
  xKey: keyof T,
  yKey: string,
  xRange: { min: number; max: number },
  yRange: { min: number; max: number },
  width: number,
  height: number,
  paddingLeft: number,
  paddingRight: number,
  paddingTop: number,
  paddingBottom: number
): string {
  return data
    .map((point) => {
      const x = toX(Number(point[xKey]), xRange.min, xRange.max, width, paddingLeft, paddingRight);
      const y = toY(Number(point[yKey]), yRange.min, yRange.max, height, paddingTop, paddingBottom);
      return `${x},${y}`;
    })
    .join(' ');
}

export function LineChart<T extends Record<string, number>>({
  title,
  data,
  xKey,
  series,
  formatX,
  formatLeftY,
  formatRightY
}: LineChartProps<T>) {
  const width = 560;
  const height = 252;
  const paddingLeft = 22;
  const paddingRight = 22;
  const paddingTop = 24;
  const paddingBottom = 28;

  if (data.length === 0) {
    return (
      <section className="chart-card">
        <header className="chart-head">
          <div>
            <span className="card-kicker">Path probe</span>
            <h3>{title}</h3>
          </div>
        </header>
        <div className="empty-state">暂无剖面数据。</div>
      </section>
    );
  }

  const xValues = data.map((point) => Number(point[xKey]));
  const leftSeries = series.filter((item) => (item.axis ?? 'left') === 'left');
  const rightSeries = series.filter((item) => item.axis === 'right');

  const leftValues = leftSeries.flatMap((item) => data.map((point) => Number(point[item.key])));
  const rightValues = rightSeries.flatMap((item) => data.map((point) => Number(point[item.key])));

  const xRange = range(xValues);
  const leftRange = range(leftValues.length ? leftValues : [0, 1]);
  const rightRange = range(rightValues.length ? rightValues : [0, 1]);
  const yGrid = Array.from({ length: 4 }, (_, index) => paddingTop + ((height - paddingTop - paddingBottom) * index) / 3);

  return (
    <section className="chart-card">
      <header className="chart-head">
        <div>
          <span className="card-kicker">Path probe</span>
          <h3>{title}</h3>
        </div>
        <div className="legend-row">
          {series.map((item) => (
            <span key={item.key} className="legend-pill">
              <i style={{ background: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      </header>

      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart" role="img" aria-label={title}>
        <rect x="0" y="0" width={width} height={height} rx="24" fill="#f6f6f3" stroke="rgba(15, 23, 42, 0.08)" />

        {yGrid.map((y) => (
          <line key={y} x1={paddingLeft} y1={y} x2={width - paddingRight} y2={y} className="grid-line" />
        ))}

        <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} className="axis-line" />
        <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={height - paddingBottom} className="axis-line" />
        <line x1={width - paddingRight} y1={paddingTop} x2={width - paddingRight} y2={height - paddingBottom} className="axis-line subtle" />

        {leftSeries.map((item) => (
          <polyline
            key={item.key}
            fill="none"
            stroke={item.color}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={buildPath(
              data,
              xKey,
              item.key,
              xRange,
              leftRange,
              width,
              height,
              paddingLeft,
              paddingRight,
              paddingTop,
              paddingBottom
            )}
          />
        ))}

        {rightSeries.map((item) => (
          <polyline
            key={item.key}
            fill="none"
            stroke={item.color}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="8 7"
            points={buildPath(
              data,
              xKey,
              item.key,
              xRange,
              rightRange,
              width,
              height,
              paddingLeft,
              paddingRight,
              paddingTop,
              paddingBottom
            )}
          />
        ))}

        <text x={paddingLeft} y={paddingTop - 6} className="axis-text">
          {formatLeftY ? formatLeftY(leftRange.max) : leftRange.max.toFixed(2)}
        </text>
        {rightSeries.length > 0 && (
          <text x={width - paddingRight} y={paddingTop - 6} textAnchor="end" className="axis-text axis-text-right">
            {formatRightY ? formatRightY(rightRange.max) : rightRange.max.toFixed(2)}
          </text>
        )}
        <text x={paddingLeft} y={height - 8} className="axis-text">
          {formatX ? formatX(xRange.min) : xRange.min.toFixed(2)}
        </text>
        <text x={width - paddingRight} y={height - 8} textAnchor="end" className="axis-text">
          {formatX ? formatX(xRange.max) : xRange.max.toFixed(2)}
        </text>
      </svg>
    </section>
  );
}
