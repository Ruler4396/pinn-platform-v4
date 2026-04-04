import type { ScenarioMetrics } from '../types/pinn';

interface MetricCardsProps {
  metrics: ScenarioMetrics;
}

const items = (metrics: ScenarioMetrics) => [
  ['Re', metrics.reynolds.toFixed(4)],
  ['峰值速度', `${(metrics.maxSpeed * 1000).toFixed(3)} mm/s`],
  ['平均压降', `${metrics.avgPressureDrop.toFixed(3)} Pa`],
  ['壁面剪切代理', metrics.wallShearProxy.toExponential(2)],
  ['流线曲率代理', metrics.streamlineCurvatureProxy.toFixed(3)],
  ['中心线压降梯度', metrics.centerlinePressureGradient.toExponential(2)]
];

export function MetricCards({ metrics }: MetricCardsProps) {
  return (
    <div className="metric-grid compact-metric-grid">
      {items(metrics).map(([label, value]) => (
        <article className="metric-card" key={label}>
          <header>{label}</header>
          <strong>{value}</strong>
        </article>
      ))}
    </div>
  );
}
