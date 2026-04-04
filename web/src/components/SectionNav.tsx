import type { SectionKey } from '../types/pinn';

const sections: Array<{ key: SectionKey; label: string; short: string; index: string }> = [
  { key: 'overview', label: '项目概览', short: '定位', index: '01' },
  { key: 'geometry', label: '参数配置', short: '几何', index: '02' },
  { key: 'visualization', label: '流场工作台', short: '画布', index: '03' },
  { key: 'reconstruction', label: '稀疏重建', short: '重建', index: '04' },
  { key: 'analysis', label: '特征分析', short: '分析', index: '05' },
  { key: 'calibration', label: '校准与扫掠', short: '校准', index: '06' },
  { key: 'methods', label: '实现说明', short: '方法', index: '07' }
];

interface SectionNavProps {
  active: SectionKey;
  onChange: (section: SectionKey) => void;
}

export function SectionNav({ active, onChange }: SectionNavProps) {
  return (
    <nav className="section-nav" aria-label="一级导航">
      {sections.map((section) => (
        <button
          key={section.key}
          type="button"
          className={section.key === active ? 'nav-pill active' : 'nav-pill'}
          onClick={() => onChange(section.key)}
        >
          <span className="nav-index">{section.index}</span>
          <span className="nav-copy">
            <strong>{section.label}</strong>
            <small>{section.short}</small>
          </span>
        </button>
      ))}
    </nav>
  );
}
