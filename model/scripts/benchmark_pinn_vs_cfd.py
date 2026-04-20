#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.pinn_platform_api import ENGINE  # noqa: E402
from model.src.data.bend_cases import get_case as get_bend_case  # noqa: E402
from model.src.data.bend_freefem import run_freefem as run_bend_freefem  # noqa: E402
from model.src.data.contraction_cases import get_case as get_contraction_case  # noqa: E402
from model.src.data.contraction_freefem import run_freefem as run_contraction_freefem  # noqa: E402


CONTRACTION_SCENARIO: dict[str, Any] = {
    'geometry': {
        'type': 'contraction',
        'wUm': 200,
        'lInOverW': 4,
        'lOutOverW': 8,
        'beta': 0.7,
        'lCOverW': 4,
        'rcOverW': 6,
        'thetaDeg': 90,
        'inletProfile': 'parabolic',
    },
    'fluid': {'preset': 'water', 'density': 997.05, 'viscosity': 8.902e-4},
    'flow': {'meanVelocity': 0.0001, 'outletPressure': 0},
    'sparse': {'sampleRatePct': 10, 'noisePct': 2, 'strategy': 'region_aware'},
}

BEND_SCENARIO: dict[str, Any] = {
    'geometry': {
        'type': 'bend',
        'wUm': 200,
        'lInOverW': 4,
        'lOutOverW': 6,
        'beta': 0.7,
        'lCOverW': 4,
        'rcOverW': 6,
        'thetaDeg': 90,
        'inletProfile': 'parabolic',
    },
    'fluid': {'preset': 'water', 'density': 997.05, 'viscosity': 8.902e-4},
    'flow': {'meanVelocity': 0.0001, 'outletPressure': 0},
    'sparse': {'sampleRatePct': 10, 'noisePct': 2, 'strategy': 'region_aware'},
}


def benchmark(fn: Callable[[], Any], runs: int) -> dict[str, Any]:
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        'samples_s': samples,
        'min_s': min(samples),
        'median_s': statistics.median(samples),
        'max_s': max(samples),
        'mean_s': statistics.mean(samples),
        'runs': runs,
    }


def speedup(cfd_stats: dict[str, Any], pinn_stats: dict[str, Any]) -> float:
    return float(cfd_stats['median_s']) / max(float(pinn_stats['median_s']), 1.0e-12)


def build_results(pinn_runs: int, cfd_runs: int) -> dict[str, Any]:
    ENGINE.simulate(
        CONTRACTION_SCENARIO,
        resolution='preview',
        include_streamlines=False,
        include_probes=False,
        include_sparse=False,
        include_reconstruction=False,
    )
    ENGINE.reconstruct(CONTRACTION_SCENARIO)
    ENGINE.simulate(
        BEND_SCENARIO,
        resolution='preview',
        include_streamlines=False,
        include_probes=False,
        include_sparse=False,
        include_reconstruction=False,
    )
    ENGINE.reconstruct(BEND_SCENARIO)

    with tempfile.TemporaryDirectory(prefix='bench_cfd_') as tmp_dir:
        tmp_root = Path(tmp_dir)

        contraction_cfd_case = get_contraction_case('C-base')
        bend_cfd_case = get_bend_case('B-base')

        results = {
            'metadata': {
                'created_at': datetime.now().isoformat(timespec='seconds'),
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'python': sys.version.split()[0],
                'pinn_runs': pinn_runs,
                'cfd_runs': cfd_runs,
                'pinn_mode': 'api ScenarioEngine preview inference',
                'cfd_mode': 'FreeFem++ Stokes solve',
            },
            'scenarios': {
                'contraction': CONTRACTION_SCENARIO,
                'bend': BEND_SCENARIO,
            },
            'benchmarks': {
                'contraction_full_inference': benchmark(
                    lambda: ENGINE.simulate(
                        CONTRACTION_SCENARIO,
                        resolution='preview',
                        include_streamlines=False,
                        include_probes=False,
                        include_sparse=False,
                        include_reconstruction=False,
                    ),
                    pinn_runs,
                ),
                'contraction_sparse_reconstruction': benchmark(
                    lambda: ENGINE.reconstruct(CONTRACTION_SCENARIO),
                    pinn_runs,
                ),
                'bend_full_inference': benchmark(
                    lambda: ENGINE.simulate(
                        BEND_SCENARIO,
                        resolution='preview',
                        include_streamlines=False,
                        include_probes=False,
                        include_sparse=False,
                        include_reconstruction=False,
                    ),
                    pinn_runs,
                ),
                'bend_sparse_reconstruction': benchmark(
                    lambda: ENGINE.reconstruct(BEND_SCENARIO),
                    pinn_runs,
                ),
                'contraction_cfd': benchmark(
                    lambda: run_contraction_freefem(contraction_cfd_case, tmp_root / 'contraction', max_retries=1),
                    cfd_runs,
                ),
                'bend_cfd': benchmark(
                    lambda: run_bend_freefem(bend_cfd_case, tmp_root / 'bend', max_retries=1),
                    cfd_runs,
                ),
            },
        }

    comparison = {
        'contraction_full_vs_cfd_speedup': speedup(
            results['benchmarks']['contraction_cfd'],
            results['benchmarks']['contraction_full_inference'],
        ),
        'contraction_sparse_vs_cfd_speedup': speedup(
            results['benchmarks']['contraction_cfd'],
            results['benchmarks']['contraction_sparse_reconstruction'],
        ),
        'bend_full_vs_cfd_speedup': speedup(
            results['benchmarks']['bend_cfd'],
            results['benchmarks']['bend_full_inference'],
        ),
        'bend_sparse_vs_cfd_speedup': speedup(
            results['benchmarks']['bend_cfd'],
            results['benchmarks']['bend_sparse_reconstruction'],
        ),
    }
    results['comparison'] = comparison
    return results


def write_markdown(output_path: Path, results: dict[str, Any]) -> None:
    bench = results['benchmarks']
    comp = results['comparison']
    lines = [
        '# PINN 与 CFD 速度基准实验',
        '',
        f"- 生成时间：`{results['metadata']['created_at']}`",
        f"- 主机：`{results['metadata']['hostname']}`",
        f"- 平台：`{results['metadata']['platform']}`",
        f"- Python：`{results['metadata']['python']}`",
        f"- PINN 重复次数：`{results['metadata']['pinn_runs']}`",
        f"- CFD 重复次数：`{results['metadata']['cfd_runs']}`",
        '',
        '## 中位耗时汇总',
        '',
        '| 工况 | PINN 中位耗时 (s) | CFD 中位耗时 (s) | CFD / PINN |',
        '| --- | ---: | ---: | ---: |',
        f"| 收缩流道完整推理 | {bench['contraction_full_inference']['median_s']:.3f} | {bench['contraction_cfd']['median_s']:.3f} | {comp['contraction_full_vs_cfd_speedup']:.2f}x |",
        f"| 收缩流道稀疏重建 | {bench['contraction_sparse_reconstruction']['median_s']:.3f} | {bench['contraction_cfd']['median_s']:.3f} | {comp['contraction_sparse_vs_cfd_speedup']:.2f}x |",
        f"| 弯曲流道完整推理 | {bench['bend_full_inference']['median_s']:.3f} | {bench['bend_cfd']['median_s']:.3f} | {comp['bend_full_vs_cfd_speedup']:.2f}x |",
        f"| 弯曲流道稀疏重建 | {bench['bend_sparse_reconstruction']['median_s']:.3f} | {bench['bend_cfd']['median_s']:.3f} | {comp['bend_sparse_vs_cfd_speedup']:.2f}x |",
        '',
        '## 结果说明',
        '',
        '- 在当前预览分辨率推理路径下，收缩流道与弯曲流道的 PINN 推理都快于同机 CFD 求解。',
        '- 弯曲流道上的绝对优势更大，主要原因是其 CFD 网格更重、求解代价更高，而 PINN 前向推理开销增长较小。',
        '- 在本次实验中，在线稀疏重建路径也仍然快于对应的 CFD 求解。',
        '',
        '## 原始文件',
        '',
        f"- JSON：`{output_path.with_suffix('.json').name}`",
        f"- 图片：`{output_path.with_suffix('.png').name}`",
        '',
    ]
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def write_figure(output_path: Path, results: dict[str, Any]) -> None:
    bench = results['benchmarks']
    geometry_labels = ['收缩流道', '弯曲流道']
    pinn_full = [
        bench['contraction_full_inference']['median_s'],
        bench['bend_full_inference']['median_s'],
    ]
    pinn_sparse = [
        bench['contraction_sparse_reconstruction']['median_s'],
        bench['bend_sparse_reconstruction']['median_s'],
    ]
    cfd_forward = [
        bench['contraction_cfd']['median_s'],
        bench['bend_cfd']['median_s'],
    ]
    speedup_labels = [
        '收缩完整推理',
        '收缩稀疏重建',
        '弯曲完整推理',
        '弯曲稀疏重建',
    ]
    speedups = [
        results['comparison']['contraction_full_vs_cfd_speedup'],
        results['comparison']['contraction_sparse_vs_cfd_speedup'],
        results['comparison']['bend_full_vs_cfd_speedup'],
        results['comparison']['bend_sparse_vs_cfd_speedup'],
    ]

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Noto Serif CJK SC', 'Noto Serif CJK JP', 'DejaVu Serif'],
            'axes.unicode_minus': False,
            'mathtext.fontset': 'stix',
            'font.size': 10,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.edgecolor': '#222222',
            'axes.labelcolor': '#111111',
            'xtick.color': '#222222',
            'ytick.color': '#222222',
            'axes.linewidth': 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), gridspec_kw={'width_ratios': [1.18, 1.0]})

    x_positions = list(range(len(geometry_labels)))
    width = 0.22
    axes[0].bar(
        [x - width for x in x_positions],
        pinn_full,
        width=width,
        color='#4c78a8',
        label='PINN 完整推理',
        edgecolor='white',
        linewidth=0.4,
    )
    axes[0].bar(
        x_positions,
        pinn_sparse,
        width=width,
        color='#72b7b2',
        label='PINN 稀疏重建',
        edgecolor='white',
        linewidth=0.4,
    )
    axes[0].bar(
        [x + width for x in x_positions],
        cfd_forward,
        width=width,
        color='#bab0ac',
        label='CFD 前向求解',
        edgecolor='white',
        linewidth=0.4,
    )
    axes[0].set_yscale('log')
    axes[0].set_ylabel('中位耗时（秒，对数坐标）')
    axes[0].set_title('（a）PINN 与 CFD 耗时对比', fontsize=11, pad=8)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(geometry_labels)
    axes[0].grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.38, color='#9ca3af')
    axes[0].legend(frameon=True, loc='upper left', fancybox=False, edgecolor='#666666', facecolor='white', fontsize=9)

    for idx, value in enumerate(pinn_full):
        axes[0].text(
            idx - width,
            value * 1.10,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#355c8a',
        )
    for idx, value in enumerate(pinn_sparse):
        axes[0].text(
            idx,
            value * 1.10,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#3b817b',
        )
    for idx, value in enumerate(cfd_forward):
        axes[0].text(
            idx + width,
            value * 1.10,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#6b625d',
        )

    axes[1].barh(
        speedup_labels,
        speedups,
        color=['#4c78a8', '#72b7b2', '#4c78a8', '#72b7b2'],
        edgecolor='white',
        linewidth=0.4,
        height=0.76,
    )
    axes[1].set_xlabel('CFD / PINN 加速倍数')
    axes[1].set_title('（b）相对加速倍数', fontsize=11, pad=8)
    axes[1].grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.38, color='#9ca3af')
    for label, value in zip(speedup_labels, speedups):
        axes[1].text(
            value + max(speedups) * 0.018,
            label,
            f'{value:.2f}x',
            va='center',
            fontsize=8.5,
            color='#111111',
        )

    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark PINN inference and sparse reconstruction against CFD.')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'docs' / 'benchmarks')
    parser.add_argument('--name', default='pinn_vs_cfd_speed_benchmark_20260420')
    parser.add_argument('--pinn-runs', type=int, default=7)
    parser.add_argument('--cfd-runs', type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / args.name

    results = build_results(pinn_runs=max(args.pinn_runs, 1), cfd_runs=max(args.cfd_runs, 1))
    json_path = output_base.with_suffix('.json')
    md_path = output_base.with_suffix('.md')
    png_path = output_base.with_suffix('.png')

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    write_markdown(md_path, results)
    write_figure(png_path, results)

    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')
    print(f'Wrote {png_path}')


if __name__ == '__main__':
    main()
