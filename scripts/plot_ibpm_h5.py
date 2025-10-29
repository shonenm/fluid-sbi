#!/usr/bin/env python3
"""
IBPM HDF5データの検証・可視化スクリプト

変換後のHDF5データが正しいかを確認します。
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import argparse

def inspect_h5_structure(h5_file):
    """HDF5ファイルの構造を表示"""
    print(f"\n{'='*60}")
    print(f"File: {h5_file}")
    print(f"{'='*60}")

    with h5py.File(h5_file, 'r') as f:
        print(f"\nDatasets:")
        for key in f.keys():
            dset = f[key]
            print(f"  {key}:")
            print(f"    Shape: {dset.shape}")
            print(f"    Dtype: {dset.dtype}")
            print(f"    Size: {dset.size * dset.dtype.itemsize / 1024 / 1024:.2f} MB")

            # 属性を表示
            if len(dset.attrs) > 0:
                print(f"    Attributes:")
                for attr_name, attr_val in dset.attrs.items():
                    print(f"      {attr_name}: {attr_val}")

            # 統計情報
            data = dset[:]
            print(f"    Range: [{data.min():.6f}, {data.max():.6f}]")
            print(f"    Mean: {data.mean():.6f}")
            print(f"    Std: {data.std():.6f}")
            print(f"    NaN count: {np.isnan(data).sum()}")

def plot_h5_samples(h5_file, output_dir, sample_idx=0, time_idx=0):
    """HDF5データのサンプルをプロット"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, 'r') as f:
        data = f['x'][:]  # (T, N, C, H, W)

    print(f"\nData shape: {data.shape}")
    T, N, C, H, W = data.shape

    # サンプルを取得
    if time_idx >= T:
        time_idx = 0
        print(f"Warning: time_idx {time_idx} out of range, using 0")

    if sample_idx >= N:
        sample_idx = 0
        print(f"Warning: sample_idx {sample_idx} out of range, using 0")

    # (C, H, W) を取得
    sample = data[time_idx, sample_idx]  # (2, H, W)
    u = sample[0]  # (H, W)
    v = sample[1]  # (H, W)

    # 速度の大きさと渦度を計算
    speed = np.sqrt(u**2 + v**2)

    # 渦度を計算（中心差分、グリッド間隔を考慮）
    # ω = ∂v/∂x - ∂u/∂y
    dx = 4.0 / (W + 1)
    dy = 4.0 / (H + 1)
    vort = np.zeros_like(u)
    # 中心差分: ∂v/∂x ≈ (v[i+1] - v[i-1]) / (2*dx)
    vort[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx) - (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy)

    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'HDF5 Data - Time={time_idx}, Sample={sample_idx}\n'
                 f'Resolution: {H}×{W}', fontsize=16, fontweight='bold')

    # 座標グリッドを作成（IBPMのセル中心座標）
    # IBPMは nx=128 で [-2, 2] の領域を分割
    # セル中心は (i+0.5)*dx + xoffset
    dx = 4.0 / (W + 1)  # W+1 セルで 4.0 の長さ
    dy = 4.0 / (H + 1)
    x = np.arange(W) * dx - 2.0 + dx/2  # セル中心
    y = np.arange(H) * dy - 2.0 + dy/2
    X, Y = np.meshgrid(x, y)

    # 円柱マスク（中心=(0,0), 半径=0.5）
    R = np.sqrt(X**2 + Y**2)
    cylinder_mask = R < 0.5

    # (0,0) 速度の大きさ
    ax = axes[0, 0]
    im0 = ax.contourf(X, Y, speed, levels=50, cmap='jet')
    ax.contour(X, Y, speed, levels=10, colors='k', linewidths=0.3, alpha=0.3)
    # 円柱を白で表示
    speed_masked = speed.copy()
    speed_masked[cylinder_mask] = np.nan
    ax.contourf(X, Y, speed_masked, levels=50, cmap='jet')
    circle = Circle((0, 0), 0.5, color='white', fill=True, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Velocity Magnitude |u|', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im0, ax=ax, label='|u|')

    # (0,1) u成分
    ax = axes[0, 1]
    im1 = ax.contourf(X, Y, u, levels=50, cmap='RdBu_r')
    ax.contour(X, Y, u, levels=10, colors='k', linewidths=0.3, alpha=0.3)
    circle = Circle((0, 0), 0.5, color='gray', fill=True, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('u-velocity', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax, label='u')

    # (1,0) v成分
    ax = axes[1, 0]
    im2 = ax.contourf(X, Y, v, levels=50, cmap='RdBu_r')
    ax.contour(X, Y, v, levels=10, colors='k', linewidths=0.3, alpha=0.3)
    circle = Circle((0, 0), 0.5, color='gray', fill=True, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('v-velocity', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax, label='v')

    # (1,1) 渦度
    ax = axes[1, 1]
    vort_levels = np.linspace(-5, 5, 51)
    im3 = ax.contourf(X, Y, vort, levels=vort_levels, cmap='RdBu_r', extend='both')
    ax.contour(X, Y, vort, levels=10, colors='k', linewidths=0.3, alpha=0.3)
    circle = Circle((0, 0), 0.5, color='gray', fill=True, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Vorticity ω (approximate)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im3, ax=ax, label='ω')

    plt.tight_layout()

    # 保存
    split_name = Path(h5_file).stem  # train, valid, test
    output_file = output_dir / f'h5_{split_name}_t{time_idx}_s{sample_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_time_series(h5_file, output_dir, sample_idx=0, time_indices=None):
    """時系列の発展をプロット"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_file, 'r') as f:
        data = f['x'][:]  # (T, N, C, H, W)

    T, N, C, H, W = data.shape

    if time_indices is None:
        # 均等に5時刻を選択
        time_indices = np.linspace(0, T-1, min(5, T), dtype=int)

    n_times = len(time_indices)
    fig, axes = plt.subplots(n_times, 3, figsize=(16, 5*n_times))
    if n_times == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'HDF5 Time Series - Sample {sample_idx}\nResolution: {H}×{W}',
                 fontsize=16, fontweight='bold')

    # 座標グリッド（IBPMのセル中心座標）
    dx = 4.0 / (W + 1)
    dy = 4.0 / (H + 1)
    x = np.arange(W) * dx - 2.0 + dx/2
    y = np.arange(H) * dy - 2.0 + dy/2
    X, Y = np.meshgrid(x, y)

    for i, t_idx in enumerate(time_indices):
        sample = data[t_idx, sample_idx]
        u = sample[0]
        v = sample[1]
        speed = np.sqrt(u**2 + v**2)

        # 渦度（中心差分、グリッド間隔を考慮）
        vort = np.zeros_like(u)
        vort[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx) - (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy)

        # 速度の大きさ
        ax = axes[i, 0]
        im0 = ax.contourf(X, Y, speed, levels=30, cmap='jet')
        circle = Circle((0, 0), 0.5, color='white', fill=True, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.set_ylabel(f't_idx = {t_idx}', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_title('Velocity Magnitude', fontsize=14)
        ax.set_aspect('equal')
        plt.colorbar(im0, ax=ax)

        # u成分
        ax = axes[i, 1]
        im1 = ax.contourf(X, Y, u, levels=30, cmap='RdBu_r')
        circle = Circle((0, 0), 0.5, color='gray', fill=True, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        if i == 0:
            ax.set_title('u-velocity', fontsize=14)
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax)

        # 渦度
        ax = axes[i, 2]
        vort_levels = np.linspace(-5, 5, 31)
        im2 = ax.contourf(X, Y, vort, levels=vort_levels, cmap='RdBu_r', extend='both')
        circle = Circle((0, 0), 0.5, color='gray', fill=True, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        if i == 0:
            ax.set_title('Vorticity', fontsize=14)
        ax.set_aspect('equal')
        plt.colorbar(im2, ax=ax)

    plt.tight_layout()

    split_name = Path(h5_file).stem
    output_file = output_dir / f'h5_{split_name}_timeseries_s{sample_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Inspect and plot IBPM HDF5 data')
    parser.add_argument('--data-dir', type=str, default='/workspace/data/ibpm_h5_128',
                        help='Directory containing HDF5 files')
    parser.add_argument('--output-dir', type=str, default='/workspace/data/ibpm_h5_plots',
                        help='Directory to save plots')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test', 'all'], default='all',
                        help='Which split to plot')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index to plot')
    parser.add_argument('--time-idx', type=int, default=0,
                        help='Time index for single sample plot')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 処理する分割を決定
    if args.split == 'all':
        splits = ['train', 'valid', 'test']
    else:
        splits = [args.split]

    print("\n" + "="*60)
    print("IBPM HDF5 Data Inspection")
    print("="*60)

    for split in splits:
        h5_file = data_dir / f'{split}.h5'

        if not h5_file.exists():
            print(f"\nWarning: {h5_file} not found, skipping...")
            continue

        # 構造を表示
        inspect_h5_structure(h5_file)

        # 単一サンプルをプロット
        print(f"\nPlotting single sample (time={args.time_idx}, sample={args.sample_idx})...")
        plot_h5_samples(h5_file, output_dir,
                       sample_idx=args.sample_idx,
                       time_idx=args.time_idx)

        # 時系列をプロット
        print(f"\nPlotting time series (sample={args.sample_idx})...")
        plot_time_series(h5_file, output_dir, sample_idx=args.sample_idx)

    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
