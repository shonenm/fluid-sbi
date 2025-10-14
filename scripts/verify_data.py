#!/usr/bin/env python
"""
SDA形式のHDF5データを検証するスクリプト

Usage:
    python verify_data.py \
        --data /path/to/train.h5 \
        --output verification_report.png
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(file_path):
    """HDF5ファイルからデータを読み込む"""
    with h5py.File(file_path, 'r') as f:
        data = f['x'][:]
        attrs = dict(f['x'].attrs)
    return data, attrs


def print_statistics(data):
    """データの統計情報を表示"""
    print("\n" + "="*60)
    print("データ統計情報")
    print("="*60)
    print(f"形状: {data.shape}")
    print(f"  - サンプル数: {data.shape[0]}")
    print(f"  - 時系列長: {data.shape[1]}")
    print(f"  - チャネル数: {data.shape[2]}")
    print(f"  - 空間解像度: {data.shape[3]}×{data.shape[4]}")
    print(f"dtype: {data.dtype}")
    print(f"データ範囲: [{data.min():.6f}, {data.max():.6f}]")
    print(f"平均: {data.mean():.6f}")
    print(f"標準偏差: {data.std():.6f}")
    print(f"NaN数: {np.isnan(data).sum()}")
    print(f"Inf数: {np.isinf(data).sum()}")

    # チャネル別統計
    u = data[:, :, 0, :, :]
    v = data[:, :, 1, :, :]

    print(f"\nu速度 (チャネル0):")
    print(f"  範囲: [{u.min():.6f}, {u.max():.6f}]")
    print(f"  平均: {u.mean():.6f}, 標準偏差: {u.std():.6f}")

    print(f"\nv速度 (チャネル1):")
    print(f"  範囲: [{v.min():.6f}, {v.max():.6f}]")
    print(f"  平均: {v.mean():.6f}, 標準偏差: {v.std():.6f}")

    # 速度の大きさ
    velocity_magnitude = np.sqrt(u**2 + v**2)
    print(f"\n速度の大きさ:")
    print(f"  範囲: [{velocity_magnitude.min():.6f}, {velocity_magnitude.max():.6f}]")
    print(f"  平均: {velocity_magnitude.mean():.6f}")

    print("="*60 + "\n")


def check_temporal_continuity(data):
    """時間連続性をチェック"""
    print("時間連続性チェック:")

    # 最初のサンプルの時間差分
    sample0 = data[0]  # (timesteps, channels, H, W)

    time_diffs = []
    for t in range(1, len(sample0)):
        diff = np.abs(sample0[t] - sample0[t-1]).mean()
        time_diffs.append(diff)

    print(f"  平均差分: {np.mean(time_diffs):.6f}")
    print(f"  最大差分: {np.max(time_diffs):.6f}")
    print(f"  最小差分: {np.min(time_diffs):.6f}")

    if np.mean(time_diffs) < 0.1:
        print("  結果: ✓ 時間的に滑らか")
    else:
        print("  結果: ⚠ 大きな時間変化あり")

    return time_diffs


def check_energy_conservation(data):
    """エネルギー保存の簡易チェック"""
    print("\nエネルギーチェック:")

    # 各タイムステップでの運動エネルギー
    u = data[:, :, 0, :, :]
    v = data[:, :, 1, :, :]

    # 運動エネルギー (0.5 * |v|^2)
    kinetic_energy = 0.5 * (u**2 + v**2)

    # 各サンプルの時系列平均エネルギー
    energy_time_series = kinetic_energy.mean(axis=(2, 3))  # (n_samples, timesteps)

    # サンプル0のエネルギー時間変化
    energy_sample0 = energy_time_series[0]

    print(f"  エネルギー範囲: [{energy_sample0.min():.6f}, {energy_sample0.max():.6f}]")
    print(f"  エネルギー平均: {energy_sample0.mean():.6f}")
    print(f"  エネルギー変動: {energy_sample0.std():.6f}")

    # エネルギーの相対変化
    energy_change = (energy_sample0.max() - energy_sample0.min()) / energy_sample0.mean()
    print(f"  相対変化: {energy_change*100:.2f}%")

    if energy_change < 0.3:
        print("  結果: ✓ エネルギーがほぼ保存されている")
    else:
        print("  結果: ⚠ エネルギーが大きく変化している")

    return energy_time_series


def compute_vorticity(u, v):
    """渦度を計算 (∂v/∂x - ∂u/∂y)"""
    # 中心差分で導関数を計算
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    vorticity = dvdx - dudy
    return vorticity


def visualize_samples(data, output_path, n_samples=3):
    """ランダムサンプルを可視化"""
    print(f"\n可視化: {n_samples}サンプルをランダム選択")

    n_total = len(data)
    if n_total < n_samples:
        n_samples = n_total
        print(f"  サンプル数が少ないため{n_samples}サンプルを表示")

    # ランダムにサンプルを選択
    sample_indices = np.random.choice(n_total, size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        sample = data[idx]  # (timesteps, channels, H, W)

        # 最初のタイムステップを表示
        t = 0
        u = sample[t, 0, :, :]
        v = sample[t, 1, :, :]
        vorticity = compute_vorticity(u, v)
        velocity_mag = np.sqrt(u**2 + v**2)

        # u速度
        im0 = axes[i, 0].imshow(u, cmap='RdBu_r', origin='lower')
        axes[i, 0].set_title(f'Sample {idx}, t={t}: u velocity')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0])

        # v速度
        im1 = axes[i, 1].imshow(v, cmap='RdBu_r', origin='lower')
        axes[i, 1].set_title(f'Sample {idx}, t={t}: v velocity')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1])

        # 渦度
        im2 = axes[i, 2].imshow(vorticity, cmap='RdBu_r', origin='lower')
        axes[i, 2].set_title(f'Sample {idx}, t={t}: Vorticity')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2])

        # 速度の大きさ
        im3 = axes[i, 3].imshow(velocity_mag, cmap='viridis', origin='lower')
        axes[i, 3].set_title(f'Sample {idx}, t={t}: |v|')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  保存: {output_path}")


def plot_time_series(data, energy_time_series, output_path):
    """時系列データをプロット"""
    print("\n時系列プロット作成中...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # サンプル0の速度時系列
    sample0 = data[0]
    u_mean = sample0[:, 0, :, :].mean(axis=(1, 2))
    v_mean = sample0[:, 1, :, :].mean(axis=(1, 2))

    # u速度の時間変化
    axes[0, 0].plot(u_mean, label='u (spatial mean)')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('u velocity')
    axes[0, 0].set_title('u velocity time series (Sample 0)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # v速度の時間変化
    axes[0, 1].plot(v_mean, label='v (spatial mean)', color='orange')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('v velocity')
    axes[0, 1].set_title('v velocity time series (Sample 0)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # エネルギーの時間変化
    axes[1, 0].plot(energy_time_series[0], label='Kinetic energy', color='green')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Kinetic energy time series (Sample 0)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 速度の大きさの時間変化
    velocity_mag_mean = np.sqrt(sample0[:, 0, :, :]**2 + sample0[:, 1, :, :]**2).mean(axis=(1, 2))
    axes[1, 1].plot(velocity_mag_mean, label='|v| (spatial mean)', color='red')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('Velocity magnitude')
    axes[1, 1].set_title('Velocity magnitude time series (Sample 0)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()

    # ファイル名を変更
    output_dir = Path(output_path).parent
    output_name = Path(output_path).stem + '_timeseries.png'
    timeseries_path = output_dir / output_name

    plt.savefig(timeseries_path, dpi=150, bbox_inches='tight')
    print(f"  保存: {timeseries_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify SDA HDF5 data"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to HDF5 file (e.g., train.h5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='verification_report.png',
        help='Output path for visualization (default: verification_report.png)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=3,
        help='Number of samples to visualize (default: 3)'
    )

    args = parser.parse_args()

    print(f"データ検証開始: {args.data}")

    # データ読み込み
    data, attrs = load_data(args.data)

    print(f"\nHDF5属性:")
    for key, value in attrs.items():
        print(f"  {key}: {value}")

    # 統計情報
    print_statistics(data)

    # 時間連続性チェック
    time_diffs = check_temporal_continuity(data)

    # エネルギーチェック
    energy_time_series = check_energy_conservation(data)

    # 異常値チェック
    print("\n異常値チェック:")
    has_nan = np.isnan(data).any()
    has_inf = np.isinf(data).any()

    if has_nan:
        print("  ✗ NaN が検出されました")
    else:
        print("  ✓ NaN なし")

    if has_inf:
        print("  ✗ Inf が検出されました")
    else:
        print("  ✓ Inf なし")

    # 可視化
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualize_samples(data, args.output, n_samples=args.n_samples)
    plot_time_series(data, energy_time_series, args.output)

    print("\n" + "="*60)
    print("検証完了")
    print("="*60)


if __name__ == '__main__':
    main()
