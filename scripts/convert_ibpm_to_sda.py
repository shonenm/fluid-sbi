#!/usr/bin/env python
"""
IBPM出力をSDA形式に変換するスクリプト

Usage:
    python convert_ibpm_to_sda.py \
        --input /path/to/ibpm_output_YYYYMMDD_HHMMSS \
        --output /path/to/sda/data \
        --coarsen 4 \
        --window 64 \
        --stride 8
"""

import argparse
import numpy as np
import h5py
import re
from pathlib import Path
from tqdm import tqdm
# from scipy.ndimage import zoom  # 不要（圧縮処理を削除したため）


def parse_tecplot(file_path):
    """
    Tecplotファイルをパース

    Returns:
        data_grid: (I, J, n_vars) の配列
        header: ヘッダー情報の辞書
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # ヘッダー解析
    header = {}
    data_start_idx = 0

    for i, line in enumerate(lines):
        # グリッドサイズの取得
        if 'I=' in line and 'J=' in line:
            match = re.search(r'I=(\d+),\s*J=(\d+)', line)
            if match:
                header['I'] = int(match.group(1))
                header['J'] = int(match.group(2))

        # 変数名の取得
        if 'VARIABLES' in line:
            vars_match = re.findall(r'"([^"]+)"', line)
            header['variables'] = vars_match

        # データ開始位置
        if 'DATAPACKING' in line:
            data_start_idx = i + 1
            break

    # データ読み込み
    data_lines = lines[data_start_idx:]
    data = []

    for line in data_lines:
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                values = list(map(float, line.split()))
                if len(values) == len(header['variables']):
                    data.append(values)
            except ValueError:
                # 変換できない行はスキップ
                continue

    if len(data) == 0:
        raise ValueError(f"No valid data found in {file_path}")

    data = np.array(data)

    # グリッド形状に整形
    I, J = header['I'], header['J']
    expected_points = I * J

    if len(data) != expected_points:
        raise ValueError(
            f"Data length mismatch in {file_path}: "
            f"expected {expected_points} (I={I}, J={J}), got {len(data)}"
        )

    # (I*J, n_vars) → (J, I, n_vars) → (I, J, n_vars)
    data_grid = data.reshape(J, I, -1).transpose(1, 0, 2)

    return data_grid, header


def extract_velocity(data_grid, header):
    """
    速度成分 (u, v) を抽出

    Returns:
        velocity: (2, H, W) の配列
    """
    variables = header['variables']

    # 変数のインデックスを取得
    if 'u' not in variables or 'v' not in variables:
        raise ValueError(
            f"Required variables 'u' and 'v' not found. "
            f"Available: {variables}"
        )

    u_idx = variables.index('u')
    v_idx = variables.index('v')

    # (H, W, n_vars) → (2, H, W)
    u = data_grid[:, :, u_idx]
    v = data_grid[:, :, v_idx]

    velocity = np.stack([u, v], axis=0)

    return velocity


def coarsen(x, r):
    """
    空間解像度を削減 (r×r平均プーリング)

    サイズが割り切れない場合は中央部分をトリミングする

    Args:
        x: (C, H, W) または (H, W) の配列
        r: coarsening factor

    Returns:
        coarsened: (C, H//r, W//r) または (H//r, W//r)
    """
    if x.ndim == 2:
        h, w = x.shape
        # トリミング（中央部分を使用）
        h_new = (h // r) * r
        w_new = (w // r) * r
        h_start = (h - h_new) // 2
        w_start = (w - w_new) // 2
        x = x[h_start:h_start+h_new, w_start:w_start+w_new]

        x = x.reshape(h_new // r, r, w_new // r, r)
        x = x.mean(axis=(1, 3))

    elif x.ndim == 3:
        c, h, w = x.shape
        # トリミング（中央部分を使用）
        h_new = (h // r) * r
        w_new = (w // r) * r
        h_start = (h - h_new) // 2
        w_start = (w - w_new) // 2
        x = x[:, h_start:h_start+h_new, w_start:w_start+w_new]

        x = x.reshape(c, h_new // r, r, w_new // r, r)
        x = x.mean(axis=(2, 4))
    else:
        raise ValueError(f"Expected 2D or 3D array, got {x.ndim}D")

    return x


def resize_to_target(x, target_size=64):
    """
    空間解像度を指定サイズにリサイズ（バイリニア補間）

    Args:
        x: (C, H, W) の配列
        target_size: 目標解像度 (default: 64)

    Returns:
        resized: (C, target_size, target_size) の配列
    """
    c, h, w = x.shape

    # zoom factorを計算
    zoom_h = target_size / h
    zoom_w = target_size / w

    # チャネルごとにリサイズ（order=1 でバイリニア補間）
    resized = zoom(x, (1, zoom_h, zoom_w), order=1)

    return resized


def sliding_window_split(timeseries, window, stride):
    """
    スライディングウィンドウで時系列を分割

    Args:
        timeseries: (n_timesteps, C, H, W) の配列
        window: ウィンドウサイズ
        stride: ストライド

    Returns:
        samples: (n_samples, window, C, H, W) のリスト
    """
    n_timesteps = len(timeseries)

    if n_timesteps < window:
        raise ValueError(
            f"Time series length {n_timesteps} < window size {window}"
        )

    samples = []
    for start in range(0, n_timesteps - window + 1, stride):
        end = start + window
        sample = timeseries[start:end]
        samples.append(sample)

    return samples


def process_ibpm_output(
    input_dir,
    coarsen_factor=1,  # デフォルトを1に変更（圧縮なし）
    window=64,
    stride=8,
):
    """
    IBPM出力ディレクトリを処理（生データをそのまま保存）

    Returns:
        samples: (n_samples, window, 2, H, W) のリスト
        stats: 統計情報の辞書
    """
    input_path = Path(input_dir)

    # .pltファイルを取得
    plt_files = sorted(input_path.glob('ibpm*.plt'))

    if len(plt_files) == 0:
        raise ValueError(f"No .plt files found in {input_dir}")

    print(f"Found {len(plt_files)} timesteps in {input_dir}")

    # タイムステップ数が不足している場合は、windowサイズを調整
    actual_window = min(window, len(plt_files))
    if len(plt_files) < window:
        print(
            f"Warning: Only {len(plt_files)} timesteps available. "
            f"Adjusting window size from {window} to {actual_window}."
        )

    timeseries = []

    for plt_file in tqdm(plt_files, desc="Parsing Tecplot files"):
        try:
            data_grid, header = parse_tecplot(plt_file)
            velocity = extract_velocity(data_grid, header)

            # ❌ 圧縮処理を削除
            # if coarsen_factor > 1:
            #     velocity = coarsen(velocity, coarsen_factor)
            # velocity = resize_to_target(velocity, target_size=64)

            # ✅ 生データをそのまま使用
            timeseries.append(velocity)

        except Exception as e:
            print(f"Error processing {plt_file.name}: {e}")
            raise

    # (n_timesteps, 2, H, W)
    timeseries = np.stack(timeseries, axis=0).astype(np.float32)

    print(f"\nFull timeseries shape: {timeseries.shape}")
    print(f"  - Timesteps: {timeseries.shape[0]}")
    print(f"  - Channels: {timeseries.shape[1]} (u, v)")
    print(f"  - Spatial resolution: {timeseries.shape[2]}×{timeseries.shape[3]} (original IBPM output)")
    print("  - No compression applied ✓")

    # スライディングウィンドウで分割
    if len(plt_files) >= actual_window:
        print(f"Splitting with window={actual_window}, stride={stride}")
        samples = sliding_window_split(timeseries, actual_window, stride)
        print(f"Generated {len(samples)} samples")
    else:
        print(f"Creating single sample with {len(plt_files)} timesteps")
        samples = [timeseries]

    # 統計情報
    stats = {
        'n_samples': len(samples),
        'n_timesteps_total': len(timeseries),
        'shape': samples[0].shape if samples else None,
        'min': timeseries.min(),
        'max': timeseries.max(),
        'mean': timeseries.mean(),
        'std': timeseries.std(),
        'nan_count': np.isnan(timeseries).sum(),
    }

    return samples, stats


def create_sda_dataset(
    samples,
    output_dir,
    train_ratio=0.7,
    valid_ratio=0.15,
):
    """
    SDA形式のHDF5データセットを作成

    Args:
        samples: (n_samples, window, 2, H, W) のサンプルリスト
        output_dir: 出力ディレクトリ
        train_ratio: 訓練データの割合
        valid_ratio: 検証データの割合
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = len(samples)
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    splits = {
        'train': samples[:n_train],
        'valid': samples[n_train:n_train+n_valid],
        'test': samples[n_train+n_valid:],
    }

    for split_name, data_list in splits.items():
        if len(data_list) == 0:
            print(f"Warning: {split_name} split is empty")
            continue

        output_file = output_path / f'{split_name}.h5'

        # データをスタック
        data = np.stack(data_list, axis=0)

        with h5py.File(output_file, 'w') as f:
            dset = f.create_dataset(
                'x',
                data=data,
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
            )

            # 属性を追加
            dset.attrs['description'] = 'IBPM velocity field data (u, v)'
            dset.attrs['shape_description'] = '(n_samples, n_timesteps, n_channels, height, width)'
            dset.attrs['channels'] = 'u, v'

            print(f"{split_name}: {dset.shape} -> {output_file}")


def print_statistics(stats):
    """統計情報を表示"""
    print("\n" + "="*60)
    print("Data Statistics")
    print("="*60)
    print(f"Number of samples: {stats['n_samples']}")
    print(f"Total timesteps: {stats['n_timesteps_total']}")
    print(f"Sample shape: {stats['shape']}")
    print(f"Data range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"Mean: {stats['mean']:.6f}")
    print(f"Std: {stats['std']:.6f}")
    print(f"NaN count: {stats['nan_count']}")

    if stats['nan_count'] > 0:
        print("\nWARNING: Data contains NaN values!")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert IBPM output to SDA HDF5 format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to IBPM output directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for HDF5 files'
    )
    parser.add_argument(
        '--coarsen',
        type=int,
        default=1,
        help='Coarsening factor (default: 1, no coarsening)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=64,
        help='Time window size (default: 64)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=8,
        help='Sliding window stride (default: 8)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training split ratio (default: 0.7)'
    )
    parser.add_argument(
        '--valid-ratio',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )

    args = parser.parse_args()

    # IBPM出力を処理
    print(f"Processing IBPM output from: {args.input}")
    samples, stats = process_ibpm_output(
        args.input,
        coarsen_factor=args.coarsen,
        window=args.window,
        stride=args.stride,
    )

    # 統計情報を表示
    print_statistics(stats)

    # HDF5データセットの作成
    print(f"Creating SDA dataset in: {args.output}")
    create_sda_dataset(
        samples,
        args.output,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    print("\n✓ Conversion completed successfully!")


if __name__ == '__main__':
    main()
