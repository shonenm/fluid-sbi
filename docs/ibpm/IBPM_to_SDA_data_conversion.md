# IBPMデータからSDAデータへの変換ガイド

## 目次
1. [概要](#概要)
2. [結論: 変換可能性の確認](#結論-変換可能性の確認)
3. [データ型の比較](#データ型の比較)
4. [差分の詳細分析](#差分の詳細分析)
5. [変換処理の実装](#変換処理の実装)
6. [実装例](#実装例)
7. [トラブルシューティング](#トラブルシューティング)
8. [ディレクトリ構成](#ディレクトリ構成)
9. [推奨ワークフロー](#推奨ワークフロー)

---

## 概要

このドキュメントでは、**IBPM (Immersed Boundary Projection Method)** で生成された流体シミュレーションデータを、**SDA (Score-based Diffusion for Autoregression)** の学習・推論に使用できる形式に変換する方法を説明します。

### 変換の必要性
- IBPMは流体力学シミュレーション用のC++ライブラリで、Tecplot形式(.plt)とバイナリ形式(.bin)でデータを出力
- SDAはPyTorchベースの機械学習フレームワークで、HDF5形式(.h5)のNumPy配列を入力として期待
- データ形式、次元、正規化が異なるため、明示的な変換処理が必要

---

## 結論: 変換可能性の確認

### ✅ IBPMデータはSDA形式に変換可能

IBPMの出力形式とSDAの期待形式には**構造・解像度・スケールの違い**がありますが、本ドキュメントで提示する変換パイプライン（ASCIIパース → 速度抽出 → Coarsening → 時系列化 → HDF5保存）により、**SDAモデルがそのまま学習できる形式に正しく変換可能**です。

### 🧩 データ形式の対応関係（サマリー）

| 項目 | IBPM | SDA (Kolmogorov Flow) | 変換内容 |
|------|------|----------------------|----------|
| **ファイル形式** | ASCII Tecplot (.plt) / バイナリ (.bin) | HDF5 (.h5) | .pltをNumPyに読み込み → HDF5へ書き出し |
| **出力構造** | 各タイムステップごとに個別ファイル<br>(ibpm00000.plt, ibpm00100.plt, ...) | 1ファイルに全時系列を格納<br>(train.h5 など) | .plt群を時系列方向に結合 |
| **変数** | x, y, u, v, vorticity（計5変数） | u, v（2チャネル） | u, vのみ抽出 (`extract_velocity`) |
| **グリッド構造** | 明示的座標格子 (x, y) | 暗黙的格子インデックス (i, j) | 座標情報を削除し、配列として格納 |
| **解像度** | 任意（例: 199×199, 256×256） | 固定（64×64推奨） | Coarsening（平均プーリング）で縮小 |
| **時間構造** | 各ファイルが1ステップ | 連続時系列配列 (n_timesteps, ...) | ファイルを時間順にスタック |
| **数値精度** | float64（物理単位） | float32（正規化推奨） | dtype変換 + 正規化（推奨） |
| **サンプル単位** | 1つのシミュレーション全体 | 複数サンプル（訓練/検証/テスト） | スライディングウィンドウ等で分割 |
| **保存構造** | 複数ファイル出力 | `file['x']` → (n_samples, n_timesteps, 2, H, W) | HDF5として保存 (`create_hdf5_dataset`) |

### 🔁 推奨変換フロー（概要）

```
IBPM出力 (.plt)
   ↓
[1] parse_tecplot(): ASCII → NumPy配列 (I, J, [x,y,u,v,vort])
   ↓
[2] extract_velocity(): u,v の2チャンネル抽出 → (2,H,W)
   ↓
[3] coarsen(): 平均プーリングで 256×256 → 64×64
   ↓
[4] aggregate_timeseries(): ibpm*.plt 群を時系列に結合 (n_t, 2, 64, 64)
   ↓
[5] create_hdf5_dataset(): HDF5に保存 (train/valid/test)
   ↓
SDA入力形式完成 → (n_samples, n_timesteps, 2, 64, 64)
```

### ⚙️ SDA学習への適合性

#### ✅ 形式互換
SDAの`TrajectoryDataset`クラスが期待する形式に完全一致:
```python
# SDAが期待する形状
(n_samples, window, n_channels, height, width)
# 例: (819, 64, 2, 64, 64)
```

#### ✅ 内容互換
Kolmogorov Flowと同じデータ特性:
- 速度場(u, v)のみ
- float32型
- 低解像度（64×64）
- 正規化済み（推奨）

#### ✅ 学習可能性
- **短時間確認**: 最低限の学習確認が可能（1サンプル、window=64）
- **本格学習**: 複数サンプル化が必要
  - 複数のIBPM実験を実行
  - または単一実験からスライディングウィンドウで複数サンプル生成
  - 推奨: 最低100サンプル以上

### ⚠️ 注意点

1. **サンプル数の不足**: 単一IBPM実験（1サンプル）では過学習のリスクあり
2. **時系列長の確保**: 最低でもwindow分（64ステップ）のタイムステップが必要
3. **正規化の重要性**: 物理単位のまま学習すると収束しない可能性

---

## データ型の比較

### IBPMの出力データ

IBPMは`ibpm_output_YYYYMMDD_HHMMSS/`ディレクトリに以下のファイルを生成します:

#### 1. **ibpm.force** (揚力・抗力データ)
- **形式**: ASCII テキスト
- **内容**: 時系列の力学データ
- **構造**:
  ```
  列0: タイムステップ番号 (整数)
  列1: 時刻 (浮動小数点)
  列2: x方向の力 (Fx)
  列3: y方向の力 (Fy)
  ```
- **データ型**: `float64`
- **サイズ例**: `(n_timesteps, 4)` where n_timesteps ≈ 100-500

**サンプルデータ**:
```
    0 0.00000e+00 0.00000e+00 0.00000e+00
    1 2.00000e-02 6.96251e+00 1.39636e-14
    2 4.00000e-02 4.41590e+00 -3.08091e-15
```

#### 2. **ibpmXXXXX.plt** (Tecplot可視化ファイル)
- **形式**: ASCII Tecplot
- **内容**: 空間グリッド上の流体物理量
- **変数**:
  - `x, y`: 座標 (空間位置)
  - `u, v`: 速度ベクトル成分 (x, y方向)
  - `Vorticity`: 渦度 (∂v/∂x - ∂u/∂y)
- **グリッド構造**: `ZONETYPE=Ordered`, `I × J` 格子点
- **データ型**: ASCII形式の`float32`/`float64`
- **サイズ例**: `(199, 199, 5)` → 199×199グリッド、5変数

**ヘッダー構造**:
```
TITLE = "Test run, step00000"
VARIABLES = "x" "y" "u" "v" "Vorticity"
ZONE T="Rectangular zone"
I=199, J=199, K=1, ZONETYPE=Ordered
DATAPACKING=POINT
```

**データ行の例**:
```
-1.98000e+00 -1.98000e+00 1.00000e+00 0.00000e+00 0.00000e+00
-1.96000e+00 -1.98000e+00 1.00000e+00 0.00000e+00 0.00000e+00
```
各行: `x y u v vorticity`

#### 3. **ibpmXXXXX.bin** (バイナリリスタートファイル)
- **形式**: IBPM固有のバイナリ形式
- **内容**: 計算の完全な状態（速度場、圧力、境界力など）
- **用途**: 計算の再開、詳細解析
- **データ型**: バイナリ (読み取りにIBPMライブラリが必要)
- **サイズ例**: 約1MB per timestep

---

### SDAの期待データ形式

SDAは **HDF5形式** のデータセットを読み込みます。

#### データ構造

**ファイル**: `train.h5`, `valid.h5`, `test.h5`

**HDF5データセット構造**:
```python
file['x']: numpy.ndarray
    shape: (n_samples, n_timesteps, n_channels, height, width)
    dtype: float32
```

#### 具体例: Kolmogorov Flow

**Kolmogorov実験のデータ形状**:
```python
# generate.py:46-49
dset = f.create_dataset(
    'x',
    shape=(len(files), 64, 2, 64, 64),
    dtype=np.float32,
)
```

- **n_samples**: サンプル数 (例: 819 for train, 102 for valid, 103 for test)
- **n_timesteps**: 時系列長 (例: 64)
- **n_channels**: チャンネル数 (例: 2 → u, vの2成分)
- **height, width**: 空間解像度 (例: 64×64)

**実際の形状**: `(819, 64, 2, 64, 64)` → 約64×64×2×64×819 ≈ 326MB (float32)

#### データの生成フロー

SDAの`experiments/kolmogorov/generate.py`では以下の処理を行います:

```python
# 1. シミュレーション実行
x = chain.trajectory(x, length=128)  # 128ステップのtrajectory生成
x = x[64:]  # 後半64ステップを使用 (ウォームアップ削除)

# 2. Coarsen処理 (解像度削減)
arr = KolmogorovFlow.coarsen(torch.from_numpy(x), 4)  # 256x256 → 64x64

# 3. 型変換とHDF5保存
arr = arr.detach().cpu().numpy().astype(np.float32)
dset[i, ...] = arr  # shape: (64, 2, 64, 64)
```

---

## 差分の詳細分析

### 1. **ファイル形式の違い**

| 項目 | IBPM | SDA |
|------|------|-----|
| フォーマット | ASCII Tecplot (.plt) / バイナリ (.bin) | HDF5 (.h5) |
| データ構造 | フラット (x, y, u, v, vort) | 多次元配列 (samples, time, channels, H, W) |
| メタデータ | ヘッダーにタイトル、変数名 | HDF5 attributes |
| 圧縮 | なし (ASCII) | HDF5圧縮可能 |

### 2. **空間次元の違い**

| 項目 | IBPM | SDA |
|------|------|-----|
| 座標系 | (x, y) の物理座標 | 暗黙的なグリッドインデックス (i, j) |
| グリッド形式 | `DATAPACKING=POINT` (各点にすべての変数) | チャネル分離 `(C, H, W)` |
| 格子点数 | 任意 (例: 199×199) | 固定 (例: 64×64) |
| 境界条件 | 多様 (遠方場、周期境界など) | 周期境界 (circular padding) |

### 3. **変数の違い**

| 項目 | IBPM | SDA (Kolmogorov) |
|------|------|------------------|
| 変数 | x, y, u, v, vorticity | u, v (速度成分のみ) |
| チャネル数 | 5 (座標2 + 速度2 + 渦度1) | 2 (速度成分のみ) |
| 渦度の扱い | 直接出力 | `KolmogorovFlow.vorticity()`で計算 |

### 4. **時系列構造の違い**

| 項目 | IBPM | SDA |
|------|------|-----|
| 時系列保存 | 個別ファイル (ibpm00000.plt, ibpm00100.plt, ...) | 単一HDF5内に連続配列 |
| タイムステップ情報 | ファイル名とibpm.force | 配列のインデックス |
| サンプリング間隔 | 設定可能 (`-dt`, `-nsteps`) | 固定 (generate.pyで制御) |

### 5. **データ型と精度**

| 項目 | IBPM | SDA |
|------|------|-----|
| 数値型 | float64 (倍精度) | float32 (単精度) |
| 正規化 | なし (物理単位そのまま) | 必要に応じて正規化 |
| スケール | O(1-10) 程度の物理量 | O(-1 ~ 1) or standardized |

### 6. **解像度とCoarsening**

IBPMは高解像度シミュレーション (例: 256×256) を実行しますが、SDAの学習では計算コスト削減のため低解像度 (64×64) にダウンサンプリングします。

**Coarseningの実装**:
```python
# sda/mcs.py:340-347
@staticmethod
def coarsen(x: Tensor, r: int = 2) -> Tensor:
    *batch, h, w = x.shape
    x = x.reshape(*batch, h // r, r, w // r, r)
    x = x.mean(dim=(-3, -1))  # r×rブロックの平均を取る
    return x
```

---

## 変換処理の実装

### アーキテクチャ

```
IBPM出力
    ↓
[1] Tecplotファイルのパース
    ↓
[2] グリッドデータの抽出
    ↓
[3] 座標系から配列へ変換
    ↓
[4] チャネル分離 (u, v)
    ↓
[5] 解像度調整 (Coarsening)
    ↓
[6] 時系列の集約
    ↓
[7] 正規化 (オプション)
    ↓
[8] HDF5形式で保存
    ↓
SDA入力データ
```

### 必要な処理ステップ

#### ステップ1: Tecplotファイルのパース

**課題**: ASCII Tecplotファイルから数値データを抽出

**解決策**:
```python
import numpy as np
import re

def parse_tecplot(file_path):
    """Tecplot .pltファイルをパースする"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # ヘッダーのパース
    header = {}
    for line in lines[:10]:
        if 'I=' in line:
            match = re.search(r'I=(\d+),\s*J=(\d+)', line)
            if match:
                header['I'] = int(match.group(1))
                header['J'] = int(match.group(2))
        elif 'VARIABLES' in line:
            # VARIABLES = "x" "y" "u" "v" "Vorticity"
            vars_match = re.findall(r'"([^"]+)"', line)
            header['variables'] = vars_match

    # データ部分の抽出 (ヘッダー後から)
    data_start = 0
    for i, line in enumerate(lines):
        if 'DATAPACKING' in line:
            data_start = i + 1
            break

    # 数値データの読み込み
    data_lines = lines[data_start:]
    data = []
    for line in data_lines:
        if line.strip():
            values = list(map(float, line.split()))
            data.append(values)

    data = np.array(data)

    # グリッド形状に整形
    I, J = header['I'], header['J']
    assert len(data) == I * J, f"Data length mismatch: {len(data)} vs {I*J}"

    # (I*J, n_vars) → (J, I, n_vars) → (I, J, n_vars)
    data_grid = data.reshape(J, I, -1).transpose(1, 0, 2)

    return data_grid, header
```

#### ステップ2: 速度成分の抽出とチャネル分離

**課題**: (H, W, 5) → (2, H, W)

```python
def extract_velocity(data_grid, header):
    """速度成分 (u, v) を抽出"""
    variables = header['variables']

    # 変数のインデックスを取得
    u_idx = variables.index('u')
    v_idx = variables.index('v')

    # (H, W, n_vars) → (H, W, 2) → (2, H, W)
    u = data_grid[:, :, u_idx]
    v = data_grid[:, :, v_idx]

    velocity = np.stack([u, v], axis=0)  # (2, H, W)

    return velocity
```

#### ステップ3: 解像度調整 (Coarsening)

**課題**: 256×256 → 64×64

```python
import torch

def coarsen_numpy(x, r=4):
    """NumPy配列をCoarsenする (r×rブロックの平均)"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    *batch, h, w = x.shape
    assert h % r == 0 and w % r == 0, f"Size {h}x{w} not divisible by {r}"

    x = x.reshape(*batch, h // r, r, w // r, r)
    x = x.mean(dim=(-3, -1))

    return x.numpy()
```

#### ステップ4: 時系列データの集約

**課題**: 複数の.pltファイル → 単一の時系列配列

```python
from pathlib import Path
import h5py

def aggregate_timeseries(output_dir, coarsen_factor=4):
    """IBPMの出力ディレクトリから時系列データを集約"""
    output_path = Path(output_dir)

    # .pltファイルを時刻順にソート
    plt_files = sorted(output_path.glob('ibpm*.plt'))

    timeseries = []

    for plt_file in plt_files:
        # パース
        data_grid, header = parse_tecplot(plt_file)

        # 速度成分を抽出 (2, H, W)
        velocity = extract_velocity(data_grid, header)

        # Coarsen
        velocity_coarse = coarsen_numpy(velocity, r=coarsen_factor)

        timeseries.append(velocity_coarse)

    # (n_timesteps, 2, H, W)
    timeseries = np.stack(timeseries, axis=0).astype(np.float32)

    return timeseries
```

#### ステップ5: HDF5ファイルの作成

**課題**: SDA形式のHDF5データセット作成

```python
def create_hdf5_dataset(
    timeseries_list,
    output_file,
    train_ratio=0.8,
    valid_ratio=0.1
):
    """複数の時系列からHDF5データセットを作成"""

    n_samples = len(timeseries_list)
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    splits = {
        'train': timeseries_list[:n_train],
        'valid': timeseries_list[n_train:n_train+n_valid],
        'test': timeseries_list[n_train+n_valid:],
    }

    for split_name, data_list in splits.items():
        with h5py.File(f'{output_file}_{split_name}.h5', 'w') as f:
            # データセット作成
            if len(data_list) > 0:
                sample_shape = data_list[0].shape
                full_shape = (len(data_list),) + sample_shape

                dset = f.create_dataset(
                    'x',
                    shape=full_shape,
                    dtype=np.float32,
                    compression='gzip',  # オプション: 圧縮
                )

                for i, data in enumerate(data_list):
                    dset[i] = data

                print(f"{split_name}: {dset.shape}")
```

---

## 実装例

### 完全な変換スクリプト

```python
#!/usr/bin/env python
"""
IBPM出力をSDA形式に変換するスクリプト

Usage:
    python convert_ibpm_to_sda.py \
        --input /path/to/ibpm_output_20251014_123123 \
        --output /path/to/sda/data \
        --coarsen 4 \
        --window 64
"""

import argparse
import numpy as np
import h5py
import torch
import re
from pathlib import Path
from tqdm import tqdm


def parse_tecplot(file_path):
    """Tecplotファイルをパース"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # ヘッダー解析
    header = {}
    data_start_idx = 0

    for i, line in enumerate(lines):
        if 'I=' in line and 'J=' in line:
            match = re.search(r'I=(\d+),\s*J=(\d+)', line)
            if match:
                header['I'] = int(match.group(1))
                header['J'] = int(match.group(2))

        if 'VARIABLES' in line:
            vars_match = re.findall(r'"([^"]+)"', line)
            header['variables'] = vars_match

        if 'DATAPACKING' in line:
            data_start_idx = i + 1
            break

    # データ読み込み
    data_lines = lines[data_start_idx:]
    data = []
    for line in data_lines:
        if line.strip():
            values = list(map(float, line.split()))
            if len(values) == len(header['variables']):
                data.append(values)

    data = np.array(data)

    # グリッド形状に整形
    I, J = header['I'], header['J']
    data_grid = data.reshape(J, I, -1).transpose(1, 0, 2)

    return data_grid, header


def extract_velocity(data_grid, header):
    """速度場 (u, v) を抽出"""
    variables = header['variables']

    u_idx = variables.index('u')
    v_idx = variables.index('v')

    u = data_grid[:, :, u_idx]
    v = data_grid[:, :, v_idx]

    velocity = np.stack([u, v], axis=0)  # (2, H, W)

    return velocity


def coarsen(x, r):
    """空間解像度を削減 (r×r平均)"""
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x

    *batch, h, w = x.shape
    x = x.reshape(*batch, h // r, r, w // r, r)
    x = x.mean(dim=(-3, -1))

    return x.numpy()


def process_ibpm_output(
    input_dir,
    coarsen_factor=4,
    window=64,
):
    """IBPM出力ディレクトリを処理"""
    input_path = Path(input_dir)

    # .pltファイルを取得
    plt_files = sorted(input_path.glob('ibpm*.plt'))

    print(f"Found {len(plt_files)} timesteps in {input_dir}")

    if len(plt_files) < window:
        raise ValueError(
            f"Not enough timesteps: {len(plt_files)} < {window}"
        )

    timeseries = []

    for plt_file in tqdm(plt_files, desc="Parsing"):
        data_grid, header = parse_tecplot(plt_file)
        velocity = extract_velocity(data_grid, header)

        if coarsen_factor > 1:
            velocity = coarsen(velocity, coarsen_factor)

        timeseries.append(velocity)

    # (n_timesteps, 2, H, W)
    timeseries = np.stack(timeseries, axis=0).astype(np.float32)

    # ウィンドウサイズに切り出し
    if len(timeseries) > window:
        # 後半を使用 (初期過渡を除く)
        timeseries = timeseries[-window:]

    return timeseries


def create_sda_dataset(
    timeseries_list,
    output_dir,
    train_ratio=0.8,
    valid_ratio=0.1,
):
    """SDA形式のHDF5データセットを作成"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = len(timeseries_list)
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    splits = {
        'train': timeseries_list[:n_train],
        'valid': timeseries_list[n_train:n_train+n_valid],
        'test': timeseries_list[n_train+n_valid:],
    }

    for split_name, data_list in splits.items():
        if len(data_list) == 0:
            print(f"Warning: {split_name} split is empty")
            continue

        output_file = output_path / f'{split_name}.h5'

        with h5py.File(output_file, 'w') as f:
            sample_shape = data_list[0].shape
            full_shape = (len(data_list),) + sample_shape

            dset = f.create_dataset(
                'x',
                shape=full_shape,
                dtype=np.float32,
                compression='gzip',
                compression_opts=4,
            )

            for i, data in enumerate(data_list):
                dset[i] = data

            print(f"{split_name}: {dset.shape} -> {output_file}")


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
        default=4,
        help='Coarsening factor (default: 4)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=64,
        help='Time window size (default: 64)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--valid-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )

    args = parser.parse_args()

    # IBPM出力を処理
    print(f"Processing IBPM output from: {args.input}")
    timeseries = process_ibpm_output(
        args.input,
        coarsen_factor=args.coarsen,
        window=args.window,
    )

    print(f"Timeseries shape: {timeseries.shape}")

    # 単一サンプルとしてデータセットを作成
    # 複数のIBPM実験がある場合は、ここでリストに追加
    timeseries_list = [timeseries]

    # HDF5データセットの作成
    print(f"Creating SDA dataset in: {args.output}")
    create_sda_dataset(
        timeseries_list,
        args.output,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    print("Conversion completed successfully!")


if __name__ == '__main__':
    main()
```

### 使用例

```bash
# 基本的な使用法
python convert_ibpm_to_sda.py \
    --input /workspace/ibpm_output_20251014_123123 \
    --output /workspace/sda/experiments/ibpm/data \
    --coarsen 4 \
    --window 64

# 複数の実験を処理
for dir in ibpm_output_*/; do
    python convert_ibpm_to_sda.py \
        --input "$dir" \
        --output /workspace/sda/experiments/ibpm/data \
        --coarsen 4 \
        --window 64
done
```

### 変換後の確認

```python
import h5py
import numpy as np

# HDF5ファイルを確認
with h5py.File('train.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Shape:", f['x'].shape)
    print("Dtype:", f['x'].dtype)
    print("Min/Max:", f['x'][:].min(), f['x'][:].max())

    # サンプルを可視化
    sample = f['x'][0]  # (n_timesteps, 2, H, W)
    print(f"Sample shape: {sample.shape}")
```

---

## トラブルシューティング

### 問題1: パース時のエラー

**症状**:
```
IndexError: list index out of range
```

**原因**: Tecplotファイルのフォーマットが想定と異なる

**解決策**:
- ヘッダー行数を確認
- `DATAPACKING`の位置を調整
- デバッグ出力で`header`と`data_start_idx`を確認

```python
# デバッグ用
print(f"Header: {header}")
print(f"Data start: {data_start_idx}")
print(f"First data line: {lines[data_start_idx]}")
```

### 問題2: 形状の不一致

**症状**:
```
AssertionError: Data length mismatch: 39601 vs 39600
```

**原因**: グリッド点数の計算ミス、または空行の混入

**解決策**:
- 空行をスキップする処理を追加
- `I × J`の積を確認

```python
# 修正例
for line in data_lines:
    if line.strip() and not line.startswith('#'):  # コメント行も除外
        values = list(map(float, line.split()))
        if len(values) == len(header['variables']):  # 正しい列数のみ
            data.append(values)
```

### 問題3: メモリ不足

**症状**:
```
MemoryError: Unable to allocate array
```

**原因**: 高解像度データを一度にメモリに読み込んでいる

**解決策**:
- Coarseningを早めに適用
- チャンク単位で処理
- HDF5の圧縮を有効化

```python
# メモリ効率的な処理
def process_in_chunks(plt_files, chunk_size=10):
    for i in range(0, len(plt_files), chunk_size):
        chunk = plt_files[i:i+chunk_size]
        yield process_chunk(chunk)
```

### 問題4: 正規化の必要性

**症状**: 学習が収束しない、loss が NaN

**原因**: 速度場のスケールがSDAの期待範囲外

**解決策**: データを正規化

```python
def normalize_velocity(velocity, method='standardize'):
    """速度場を正規化"""
    if method == 'standardize':
        # 平均0、標準偏差1
        mean = velocity.mean()
        std = velocity.std()
        return (velocity - mean) / (std + 1e-8)

    elif method == 'minmax':
        # [0, 1] にスケーリング
        vmin = velocity.min()
        vmax = velocity.max()
        return (velocity - vmin) / (vmax - vmin + 1e-8)

    elif method == 'clip':
        # [-3, 3] にクリップ
        return np.clip(velocity, -3, 3) / 3
```

### 問題5: 時系列長の不足

**症状**:
```
ValueError: Not enough timesteps: 50 < 64
```

**原因**: IBPM実行時のタイムステップ数が不足

**解決策**:
- IBPMの`-nsteps`オプションを増やす
- または`window`パラメータを減らす

```bash
# より長い時系列を生成
ibpm -geom cylinder.geom -nsteps 500 -dt 0.01
```

---

## ディレクトリ構成

### 推奨ディレクトリ構成

IBPMデータからSDAへの変換を行う際には、以下のディレクトリ構成を推奨します。この構成により、**データの責任分離**、**スケーラビリティ**、**Kolmogorov実験との一貫性**が確保されます。

```
/workspace/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
│
├── docs/                           # ドキュメント
│   ├── IBPM.md
│   ├── IBPM_SDA_implementation_tasks.md
│   ├── IBPM_to_SDA_data_conversion.md
│   └── ...
│
├── data/                           # すべての生データ
│   ├── ibpm/                       # IBPM生データ（Tecplot等）
│   │   └── ibpm_output_20251014_123123/
│   │       ├── ibpm.cmd
│   │       ├── ibpm.force
│   │       ├── ibpm00000.plt
│   │       ├── ibpm00000.bin
│   │       └── ...
│   └── kolmogorov_flow/            # Kolmogorov生データ
│       ├── x_000000.npy
│       └── ...
│
├── scripts/                        # ユーティリティスクリプト
│   ├── convert_ibpm_to_sda.py     # データ変換（IBPMからSDA形式へ）
│   ├── verify_data.py              # データ検証
│   ├── setup.sh
│   └── slurm/
│       └── ...
│
├── sda/                            # SDAパッケージ本体
│   ├── README.md
│   ├── setup.py
│   ├── pyproject.toml
│   ├── environment.yml
│   │
│   ├── sda/                        # SDAコアライブラリ
│   │   ├── __init__.py
│   │   ├── mcs.py
│   │   ├── nn.py
│   │   ├── score.py
│   │   └── utils.py
│   │
│   └── experiments/                # 実験ディレクトリ
│       ├── __init__.py
│       │
│       ├── kolmogorov/             # Kolmogorov Flow実験
│       │   ├── __init__.py
│       │   ├── utils.py
│       │   ├── generate.py
│       │   ├── train.py
│       │   ├── eval.py
│       │   └── data/               # Kolmogorov変換済みデータ（HDF5）
│       │       ├── train.h5
│       │       ├── valid.h5
│       │       └── test.h5
│       │
│       ├── lorenz/                 # Lorenz実験
│       │   ├── __init__.py
│       │   ├── utils.py
│       │   ├── generate.py
│       │   ├── train.py
│       │   └── data/
│       │
│       └── ibpm/                   # IBPM実験（新規）★
│           ├── __init__.py
│           ├── utils.py            # IBPM用スコアモデル定義（必要時）
│           ├── train.py            # IBPM用学習スクリプト（必要時）
│           ├── README.md           # IBPM実験の説明
│           └── data/               # IBPM変換済みデータ（HDF5）★
│               ├── train.h5
│               ├── valid.h5
│               └── test.h5
│
├── runs/                           # 学習結果（チェックポイント等）
│   ├── northern-forest-6_x70kk1jw/
│   └── ...
│
├── results/                        # 実験結果（画像、ログ等）
│   ├── imgs/
│   └── slurm/
│
├── wandb/                          # Weights & Biases ログ
│   └── ...
│
└── infra/                          # インフラ設定
    └── slurm/
        ├── bin/
        └── config/
```

### ディレクトリ構成のメリット

#### 1. **明確な責任分離**
- `data/`: 生データ（変更不可）
- `scripts/`: データ処理・変換ツール
- `sda/experiments/*/data/`: 実験用の変換済みデータ
- `runs/`: 学習結果
- `results/`: 最終成果物

#### 2. **スケーラビリティ**
- 新しい実験（例: `experiments/cylinder/`）を追加しやすい
- 複数のIBPM実験を管理しやすい

#### 3. **一貫性**
- Kolmogorov実験と同じ構造
- SDAパッケージの規約に従う

#### 4. **移植性**
- スクリプトが`scripts/`にまとまっている
- 相対パスではなく明示的なパスを使用

### ディレクトリ構成のセットアップ

プロジェクトのセットアップ時に、以下のコマンドを実行してディレクトリ構造を整備します:

```bash
# 1. IBPM実験ディレクトリの作成（必須）
mkdir -p /workspace/sda/experiments/ibpm/data
touch /workspace/sda/experiments/ibpm/__init__.py

# 2. スクリプトディレクトリの作成とファイル配置（推奨）
mkdir -p /workspace/scripts

# 3. IBPMデータディレクトリの作成（推奨）
mkdir -p /workspace/data/ibpm
```

### パス指定の例

この構成により、コマンド実行時のパス指定は以下のようになります:

```bash
# 変換スクリプトの実行例
python scripts/convert_ibpm_to_sda.py \
  --input /workspace/data/ibpm/ibpm_output_20251014_123123 \
  --output /workspace/sda/experiments/ibpm/data

# データ検証スクリプトの実行例
python scripts/verify_data.py \
  --data /workspace/sda/experiments/ibpm/data/train.h5
```

---

## 推奨ワークフロー

### 📋 ステップバイステップガイド

#### フェーズ1: IBPM実験の準備と実行

```bash
# 1. IBPMで複数の実験を実行（異なる初期条件やパラメータ）
cd /workspace/ibpm/examples

# 実験1: 標準設定
../build/ibpm -geom cylinder.geom -Re 100 -nsteps 300 -dt 0.01

# 実験2: レイノルズ数を変更
../build/ibpm -geom cylinder.geom -Re 150 -nsteps 300 -dt 0.01

# 実験3: より長時間
../build/ibpm -geom cylinder.geom -Re 100 -nsteps 500 -dt 0.01

# 出力を整理
mkdir -p /workspace/ibpm_experiments
mv ibpm_output_* /workspace/ibpm_experiments/
```

**推奨設定**:
- `-nsteps`: 最低200以上（window=64を確保し、初期過渡を除外するため）
- `-dt`: 0.01 ~ 0.02（時間解像度と計算コストのバランス）
- `-Re`: 100 ~ 200（物理的に興味深い範囲）

#### フェーズ2: データ変換

```bash
# 2. 変換スクリプトを実行（単一実験の場合）
python scripts/convert_ibpm_to_sda.py \
    --input /workspace/data/ibpm/ibpm_output_20251014_123123 \
    --output /workspace/sda/experiments/ibpm/data \
    --coarsen 4 \
    --window 64 \
    --stride 8

# 3. 複数実験を一括変換（複数実験がある場合）
for dir in /workspace/data/ibpm/ibpm_output_*/; do
    python scripts/convert_ibpm_to_sda.py \
        --input "$dir" \
        --output /workspace/sda/experiments/ibpm/data \
        --coarsen 4 \
        --window 64 \
        --stride 8
done
```

**推奨パラメータ**:
- `--coarsen 4`: 256×256 → 64×64（Kolmogorovと同じ）
- `--window 64`: SDAの標準時系列長
- `--stride 8`: スライディングウィンドウのストライド（小さいほどサンプル数増加）
- `--train-ratio 0.7`: 70% 訓練、15% 検証、15% テスト

#### フェーズ3: データ検証

```python
# 4. HDF5ファイルの確認
import h5py
import numpy as np
import matplotlib.pyplot as plt

# データ読み込み
with h5py.File('/workspace/sda/experiments/ibpm/data/train.h5', 'r') as f:
    data = f['x'][:]
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Mean: {data.mean():.3f}, Std: {data.std():.3f}")

    # サンプル可視化
    sample = data[0, 0]  # 最初のサンプルの最初のタイムステップ
    u, v = sample[0], sample[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(u, cmap='RdBu_r')
    axes[0].set_title('u velocity')
    axes[1].imshow(v, cmap='RdBu_r')
    axes[1].set_title('v velocity')

    # 渦度を計算
    vorticity = np.gradient(v, axis=0) - np.gradient(u, axis=1)
    axes[2].imshow(vorticity, cmap='RdBu_r')
    axes[2].set_title('Vorticity')

    plt.savefig('ibpm_data_check.png')
    print("Saved visualization to ibpm_data_check.png")
```

#### フェーズ4: SDA学習スクリプトの作成

```bash
# 5. IBPM実験用のディレクトリを作成
mkdir -p sda/experiments/ibpm
cd sda/experiments/ibpm
```

```python
# 6. utils.py を作成
# sda/experiments/ibpm/utils.py
import os
from pathlib import Path
from sda.mcs import *
from sda.score import *
from sda.utils import *

if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/ibpm'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def make_score(
    window: int = 5,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (96, 192, 384),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    score = MCScoreNet(2, order=window // 2)
    score.kernel = ScoreUNet(
        channels=window * 2,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        padding_mode='circular',  # 周期境界条件
    )
    return score


def load_score(file: Path, device: str = 'cpu', **kwargs) -> nn.Module:
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)
    config.update(kwargs)
    score = make_score(**config)
    score.load_state_dict(state)
    return score
```

```python
# 7. train.py を作成
# sda/experiments/ibpm/train.py
#!/usr/bin/env python

import wandb
from dawgz import job, schedule

from sda.mcs import *
from sda.score import *
from sda.utils import *
from .utils import *

CONFIG = {
    # Architecture
    'window': 5,
    'embedding': 64,
    'hidden_channels': (96, 192, 384),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Training
    'epochs': 2048,
    'batch_size': 16,  # 小さめに設定（サンプル数が少ない場合）
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,
    'scheduler': 'cosine',
}


@job(array=3, cpus=4, gpus=1, ram='16GB', time='24:00:00')
def train(i: int):
    run = wandb.init(project='sda-ibpm', config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    window = CONFIG['window']
    score = make_score(**CONFIG)
    shape = torch.Size((window * 2, 64, 64))
    sde = VPSDE(score.kernel, shape=shape).cuda()

    # Data
    trainset = TrajectoryDataset(
        PATH / 'data/train.h5',
        window=window,
        flatten=True
    )
    validset = TrajectoryDataset(
        PATH / 'data/valid.h5',
        window=window,
        flatten=True
    )

    print(f"Train samples: {len(trainset)}")
    print(f"Valid samples: {len(validset)}")

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
        **CONFIG,
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(score.state_dict(), runpath / 'state.pth')

    # Sample generation
    x = sde.sample(torch.Size([4]), steps=64).cpu()
    x = x.unflatten(1, (-1, 2))

    # 渦度を計算して可視化
    vorticity = []
    for sample in x:
        w = torch.gradient(sample[:, 1], dim=1)[0] - \
            torch.gradient(sample[:, 0], dim=2)[0]
        vorticity.append(w)
    vorticity = torch.stack(vorticity)

    # 簡易可視化（Kolmogorov用のdraw関数がない場合）
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(vorticity[idx, 0].numpy(), cmap='RdBu_r')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(runpath / 'samples.png')
    run.log({'samples': wandb.Image(str(runpath / 'samples.png'))})

    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='IBPM Training',
        backend='slurm',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )
```

#### フェーズ5: 学習実行

```bash
# 8. 学習を開始
cd /workspace/sda/experiments/ibpm
python train.py
```

### 📊 期待される結果

#### 学習初期（epoch 0-100）
- Loss: 高い（~10-100）
- 速度場の大まかなパターンを学習

#### 学習中期（epoch 100-500）
- Loss: 安定化（~1-10）
- 円柱周りの流れのトポロジーを獲得

#### 学習後期（epoch 500-2000）
- Loss: 収束（~0.1-1）
- 詳細な渦構造の再現

### 🎯 成功の指標

1. **Loss収束**: validation lossが安定して下降
2. **サンプル品質**: 生成された速度場が物理的に妥当
3. **渦度パターン**: カルマン渦列などの特徴的構造を再現

---

## まとめ

### 変換の要点

1. **Tecplotファイルのパース**: ASCII形式から数値配列へ
2. **速度成分の抽出**: (H, W, 5) → (2, H, W)
3. **解像度調整**: 256×256 → 64×64 (Coarsening)
4. **時系列の集約**: 複数ファイル → 単一配列
5. **HDF5形式での保存**: SDAが読み込める形式

### データ形状の遷移

```
IBPM: ibpm00000.plt (ASCII)
  ↓ parse_tecplot()
(199, 199, 5) float64  [x, y, u, v, vorticity]
  ↓ extract_velocity()
(2, 199, 199) float64  [u, v]
  ↓ coarsen(r=3)
(2, 66, 66) float64
  ↓ aggregate_timeseries()
(n_timesteps, 2, 66, 66) float32
  ↓ create_hdf5_dataset()
SDA: train.h5
  Dataset 'x': (n_samples, n_timesteps, 2, 66, 66) float32
```

### 推奨設定

| パラメータ | 推奨値 | 理由 |
|-----------|--------|------|
| IBPM `-nsteps` | 200-500 | 十分な時系列長と初期過渡の除外 |
| IBPM `-Re` | 100-200 | 興味深い流れ構造（カルマン渦列） |
| Coarsen factor | 4 | 256×256 → 64×64 (Kolmogorov と同じ) |
| Window size | 64 | SDAの標準ウィンドウサイズ |
| Batch size | 16-32 | サンプル数に応じて調整 |
| Learning rate | 1e-4 | 安定した学習のため |
| Epochs | 2048 | 十分な収束時間 |
| Dtype | float32 | 学習効率とメモリのバランス |
| HDF5 compression | gzip level 4 | ファイルサイズ削減 |

### 最終チェックリスト

- [ ] IBPMで複数実験を実行（最低3つ以上推奨）
- [ ] 各実験で200ステップ以上のデータを生成
- [ ] 変換スクリプトでHDF5形式に変換完了
- [ ] データ形状が`(n_samples, 64, 2, 64, 64)`であることを確認
- [ ] 正規化を適用（必要に応じて）
- [ ] SDA学習スクリプト（`utils.py`, `train.py`）を作成
- [ ] 学習を開始し、loss曲線をモニタリング
- [ ] 生成サンプルが物理的に妥当であることを確認

### 次のステップ: さらなる改善

1. **データ拡張**: 複数のレイノルズ数、幾何形状でIBPM実験
2. **正規化の最適化**: 各物理量に応じた正規化手法
3. **ハイパーパラメータチューニング**: learning rate, batch size等
4. **長時間予測**: より長いwindowサイズでの学習
5. **高解像度化**: 128×128や256×256での学習

これで、IBPMで生成した流体シミュレーションデータをSDAで学習し、新しい流れパターンを生成できるようになります！
