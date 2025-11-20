# IBPM解像度設定ガイド

## 概要

物理的特徴を保持するため、**IBPM実行時に目標解像度を直接指定**します。
変換スクリプト（`convert_ibpm_to_sda.py`）ではリサイズを行いません。

---

## 推奨解像度

### 解像度の決定基準

円柱直径 D = 1.0 に対して、**最低10-20セル/直径**が必要です。

| 解像度 | 格子間隔 Δx | 円柱の解像度 | 計算コスト | 推奨用途 |
|-------|-----------|------------|-----------|---------|
| **64×64** | 0.0625 | D ≈ 16セル | 最小 | プロトタイプ検証 |
| **128×128** | 0.03125 | D ≈ 32セル | 中程度 | **推奨（バランス良好）** |
| **199×199** | 0.02 | D ≈ 50セル | 高 | 最高品質 |
| **256×256** | 0.015625 | D ≈ 64セル | 最高 | 高精度計算 |

**計算式**:
```
領域サイズ: [-2, 2] × [-2, 2] = 4 × 4
格子間隔: Δx = 4 / nx
円柱直径: D = 1.0
円柱の解像度: D / Δx = nx / 4 セル
```

---

## IBPM実行コマンド

### 1. 64×64（最小構成）

```bash
ibpm -nx 64 -ny 64 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.02 -nsteps 250 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_64
```

**特徴**:
- ✅ 計算時間: 約5分（最速）
- ✅ メモリ使用量: 最小
- ⚠️ 円柱の解像度: 16セル（ギリギリ）
- ⚠️ 渦構造の捕捉: 粗い

**生成データ**: 63×63グリッド（セル中心）

---

### 2. 128×128（推奨）⭐

```bash
ibpm -nx 128 -ny 128 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.02 -nsteps 250 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_128
```

**特徴**:
- ✅ 円柱の解像度: 32セル（良好）
- ✅ 渦構造の捕捉: 十分
- ✅ 計算コスト: 許容範囲（約15-20分）
- ✅ メモリ: 適度

**生成データ**: 127×127グリッド

**推奨理由**:
- 物理的特徴を十分に捕捉
- 機械学習での計算コストとのバランス
- SDA実験に最適

---

### 3. 199×199（デフォルト、高品質）

```bash
ibpm -nx 200 -ny 200 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.02 -nsteps 250 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_full
```

**特徴**:
- ✅ 円柱の解像度: 50セル（最高）
- ✅ 渦構造: 詳細に捕捉
- ⚠️ 計算時間: 約30-40分
- ⚠️ ファイルサイズ: 大きい（1時刻あたり2.5MB）

**生成データ**: 199×199グリッド

---

### 4. 256×256（超高解像度）

```bash
ibpm -nx 256 -ny 256 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.01 -nsteps 500 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_256
```

**特徴**:
- ✅ 円柱の解像度: 64セル（極めて高精度）
- ✅ 小スケール渦も捕捉
- ⚠️ 計算時間: 1時間以上
- ⚠️ ストレージ: 大量（1時刻あたり5MB以上）
- ⚠️ dt調整推奨: CFL条件により dt=0.01

**注意**: 時間刻みを小さくする必要があります（dt=0.01推奨）

---

## HDF5変換

解像度に関わらず、変換コマンドは同じです：

```bash
python /workspace/scripts/convert_ibpm_to_sda.py \
    --input /workspace/data/ibpm_128 \
    --output /workspace/data/ibpm_h5_128 \
    --window 64 \
    --stride 8
```

**重要**: `--coarsen` は指定しない（デフォルト=1、圧縮なし）

---

## データサイズの比較

### ディスク使用量

| 解像度 | 1時刻あたり | 250時刻 | HDF5 (gzip圧縮) |
|-------|-----------|--------|----------------|
| 64×64 | ~200 KB | ~50 MB | ~15 MB |
| 128×128 | ~800 KB | ~200 MB | ~60 MB |
| 199×199 | ~2.5 MB | ~625 MB | ~180 MB |
| 256×256 | ~5.0 MB | ~1.25 GB | ~350 MB |

### メモリ使用量（学習時）

| 解像度 | 1バッチ(32サンプル) | GPU RAM |
|-------|------------------|---------|
| 64×64 | ~80 MB | 2-4 GB |
| 128×128 | ~320 MB | 6-8 GB |
| 199×199 | ~1 GB | 12-16 GB |
| 256×256 | ~2 GB | 24+ GB |

---

## 実践的なワークフロー

### フェーズ1: プロトタイプ（64×64）

```bash
# 高速検証用
ibpm -nx 64 -ny 64 -geom cylinder.geom -nsteps 250 -tecplot 1 -outdir ibpm_64_test
python convert_ibpm_to_sda.py --input ibpm_64_test --output data_64_test
# 学習・評価の動作確認
```

### フェーズ2: 開発（128×128）⭐

```bash
# バランスの良い解像度で開発
ibpm -nx 128 -ny 128 -geom cylinder.geom -nsteps 250 -tecplot 1 -outdir ibpm_128
python convert_ibpm_to_sda.py --input ibpm_128 --output data_128
# 本格的な学習・評価
```

### フェーズ3: 最終検証（199×199または256×256）

```bash
# 高品質データで最終評価
ibpm -nx 200 -ny 200 -geom cylinder.geom -nsteps 250 -tecplot 1 -outdir ibpm_full
python convert_ibpm_to_sda.py --input ibpm_full --output data_full
# 論文用の結果生成
```

---

## 注意事項

### 1. dt（時間刻み）の調整

**CFL条件**: CFL = U∞ × dt / Δx ≤ 1.0

| 解像度 | Δx | 推奨 dt |
|-------|-----|--------|
| 64×64 | 0.0625 | 0.02 ✓ |
| 128×128 | 0.03125 | 0.02 ✓ |
| 199×199 | 0.02 | 0.02 ✓ |
| 256×256 | 0.015625 | **0.01** ⚠️ |

256×256では dt を 0.01 に変更：
```bash
ibpm -nx 256 -ny 256 -dt 0.01 ...
```

### 2. ファイル数の調整

高解像度では出力を間引くことを推奨：

```bash
# 10時刻ごとに出力（ファイル数を1/10に削減）
ibpm -nx 256 -ny 256 -tecplot 10 -nsteps 500 ...
```

### 3. 円柱の解像度確認

```python
# 生成後に確認
import h5py
with h5py.File('data_128/train.h5', 'r') as f:
    print(f'Shape: {f["x"].shape}')
    # 期待: (n_samples, window, 2, 127, 127)
```

---

## トラブルシューティング

### Q: 64×64で円柱がぼやける

**A**: 解像度不足です。128×128以上を使用してください。

### Q: 256×256で計算が進まない

**A**: dt を 0.01 に減らしてください（CFL条件）。

### Q: HDF5ファイルが巨大

**A**:
- IBPM側で `-tecplot 10` で出力を間引く
- または `--stride` を大きくする（例: 16）

### Q: 変換後の解像度が1ピクセル小さい

**A**: 正常です。200セル → 199グリッド点（セル中心）。

---

## まとめ

### ✅ 推奨設定

**標準開発**: 128×128
```bash
ibpm -nx 128 -ny 128 -geom cylinder.geom -Re 100 -dt 0.02 -nsteps 250 -tecplot 1
```

**高品質**: 199×199（デフォルトのまま）
```bash
ibpm -nx 200 -ny 200 -geom cylinder.geom -Re 100 -dt 0.02 -nsteps 250 -tecplot 1
```

### ❌ 非推奨

- ~~リサイズ処理（`--coarsen`）~~
- ~~64×64未満の解像度~~
- ~~後処理での補間~~

---

**更新日**: 2024年10月29日
**対象バージョン**: IBPM 1.0, convert_ibpm_to_sda.py v2.0
