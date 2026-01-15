# Latent Space SDA (Score-based Data Assimilation)

## 概要

Latent Space SDA は、高次元の流体力学データに対するデータ同化を、
低次元の潜在空間で効率的に行う手法です。

### 通常のSDAとの違い

| 項目 | 通常のSDA | Latent SDA |
|------|-----------|------------|
| 操作空間 | 元データ空間 (C×H×W) | 潜在空間 (D_latent) |
| 計算コスト | 高い | 低い |
| メモリ使用量 | 大きい | 小さい |
| 学習時間 | 長い | 短い |
| 観測オペレータ | 直接適用 | デコード後に適用 |

### 利点

1. **計算効率**: 64×64×2 = 8192次元 → 64次元に圧縮
2. **高速な学習**: スコアモデルの学習が軽量
3. **柔軟な観測**: 様々な観測オペレータに対応

---

## アーキテクチャ

### 全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│                        LatentSDA                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    │
│  │   Encoder   │    │ Score Model │    │    VPSDE         │    │
│  │ (PCA / VAE) │    │ (ScoreNet)  │    │ (拡散過程)        │    │
│  └──────┬──────┘    └──────┬──────┘    └────────┬─────────┘    │
│         │                  │                    │              │
│         └──────────────────┴────────────────────┘              │
│                            │                                   │
│                   ┌────────▼────────┐                          │
│                   │ LatentGaussian  │                          │
│                   │     Score       │                          │
│                   │ (データ同化)     │                          │
│                   └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### データフロー

```
入力データ (N, T, C, H, W)
        │
        ▼ flatten
(N×T, C, H, W)
        │
        ▼ Encoder.encode()
潜在表現 (N×T, D_latent)
        │
        ▼ Score Model 学習
ノイズ予測 ε(z_t, t)
        │
        ▼ VPSDE reverse sampling
        │   + LatentGaussianScore (観測誘導)
        │
        ▼ Encoder.decode()
復元データ (B, C, H, W)
```

### コンポーネント詳細

#### 1. LatentSDA クラス

統合インターフェース。エンコーダ、スコアモデル、SDEを管理。

```python
class LatentSDA:
    def __init__(
        self,
        encoder: LatentEncoder,          # PCA or VAE
        score_hidden: tuple = (256, 256, 256),  # スコアネットワーク隠れ層
        score_embedding: int = 64,        # 時間埋め込み次元
        sde_alpha: str = "cos",           # ノイズスケジュール
        device: str = "cpu",
    )
```

#### 2. VPSDE (Variance-Preserving SDE)

拡散過程を定義。ノイズスケジュールにより分散の変化を制御。

| スケジュール | 説明 |
|-------------|------|
| `lin` | 線形（β_t が線形増加） |
| `cos` | コサイン（滑らかな変化、推奨） |
| `exp` | 指数関数 |

#### 3. LatentGaussianScore

観測に基づくスコア誘導。Tweedie推定でデノイズし、
観測空間での尤度勾配を計算。

```python
# 誘導付きスコア = 事前スコア - σ × ∇_z log p(y|z)
```

---

## エンコーダ

### PCAEncoder

線形次元削減。高速で解釈可能。

```python
from experiments.latent_sda.encoders.pca import PCAEncoder

encoder = PCAEncoder(
    n_components=64,    # 潜在次元
    whiten=True,        # 白色化（推奨）
    random_state=42,    # 再現性のためのシード
)

# フィッティング
encoder.fit(X_train)  # (N, C, H, W)

# 変換
z = encoder.encode(x)        # (B, D_latent)
x_recon = encoder.decode(z)  # (B, C, H, W)

# 説明分散の確認
print(f"説明分散率: {encoder.total_explained_variance:.2%}")
```

**特徴:**
- 瞬時にフィット（SVD分解）
- 白色化により潜在空間が正規化
- `explained_variance_ratio` で各成分の寄与率を確認可能

### VAEEncoder

非線形次元削減。複雑なパターンを学習可能。

```python
from experiments.latent_sda.encoders.vae import VAEEncoder

encoder = VAEEncoder(
    in_channels=2,                    # 入力チャネル（速度場 u, v）
    latent_dim=64,                    # 潜在次元
    hidden_channels=(32, 64, 128),    # Conv層のチャネル数
    image_size=64,                    # 入力画像サイズ
    kl_weight=1e-4,                   # KL損失の重み（β-VAE）
)

# 学習
encoder.fit(
    X_train,
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    device="cuda",
)
```

**アーキテクチャ:**
```
Encoder: Conv2d(stride=2) × 3 → Flatten → Linear → (μ, log σ²)
Decoder: Linear → Reshape → ConvTranspose2d × 3
```

---

## 使用方法

### 学習 (train.py)

#### コマンドラインオプション

```bash
uv run python sda/experiments/latent_sda/train.py [OPTIONS]
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--method` | pca | エンコーダ種類 (pca / vae) |
| `--latent-dim` | 64 | 潜在空間の次元 |
| `--epochs` | 256 | スコアモデルの学習エポック数 |
| `--vae-epochs` | 100 | VAEの学習エポック数 |
| `--batch-size` | 64 | バッチサイズ |
| `--learning-rate` | 1e-3 | 学習率 |
| `--name` | (自動) | run名（省略時は自動生成） |
| `--device` | cuda | 計算デバイス |
| `--seed` | 42 | 乱数シード |
| `--dry-run` | - | テスト実行（2エポックのみ） |

#### 実行例

```bash
# PCA (64次元) で学習
uv run python sda/experiments/latent_sda/train.py \
    --method pca --latent-dim 64

# VAE (128次元) で学習
uv run python sda/experiments/latent_sda/train.py \
    --method vae --latent-dim 128 --vae-epochs 200

# テスト実行
uv run python sda/experiments/latent_sda/train.py --dry-run
```

#### 出力ファイル

```
runs/latent_sda/{method}/{run_name}/
├── encoder.pt      # エンコーダの重み/パラメータ
├── score.pt        # スコアモデルの重み
├── config.pt       # アーキテクチャ設定
├── history.pt      # 学習履歴（損失推移）
└── metrics.pt      # 評価メトリクス
```

---

### 評価 (evaluate.py)

#### コマンドラインオプション

```bash
uv run python sda/experiments/latent_sda/evaluate.py [OPTIONS]
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--method` | pca | 自動検出時のメソッド |
| `--run-id` | (最新) | 特定のrun IDを指定 |
| `--checkpoint` | - | フルパスで指定（レガシー） |
| `--n-samples` | 5 | 事後分布のサンプル数 |
| `--steps` | 64 | 拡散サンプリングのステップ数 |
| `--device` | cuda | 計算デバイス |

#### 実行例

```bash
# 最新のPCA runを自動検出して評価
uv run python sda/experiments/latent_sda/evaluate.py

# 特定のrunを評価
uv run python sda/experiments/latent_sda/evaluate.py \
    --run-id dim64_20260115_140100

# VAEモデルを評価（サンプル数増加）
uv run python sda/experiments/latent_sda/evaluate.py \
    --method vae --n-samples 10 --steps 128
```

#### 観測シナリオ

| シナリオ | 説明 | パラメータ |
|---------|------|-----------|
| `sub` | スパース観測（間引き） | rate=2, noise=0.1 |
| `coarse` | 荒い観測（空間平均） | factor=4, noise=0.1 |

#### 出力ファイル

```
results/latent_sda/{method}/{run_name}/
├── eval_config.yaml   # 評価時のパラメータ
├── results.pt         # 全結果（PyTorch形式）
├── stats.csv          # メトリクス（CSV形式）
├── da_sub_0.png       # subsample観測の結果
├── da_sub_1.png
├── da_coarse_0.png    # coarse観測の結果
└── da_coarse_1.png
```

#### 可視化の見方

各PNGファイルは3列構成：

| 左 | 中央 | 右 |
|----|------|-----|
| Ground Truth | 観測 | 復元結果 |
| （正解の渦度場） | （観測点のみ/低解像度） | （データ同化後） |

---

## ファイル構成

```
sda/experiments/latent_sda/
├── __init__.py           # パッケージエクスポート
├── latent_sda.py         # LatentSDA メインクラス
├── latent_score.py       # LatentGaussianScore, LatentDPSScore
├── train.py              # 学習スクリプト
├── evaluate.py           # 評価スクリプト
└── encoders/
    ├── __init__.py       # エンコーダエクスポート
    ├── base.py           # LatentEncoder 抽象基底クラス
    ├── pca.py            # PCAEncoder 実装
    └── vae.py            # VAEEncoder 実装
```

---

## コード例

### 基本的な学習

```python
from experiments.latent_sda import LatentSDA
from experiments.latent_sda.encoders import PCAEncoder
import torch

# データ読み込み（Kolmogorov flow）
X_train = torch.randn(1000, 2, 64, 64)  # (N, C, H, W)

# エンコーダ作成
encoder = PCAEncoder(n_components=64, whiten=True)

# LatentSDA 初期化
latent_sda = LatentSDA(
    encoder=encoder,
    score_hidden=(256, 256, 256),
    sde_alpha="cos",
    device="cuda",
)

# エンコーダをフィット
latent_sda.fit_encoder(X_train)
print(f"説明分散: {encoder.total_explained_variance:.2%}")

# スコアモデルを学習
history = latent_sda.train_score(
    X_train,
    epochs=256,
    batch_size=64,
    learning_rate=1e-3,
)

# モデル保存
latent_sda.save("./my_model")
```

### データ同化

```python
# モデル読み込み
latent_sda = LatentSDA.load("./my_model", device="cuda")

# 観測オペレータを定義（例：2倍間引き）
def subsample_op(x):
    """x: (B, C, H, W) or (C, H, W)"""
    if x.ndim == 4:
        return x[:, :, ::2, ::2]
    return x[:, ::2, ::2]

# 観測データ（ノイズ付き）
x_true = torch.randn(2, 64, 64)  # 真の状態
y_obs = subsample_op(x_true) + 0.1 * torch.randn_like(subsample_op(x_true))

# データ同化実行
x_samples = latent_sda.assimilate(
    y=y_obs,
    A=subsample_op,
    std=0.1,           # 観測ノイズの標準偏差
    steps=64,          # サンプリングステップ数
    n_samples=5,       # 事後サンプル数
)
# x_samples: (5, 2, 64, 64)

# 事後平均を使用
x_mean = x_samples.mean(dim=0)  # (2, 64, 64)
```

### カスタム観測オペレータ

```python
# 荒い観測（空間平均）
def coarsen_op(x, factor=4):
    """factor倍に荒くする"""
    from sda.mcs import KolmogorovFlow
    return KolmogorovFlow.coarsen(x, factor)

# 部分観測（特定領域のみ）
def partial_op(x, region=(16, 48, 16, 48)):
    """指定領域のみ観測"""
    y1, y2, x1, x2 = region
    if x.ndim == 4:
        return x[:, :, y1:y2, x1:x2]
    return x[:, y1:y2, x1:x2]

# 線形変換
def linear_op(x, A_matrix):
    """任意の線形観測"""
    x_flat = x.flatten(-2)  # (B, C, H*W)
    return (A_matrix @ x_flat.unsqueeze(-1)).squeeze(-1)
```

---

## パラメータチューニングガイド

### 精度向上のための調整

| パラメータ | 効果 | トレードオフ |
|-----------|------|-------------|
| `--latent-dim` ↑ | 情報保持量増加 | 学習が難しくなる |
| `--epochs` ↑ | スコア精度向上 | 学習時間増加 |
| `--steps` ↑ | サンプリング精度向上 | 推論時間増加 |
| `--n-samples` ↑ | 不確実性推定改善 | メモリ使用量増加 |

### 推奨設定

| ケース | latent-dim | epochs | steps |
|--------|------------|--------|-------|
| 高速テスト | 32 | 100 | 32 |
| バランス（推奨） | 64 | 256 | 64 |
| 高精度 | 128 | 512 | 128 |

---

## 依存関係

**必須:**
- PyTorch
- scikit-learn (PCA)
- h5py (データ読み込み)
- PIL (可視化)
- PyYAML (設定保存)

**内部依存:**
- `sda.score.VPSDE` - 拡散SDE
- `sda.score.ScoreNet` - スコアネットワーク
- `sda.mcs.KolmogorovFlow` - 流体場ユーティリティ

---

## 参考文献

- [Score-based Data Assimilation](https://arxiv.org/abs/2306.10574) - Rozet & Louppe, NeurIPS 2023
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687) - Chung et al., 2022
