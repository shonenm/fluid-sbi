# SDA Kolmogorov Flow: Model Architecture

Score-based Data Assimilation (SDA) for Kolmogorov Flow の詳細なモデルアーキテクチャ解説

---

## 📋 目次

1. [全体構造](#全体構造)
2. [時間埋め込み (TimeEmbedding)](#時間埋め込み-timeembedding)
3. [U-Net アーキテクチャ](#u-net-アーキテクチャ)
4. [Markov Chain Score Network](#markov-chain-score-network)
5. [VPSDE (Variance Preserving SDE)](#vpsde-variance-preserving-sde)
6. [Kolmogorov Flow 固有の設計](#kolmogorov-flow-固有の設計)
7. [学習とサンプリング](#学習とサンプリング)

---

## 全体構造

### レイヤー構成

```
Input: x (B, L, C, H, W)  ← 時系列の速度場
  ↓
MCScoreNet (Markov Chain ラッパー)
  ├─ unfold: (B, L, C, H, W) → (B, L', C*(2*order+1), H, W)
  ↓
  └─ kernel: LocalScoreUNet (実際のスコア関数)
      ├─ TimeEmbedding: t → time_emb (embedding次元)
      ├─ Forcing: sin(4x) を constant channel として追加
      └─ UNet: 階層的な畳み込みネット
          ├─ Encoder (descent): 3段階のダウンサンプリング
          ├─ Bottleneck: 最深層の特徴抽出
          └─ Decoder (ascent): 3段階のアップサンプリング + skip connections
  ↓
  ├─ fold: (B, L', C*(2*order+1), H, W) → (B, L, C, H, W)
  ↓
Output: score (B, L, C, H, W)  ← 推定されたスコア関数
```

---

## 時間埋め込み (TimeEmbedding)

### 目的
拡散過程の時刻 `t ∈ [0, 1]` をネットワークに条件付けるための高次元表現を生成

### 実装

```python
class TimeEmbedding(nn.Sequential):
    def __init__(self, features: int):
        super().__init__(
            nn.Linear(32, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )
        self.register_buffer('freqs', torch.pi * torch.arange(1, 16 + 1))

    def forward(self, t: Tensor) -> Tensor:
        # Sinusoidal encoding
        t = self.freqs * t.unsqueeze(dim=-1)  # (B,) → (B, 16)
        t = torch.cat((t.cos(), t.sin()), dim=-1)  # → (B, 32)

        # MLP projection
        return super().forward(t)  # → (B, features)
```

### 特徴

1. **Sinusoidal Encoding**:
   - 周波数: `π, 2π, 3π, ..., 16π`
   - cos と sin の両方を使用 → 32次元

2. **MLP Projection**:
   - 32 → 256 → features (デフォルト: 64)
   - SiLU 活性化関数

3. **利点**:
   - 時間の連続性を保持
   - 異なる時間スケールを捉える
   - Transformer の positional encoding と類似

---

## U-Net アーキテクチャ

### 基本構造

```
入力: (B, C_in, H, W)
  ↓
┌─────────────────────────────────────────┐
│ Encoder (Descent Path)                  │
├─────────────────────────────────────────┤
│ Level 0: C_in → 96                      │  ← skip_0
│   ├─ Conv2d(3x3)                        │
│   └─ ResBlock x 3                       │
├─────────────────────────────────────────┤
│ Level 1: 96 → 192 (stride=2)            │  ← skip_1
│   ├─ Conv2d(3x3, stride=2)              │
│   └─ ResBlock x 3                       │
├─────────────────────────────────────────┤
│ Level 2: 192 → 384 (stride=2)           │  ← skip_2
│   ├─ Conv2d(3x3, stride=2)              │
│   └─ ResBlock x 3                       │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ Decoder (Ascent Path)                   │
├─────────────────────────────────────────┤
│ Level 2: 384 → 192                      │
│   ├─ ResBlock x 3                       │
│   ├─ Upsample(2x) + Conv2d              │
│   └─ + skip_2                           │
├─────────────────────────────────────────┤
│ Level 1: 192 → 96                       │
│   ├─ ResBlock x 3                       │
│   ├─ Upsample(2x) + Conv2d              │
│   └─ + skip_1                           │
├─────────────────────────────────────────┤
│ Level 0: 96 → C_out                     │
│   ├─ ResBlock x 3                       │
│   └─ Conv2d(3x3)                        │
└─────────────────────────────────────────┘
  ↓
出力: (B, C_out, H, W)
```

### パラメータ詳細

**Kolmogorov の CONFIG:**
```python
{
    'window': 5,                      # Markov chain の window サイズ
    'embedding': 64,                  # 時間埋め込み次元
    'hidden_channels': (96, 192, 384),  # 各階層のチャネル数
    'hidden_blocks': (3, 3, 3),       # 各階層の ResBlock 数
    'kernel_size': 3,                 # 畳み込みカーネルサイズ
    'activation': 'SiLU',             # 活性化関数
    'spatial': 2,                     # 空間次元 (2D)
    'padding_mode': 'circular',       # 周期境界条件
}
```

### ModResidualBlock

各 ResBlock は **時間埋め込みによる変調** (modulation) を持つ：

```python
class ModResidualBlock(nn.Module):
    def __init__(self, project: nn.Module, residue: nn.Module):
        self.project = project  # time_emb → channel_scale
        self.residue = residue  # 畳み込み層

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # y = time_emb (B, embedding)
        # project(y) → (B, C, 1, 1)
        return x + self.residue(x + self.project(y))
```

**構造:**
```
x (B, C, H, W)
  ├─ project(time_emb) → (B, C, 1, 1)
  │
  ├─ + で加算
  ↓
  LayerNorm
  ↓
  Conv2d(C, C, 3x3)
  ↓
  SiLU
  ↓
  Conv2d(C, C, 3x3)
  ↓
  + 残差接続
  ↓
出力 (B, C, H, W)
```

---

## Markov Chain Score Network

### MCScoreNet の役割

時系列データ `x = (x_0, x_1, ..., x_L)` に対して、Markov性を利用してスコア関数を効率的に計算。

### Unfold/Fold 操作

**Unfold: 時間窓を作成**
```python
@staticmethod
def unfold(x: Tensor, order: int) -> Tensor:
    # x: (B, L, C, H, W)
    # order = 2 (window=5 の場合)

    x = x.unfold(1, 2 * order + 1, 1)  # (B, L-4, C, H, W, 5)
    x = x.movedim(-1, 2)               # (B, L-4, 5, C, H, W)
    x = x.flatten(2, 3)                # (B, L-4, 5*C, H, W)

    return x
```

**例 (window=5, order=2):**
```
入力: x = [x_0, x_1, x_2, x_3, x_4, x_5, x_6]

unfold 後:
  位置 0: [x_0, x_1, x_2, x_3, x_4]  ← 5つのフレームを1つのチャネルに
  位置 1: [x_1, x_2, x_3, x_4, x_5]
  位置 2: [x_2, x_3, x_4, x_5, x_6]

→ (B, 3, 10, 64, 64)  ← チャネル数 = 5 * 2 = 10
```

**Fold: 元の時系列に戻す**
```python
@staticmethod
def fold(x: Tensor, order: int) -> Tensor:
    # x: (B, L-4, 5*C, H, W)

    x = x.unflatten(2, (2 * order + 1, -1))  # (B, L-4, 5, C, H, W)

    # 端の処理 + 中央の全フレーム
    return torch.cat((
        x[:, 0, :order],      # 最初の order 個
        x[:, :, order],       # すべての中央フレーム
        x[:, -1, -order:],    # 最後の order 個
    ), dim=1)
```

---

## VPSDE (Variance Preserving SDE)

### 数学的定義

**Forward SDE:**
```
dx(t) = -β(t)/2 * x(t) dt + √β(t) dw
```

**Perturbation Kernel:**
```
p(x(t) | x) = N(x(t) | μ(t)x, σ²(t)I)

μ(t) = α(t)
σ²(t) = 1 - α²(t) + η²
```

**α(t) の選択:**
```python
if alpha == 'cos':  # デフォルト
    α(t) = cos²(arccos(√η) * t)
```

### Denoising Score Matching

**学習目標:**
```
L = E_{x, t, ε} [||ε_θ(x(t), t) - ε||²]

where:
  x(t) = μ(t)x + σ(t)ε
  ε ~ N(0, I)
```

**実装:**
```python
def loss(self, x: Tensor, c: Tensor = None) -> Tensor:
    t = torch.rand(x.shape[0])  # (B,) ← [0, 1] からランダム

    # Forward diffusion
    eps = torch.randn_like(x)
    x_t = self.mu(t) * x + self.sigma(t) * eps

    # Predict noise
    eps_pred = self.eps(x_t, t, c)

    # MSE loss
    return (eps_pred - eps).square().mean()
```

### Predictor-Corrector サンプリング

**Predictor (Reverse SDE):**
```python
r = μ(t - dt) / μ(t)
x = r * x + (σ(t - dt) - r * σ(t)) * ε_θ(x, t)
```

**Corrector (Langevin MCMC):**
```python
for _ in range(corrections):
    eps = ε_θ(x, t - dt)
    δ = τ / eps.square().mean()
    x = x - (δ * eps + √(2δ) * z) * σ(t - dt)
```

---

## Kolmogorov Flow 固有の設計

### LocalScoreUNet: Forcing Channel の追加

**Kolmogorov forcing:**
```python
class LocalScoreUNet(ScoreUNet):
    def __init__(self, channels: int, size: int = 64, **kwargs):
        super().__init__(channels, 1, **kwargs)  # context=1

        # sin(4x) の forcing を作成
        domain = 2 * π / size * (torch.arange(size) + 0.5)
        forcing = torch.sin(4 * domain).expand(1, size, size)

        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.forcing)  # forcing を context として渡す
```

**Forcing の役割:**
- Kolmogorov flow の外力項 `f = sin(4y)` を表現
- ネットワークが物理的な対称性を学習しやすくする
- context channel として ScoreUNet に入力

### Circular Padding

**周期境界条件:**
```python
UNet(..., padding_mode='circular')
```

**効果:**
- 流体の周期性を保持
- 境界でのアーティファクトを防ぐ
- 物理的に正しい境界条件

---

## 学習とサンプリング

### 学習プロセス

**1. データ準備**
```python
trainset = TrajectoryDataset(PATH / 'data/train.h5', window=5, flatten=True)
# 各サンプル: (64, 2, 64, 64)
#             window*2 チャネル, H, W
```

**2. ネットワーク構築**
```python
score = make_score(
    window=5,
    embedding=64,
    hidden_channels=(96, 192, 384),
    hidden_blocks=(3, 3, 3),
)

sde = VPSDE(score.kernel, shape=(10, 64, 64))
```

**3. 学習ループ**
```python
for epoch in range(epochs):
    for x, _ in trainloader:
        loss = sde.loss(x).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### サンプリングプロセス

**1. 初期ノイズ**
```python
x = torch.randn(batch, 10, 64, 64).cuda()  # ~ N(0, I)
```

**2. Reverse SDE**
```python
time = torch.linspace(1, 0, steps + 1)

for t in time[:-1]:
    # Predictor
    r = μ(t - dt) / μ(t)
    x = r * x + (σ(t - dt) - r * σ(t)) * score.kernel(x, t)

    # Corrector (Langevin)
    for _ in range(corrections):
        eps = score.kernel(x, t - dt)
        δ = τ / eps.square().mean()
        x = x - (δ * eps + √(2δ) * z) * σ(t - dt)
```

**3. Unflatten**
```python
x = x.unflatten(1, (-1, 2))  # (B, 10, 64, 64) → (B, 5, 2, 64, 64)
#                                 チャネル → (window, u/v, H, W)
```

---

## パラメータ数の計算

### Kolmogorov 設定でのパラメータ数

**TimeEmbedding:**
```
32 * 256 + 256 * 64 = 24,576
```

**UNet:**

**Level 0 (96 channels):**
- head: `10 * 96 * 3 * 3 = 8,640`
- ResBlock x 3: `(96 * 96 * 3 * 3 * 2) * 3 ≈ 497,664`
- tail: `96 * 10 * 3 * 3 = 8,640`

**Level 1 (192 channels):**
- downconv: `96 * 192 * 3 * 3 = 165,888`
- ResBlock x 3: `(192 * 192 * 3 * 3 * 2) * 3 ≈ 1,990,656`
- upconv: `192 * 96 * 3 * 3 = 165,888`

**Level 2 (384 channels):**
- downconv: `192 * 384 * 3 * 3 = 663,552`
- ResBlock x 3: `(384 * 384 * 3 * 3 * 2) * 3 ≈ 7,962,624`
- upconv: `384 * 192 * 3 * 3 = 663,552`

**Total: 約 12M パラメータ**

---

## データフロー全体像

```
[学習時]
x (B, 64, 2, 64, 64)  ← HDF5 から読み込み
  ↓ flatten
x (B, 10, 64, 64)  ← 5 window * 2 channels
  ↓ random t ~ U[0,1]
  ↓ forward diffusion
x_t = μ(t)x + σ(t)ε
  ↓
MCScoreNet:
  ├─ unfold → (B, L', 10, 64, 64)
  ├─ LocalScoreUNet(x_t, t, forcing)
  │   ├─ TimeEmbedding(t) → (B, 64)
  │   ├─ cat([x_t, forcing], dim=1) → (B, 11, 64, 64)
  │   └─ UNet(x_cat, time_emb) → (B, 10, 64, 64)
  └─ fold → (B, 10, 64, 64)
  ↓
ε_pred (B, 10, 64, 64)
  ↓
loss = ||ε_pred - ε||²


[サンプリング時]
x ~ N(0, I)  (B, 10, 64, 64)
  ↓
for t in [1.0 → 0.0]:
  ├─ ε_pred = LocalScoreUNet(x, t, forcing)
  ├─ x = r*x + (σ_new - r*σ)*ε_pred  (Predictor)
  └─ x = x - δ*ε_pred + √(2δ)*z  (Corrector)
  ↓
x (B, 10, 64, 64)
  ↓ unflatten
x (B, 5, 2, 64, 64)  ← 生成された速度場
```

---

## まとめ

### 主要コンポーネント

1. **MCScoreNet**: Markov chain の時系列構造を扱う
2. **LocalScoreUNet**: Forcing channel 付きの U-Net
3. **TimeEmbedding**: Sinusoidal encoding + MLP
4. **VPSDE**: Variance Preserving SDE によるノイズスケジューリング
5. **Circular Padding**: 周期境界条件

### 設計思想

- **物理的制約の組み込み**: Forcing term, circular padding
- **階層的特徴抽出**: U-Net の encoder-decoder
- **時間条件付け**: TimeEmbedding + modulation
- **Markov性の活用**: unfold/fold で効率的な時系列処理

### パフォーマンス

- パラメータ数: 約 12M
- 入力解像度: 64×64
- 時間窓: 5 frames
- 学習時間: ~24時間 (4096 epochs, GPU)

このアーキテクチャにより、Kolmogorov flow の複雑な非線形ダイナミクスを学習し、データ同化タスクに適用できます。
