# 拡散モデルによる逆問題における勾配スケーリング問題

## 概要

拡散モデルを用いた逆問題（Inverse Problems）において、観測点数が増加すると
尤度勾配が比例して大きくなり、事前スコアとのバランスが崩壊する問題。

これは **"Gradient Dominance"** または **"Guidance Strength Scaling"** と呼ばれる
一般的な課題であり、高解像度データを扱う際に顕在化する。

---

## 1. 理論的背景

### 1.1 ベイズ的スコア更新

拡散モデルによる事後分布サンプリングでは、以下のスコア更新が行われる：

```
∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)
                 ─────────────   ─────────────
                   事前スコア      尤度勾配
                 (Prior Score)  (Likelihood Gradient)
```

### 1.2 次元依存性の問題

高次元空間における **測度の集中（Concentration of Measure）** により：

- **事前スコアのノルム**: データ次元数 D に依存し、概ね √D のオーダー
  （拡散モデルの学習時に正規化されるため比較的安定）

- **尤度勾配のノルム**: 観測点数 N による二乗誤差の総和（`.sum()`）であるため、
  N または √N に比例して増大

**問題の本質**: N（観測点数）が変動する場合、尤度勾配の「強さ」が劇的に変わり、
事前スコアとのバランスが崩れる。

### 1.3 現象の分類

| 状態 | 条件 | 結果 |
|------|------|------|
| **Likelihood Dominance** | N が大きすぎる | 数値爆発、観測への過剰適合 |
| **Balanced** | 適切なバランス | 正常な再構成 |
| **Prior Dominance** | N が小さすぎる | 観測が無視される、フラット出力 |

---

## 2. 本問題での発現

### 修正前の結果画像

修正前（IBPM sub=4基準のスケーリング）の結果は以下に保存:

```
results/ibpm/evaluate/sparse_before_fix/
├── sparse_ground_truth.png
├── sparse_sub2_reconstructed.png   # 爆発（±20000）
├── sparse_sub4_reconstructed.png   # 成功
├── sparse_sub8_reconstructed.png   # 動作
└── sparse_sub16_reconstructed.png  # 動作
```

- **sub=2**: 値が±20000に爆発（Likelihood Dominance）
- **sub=4**: カルマン渦が見える（Balanced - 偶然）
- **sub=8, 16**: 動作するがPrior Dominanceの可能性

### 2.1 GaussianScoreの実装 (`sda/score.py:549`)

```python
log_p = -(err ** 2 / var).sum() / 2  # 対数尤度
s, = torch.autograd.grad(log_p, x)   # 勾配
return eps - sigma * s               # 事前スコア - σ × 尤度勾配
```

`.sum()` により **勾配の大きさが観測点数に比例** する：

```
|s| ∝ N / std²
```

### 2.2 Kolmogorov vs IBPM

**Kolmogorov（参照実装）**: `(8, 2, 64, 64)` = 8,192 要素

| subsample | 観測点数 | std | 動作 |
|-----------|---------|-----|------|
| 2 | 16,384 | 0.1 | ✓ |
| 4 | 4,096 | 0.1 | ✓ |
| 8 | 1,024 | 0.1 | ✓ |
| 16 | 256 | 0.1 | ✓ |

**IBPM**: `(32, 199, 399)` ≈ 2,500,000 要素

| subsample | 観測点数 | Kolmo比 | 結果 | 状態 |
|-----------|---------|---------|------|------|
| 2 | 630,000 | **38x** | 爆発 | Likelihood Dominance |
| 4 | 160,000 | 10x | ギリギリ | Balanced (偶然) |
| 8 | 40,000 | 2.4x | フラット | Prior Dominance |
| 16 | 9,600 | 0.6x | フラット | Prior Dominance |

---

## 3. 解決策の分類

### 3.1 Likelihood Weighting（尤度重み付け）- 現在の対応

観測点数に応じて `std` をスケーリングし、勾配の大きさを正規化する。

```python
n_obs_kolmo_ref = 16384  # Kolmogorov sub=2 を基準
std_scaled = 0.1 * math.sqrt(n_obs / n_obs_kolmo_ref)
```

**数学的根拠**:
```
勾配バランス: N_ref / std_ref² = N_target / std_target²
→ std_target = std_ref × √(N_target / N_ref)
```

| 評価項目 | 結果 |
|----------|------|
| 数学的妥当性 | ✓ 勾配バランス式から導出 |
| 実装容易性 | ✓ パラメータ変更のみ |
| 後方互換性 | ✓ 既存コード変更なし |
| 汎用性 | △ 問題ごとに調整が必要 |
| 根本解決 | ✗ `.sum()`の問題は残る |

### 3.2 Adaptive Gradient Normalization（適応的勾配正規化）

尤度勾配を事前スコアのノルムに合わせて自動正規化する。

```python
g_total = s_prior + ξ * (||s_prior|| / ||g_lik||) * g_lik
```

**利点**:
- 観測点数に依存しない自動調整
- ハイパーパラメータ（ξ）が1つで済む

**欠点**:
- 実装が複雑
- 勾配のノルム計算のオーバーヘッド

**参考文献**: DPS派生手法、Manifold Constrained Gradient (MCG)

### 3.3 Projection / Decomposition Methods（射影法）

観測空間と非観測空間を分離し、観測点は直接置換する。

```
観測点:   x_i = y_i + noise  (データで強制置換)
非観測点: x_i = diffusion output
```

**利点**:
- 勾配爆発の概念自体がなくなる
- 数学的に安定

**欠点**:
- 観測ノイズがある場合、不連続面が生じる可能性
- 流体力学的整合性（連続の式など）を壊すリスク

**参考文献**: DDRM (Denoising Diffusion Restoration Models)

### 3.4 Proximal Methods（近接法）

正則化項（拡散モデル）とデータ忠実項（観測）を分離して交互に最適化。

**利点**:
- 数学的に収束が保証されやすい
- 各ステップが明確に分離

**欠点**:
- 実装が複雑
- 計算コストが高い

**参考文献**: PnP-ADMM, RED (Regularization by Denoising)

---

## 4. 各解決策の評価（IBPM向け）

| アプローチ | 推奨度 | 理由 |
|-----------|--------|------|
| **3.1 Likelihood Weighting** | ★★★ 推奨 | 実装容易、既存コードとの整合性高い |
| 3.2 Adaptive Normalization | ★★☆ 有力 | 観測点数が動的に変わる場合に有効 |
| 3.3 DDRM (射影法) | ★☆☆ 注意 | 流体データでは物理的整合性を壊すリスク |
| 3.4 Proximal Methods | ★☆☆ 複雑 | 実装コストに見合うメリットが不明 |

---

## 5. 長期的な改善案

### 5.1 GaussianScoreへの正規化オプション追加

```python
class GaussianScore(nn.Module):
    def __init__(self, ..., normalize: bool = False):
        self.normalize = normalize

    def forward(self, x, t, c=None):
        ...
        if self.normalize:
            log_p = -(err ** 2 / var).mean() / 2  # 正規化
        else:
            log_p = -(err ** 2 / var).sum() / 2   # 後方互換
```

### 5.2 Adaptive Normalizationの組み込み

```python
class AdaptiveGaussianScore(nn.Module):
    def forward(self, x, t, c=None):
        eps = self.sde.eps(x, t, c)  # 事前スコア
        g_lik = self._likelihood_grad(x, t, c)  # 尤度勾配

        # 勾配ノルムを自動調整
        scale = eps.norm() / (g_lik.norm() + 1e-8)
        return eps - self.sigma(t) * scale * g_lik
```

---

## 6. 結論

本問題は **「高次元逆問題における勾配支配（Gradient Dominance）」** という
一般的な課題の具体例である。

**短期対応**: Likelihood Weighting（stdスケーリング）で実用上は解決可能

**長期対応**: GaussianScoreに正規化オプションを追加し、
高解像度実験では `normalize=True` をデフォルトにすることを推奨

---

## 7. 実験的検証と新知見（2025年1月追記）

### 7.1 `.mean()` 正規化の実験

セクション5.1で提案した `.sum()` → `.mean()` への変更を実験的に検証した。

**実験条件**: IBPM sub=4（観測点数 N = 160,000）

**結果**:
```
=== 勾配ノルム比較 ===
normalize=False (.sum()): 3.86e+04
normalize=True  (.mean()): 1.60e+02
事前スコアノルム:          1.60e+02

実測比率 = 160,000倍（= 観測点数 N）
```

**観察**: `.mean()` に変更すると、全ての再構成画像がフラット（崩壊）した。

**失敗した再構成結果の保存先**:
```
sda/results/ibpm/evaluate/sparse_normalize_failed/
├── sparse_ground_truth.png
├── sparse_sub2_reconstructed.png   # フラット出力
├── sparse_sub4_reconstructed.png   # フラット出力
├── sparse_sub8_reconstructed.png   # フラット出力
└── sparse_sub16_reconstructed.png  # フラット出力
```

### 7.2 尤度/事前バランスの分析

| 方式 | 尤度勾配ノルム | 事前スコアノルム | 比率 | 再構成結果 |
|------|---------------|-----------------|------|-----------|
| `.sum()` | 3.86e+04 | 1.60e+02 | **241倍** | ✓ 成功 |
| `.mean()` | 1.60e+02 | 1.60e+02 | **1倍** | ✗ 崩壊 |

### 7.3 Subsampleレート別の影響

```
sub= 2: .sum() → 4.30e+04 (事前の269倍), .mean() → 1.60e+02 (事前の1.0倍)
sub= 4: .sum() → 3.86e+04 (事前の241倍), .mean() → 1.60e+02 (事前の1.0倍)
sub= 8: .sum() → 2.40e+04 (事前の150倍), .mean() → 1.60e+02 (事前の1.0倍)
```

### 7.4 重要な発見

1. **「バランス」の誤解**
   - 従来の理解: 尤度と事前が1:1でバランスすれば正常動作
   - 実験結果: **1:1では観測データが無視され、Prior支配となる**

2. **DPS成功の条件**
   - 尤度勾配が事前スコアを **適度に圧倒する（~150-300倍）** 必要がある
   - この「黄金比」により、観測データへの強いガイダンスが維持される

3. **`.sum()` の役割**
   - `.sum()` はバグではなく、**尤度支配を実現するための必須機構**
   - 観測点数 N に比例した勾配により、強いガイダンス信号を生成

4. **`.mean()` が失敗する理由**
   - 勾配が N 分の1に縮小され、尤度の影響力が事前と同等になる
   - 結果として観測データが無視され、フラットな出力に崩壊

### 7.5 暫定対応策（不完全）

**stdスケーリング**（セクション3.1）：

- 基準となる subsample rate（例: sub=4）での ~240倍の比率を維持
- 他の subsample rate でも同じ比率になるよう std を調整
- 数式: `std_target = std_ref × √(N_target / N_ref)`

> **注意**: この対応策は**不完全**であることが判明。
> 詳細はセクション8を参照。

---

## 8. 時刻依存性の発見とDual Scaling（2025年1月追記）

### 8.1 stdスケーリングが機能しない原因

セクション7.5のstdスケーリングを実装しても、結果は改善しなかった。
詳細な数値分析により、**時刻 t による支配項の変化**が根本原因と判明。

**GaussianScoreの分散計算**:
```
var = std² + γ(σ/μ)²
      ───    ────────
      観測項   拡散項(gamma項)
```

### 8.2 gamma項の時刻依存性

| 時刻 t | σ/μ | gamma項 | sub=2 gamma比率 | sub=16 gamma比率 |
|-------|-----|---------|-----------------|------------------|
| 0.01 | 0.04 | 0.00002 | 0% | 3% |
| 0.10 | 0.34 | 0.0012 | 3% | 65% |
| 0.50 | 3.41 | 0.1165 | 74% | 99.5% |
| 0.90 | 58.8 | 34.6 | 99.9% | 100% |

**発見**:
- **t < 0.05**: std²項が支配的 → stdスケーリングが有効
- **t > 0.1**: gamma項が支配的 → stdスケーリングは**無効**
- 拡散過程の大部分（t = 0.1〜1.0）でgamma項が支配

### 8.3 勾配比率の時刻依存性

stdスケーリング適用後の勾配比率（sub=4基準）:

```
t=0.01 (拡散終了): sub2=1.00x, sub8=0.99x, sub16=0.97x  ← 均等 ✓
t=0.10:            sub2=1.08x, sub8=0.76x, sub16=0.38x  ← 差が出始める
t=0.20:            sub2=1.35x, sub8=0.48x, sub16=0.16x  ← 差が拡大
t=0.50 (中盤):     sub2=3.27x, sub8=0.26x, sub16=0.06x  ← 大きく乖離
```

**結論**:
- sub=2: 中盤で勾配が3.3倍強い → 過剰なLikelihood Dominance → **爆発**
- sub=4: 基準通り → **動作**
- sub=8,16: 中盤で勾配が1/4〜1/16 → Prior Dominance → **フラット**

### 8.4 根本原因の数学的説明

尤度勾配の大きさ S は以下に比例:

```
S ∝ N / var = N / (std² + γ(σ/μ)²)
```

**目標**: 全時刻 t で S を N によらず一定に保つ（sub=4と同じ挙動）

そのためには分母 var 全体が N に比例する必要がある:

```
var_target = var_ref × (N_target / N_ref)
```

**問題**: stdスケーリングは std² 項のみをスケール
→ gamma項 γ(σ/μ)² はスケールされない
→ t > 0.1 でgamma項が支配的になると、スケーリングが効かなくなる

### 8.5 解決策: Dual Scaling (std & gamma)

**両方の項をスケーリング**することで、全時刻でvar全体がN比例となる:

```
ratio = N_target / N_ref

# 1. std のスケーリング（既存）
#    std² が ratio 倍になるよう √ratio 倍
std_scaled = std × √ratio

# 2. gamma のスケーリング（新規）
#    gamma項が ratio 倍になるよう ratio 倍
gamma_scaled = gamma × ratio
```

**数学的根拠**:
```
var_new = std_scaled² + gamma_scaled × (σ/μ)²
        = ratio × std² + ratio × gamma × (σ/μ)²
        = ratio × (std² + gamma × (σ/μ)²)
        = ratio × var_ref
```

これにより:
```
S_new = N_target / var_new
      = N_target / (ratio × var_ref)
      = N_target / ((N_target/N_ref) × var_ref)
      = N_ref / var_ref
      = S_ref  ← 全時刻で基準と同じ
```

### 8.6 一般論としての考察

この問題は**「マルチスケールな物理制約におけるパラメータ感度の不均一性」**である。

**拡散モデルの性質**:
- t が大きい時: 大まかな構造を決定
- t が小さい時: 細部を調整

**パラメータの役割**:
- gamma項: 拡散初期〜中期（大まかな構造）で観測データと整合させる
- std項: 拡散終期（細部調整）で観測ノイズと整合させる

**今回の現象**:
- sub=2の爆発、sub=16のフラットは、画像の大まかな構造が決まる中盤（t≈0.5）で顕著
- これは gamma（大域的整合性担当）のスケーリング漏れが原因

---

## 参考文献

- Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems" (DPS)
- Kawar et al., "Denoising Diffusion Restoration Models" (DDRM)
- Song et al., "Score-Based Generative Modeling through SDEs"
- Venkatakrishnan et al., "Plug-and-Play Priors for Model Based Reconstruction" (PnP)
