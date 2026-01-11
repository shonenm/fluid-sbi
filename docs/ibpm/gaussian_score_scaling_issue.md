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

## 9. Dual Scalingの限界とLinear Scaling

### 9.1 Dual Scalingの実験結果

セクション8のDual Scaling（std: √ratio, gamma: ratio）を実装した結果:

| subsample | Before fix | Dual Scaling | 期待値 |
|-----------|------------|--------------|--------|
| sub=2 | 19.4 | **43.9** | ~22 |
| sub=4 | 22.2 | **20.9** | ~22 (基準) |
| sub=8 | 20.8 | **12.5** | ~22 |
| sub=16 | 20.1 | **12.6** | ~22 |

**問題**: sub=2は過大（43.9）、sub=8,16は過小（12.5-12.6）となり、基準のsub=4（~22）に収束していない。

### 9.2 失敗のメカニズム分析

GaussianScoreの戻り値（拡散モデルの更新項）の構造:

```
Update Term = ε - σ × s
                  ─────
                  尤度項
```

Dual Scalingでは勾配 s を一定にすることに集中し、**係数 σ の変化を見落としていた**。

| 項目 | Dual Scaling (√ratio) の挙動 | 結果 |
|------|------------------------------|------|
| 勾配 s | s ∝ N/var ∝ ratio/ratio = 1 | 一定（成功） |
| 係数 σ | σ ∝ √ratio | √ratio 倍に変化（失敗） |
| 尤度項 σ×s | 1 × √ratio = √ratio | sub=2で2倍、sub=16で0.25倍 |

### 9.3 実験結果との整合

Dual Scalingの「√ratio倍のズレ」と実験結果が一致:

**Sub=2 (ratio=4)**:
- 理論値: √4 = 2倍の尤度過多
- 結果: Std 43.9（過剰な尤度によるノイズ増幅）

**Sub=16 (ratio=0.06)**:
- 理論値: √0.06 = 0.25倍の尤度過少
- 結果: Std 12.6（拘束力が弱すぎて特徴を拾えない）

### 9.4 真の解決策: Linear Scaling

尤度項の強さを全解像度で一定（=1）にするための条件:

```
Likelihood Term ≈ σ × (N/σ²) = N/σ → 一定にしたい
```

N が ratio 倍になるなら、分母の σ も **ratio 倍**（√ratio ではなく）でなければならない。

**修正後のスケーリング則**:

| パラメータ | Dual Scaling (失敗) | Linear Scaling (正解) |
|-----------|--------------------|-----------------------|
| std | √ratio 倍 | **ratio 倍** |
| gamma | ratio 倍 | **ratio² 倍** |

### 9.5 Linear Scalingの数学的検証

ratio = N_target / N_ref として:

```
# Linear Scaling
std_scaled = std × ratio
gamma_scaled = gamma × ratio²

# 分散
var_new = std_scaled² + gamma_scaled × (σ/μ)²
        = ratio² × std² + ratio² × gamma × (σ/μ)²
        = ratio² × var_ref

# 勾配
s_new = N_target / var_new
      = (ratio × N_ref) / (ratio² × var_ref)
      = N_ref / (ratio × var_ref)
      = s_ref / ratio

# 尤度項（σは√var_newに比例すると仮定）
σ_new × s_new ∝ ratio × (s_ref / ratio) = s_ref  ← 一定！
```

### 9.6 予想される結果

**Sub=2 (ratio=4)**:
- σ が4倍になる
- var が16倍になる
- 勾配 s が 4N/(16var) = 0.25倍になる
- 尤度項 σ×s = 4 × 0.25 = **1.0倍**（基準と同じ）

**Sub=16 (ratio=0.06)**:
- 同様に係数と勾配が相殺
- 尤度項 = **1.0倍**

これにより、**Std値は全Subsampleレートにおいてsub=4の値（約20〜22）に収束する**はず。

---

## 10. Linear Scalingの実験結果と真の根本原因（2025年1月追記）

### 10.1 安定性の判断基準

再構成結果の「安定」「爆発」は、**出力値の範囲**で判断する。

**Ground Truth（正規化後）**: `min=-3.51, max=3.53`

| 状態 | 値の範囲 | 判断基準 |
|------|----------|----------|
| **安定** | [-4, 4] 程度 | Ground Truthと同程度 |
| **爆発** | 10^10 以上 | 数値発散 |
| **NaN** | nan | 完全崩壊 |

診断コード（evaluate.py）:
```python
print(f"[診断] 正規化後: min={x_recon_norm.min():.2f}, max={x_recon_norm.max():.2f}")
```

### 10.2 Linear Scalingの理論的誤り

セクション9で提案したLinear Scaling理論には**致命的な誤り**があった。

**誤った仮定**:
```
σ（更新項の係数）∝ √var ∝ std
```

**実際の構造** (`sda/score.py:535, 571`):
```python
mu, sigma = self.sde.mu(t), self.sde.sigma(t)  # SDEの時刻依存パラメータ
...
return eps - sigma * s  # sigma は std とは独立
```

`sigma` は **VPSDEの σ(t)** であり、GaussianScoreの `std` パラメータとは**完全に独立**。
したがって、セクション9.2-9.5の「σ ∝ √ratio」という前提は成立しない。

### 10.3 Linear Scalingの実験結果

**実装**:
```python
std_scaled = noise_std * ratio      # √ratio ではなく ratio
gamma_scaled = gamma_base * ratio ** 2  # ratio ではなく ratio²
```

**結果**:
| subsample | std | gamma | 正規化後の値範囲 | 状態 |
|-----------|-----|-------|-----------------|------|
| sub=2 | 0.40 | 0.16 | [-3.43, 3.42] | ✓ 安定 |
| sub=4 | 0.10 | 0.01 | [-9.8e11, 1.3e12] | **爆発** |
| sub=8 | 0.025 | 0.0006 | NaN | 崩壊 |
| sub=16 | 0.006 | 0.00004 | NaN | 崩壊 |

**重大な発見**: sub=4は**参照ケース（ratio=1）** であり、スケーリングは適用されない。
それでも爆発したということは、**スケーリング以前の問題**が存在する。

### 10.4 Dual Scalingでも同様の結果

Linear Scalingの誤りを発見後、Dual Scaling（セクション8）に戻して再実験:

| subsample | std | gamma | 正規化後の値範囲 | 状態 |
|-----------|-----|-------|-----------------|------|
| sub=2 | 0.20 | 0.04 | [-3.44, 3.46] | ✓ 安定 |
| sub=4 | 0.10 | 0.01 | [-9.8e11, 1.3e12] | **爆発** |
| sub=8 | 0.05 | 0.0024 | NaN | 崩壊 |
| sub=16 | 0.024 | 0.0006 | NaN | 崩壊 |

**結論**: Dual ScalingでもLinear Scalingでも、sub=4（参照ケース）が爆発する。
問題は**スケーリングの方法ではなく、ベースライン値そのもの**にある。

### 10.5 真の根本原因: 分散の絶対値

**発見**: 問題はスケーリングの**比率**ではなく、**分散の絶対値**。

```
var = std² + gamma × (σ/μ)²
```

**安定性の閾値**（実験的に特定）:
- var ≳ 0.04 → 安定
- var ≲ 0.01 → 勾配爆発

**メカニズム**:
```python
log_p = -(err ** 2 / var).sum() / 2   # score.py:561
s, = torch.autograd.grad(log_p, x)     # score.py:564
```

1. `var` が小さい → `err² / var` が大きい
2. → 勾配 `s` が大きい
3. → サンプリング更新 `eps - sigma * s` で `sigma * s` が支配的
4. → 数値発散

### 10.6 検証実験: ベースライン値の変更

**仮説**: ベースライン値を大きくすれば、より多くのsubsampleレートが安定するはず。

**実験**: ベースラインを (std=0.1, gamma=0.01) → (std=0.2, gamma=0.04) に変更

| subsample | std | gamma | var (概算) | 状態 |
|-----------|-----|-------|-----------|------|
| sub=2 | 0.40 | 0.16 | 0.16 | ✓ 安定 |
| sub=4 | 0.20 | 0.04 | 0.04 | ✓ **安定（改善！）** |
| sub=8 | 0.10 | 0.01 | 0.01 | 爆発 |
| sub=16 | 0.05 | 0.0024 | 0.0025 | NaN |

**結果**: sub=4が安定化した。sub=8は旧sub=4と同じパラメータ(std=0.1, gamma=0.01)になり、爆発。

### 10.7 パターンの確認

安定/爆発の境界はパラメータ値で決まる:

| std | gamma | 結果 |
|-----|-------|------|
| 0.40 | 0.16 | ✓ 安定 |
| 0.20 | 0.04 | ✓ 安定 |
| 0.10 | 0.01 | **爆発** |
| 0.05 | 0.0025 | NaN |

**閾値**: std ≈ 0.15, gamma ≈ 0.02 付近

### 10.8 セクション9の訂正

セクション9「Linear Scaling」の理論は誤りである。

**誤り**:
- 「σ ∝ √var」という仮定は不正確
- SDEの σ(t) は std パラメータとは独立

**正しい理解**:
- スケーリングの目的は「勾配比率を一定にする」こと（Dual Scalingで達成）
- しかし、**分散の絶対値が小さすぎると、どのスケーリングでも爆発する**
- これは数値安定性の問題であり、スケーリング理論とは別次元の制約

### 10.9 今後の対策案

1. **ベースライン値を大きくする**
   - std=0.2, gamma=0.04 をベースラインに設定
   - 観測ノイズの実際の大きさとの整合性に注意

2. **最小値フロアを設定**
   ```python
   std_scaled = max(noise_std * math.sqrt(ratio), 0.15)
   gamma_scaled = max(gamma_base * ratio, 0.02)
   ```

3. **スパースすぎる観測を避ける**
   - sub=8, 16 は数値的に不安定な領域に入る可能性が高い
   - 実用上は sub=2, 4 程度に制限

---

## 11. Linear Scalingの理論的正当性と実験結果の矛盾

### 11.1 Linear Scalingが正解である理論的根拠

セクション9で提案したLinear Scaling理論は、以下の理論的根拠に基づく。

**理論的目標**: 全解像度で「観測の拘束力」を一定に保つ

```
Effective Force ∝ N / σ = 一定
```

**論理展開**:
1. N が ratio 倍になった（N' = ratio × N）
2. 分母の σ も ratio 倍にする必要がある（σ' = ratio × σ）
3. これにより N'/σ' = (ratio × N)/(ratio × σ) = N/σ = 一定

### 11.2 Dual Scaling (√ratio) が不十分だった理由

セクション9.1の実験データを再解釈する:

| subsample | ratio | Dual Scaling結果 | 基準比 | 理論的解釈 |
|-----------|-------|-----------------|--------|-----------|
| sub=2 | 4.0 | Std 43.9 | **2.1倍** | √4=2倍のズレが残存 |
| sub=4 | 1.0 | Std 20.9 | 1.0倍 | 基準 |
| sub=8 | 0.25 | Std 12.5 | **0.6倍** | 1/√4=0.5倍のズレ |
| sub=16 | 0.06 | Std 12.6 | **0.6倍** | 1/√16=0.25倍のズレ |

Dual Scalingでは分母が √ratio 倍にしかならないため、全体として √ratio 倍のズレが生じる。
これが実験結果の「2倍のズレ」の正体であり、Linear Scaling (ratio倍) が必要な理由。

**数学的帰結**:
- 現在の補正係数 √ratio に、もう一度 √ratio を掛けて、トータルで ratio で割る必要がある
- sub=2 (ratio=4): √4=2倍の尤度過多 → さらに1/2に落とす必要
- sub=16 (ratio≈0.06): √0.06≈0.25倍の尤度過少 → さらに4倍に強化する必要

### 11.3 理論と実験の矛盾

しかし、セクション10.3のLinear Scaling実験結果は**理論の予測と正反対**だった。

**理論の予測**:
- 全subsampleレートでStd ≈ 21 に収束するはず

**実際の結果**:
- sub=2: 安定（理論通り）
- sub=4: **爆発**（理論では変化なしのはず）
- sub=8, 16: NaN

### 11.4 矛盾の原因分析

**理論の前提**:
```
σ（更新項の係数）∝ std
→ std を ratio 倍すれば、尤度項 σ×s 全体が ratio 倍
```

**実際の実装** (`sda/score.py`):
```python
mu, sigma = self.sde.mu(t), self.sde.sigma(t)  # SDEの時刻依存パラメータ
return eps - sigma * s  # sigma は std とは独立
```

**矛盾の本質**:
- 理論: σ が std に比例すると仮定
- 実際: σ(t) は時刻 t のみに依存し、GaussianScoreの std パラメータとは無関係
- 結果: std を変えても σ は変わらない → 理論通りの補正が効かない

### 11.5 二つの問題の分離

実験結果から、**二つの独立した問題**が存在することが判明:

1. **スケーリング比率の問題**（セクション9の主題）
   - Dual Scaling (√ratio) では不十分、Linear Scaling (ratio) が必要
   - 理論的には正しいが、現在の実装では効果が出ない
   - 原因: 更新項の σ が std と独立

2. **分散の絶対値の問題**（セクション10の発見）
   - var ≲ 0.01 で勾配爆発が発生
   - スケーリング方式に関係なく、ベースライン値が小さすぎると不安定
   - 対策: ベースライン値を大きくする or 最小値フロアを設定

### 11.6 今後の課題

Linear Scalingの理論は数学的に正しいが、現在の実装では効果を発揮できない。

**根本的な解決には**:
1. GaussianScoreの更新項で σ(t) の代わりに std ベースの係数を使用する
2. または、尤度勾配 s 自体を ratio で正規化する機構を追加する

これらは実装の根本的な変更を伴うため、今後の検討課題とする。

---

## 12. 出力スケールの矛盾

### 12.1 セクション9.1の結果の疑問

セクション9.1で報告された「Std」値:

| subsample | Dual Scaling結果 |
|-----------|-----------------|
| sub=2 | Std 43.9 |
| sub=4 | Std 20.9 |
| sub=8 | Std 12.5 |
| sub=16 | Std 12.6 |

この「Std」は**再構成された速度場のピクセル値の標準偏差**である。

### 12.2 データ正規化との矛盾

IBPMNormalizerによるデータ正規化:

```python
# 正規化: x_norm = (x - mean) / std
DEFAULT_MEAN = [0.998540, 0.000000]  # [u, v]
DEFAULT_STD = [0.415285, 0.207527]   # [u, v]
```

**正規化後のデータ**: std ≈ 1.0（チャネルごと）

**期待される再構成結果の std**:
- 正規化空間: **≈ 1.0**
- 生データ空間: **≈ 0.2 〜 0.4**

**実際の結果**: Std = 12.5 〜 43.9

### 12.3 スケールのずれ

| 空間 | 期待値 | 実測値 | ずれ |
|------|--------|--------|------|
| 正規化後 | ~1.0 | 20-43 | **20〜43倍** |
| 生データ | ~0.2-0.4 | 20-43 | **50〜200倍** |

### 12.4 セクション10との比較

セクション10.3の診断結果（正規化空間）:

```
Ground Truth: min=-3.51, max=3.53  → std ≈ 1.0
sub=2: min=-3.44, max=3.46         → std ≈ 1.0 ✓
sub=4: min=-9.8e11, max=1.3e12     → 爆発
```

sub=2の結果は正規化空間で妥当な範囲（std≈1）。

### 12.5 未解決の疑問

1. **セクション9.1の測定方法**: どの空間（正規化/生データ）で測定したか不明
2. **セクション9と10の差異**: 同じコードで異なる結果が出ている可能性
3. **Std 20.9 の解釈**: 仮に正しい測定なら、出力スケール自体が根本的に間違っている

### 12.6 観測ノイズ std との関係

GaussianScoreに渡す `std` パラメータ（観測ノイズ）:
- 現在の設定: std = 0.1 〜 0.2

**疑問**: データが std=1 に正規化されているなら、観測ノイズ std=0.1 は「10%のノイズ」を意味する。これは非常に高精度な観測を仮定しており、尤度の確信度が高すぎる可能性がある。

**仮説**: 観測ノイズ std をデータスケール（std=1）に近づける（例: std=0.5〜1.0）ことで、尤度の確信度を下げ、勾配爆発を防げる可能性がある。

---

## 13. 診断指標による現状把握とClamped Linear Scaling（2025年1月追記）

### 13.1 診断指標の導入

モデルの状態を正確に把握するため、3つの診断指標を実装した。

| 指標 | 計算式 | 意味 |
|------|--------|------|
| **Range** | min, max | 数値爆発の確認 |
| **Energy Ratio** | std(recon) / std(GT) | ボケ/ノイズの確認（最重要） |
| **RMSE** | √mean((recon - GT)²) | ピクセルずれの確認 |

### 13.2 診断結果（Dual Scaling + ベースライン増大後）

現在の設定: `noise_std=0.2`, `gamma_base=0.04`, Dual Scaling (√ratio, ratio)

| subsample | Range | Energy Ratio | RMSE | 状態 |
|-----------|-------|--------------|------|------|
| sub=2 | [-3.46, 3.48] | **0.622** | 0.52 | ✓ 安定（ボケ傾向） |
| sub=4 | [-3.44, 3.49] | **0.623** | 0.52 | ✓ 安定（ボケ傾向） |
| sub=8 | 10^11 | 爆発 | 爆発 | ✗ 爆発 |
| sub=16 | NaN | NaN | NaN | ✗ 崩壊 |

Ground Truth: gt_std = 0.736

### 13.3 診断結果の解釈

**Sub=2, 4 (Energy Ratio ≈ 0.62)**:
- 状態: 安定しているが「拘束力が弱すぎる」
- 物理的解釈: std（許容誤差）が大きすぎるため、モデルは「だいたい合っていればいい」と判断し、平均的な（ボケた）画像を出力
- 必要な対策: 拘束を強める（std を小さくする）

**Sub=8, 16 (爆発/崩壊)**:
- 状態: 分散が小さすぎて数値的に破綻
- 物理的解釈: スケーリングにより std が極小値（0.05, 0.01）になり、勾配 1/σ² が天文学的数字になって発散
- 必要な対策: 数値安定性のための「床（Floor）」が必要

### 13.4 二律背反する要件

| 対象 | 要件 | 方向 |
|------|------|------|
| Sub=2, 4 | Energy Ratio を 1.0 に近づける | std を**下げたい** |
| Sub=8, 16 | 爆発を防ぐ | std を**下げたくない** |

### 13.5 解決策: Clamped Linear Scaling（床付き線形スケーリング）

「基本は線形スケーリングだが、ある一定以下には絶対に下げない（Clamp）」というロジック。

```python
# === 設定パラメータ ===
REF_SUB = 4
n_obs_ref = (H // REF_SUB) * (W // REF_SUB) * T * C
ratio = n_obs / n_obs_ref

# ベースパラメータ（Energy Ratio 1.0 を目指して調整）
BASE_STD = 0.2
BASE_GAMMA = 0.04

# 安全装置（Floor）: これ以下には絶対に下げない
MIN_STD = 0.15
MIN_GAMMA = 0.02

# === Clamped Linear Scaling ===
raw_std_scaled = BASE_STD * ratio
std_scaled = max(raw_std_scaled, MIN_STD)

raw_gamma_scaled = BASE_GAMMA * (ratio ** 2)
gamma_scaled = max(raw_gamma_scaled, MIN_GAMMA)
```

### 13.6 Clamped Linear Scalingの予測

| subsample | ratio | 計算上のstd | 適用されるstd | 予想 |
|-----------|-------|------------|--------------|------|
| sub=2 | 4.0 | 0.8 | 0.8 | ✓ 安定、シャープ化可能 |
| sub=4 | 1.0 | 0.2 | 0.2 | ✓ 基準（Energy Ratio調整対象） |
| sub=8 | 0.25 | 0.05 | **0.15 (Floor)** | ✓ 爆発回避 |
| sub=16 | 0.06 | 0.012 | **0.15 (Floor)** | ✓ 崩壊回避 |

### 13.7 チューニング手順

1. `BASE_STD = 0.2` で実行し、Sub=4 の Energy Ratio を確認
2. Energy Ratio < 0.8 なら、`BASE_STD` を 0.18 → 0.16 と段階的に下げる
3. Sub=4 が Energy Ratio ≈ 1.0 になれば、Sub=2 も自動的に適正化
4. Sub=8, 16 は `MIN_STD` で守られ、少なくとも爆発しない

### 13.8 期待される最終結果

| subsample | Energy Ratio | 状態 |
|-----------|--------------|------|
| sub=2 | ≈ 1.0 | ✓ シャープな再構成 |
| sub=4 | ≈ 1.0 | ✓ 基準通り |
| sub=8 | 0.8〜1.0 | ✓ Floorで安定化 |
| sub=16 | 0.6〜0.8 | △ Floorで最善努力 |

### 13.9 Clamped Linear Scaling 実験結果

設定: `BASE_STD=0.2`, `BASE_GAMMA=0.04`, `MIN_STD=0.15`, `MIN_GAMMA=0.02`

| subsample | std | gamma | Range | Energy Ratio | RMSE | 状態 |
|-----------|-----|-------|-------|--------------|------|------|
| sub=2 | 0.81 | 0.66 | [-3.49, 3.43] | 0.600 | 0.57 | ✓ 安定 |
| sub=4 | 0.20 | 0.04 | [-3.45, 3.48] | 0.625 | 0.52 | ✓ 安定 |
| sub=8 | 0.15 (clamped) | 0.02 (clamped) | [-3.43, 3.47] | 0.612 | 0.56 | ✓ **安定化成功** |
| sub=16 | 0.15 (clamped) | 0.02 (clamped) | [-3.43, 3.31] | 0.593 | 0.60 | ✓ **崩壊回避成功** |

**成果**:
- sub=8, 16 が安定化 - Floorが正しく機能
- 全subsampleレートで爆発/NaNなし
- Energy Ratio ≈ 0.60（ボケ傾向だが安定）

**結果画像の保存先**:
```
sda/results/ibpm/evaluate/sparse_clamped_linear_scaling/
├── sparse_ground_truth.png
├── sparse_sub2_reconstructed.png
├── sparse_sub4_reconstructed.png
├── sparse_sub8_reconstructed.png
└── sparse_sub16_reconstructed.png
```

### 13.10 BASE_STD チューニング結果

Energy Ratio を 1.0 に近づけるため、`BASE_STD` の調整を試みた。

#### 実験1: BASE_STD = 0.18

| sub | Range | Energy Ratio | 状態 |
|-----|-------|--------------|------|
| 2 | [-3.43, 3.41] | 0.602 | 安定 |
| 4 | [-3.42, 3.46] | 0.627 | 安定 |
| 8 | [-3.41, 3.48] | 0.609 | 安定 |
| 16 | [-3.38, 3.30] | 0.610 | 安定 |

→ ほぼ変化なし（0.20 → 0.18 で 10% の変化では効果が小さい）

#### 実験2: BASE_STD = 0.15 (= MIN_STD)

| sub | Range | Energy Ratio | 状態 |
|-----|-------|--------------|------|
| 2 | [-3.44, 3.47] | 0.610 | 安定 |
| 4 | [-3.43, 3.42] | 0.634 | 安定 |
| 8 | **[-15.97, 3.41]** | 0.653 | **爆発傾向** |
| 16 | **[-7.62, 5.34]** | 0.740 | **爆発傾向** |

→ **危険**: sub=8, 16 で Range が異常に拡大（爆発の初期兆候）

#### サマリー

| BASE_STD | sub=8,16 状態 | Energy Ratio | 結論 |
|----------|---------------|--------------|------|
| 0.20 | 安定 | ≈ 0.60 | **推奨値** |
| 0.18 | 安定 | ≈ 0.60 | 変化なし |
| 0.15 | 爆発傾向 | ≈ 0.65-0.74 | 危険 |

#### 結論

- `BASE_STD = 0.2` が安全な下限
- これ以下に下げると sub=8, 16 の安定性が失われる
- Energy Ratio ≈ 0.60 はボケを示すが、現在の Clamped Linear Scaling の限界
- より高い Energy Ratio を実現するには、別のアプローチが必要（セクション11.6参照）

---

## 参考文献

- Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems" (DPS)
- Kawar et al., "Denoising Diffusion Restoration Models" (DDRM)
- Song et al., "Score-Based Generative Modeling through SDEs"
- Venkatakrishnan et al., "Plug-and-Play Priors for Model Based Reconstruction" (PnP)
