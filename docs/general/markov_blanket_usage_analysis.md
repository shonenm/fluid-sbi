# マルコフブランケットの使用状況分析

## 質問

**「マルコフブランケット（MCScoreNetのunfold/fold機能）は実際に使われているのか？」**

---

## 発見: 学習時と推論時で異なる使い方

### 1. 学習時（train.py）: マルコフブランケットは**使われていない** ❌

```python
# train.py: 76-79行目
score = make_score(**CONFIG)  # MCScoreNet全体を作成
# ↓ しかし score.kernel だけを使う！
sde = VPSDE(score.kernel, shape=(window * 2, 64, 64)).cuda()
```

**理由**:
1. `VPSDE` に渡しているのは `score.kernel`（LocalScoreUNet）のみ
2. `MCScoreNet` の `unfold`/`fold` メソッドは完全にバイパスされる
3. データも `flatten=True` で時系列軸を潰している

```python
# train.py: 82行目
trainset = TrajectoryDataset(PATH / 'data/train.h5', window=window, flatten=True)
```

**データ形状の変化**:
```
元のHDF5データ:    (N, T, C, H, W) = (819, 64, 2, 64, 64)
↓ window=5 で切り出し
TrajectoryDataset: (N, 5, 2, 64, 64)
↓ flatten=True
実際の入力:        (N, 10, 64, 64)  # 5時刻×2チャネル=10チャネル
```

時系列軸 `L` が消えているため、`MCScoreNet.unfold/fold` は使えません。

---

### 2. 推論時（notebooks）: マルコフブランケットは**使われている** ✅

```python
# figures.ipynb, figures_bis.ipynb
sde = VPSDE(
    GaussianScore(
        y=y_star,
        A=A,
        std=0.1,
        sde=VPSDE(score, shape=()),  # ← score全体を使用！
        gamma=3e-2,
    ),
    shape=x_star.shape,
).cuda()
```

**重要な違い**:
- `VPSDE(score, shape=())` で `MCScoreNet` **全体** を使用
- `shape=()` は「スカラー」を意味（時系列データではない？）

---

## shape=() の意味を確認

### VPSDE の実装

```python
# sda/score.py: 227-273行目
class VPSDE(nn.Module):
    def __init__(
        self,
        eps: nn.Module,        # スコアネットワーク
        shape: Size,           # データ形状（バッチを除く）
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        self.eps = eps
        self.shape = shape
```

`shape` はデータの形状を指定します。

**Kolmogorovでの使用例**:
1. **学習時**: `shape=(10, 64, 64)` - flattenされた2D画像
2. **推論時**: `shape=()` - スカラー？

---

## GaussianScore の役割

```python
# sda/score.py: 468-536行目
class GaussianScore(nn.Module):
    def __init__(
        self,
        y: Tensor,                            # 観測データ
        A: Callable[[Tensor], Tensor],        # 観測演算子
        std: Union[float, Tensor],            # 観測ノイズ
        sde: VPSDE,                           # 事前分布のスコア ← ここに score を渡す
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
    ):
        self.sde = sde  # 内部でスコアネットワークを使う

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        # 事前分布のスコアを計算
        with torch.enable_grad():
            eps = self.sde.eps(x, t, c)  # ← ここで score が呼ばれる
            # ...
```

`GaussianScore` は内部で `self.sde.eps(x, t, c)` を呼び出します。

これが `MCScoreNet` 全体なら：
```python
MCScoreNet.forward(x, t, c) → unfold → kernel → fold
```

---

## MCScoreNet.forward の実装確認

```python
# sda/score.py: 159-225行目
class MCScoreNet(nn.Module):
    def __init__(self, features: int, context: int = 0, order: int = 1, **kwargs):
        self.order = order
        # 入力は 2*order+1 個の時刻を含む
        self.kernel = build(features * (2 * order + 1), context, **kwargs)

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W) 時系列データ
        t: Tensor,
        c: Tensor = None,
    ) -> Tensor:
        # 時間窓を展開：各時刻について前後order個の状態を連結
        x = self.unfold(x, self.order)  # (B, L-2*order, (2*order+1)*C, H, W)
        # スコア関数を適用
        s = self.kernel(x, t, c)
        # 元の時系列形状に戻す
        s = self.fold(s, self.order)
        return s

    @staticmethod
    def unfold(x: Tensor, order: int) -> Tensor:
        # (B, L, C, H, W) → (B, L-2*order, (2*order+1)*C, H, W)
        x = x.unfold(1, 2 * order + 1, 1)  # 時間軸に沿ってスライディングウィンドウ
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)
        return x
```

**重要**: `unfold` は `x.unfold(1, ...)` で **dim=1（時系列軸）** に対して動作します。

---

## 推論時のデータ形状を確認

notebookでは観測データから時系列全体を復元しようとしています：

```python
# 観測データ y_star の形状
y_star.shape  # 例: (1, 127, 1, 64, 64) または (127, 1, 64, 64)?

# 復元したいデータ x_star の形状
x_star.shape  # 例: (1, 127, 2, 64, 64) または (127, 2, 64, 64)?
```

もし `x` が `(B, L, C, H, W)` 形式なら、`MCScoreNet.unfold` は動作します！

---

## 結論: マルコフブランケットは推論時のみ使用

| フェーズ | 使用するモジュール | データ形状 | unfold/fold | マルコフブランケット |
|---------|-------------------|-----------|-------------|---------------------|
| **学習** | `score.kernel` (LocalScoreUNet) | `(B, 10, 64, 64)` flattenされた画像 | ❌ 使われない | ❌ 無効 |
| **推論** | `score` (MCScoreNet全体) | `(B, L, 2, 64, 64)` 時系列データ | ✅ 使われる | ✅ 有効 |

---

## なぜこの設計なのか？

### 学習時に flatten する理由

1. **効率性**:
   - 時系列全体 `(B, L, C, H, W)` を処理するとメモリ消費が大きい
   - flattenすると `(B, L*C, H, W)` で通常の2D画像として扱える

2. **実装の簡便さ**:
   - 既存の画像用SDEコードをそのまま使える
   - バッチサイズを大きくできる（GPU効率向上）

3. **局所的な時間依存性**:
   - window=5程度なら、各時刻の速度場を並べて2D画像として扱っても学習可能
   - 時系列の因果性は学習データの順序で暗黙的に学習

### 推論時に unfold/fold を使う理由

1. **柔軟な時系列長**:
   - 学習時のwindow（5時刻）に縛られない
   - 任意の長さの時系列を生成・復元可能

2. **時間的整合性**:
   - `unfold` で各時刻に対して前後の文脈を明示的に与える
   - より一貫性のある時系列生成

3. **データ同化**:
   - 観測が飛び飛びでも、時系列全体を一度に復元できる
   - `GaussianScore` で観測制約を加えながらサンプリング

---

## 学習時に unfold/fold を使わない問題点

### 1. 時間的依存性の学習が不十分

**現状の学習**:
```
入力:  [t-2, t-1, t0, t+1, t+2] の5時刻をflattenして10チャネル
出力: ノイズ推定（全時刻同時）
```

これでは各時刻が独立に扱われがちです。

**unfold/foldを使った学習**:
```
入力:  各時刻について前後order個の窓を明示的に作成
処理: 各時刻で前後の文脈を考慮したスコア計算
出力: 時系列として整合性のあるスコア
```

### 2. 学習と推論のミスマッチ

- **学習**: `score.kernel` で flatten された画像として学習
- **推論**: `score` 全体で時系列として使用

このミスマッチが性能に影響する可能性があります。

---

## 推奨される修正（Kolmogorovの場合）

### オプション1: 学習時も MCScoreNet 全体を使う

```python
# train.py の修正
score = make_score(**CONFIG)
# ✅ score 全体を使う
sde = VPSDE(score, shape=(window, 2, 64, 64)).cuda()  # flatten しない形状

# データも flatten しない
trainset = TrajectoryDataset(PATH / 'data/train.h5', window=window, flatten=False)
validset = TrajectoryDataset(PATH / 'data/valid.h5', window=window, flatten=False)
```

**メリット**:
- 学習と推論で同じ処理パス
- 時間的依存性を明示的に学習
- マルコフブランケットの効果を最大化

**デメリット**:
- メモリ消費増加（バッチサイズ削減が必要）
- 学習が遅くなる可能性

### オプション2: 推論時も score.kernel を使う（現在の学習に合わせる）

```python
# notebook での修正
# ❌ 現状
sde=VPSDE(score, shape=())

# ✅ 修正後
sde=VPSDE(score.kernel, shape=(10, 64, 64))  # flatten された形状
```

**メリット**:
- 学習と推論で一貫性
- 既存の学習済みモデルがそのまま使える

**デメリット**:
- マルコフブランケットの利点を放棄
- 長い時系列の生成が難しい

---

## IBPMへの影響

IBPM実装（eval_coarse.py, eval_sparse.py）も同じパターンです：

```python
# eval_coarse.py: 203-208行目
sde_prior = VPSDE(
    GaussianScore(
        # ...
        sde=VPSDE(score, shape=()),  # ← score 全体
    ),
    shape=x_gt.shape,
)
```

**確認事項**:
1. IBPMの学習（train.py）でも `score.kernel` を使っているか？
2. データは flatten されているか？
3. 学習と推論で同じ処理パスを通っているか？

---

## まとめ

### 現状の問題点

1. **学習時**: `score.kernel` のみ使用 → マルコフブランケット無効
2. **推論時**: `score` 全体使用 → マルコフブランケット有効
3. **ミスマッチ**: 学習と推論で異なる処理パス

### 影響

- マルコフブランケットの効果が十分に発揮されていない
- 時間的依存性の学習が不十分な可能性
- 推論時のモデル動作が学習時と異なる

### 推奨

**短期**: 現在のコードでも動作しているので、まずは実験を進める

**中期**: 学習時も `score` 全体を使うように修正を検討
- メモリ効率とのトレードオフを評価
- Ablation studyで性能差を測定

**長期**: より効率的な時系列スコアネットワークの設計を検討

---

**作成日**: 2024-10-29
**分析対象**: Kolmogorov実験（/workspace/sda/experiments/kolmogorov）
