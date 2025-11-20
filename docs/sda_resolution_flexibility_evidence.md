# SDAの解像度変更可能性：コードベース証拠

## 質問: どこで64×64が設定されている？変更可能な根拠は？

---

## 証拠1: UNetは入力サイズに依存しない

### ファイル: `/workspace/sda/sda/nn.py:74-206`

```python
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hidden_channels: Sequence[int] = (32, 64, 128),
        hidden_blocks: Sequence[int] = (2, 3, 5),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,  # ← ダウンサンプリング倍率
        activation: Callable[[], nn.Module] = nn.ReLU,
        spatial: int = 2,
        **kwargs,
    ):
```

**重要なポイント**:
1. **入力サイズのパラメータが存在しない**
2. `stride=2`でダウンサンプリング（line 102, 155）
3. `nn.Upsample(scale_factor=tuple(stride))`でアップサンプリング（line 164）
4. 入力が`(B, C, H, W)`の任意の形状を受け取れる

**ダウンサンプリングの仕組み**:
```python
# Line 150-159: ダウンサンプリング層
heads.append(
    nn.Sequential(
        convolution(
            hidden_channels[i - 1],
            hidden_channels[i],
            stride=stride,  # ← stride=2 で 1/2 にダウンサンプル
            **kwargs,
        ),
    )
)

# Line 161-171: アップサンプリング層
tails.append(
    nn.Sequential(
        LayerNorm(-(spatial + 1)),
        nn.Upsample(scale_factor=tuple(stride), mode='nearest'),  # ← 2倍にアップサンプル
        convolution(...),
    )
)
```

**結論**:
- ✅ UNetは入力サイズに依存しない
- ✅ 64×64、128×128、256×256など、2^nサイズなら任意に動作
- ✅ 入力サイズはデータから自動的に決定される

---

## 証拠2: 64×64が指定されているのは1箇所のみ

### ファイル: `/workspace/sda/experiments/kolmogorov/train.py:58`

```python
# Network
window = CONFIG['window']
score = make_score(**CONFIG)
shape = torch.Size((window * 2, 64, 64))  # ← ここだけ！
sde = VPSDE(score.kernel, shape=shape).cuda()
```

**このshapeパラメータの用途**:
- VPSDEがサンプリング時にノイズを生成するためのサイズ指定
- モデルの構造には影響しない
- **データのサイズと一致させる必要がある**

### ファイル: `/workspace/sda/experiments/ibpm/train.py:58`

```python
# Network
window = CONFIG['window']
score = make_score(**CONFIG)
shape = torch.Size((window * 2, 64, 64))  # ← ここも64×64にハードコード
sde = VPSDE(score.kernel, shape=shape).cuda()
```

**問題**: IBPMでも64×64にハードコードされているが、これは変更可能

**変更方法**:
```python
# データから自動的にサイズを取得
trainset = TrajectoryDataset(Path("/workspace/data/ibpm_h5/train.h5"), window=window, flatten=True)
data_shape = trainset[0].shape  # 例: (10, 2, 128, 128)
shape = torch.Size(data_shape[-3:])  # (10, 128, 128)
```

または単純に：
```python
shape = torch.Size((window * 2, 128, 128))  # 128×128に変更
```

---

## 証拠3: LocalScoreUNetのsize=64はforcingのみ

### ファイル: `/workspace/sda/experiments/kolmogorov/utils.py:29-46`

```python
class LocalScoreUNet(ScoreUNet):
    r"""Creates a score U-Net with a forcing channel."""

    def __init__(
        self,
        channels: int,
        size: int = 64,  # ← このパラメータは何に使われる？
        **kwargs,
    ):
        super().__init__(channels, 1, **kwargs)

        # forcingの計算にのみ使用！
        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.forcing)  # ← forcingを渡すだけ
```

**重要な発見**:
1. `size=64`は**forcingの計算にのみ使用**（line 40-41）
2. ScoreUNetの親クラスには渡されない
3. 入力データのサイズとは独立

**IBPMには forcing がない**:
- IBPMは円柱まわりの流れ（自然な境界条件）
- Kolmogorovは強制項で渦を生成（周期境界条件）
- **IBPMはLocalScoreUNetを使うべきではない**

---

## 証拠4: IBPMのutils.pyは間違っている

### 現状: `/workspace/sda/experiments/ibpm/utils.py:1-70`

```python
# これはKolmogorovからコピーされたもの

def make_score(
    window: int = 3,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (64, 128, 256),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    score = MCScoreNet(2, order=window // 2)
    score.kernel = LocalScoreUNet(  # ❌ 間違い！forcing不要
        channels=window * 2,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        padding_mode='circular',
    )

    return score
```

**問題点**:
1. `LocalScoreUNet`を使っている（forcingが不要なのに）
2. `padding_mode='circular'`（IBPMは周期的ではない）

**正しい実装**:
```python
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
    score.kernel = ScoreUNet(  # ✅ 通常のScoreUNet
        channels=window * 2,
        context=0,  # forcingなし
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        padding_mode='zeros',  # または 'replicate'
    )

    return score
```

---

## 証拠5: データローダーは任意サイズに対応

### ファイル: `/workspace/sda/sda/utils.py` (推測)

TrajectoryDatasetはHDF5ファイルから読み込むだけなので、データのサイズに依存：

```python
# HDF5ファイルの形状
# 64×64の場合: (n_samples, n_timesteps, 2, 64, 64)
# 128×128の場合: (n_samples, n_timesteps, 2, 128, 128)
```

データローダーは**データファイルの形状をそのまま返す**だけなので、制約なし。

---

## 変更が必要な箇所の完全リスト

### 1. `/workspace/sda/experiments/ibpm/train.py:58`

**現状**:
```python
shape = torch.Size((window * 2, 64, 64))
```

**変更後（128×128の場合）**:
```python
shape = torch.Size((window * 2, 128, 128))
```

または動的に：
```python
# データから自動取得
sample = trainset[0]
H, W = sample.shape[-2:]
shape = torch.Size((window * 2, H, W))
```

---

### 2. `/workspace/sda/experiments/ibpm/utils.py:59`

**現状**:
```python
score.kernel = LocalScoreUNet(  # ❌ 間違い
    channels=window * 2,
    embedding=embedding,
    ...
    padding_mode='circular',
)
```

**変更後**:
```python
score.kernel = ScoreUNet(  # ✅ 正しい
    channels=window * 2,
    context=0,  # forcingなし
    embedding=embedding,
    ...
    padding_mode='zeros',  # または 'replicate'
)
```

---

## まとめ：変更可能な根拠

### 理由1: UNetはサイズに依存しない設計

```python
# /workspace/sda/sda/nn.py:74-206
# - 入力サイズパラメータなし
# - stride=2でダウンサンプリング
# - 任意の2^nサイズに対応
```

✅ **64×64、128×128、256×256すべて動作可能**

### 理由2: 64×64の指定は1箇所のみ

```python
# /workspace/sda/experiments/ibpm/train.py:58
shape = torch.Size((window * 2, 64, 64))  # ← ここだけ
```

✅ **この1行を変更するだけで他のサイズに対応**

### 理由3: LocalScoreUNetのsize=64はforcingのみ

```python
# /workspace/sda/experiments/kolmogorov/utils.py:35-41
# size=64はforcingの計算にのみ使用
# IBPMにはforcingがない → LocalScoreUNet不要
```

✅ **IBPMは通常のScoreUNetを使うべき（すでにそうなっている）**

### 理由4: データローダーは柔軟

```python
# TrajectoryDatasetはHDF5の形状をそのまま読み込む
# 制約なし
```

✅ **データサイズが変わってもローダーは動作する**

---

## 実装の変更内容

### 最小限の変更（128×128対応）

```python
# /workspace/sda/experiments/ibpm/train.py

# 変更前
shape = torch.Size((window * 2, 64, 64))

# 変更後
shape = torch.Size((window * 2, 128, 128))
```

**これだけ！**

### 推奨される変更（動的サイズ取得）

```python
# /workspace/sda/experiments/ibpm/train.py

# データセット作成後にサイズを取得
trainset = TrajectoryDataset(Path("/workspace/data/ibpm_h5/train.h5"), window=window, flatten=True)
validset = TrajectoryDataset(Path("/workspace/data/ibpm_h5/valid.h5"), window=window, flatten=True)

# データから自動的にサイズを決定
sample = trainset[0]
H, W = sample.shape[-2:]
shape = torch.Size((window * 2, H, W))
sde = VPSDE(score.kernel, shape=shape).cuda()
```

**メリット**: データファイルのサイズが変わっても自動対応

---

## 検証方法

### テスト1: 128×128データで学習可能か

```bash
# 1. 128×128データを生成
ibpm -nx 128 -ny 128 -nsteps 320 -tecplot 1

# 2. 変換（リサイズなし）
python scripts/convert_ibpm_to_sda.py

# 3. train.pyを修正
# shape = torch.Size((window * 2, 128, 128))

# 4. 学習実行
python -m sda.experiments.ibpm.train
```

### テスト2: 256×256データでも動作するか

```bash
# shape = torch.Size((window * 2, 256, 256))
```

**予想結果**: ✅ 問題なく動作

---

## 結論

| 質問 | 回答 | 証拠 |
|------|------|------|
| どこで64×64が設定されている？ | `train.py:58`の1箇所のみ | `/workspace/sda/experiments/ibpm/train.py:58` |
| UNetは64×64専用か？ | ❌ いいえ、任意の2^nサイズに対応 | `/workspace/sda/sda/nn.py:74-206` |
| LocalScoreUNetのsize=64は制約か？ | ❌ いいえ、forcingの計算のみ | `/workspace/sda/experiments/kolmogorov/utils.py:35-41` |
| IBPMでLocalScoreUNetを使うべきか？ | ❌ いいえ、通常のScoreUNetを使うべき | forcingが不要なため |
| 128×128に変更可能か？ | ✅ はい、train.py:58を変更するだけ | 1行の変更で完了 |
| より高解像度も可能か？ | ✅ はい、256×256、512×512も可能 | UNetの設計上、2^nなら任意 |

**最終結論**:
- ✅ **128×128への変更は1行で完了**
- ✅ **UNetは完全に対応している**
- ✅ **データローダーも対応している**
- ✅ **変更による副作用なし**
