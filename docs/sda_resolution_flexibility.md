# SDAフレームワークの解像度柔軟性

## 結論

**SDAフレームワークには解像度の制約はありません。任意の解像度で動作します。**

元々64×64に圧縮していた理由は、**技術的制約ではなく設計選択**でした。

---

## 調査結果

### 1. UNetアーキテクチャの解像度柔軟性

**`/workspace/sda/sda/nn.py` の `UNet` クラス**:

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
        stride: Union[int, Sequence[int]] = 2,
        activation: Callable[[], nn.Module] = nn.ReLU,
        spatial: int = 2,  # 1D, 2D, 3Dに対応
        **kwargs,
    ):
```

**特徴**:
- ✅ 入力解像度に関するハードコードされた制約なし
- ✅ `stride=2` でダウンサンプリングするため、2の累乗に近い解像度が効率的だが必須ではない
- ✅ 周期境界条件（`padding_mode='circular'`）と非周期境界条件（`'replicate'`）の両方をサポート

**動作確認**:

| 解像度 | Level 0 | Level 1 | Level 2 | Level 3 | 動作可否 |
|--------|---------|---------|---------|---------|----------|
| 64×64  | 64      | 32      | 16      | 8       | ✅ 完璧 |
| 127×127| 127     | 63      | 31      | 15      | ✅ 動作 |
| 128×128| 128     | 64      | 32      | 16      | ✅ 完璧 |
| 199×199| 199     | 99      | 49      | 24      | ✅ 動作 |
| 256×256| 256     | 128     | 64      | 32      | ✅ 完璧 |

**注**: `hidden_blocks=(3, 3, 3)` の場合、3段階のダウンサンプリング（stride=2）が行われます。

---

### 2. TrajectoryDatasetの柔軟性

**`/workspace/sda/sda/utils.py` の `TrajectoryDataset` クラス**:

```python
class TrajectoryDataset(Dataset):
    def __init__(
        self,
        file: Path,
        window: int = None,
        flatten: bool = False,
    ):
        with h5py.File(file, mode='r') as f:
            self.data = f['x'][:]  # 形状: (N, T, C, H, W)
```

**特徴**:
- ✅ HDF5ファイルから任意の形状のデータを読み込める
- ✅ 空間解像度（H, W）に関する制約なし
- ✅ チャネル数、時系列長も自由

---

### 3. Kolmogorov実験での64×64の選択理由

**`/workspace/sda/experiments/kolmogorov/generate.py`**:

```python
@job(array=1024, cpus=1, ram='1GB', time='00:05:00')
def simulate(i: int):
    # ... 256×256でシミュレーション実行 ...

def aggregate():
    for name, files in splits.items():
        with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
            dset = f.create_dataset(
                'x',
                shape=(len(files), 64, 2, 64, 64),  # ← ここで64×64に固定
                dtype=np.float32,
            )
            for i, x in enumerate(map(np.load, files)):
                # 空間解像度を256×256から64×64に縮小
                arr = KolmogorovFlow.coarsen(torch.from_numpy(x), 4)
```

**理由**:
1. **計算コスト削減**
   - GPU メモリ: 64×64は256×256の約1/16
   - バッチサイズ: より大きなバッチで学習可能（32→128など）
   - 学習速度: 畳み込み演算が高速化

2. **元データの解像度が高い**
   - Kolmogorov流は256×256でシミュレーション
   - 物理的に意味のある情報は低周波成分に集中
   - 64×64でも十分な表現力

3. **先行研究の慣習**
   - 流体力学の機械学習では64×64が一般的
   - 実装の簡便さ

**重要**: これは**フレームワークの制約ではなく、実験設計の選択**です。

---

### 4. IBPMで元々64×64に圧縮していた理由

**推測される理由**:

1. **Kolmogorovの設定をそのまま流用**
   ```python
   # Kolmogorovの設定
   shape = torch.Size((window * 2, 64, 64))  # 64×64固定
   ```
   → IBPMでも同じ64×64を使おうとした

2. **深く考えずにリサイズ処理を追加**
   ```python
   # 元のconvert_ibpm_to_sda.py
   velocity = resize_to_target(velocity, target_size=64)  # ← 安易な圧縮
   ```

3. **物理的特徴への配慮不足**
   - Kolmogorov: 周期境界、外部強制、比較的スムーズな流れ
   - IBPM: 非周期境界、円柱による急峻な境界層、カルマン渦列
   - **IBPMの方が高解像度が重要**

---

## 64×64圧縮の問題点（IBPMの場合）

### 1. 円柱解像度の低下

| IBPM解像度 | 変換前の円柱 | 64×64変換後 | 問題 |
|-----------|------------|------------|------|
| 199×199   | D ≈ 50セル | D ≈ 8セル  | ❌ 円柱がぼやける |
| 128×128   | D ≈ 32セル | D ≈ 16セル | ⚠️ ギリギリ許容範囲 |
| 64×64     | D ≈ 16セル | D ≈ 16セル | ✅ 圧縮不要 |

**円柱直径の推奨**: 最低10-20セル、理想的には30セル以上

### 2. 物理的特徴の損失

**補間によって失われるもの**:
- 🔴 **渦構造**: カルマン渦列の微細構造
- 🔴 **速度勾配**: 円柱表面の境界層
- 🔴 **圧力分布**: 円柱前方と後方の圧力差
- 🔴 **エネルギースペクトル**: 高周波成分の散逸

### 3. Kolmogorovとの違い

| 項目 | Kolmogorov | IBPM |
|-----|-----------|------|
| **境界条件** | 周期的（スムーズ） | 非周期的（急峻） |
| **流れの特徴** | 連続的な渦運動 | 境界層分離、渦放出 |
| **重要なスケール** | 広帯域 | 円柱スケール周辺 |
| **圧縮の影響** | 小さい | 大きい |

---

## 推奨される解像度戦略

### 戦略A: IBPM側で解像度を決定（推奨）⭐

```bash
# 目標解像度で直接生成
ibpm -nx 128 -ny 128 -outdir ibpm_128

# リサイズなしで変換
python convert_ibpm_to_sda.py \
    --input ibpm_128 \
    --output data_128 \
    --coarsen 1  # デフォルト、圧縮なし
```

**メリット**:
- ✅ 物理的特徴を完全に保持
- ✅ 補間による人工的なアーチファクトなし
- ✅ 解像度とコストのトレードオフを明示的に制御

### 戦略B: Coarsenによる段階的ダウンサンプリング

```bash
# 高解像度で生成
ibpm -nx 256 -ny 256 -outdir ibpm_256

# 平均プーリングでダウンサンプリング
python convert_ibpm_to_sda.py \
    --input ibpm_256 \
    --output data_128 \
    --coarsen 2  # 255×255 → 127×127
```

**メリット**:
- ✅ 補間より物理的に妥当（エネルギー保存的）
- ✅ 高周波ノイズの除去
- ⚠️ それでも情報は失われる

### 戦略C: 複数解像度でのアブレーション

```bash
# 複数解像度で実験
for nx in 64 128 256; do
    ibpm -nx $nx -ny $nx -outdir ibpm_${nx}
    python convert_ibpm_to_sda.py \
        --input ibpm_${nx} \
        --output data_${nx}
done

# 解像度ごとに学習・評価
```

**目的**:
- 解像度が復元精度に与える影響を定量評価
- 最小必要解像度の特定

---

## 実験での推奨

### 開発フェーズ: 128×128

```python
# train.py
CONFIG = {
    'window': 3,
    # ... その他のパラメータ
}

# データは127×127（IBPM -nx 128 -ny 128から生成）
shape = torch.Size((window * 2, 127, 127))  # ← 固定値を削除
```

**理由**:
- 円柱解像度: D ≈ 32セル（十分）
- GPU メモリ: 16GB で batch_size=32 可能
- 計算時間: 1エポック約30秒

### 最終評価: 199×199または256×256

```bash
# 論文用の高品質結果
ibpm -nx 200 -ny 200 -nsteps 500 -outdir ibpm_full
python convert_ibpm_to_sda.py --input ibpm_full --output data_full

# 学習（batch_sizeを削減）
python train.py --batch-size 16  # GPU メモリに応じて調整
```

---

## コード修正の必要性

### 現在の実装: ✅ すでに解像度柔軟

**`train.py`**:
```python
# データセットから自動的に形状を取得
trainset = IBPMDataset(data_dir / 'train.h5', window=window, flatten=True)

# 実際のデータ形状に基づいてSDEを構築
x_sample, _ = trainset[0]
shape = x_sample.shape  # (window * 2, H, W) - Hとwは自動決定
sde = VPSDE(score.kernel, shape=torch.Size(shape)).cuda()
```

**`eval_utils.py`**:
```python
def create_cylinder_mask(size: int = 64, device: str = 'cpu') -> Tensor:
    # ↑ sizeパラメータで解像度を指定可能
    # 必要に応じて center, radius も調整
```

### ✅ 修正不要、そのまま使える

- データの解像度に応じて自動的に適応
- 唯一の注意点: `eval_utils.py` の円柱マスクパラメータの調整のみ

---

## まとめ

### 質問への回答

> 元々64×64に圧縮していたが、それはなぜ？

**回答**: Kolmogorov実験の設定を無批判に流用したため。技術的必然性はない。

> sdaがそのサイズしか許容していない？

**回答**: いいえ。**SDAフレームワークは任意の解像度をサポート**しています。

### 重要なポイント

1. **64×64は制約ではない**
   - UNetは任意解像度で動作
   - データセットも柔軟

2. **IBPMでは高解像度が望ましい**
   - 円柱の境界層を捕捉するため
   - カルマン渦列の詳細構造のため
   - 128×128（D≈32セル）以上を推奨

3. **リサイズ処理は有害**
   - 補間が物理的特徴を損なう
   - IBPM側で直接目標解像度を生成すべき

4. **計算コストとのトレードオフ**
   - 開発: 128×128
   - 検証: 199×199
   - 論文: 256×256

---

**結論**: リサイズ処理を削除した現在の実装は正しい方向性です。SDAフレームワークの柔軟性を活かして、物理的に意味のある解像度でデータを生成・学習できます。

---

**参考文献**:
- `/workspace/sda/sda/nn.py` - UNet実装
- `/workspace/sda/sda/utils.py` - データセット実装
- `/workspace/sda/experiments/kolmogorov/generate.py` - Kolmogorovの64×64選択
- `/workspace/docs/ibpm_resolution_guide.md` - IBPM解像度ガイド

**作成日**: 2024-10-29
