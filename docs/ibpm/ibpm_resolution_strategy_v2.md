# IBPM解像度戦略の再検討（変換なし前提）

## ユーザー要件

**重要制約**: 変換（ダウンサンプリング・補間）は情報が失われるため使用しない

---

## 問題の本質

現状のワークフロー:
```
IBPM (200×200) → 粗視化 (49×49) → 補間 (64×64) → SDA
                 ❌ 情報損失     ❌ 情報損失
```

**目標**: 変換なしで直接使えるデータを生成する

---

## 新戦略: SDA側を拡張する

### 重要な発見

SDAのコード調査結果：

1. **UNetは空間解像度に柔軟**
   ```python
   # /workspace/sda/sda/nn.py: UNet
   def __init__(self, spatial: int = 2, stride: int = 2, ...)
   # → ダウンサンプリングはstride=2
   # → 入力サイズは2^nであれば任意（64, 128, 256, ...）
   ```

2. **LocalScoreUNetの`size=64`はforcingのみ**
   ```python
   # /workspace/sda/experiments/kolmogorov/utils.py
   class LocalScoreUNet(ScoreUNet):
       def __init__(self, channels: int, size: int = 64, **kwargs):
           # sizeはフォーシング項の計算にのみ使用
           domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
           forcing = torch.sin(4 * domain).expand(1, size, size).clone()
   ```

3. **IBPMには埋め込み境界法のforcingがない**
   - Kolmogorovの`LocalScoreUNet`は使わない
   - 通常の`ScoreUNet`を使用
   - → `size`パラメータが不要

---

## 推奨方針: IBPMを128×128で実行 + SDAを128×128対応

### ステップ1: IBPMを128×128で実行（変換なし）

```bash
# 1. IBPMを128×128で実行
ibpm -nx 128 -ny 128 \
     -length 4.0 -xoffset -2.0 -yoffset -2.0 \
     -geom /opt/ibpm/examples/cylinder.geom \
     -Re 100 -dt 0.015625 \
     -nsteps 320 \
     -tecplot 1 \
     -force 1 \
     -outdir /workspace/data/ibpm_128
```

**物理的妥当性**:
- グリッド解像度: 128×128
- 格子幅: dx = 4.0 / 128 = 0.03125
- 円柱解像度: 1.0 / 0.03125 = **32点/D** ✅ (最低限許容)
- CFL数: 1.0 × 0.015625 / 0.03125 = **0.5** ✅ (安定)

### ステップ2: 変換スクリプトを修正（補間・粗視化なし）

```python
# scripts/convert_ibpm_to_sda.py の修正
# 128×128 Tecplot → 128×128 HDF5（そのまま保存）

def process_ibpm_output(input_dir, output_dir, train_ratio=0.8):
    """128×128データをそのまま変換（リサイズなし）"""

    # Tecplotファイル読み込み
    velocity = read_tecplot(filepath)  # shape: (2, 127, 127)

    # ❌ 粗視化なし
    # ❌ 補間なし

    # ✅ そのまま保存
    train_data.append(velocity)
```

### ステップ3: SDAのIBPM実験を128×128対応

```python
# /workspace/sda/experiments/ibpm/utils.py

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
    score.kernel = ScoreUNet(  # LocalScoreUNetではなくScoreUNet
        channels=window * 2,
        context=0,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        padding_mode='circular',  # or 'replicate' for IBPM
    )
    return score
```

**変更点**:
- `LocalScoreUNet` → `ScoreUNet`（forcingなし）
- データサイズは自動的に入力から決定される
- 128×128でも256×256でも動作する

---

## アーキテクチャの互換性検証

### UNetのダウンサンプリング階層

```python
# hidden_channels = (96, 192, 384)
# hidden_blocks = (3, 3, 3)
# stride = 2
```

**64×64の場合**:
```
入力: 64×64
Level 1: 64×64 → 32×32 (stride=2でダウンサンプル)
Level 2: 32×32 → 16×16 (stride=2でダウンサンプル)
Level 3: 16×16 → 8×8 (stride=2でダウンサンプル)
Level 2: 8×8 → 16×16 (アップサンプル)
Level 1: 16×16 → 32×32 (アップサンプル)
出力: 32×32 → 64×64 (アップサンプル)
```

**128×128の場合**:
```
入力: 128×128
Level 1: 128×128 → 64×64 (stride=2でダウンサンプル)
Level 2: 64×64 → 32×32 (stride=2でダウンサンプル)
Level 3: 32×32 → 16×16 (stride=2でダウンサンプル)
Level 2: 16×16 → 32×32 (アップサンプル)
Level 1: 32×32 → 64×64 (アップサンプル)
出力: 64×64 → 128×128 (アップサンプル)
```

✅ **完全に互換性あり** - 入力サイズが2^nであればどのサイズでも動作

---

## 代替案: 64×64で物理的に妥当な設定を探す

変換なしで64×64を使いたい場合の選択肢：

### Option A: 計算領域を縮小

```bash
ibpm -nx 64 -ny 64 \
     -length 3.2 -xoffset -1.6 -yoffset -1.6 \
     -geom cylinder.geom -Re 100
```

**設定**:
- 計算領域: 3.2 × 3.2 (3.2D × 3.2D)
- 格子幅: dx = 0.05
- 円柱解像度: **20点/D** ⚠️ (ギリギリ)
- 領域サイズ: **3.2D** ⚠️ (やや狭い、推奨は4D以上)

**懸念点**:
- 境界条件の影響が大きい
- 円柱後方の渦構造が境界に干渉する可能性
- 物理的妥当性に疑問

### Option B: 非正方形グリッド

```bash
ibpm -nx 64 -ny 64 \
     -length 4.0 -xoffset -2.0 -yoffset -2.0 \
     -geom cylinder.geom -Re 100
```

**設定**:
- 計算領域: 4.0 × 4.0
- 格子幅: dx = 0.0625
- 円柱解像度: **16点/D** ❌ (不十分)

**結論**: 物理的に不適切

---

## 推奨方針のまとめ

### 第1推奨: **128×128を採用（変換なし）**

**ワークフロー**:
```
IBPM 128×128 → Tecplot 127×127 → HDF5 127×127 → SDA学習 (128×128対応)
                                 ✅ 変換なし
```

**メリット**:
1. ✅ 変換なし（情報損失なし）
2. ✅ 物理的に妥当（32点/D）
3. ✅ SDAアーキテクチャと完全互換
4. ✅ 実装が簡単

**デメリット**:
1. ⚠️ 計算コストが64×64の4倍（ただし許容範囲）
2. ⚠️ メモリ使用量が増加

### 第2推奨: **より高解像度（256×256, 512×512）も検討可能**

SDAのUNetは2^nサイズに対応しているため、計算リソースが許せば：

```bash
# 256×256で高精度計算
ibpm -nx 256 -ny 256 -Re 100 -dt 0.0078125
# 円柱解像度: 64点/D ✅✅ (高精度)
```

---

## 実装の優先順位

### Phase 1: 128×128で実装（推奨）

1. 変換スクリプトを修正（リサイズ処理を削除）
2. IBPMを128×128で実行
3. SDAのIBPM実験でデータ読み込み確認
4. 学習実行

### Phase 2: 必要に応じて64×64も検討

もし128×128が計算リソース的に厳しい場合：
- 計算領域を3.2×3.2に縮小（Option A）
- 物理的妥当性を検証
- 学習データとして使えるか評価

---

## Kolmogorovとの比較

| 項目 | Kolmogorov | IBPM (推奨) |
|------|-----------|------------|
| グリッド解像度 | 64×64 | 128×128 |
| 境界条件 | 周期境界 | 自然流出 |
| フォーシング | あり | なし |
| ScoreNet | LocalScoreUNet | ScoreUNet |
| データサイズ | 小さい | 4倍 |
| 物理精度 | 十分 | 許容範囲 |

---

## 次のステップ

1. **変換スクリプト修正**
   - リサイズ処理を削除
   - 128×128をそのまま保存

2. **IBPM実行**
   - 128×128, dt=0.015625で実行
   - 十分な時系列データを生成（nsteps=1000以上推奨）

3. **SDA学習**
   - 128×128データで学習
   - Kolmogorovと同様の実験を実施

4. **検証**
   - 生成された流れ場の物理的妥当性
   - 渦構造の再現性
   - スパース観測での再構成精度

---

## まとめ

**結論**:
- ❌ 64×64直接計算は物理的に不適切
- ❌ ダウンサンプリング・補間は情報損失
- ✅ **128×128を採用し、変換なしで使用**
- ✅ SDAは128×128に完全対応可能

**理由**:
1. 変換による情報損失を回避
2. 物理的に妥当な解像度（32点/D）
3. SDAアーキテクチャと完全互換
4. 実装がシンプル
