# データ・実験管理体制

## 概要

本リポジトリでは、シミュレーションデータ・学習実験・評価結果を一貫して追跡可能にするため、以下の管理システムを導入している。

| 領域 | ツール/モジュール | 目的 |
|------|------------------|------|
| データバージョニング | `sda.data.DataRegistry` | データセットの版管理・チェックサム |
| 実験追跡 | `sda.tracking.RunManager` | 学習runのメタデータ自動収集 |
| 設定管理 | `Pydantic` + YAML | 型安全な設定定義 |
| 結果管理 | run_idベースディレクトリ | 評価結果とrunの紐付け |
| 外部連携 | WandB Artifacts | クラウド上での追跡・共有 |

---

## ディレクトリ構造

```
data/
├── .registry/
│   └── datasets.yaml          # データセットレジストリ（Git管理）
├── raw/                        # 生シミュレーション出力（大容量）
│   └── ibpm_{params}_{date}/
└── processed/                  # 処理済みHDF5（バージョン付き）
    └── {dataset}_v{N}/
        ├── train.h5
        ├── valid.h5
        ├── test.h5
        └── metadata.yaml       # 生成パラメータ + チェックサム

runs/{experiment}/{run_id}/     # 学習run
├── config.yaml                 # 学習設定
├── metadata.yaml               # 自動収集メタデータ
└── state_*.pth                 # モデルチェックポイント

results/{experiment}/{run_id}/  # 評価結果（run_idに紐付け）
└── {mode}_{timestamp}/
    ├── *.png                   # 可視化
    └── report.json             # 評価指標
```

---

## 1. データ管理

### 1.1 DataRegistry

`sda.data.DataRegistry` がデータセットの中央レジストリを管理。

```python
from sda.data import DataRegistry

registry = DataRegistry()

# データセット登録（自動でチェックサム計算・バージョン採番）
dataset = registry.register_dataset(
    name="ibpm_128x128",
    path=Path("data/processed/ibpm_128x128_v1"),
    metadata={
        "resolution": [128, 128],
        "reynolds": 100,
        "source_script": "scripts/convert_ibpm_to_sda.py",
    },
)
print(dataset.full_name)  # "ibpm_128x128_v1"

# 最新バージョン取得
latest = registry.get_latest("ibpm_128x128")

# 特定バージョン取得
v2 = registry.get_version("ibpm_128x128", 2)

# チェックサム検証
results = registry.verify_checksums(latest)
```

### 1.2 データセットメタデータ (`metadata.yaml`)

各データセットディレクトリに自動生成される。

```yaml
name: ibpm_128x128
version: 3
created_at: "2026-01-12T14:30:22Z"
checksums:
  train.h5: "sha256:abc123..."
  valid.h5: "sha256:def456..."
  test.h5: "sha256:789ghi..."
# カスタムメタデータ
resolution: [128, 128]
reynolds: 100
domain:
  x: [-2, 6]
  y: [-2, 2]
source_script: scripts/convert_ibpm_to_sda.py
git_hash: "abc123def"
```

### 1.3 レジストリファイル (`data/.registry/datasets.yaml`)

全データセットの一覧。Gitで管理し、チーム間で共有可能。

```yaml
datasets:
  ibpm_128x128:
    - name: ibpm_128x128
      version: 1
      path: /path/to/data/processed/ibpm_128x128_v1
      checksums: {...}
      created_at: "2026-01-10T..."
    - name: ibpm_128x128
      version: 2
      path: /path/to/data/processed/ibpm_128x128_v2
      checksums: {...}
      created_at: "2026-01-12T..."
  lorenz:
    - ...
updated_at: "2026-01-12T15:00:00Z"
```

---

## 2. 実験（Run）管理

### 2.1 RunManager

`sda.tracking.RunManager` が学習runを統合管理。

```python
from sda.tracking import RunManager

manager = RunManager(
    experiment="ibpm",
    name="vpsde_baseline",
    config=CONFIG,
    dataset_name="ibpm_128x128",  # レジストリから自動取得
    wandb_project="sda-ibpm",
    wandb_tags=["vpsde", "baseline"],
)

with manager.create_run() as run:
    print(run.run_id)      # "ibpm_vpsde_baseline_20260112_143022_abc123"
    print(run.run_dir)     # runs/ibpm/ibpm_vpsde_baseline_20260112_143022_abc123/

    for epoch in range(epochs):
        # 学習...
        run.log_metrics({"loss": loss, "lr": lr}, step=epoch)

        if epoch % 50 == 0:
            run.save_checkpoint(epoch, model.state_dict())

    run.save_checkpoint(epochs, model.state_dict(), is_final=True)
```

### 2.2 自動収集メタデータ (`metadata.yaml`)

```yaml
run_id: "ibpm_vpsde_baseline_20260112_143022_abc123"
experiment: ibpm
name: vpsde_baseline
created_at: "2026-01-12T14:30:22Z"

git:
  commit: "abc123def456..."
  branch: main
  dirty: false

environment:
  hostname: gpu-server-01
  platform: Linux-5.15.0
  python: "3.11.5"
  pytorch: "2.1.0"
  cuda: "12.1"
  gpu: "NVIDIA A100"
  gpu_count: 1

data:
  dataset: ibpm_128x128
  version: 3
  path: /path/to/data/processed/ibpm_128x128_v3
  checksums:
    train.h5: "sha256:abc123..."

config:
  epochs: 2000
  batch_size: 2
  learning_rate: 0.0001
  # ... 全学習設定

wandb:
  run_id: "abc123"
  project: sda-ibpm
  url: "https://wandb.ai/..."
```

### 2.3 Run ID命名規則

```
{experiment}_{name}_{timestamp}_{wandb_id}

例: ibpm_vpsde_baseline_20260112_143022_abc123
    ├── ibpm              : 実験種別
    ├── vpsde_baseline    : run名
    ├── 20260112_143022   : 開始時刻
    └── abc123            : WandB ID（8文字）
```

---

## 3. 設定（パラメータ）管理

### 3.1 学習設定 (`config.yaml`)

各runディレクトリに保存。再現に必要な全パラメータを含む。

```yaml
# アーキテクチャ
window: 16
cond_channels: 2
embedding: 64
hidden_channels: [64, 128, 256]
hidden_blocks: [2, 2, 2]
kernel_size: 3
activation: SiLU

# 学習
epochs: 2000
batch_size: 2
optimizer: AdamW
learning_rate: 0.0001
weight_decay: 0.001
scheduler: cosine
```

### 3.2 Pydanticによる型安全設定

`sda/sda/config.py` で設定スキーマを定義。

```python
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    # ... バリデーション付き
```

---

## 4. 評価結果管理

### 4.1 Run連携出力

評価結果はrun_idに紐付いたディレクトリに保存。

```python
# evaluate.py での使用
from sda.paths import get_run_results_dir

# run_idから結果ディレクトリを取得
results_dir = get_run_results_dir("ibpm", run_id, "sparse")
# -> results/ibpm/{run_id}/sparse_20260112_150000/
```

### 4.2 結果ディレクトリ構造

```
results/ibpm/ibpm_vpsde_baseline_20260112_143022_abc123/
├── evaluate_20260112_150000/
│   ├── data/
│   │   └── data_train_samples.png
│   ├── sample/
│   │   └── sample_uncond_*.png
│   ├── sparse/
│   │   └── sparse_sub*_reconstructed.png
│   └── trajectory/
│       └── trajectory.png
└── generalization_20260112_160000/
    ├── grid_offset/
    ├── perturbation/
    ├── geometry/
    ├── reynolds/
    └── report.json
```

---

## 5. トレーサビリティ

### 5.1 実験→データの追跡

```
metadata.yaml (run)
└── data.dataset: "ibpm_128x128"
└── data.version: 3
└── data.checksums: {...}
    ↓
datasets.yaml (registry)
└── ibpm_128x128[version=3]
    └── path, checksums, metadata
```

### 5.2 結果→実験の追跡

```
results/ibpm/{run_id}/evaluate_*/
    ↓ run_id
runs/ibpm/{run_id}/
├── config.yaml      # 学習設定
├── metadata.yaml    # データ版、環境、git
└── state_final.pth  # モデル
```

### 5.3 WandB連携

- **Config Artifact**: `config.yaml` + `metadata.yaml`
- **Model Artifact**: チェックポイント（epoch毎 + final）
- 全てrun_idで紐付け

---

## 6. ワークフロー

### 6.1 データ生成→登録

```bash
# 1. シミュレーション実行
python experiments/kolmogorov/generate.py

# 2. 自動でレジストリに登録
# -> data/.registry/datasets.yaml に追加
# -> data/processed/kolmogorov_v1/metadata.yaml 生成
```

### 6.2 学習→評価

```bash
# 1. 学習（自動でrun作成・メタデータ収集）
python experiments/ibpm/train.py
# -> runs/ibpm/{run_id}/ 作成
# -> WandBにconfig/model artifact登録

# 2. 評価（結果をrun_idに紐付け）
python experiments/ibpm/evaluate.py --run-dir runs/ibpm/{run_id}
# -> results/ibpm/{run_id}/evaluate_*/ に出力
```

### 6.3 既存データの移行

```bash
# 既存のdata/ディレクトリをスキャンしてレジストリに登録
python scripts/register_existing_datasets.py --dry-run  # 確認
python scripts/register_existing_datasets.py            # 実行
```

---

## 7. 設計判断

| 判断 | 採用 | 不採用 | 理由 |
|------|------|--------|------|
| データバージョニング | チェックサム+YAML | DVC | シンプル、Git管理可能 |
| 設定管理 | Pydantic | Hydra | 既存コードとの互換性 |
| 実験追跡 | WandB Artifacts強化 | MLflow | 既に使用中 |
| レジストリ形式 | YAML | SQLite | 可読性、Git差分 |

---

## 8. 関連ファイル

| ファイル | 役割 |
|----------|------|
| `sda/sda/data/registry.py` | DataRegistry, DatasetVersion |
| `sda/sda/tracking/run_manager.py` | RunManager, RunContext |
| `sda/sda/paths.py` | パス関数 (get_*_dir) |
| `scripts/register_existing_datasets.py` | 既存データ登録 |
| `data/.registry/datasets.yaml` | データセット一覧 |
