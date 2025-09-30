# Kolmogorovトレーニング実行の出力ディレクトリ構造

## 概要
`sda/experiments/kolmogorov/train.py`を実行すると、`runs/`ディレクトリに学習結果が保存されます。

## ディレクトリ命名規則

### ルートディレクトリ
- **パス**: `/workspace/runs/`
- **説明**: すべてのトレーニング実行の結果を格納する親ディレクトリ

### 各実行ディレクトリ
- **命名形式**: `{wandb_run_name}_{wandb_run_id}`
- **例**:
  - `crimson-cloud-4_6k7cjnx7`
  - `distinctive-spaceship-2_hcpd7r7z`
  - `glorious-sunset-1_hvvqc6tu`
  - `treasured-durian-3_acsay1ip`

#### 命名規則の詳細
- **wandb_run_name**: Weights & Biases (wandb)が自動生成する実行名
  - 形式: `{形容詞}-{名詞}-{番号}`
  - 例: `crimson-cloud-4`, `glorious-sunset-1`
- **wandb_run_id**: wandbが生成する一意の実行ID（8文字のランダム文字列）
  - 例: `6k7cjnx7`, `hvvqc6tu`

## ファイル構造

各実行ディレクトリ内には以下のファイルが生成されます：

### 1. config.json
- **説明**: トレーニング設定を記録したJSONファイル
- **作成タイミング**: トレーニング開始時（train.py:39）
- **内容**: CONFIG辞書の全パラメータ
  ```json
  {
    "window": 5,
    "embedding": 64,
    "hidden_channels": [96, 192, 384],
    "hidden_blocks": [3, 3, 3],
    "kernel_size": 3,
    "activation": "SiLU",
    "epochs": 4096,
    "batch_size": 32,
    "optimizer": "AdamW",
    "learning_rate": 0.0002,
    "weight_decay": 0.001,
    "scheduler": "linear"
  }
  ```

### 2. state.pth
- **説明**: 学習済みモデルの重みを保存したPyTorchファイル
- **作成タイミング**: トレーニング完了後（train.py:68-71）
- **内容**: スコアネットワークのstate_dict
- **サイズ**: 約90MB（モデルアーキテクチャに依存）

## コード内での生成プロセス

### ディレクトリ作成（train.py:36-37）
```python
runpath = PATH / f'runs/{run.name}_{run.id}'
runpath.mkdir(parents=True, exist_ok=True)
```

### 設定ファイル保存（train.py:39）
```python
save_config(CONFIG, runpath)
```

### モデル保存（train.py:68-71）
```python
torch.save(
    score.state_dict(),
    runpath / 'state.pth',
)
```

## 追加情報

### wandbとの連携
- 各実行はwandbプロジェクト`sda-kolmogorov`にログされます
- 以下のメトリクスがwandbに記録されます：
  - `loss_train`: トレーニング損失
  - `loss_valid`: 検証損失
  - `lr`: 学習率
  - `samples`: 生成されたサンプル画像（評価時）

### 並列実行
- `@job`デコレータで`array=3`が指定されているため、最大3つの並列実行が可能
- 各実行は独自のディレクトリに結果を保存するため、競合は発生しません

### ファイルの活用方法
- **config.json**: 実験の再現性確保、ハイパーパラメータの記録
- **state.pth**: 推論時のモデル読み込み、ファインチューニングの開始点

## ディレクトリ構造の例
```
runs/
├── crimson-cloud-4_6k7cjnx7/
│   ├── config.json
│   └── state.pth
├── distinctive-spaceship-2_hcpd7r7z/
│   ├── config.json
│   └── state.pth
├── glorious-sunset-1_hvvqc6tu/
│   ├── config.json
│   └── state.pth
└── treasured-durian-3_acsay1ip/
    ├── config.json
    └── state.pth
```