# Fluid SBI - Score-based Data Assimilation Docker Environment

[Score-based Data Assimilation (SDA)](https://github.com/francois-rozet/sda) のDocker開発環境です。

## 🎯 概要

このプロジェクトは、拡散モデルを用いた大規模動的システム（流体、大気、海洋）の状態軌道推論手法である Score-based Data Assimilation の実験環境を提供します。

## 📁 プロジェクト構成

```
fluid-sbi/
├── .devcontainer/          # VS Code Dev Container設定
├── .gitmodules            # Git submodule設定
├── Dockerfile             # Docker環境設定
├── docker-compose.yml     # Docker Compose設定
├── .env.example           # 環境変数テンプレート
├── scripts/               # 開発支援スクリプト
│   ├── setup.sh          # 初期セットアップ
│   ├── dev.sh            # 開発ヘルパー
│   ├── run-experiments.sh # 実験実行
│   ├── update-sda.sh     # SDA更新
│   └── docker-helper.sh  # Docker操作関数
├── data/                  # データディレクトリ
│   ├── inputs/           # 入力データ
│   ├── outputs/          # 出力データ
│   ├── raw/              # 生データ
│   └── processed/        # 処理済みデータ
├── results/               # 実験結果
│   ├── models/           # 学習済みモデル
│   ├── figures/          # 生成された図表
│   └── logs/             # 実験ログ
└── sda/                   # SDAサブモジュール
    ├── sda/              # SDAコアライブラリ
    ├── experiments/      # 実験スクリプト
    │   ├── lorenz/       # Lorenz attractor実験
    │   └── kolmogorov/   # Kolmogorov flow実験
    └── ...
```

## ⚡ クイックスタート

```bash
# 1. 初期セットアップ
chmod +x scripts/*.sh
./scripts/setup.sh

# 2. 環境変数設定
nano .env  # WANDB_API_KEYを設定

# 3. 開発環境開始
./scripts/dev.sh start

# 4. コンテナに入る
./scripts/dev.sh shell
```

## 🛠️ 開発コマンド

### 環境管理
```bash
./scripts/dev.sh start      # 開発環境開始
./scripts/dev.sh stop       # 環境停止
./scripts/dev.sh restart    # 環境再起動
./scripts/dev.sh status     # 状態確認
./scripts/dev.sh clean      # 環境クリーンアップ
./scripts/dev.sh rebuild    # イメージ再構築
```

### 対話的操作
```bash
./scripts/dev.sh shell      # 開発コンテナに入る
./scripts/dev.sh jupyter    # Jupyter Lab起動
./scripts/dev.sh logs       # ログ確認
./scripts/dev.sh exec 'python --version'  # コマンド実行
```

### 実験実行
```bash
./scripts/run-experiments.sh lorenz       # Lorenz実験
./scripts/run-experiments.sh kolmogorov   # Kolmogorov実験
./scripts/run-experiments.sh both         # 両方実行
./scripts/run-experiments.sh list         # 利用可能な実験一覧
./scripts/run-experiments.sh info lorenz  # 実験情報表示
```

### SDA更新
```bash
./scripts/update-sda.sh     # SDAを最新版に更新
```

## 🔬 実験の詳細

### Lorenz Attractor 実験
カオス理論で有名なLorenzアトラクターを用いた時系列予測実験。

### Kolmogorov Flow 実験  
2次元乱流のKolmogorov flowを用いた流体力学実験。

## 🐳 Docker 環境

### サービス構成
- **sda-dev**: メイン開発環境（PyTorch + JAX + SDA）
- **jupyter**: 専用Jupyter Labサーバー

### ポート
- `8888`: 開発環境のJupyter Lab
- `8889`: 専用Jupyter Labサービス
- `6006`: TensorBoard
- `8000`: 追加開発サーバー

## ⚙️ 設定

### 必須設定 (.env)
```bash
# Weights & Biases API キー（必須）
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=fluid-sbi-experiments
```

### GPU 設定
GPU使用時は `docker-compose.yml` で以下をアンコメント：
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## 💻 VS Code開発

1. Remote-Containers拡張機能をインストール
2. プロジェクトを開く
3. "Reopen in Container"を選択

自動で以下が設定されます：
- Python環境とライブラリ
- Jupyter統合
- デバッグ設定
- コード整形（Black, isort）

## 🐛 トラブルシューティング

### よくある問題と解決法

1. **submoduleが空の場合**
   ```bash
   git submodule update --init --recursive
   ```

2. **jax-cfd インストールエラー**
   ```bash
   ./scripts/dev.sh exec 'pip install --upgrade pip setuptools wheel'
   ./scripts/dev.sh exec 'pip install git+https://github.com/google/jax-cfd'
   ```

3. **WANDB認証エラー**
   ```bash
   ./scripts/dev.sh exec 'wandb login'
   ```

4. **権限エラー**
   ```bash
   chmod +x scripts/*.sh
   ```

5. **GPU認識されない**
   - NVIDIA Docker runtimeがインストールされているか確認
   - docker-compose.ymlのGPU設定をアンコメント

### ログの確認
```bash
./scripts/dev.sh logs        # 開発環境ログ
./scripts/dev.sh logs jupyter # Jupyterログ
```

## 📚 リソース

- **元論文**: [Score-based Data Assimilation](https://arxiv.org/abs/2306.10574)
- **SDAリポジトリ**: https://github.com/francois-rozet/sda
- **Weights & Biases**: https://wandb.ai
- **JAX-CFD**: https://github.com/google/jax-cfd

## 🔗 便利なリンク

開発中にアクセスできるサービス：
- Jupyter Lab: http://localhost:8889
- TensorBoard: http://localhost:6006  
- Weights & Biases: https://wandb.ai

## 🤝 コントリビューション

バグ報告や改善提案は Issue でお知らせください。