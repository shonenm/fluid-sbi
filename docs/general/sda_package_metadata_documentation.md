# SDA パッケージメタデータ (sda.egg-info) 詳細解説

## 目次
1. [概要](#概要)
2. [egg-infoとは](#egg-infoとは)
3. [ディレクトリ構造](#ディレクトリ構造)
4. [各ファイルの詳細](#各ファイルの詳細)
5. [パッケージビルドシステム](#パッケージビルドシステム)
6. [依存関係管理](#依存関係管理)
7. [バージョン管理](#バージョン管理)
8. [インストールと配布](#インストールと配布)
9. [トラブルシューティング](#トラブルシューティング)

## 概要

`sda.egg-info` ディレクトリは、SDA (Score-based Diffusion for Assimilation) パッケージのメタデータを含む、Pythonパッケージの配布情報を管理する重要なディレクトリです。このディレクトリは `pip install -e .` コマンド実行時に自動的に生成されます。

### パッケージ基本情報
- **パッケージ名**: sda
- **バージョン**: 0.0.1 (pyproject.toml) / 1.0.0 (setup.py) ※不整合あり
- **Python要件**: Python 3.10以上
- **ライセンス**: MIT License
- **開発者**: François Rozet, Gilles Louppe

## egg-infoとは

`.egg-info` ディレクトリは、Pythonパッケージのメタデータを格納する標準的な形式です。

### 主な役割
1. **パッケージ発見**: Pythonがインポート可能なパッケージを識別
2. **依存関係管理**: 必要なライブラリの追跡
3. **バージョン管理**: パッケージのバージョン情報の保持
4. **配布情報**: PyPIへのアップロードに必要な情報

### 生成タイミング
- `pip install -e .` (開発モード) 実行時
- `python setup.py install` 実行時
- `pip install .` 実行時
- `python -m build` によるビルド時

## ディレクトリ構造

```
/workspace/sda/sda.egg-info/
├── PKG-INFO              # パッケージメタデータ
├── SOURCES.txt           # ソースファイルリスト
├── dependency_links.txt  # 外部依存リンク
├── requires.txt          # 依存パッケージリスト
└── top_level.txt         # トップレベルモジュール名
```

## 各ファイルの詳細

### 1. PKG-INFO

パッケージの主要メタデータを含むファイル。

```
Metadata-Version: 2.4          # メタデータ仕様バージョン
Name: sda                       # パッケージ名
Version: 0.0.1                  # バージョン番号
Summary: Submodule package...   # 簡潔な説明
Requires-Python: >=3.10         # Python要件
Description-Content-Type: text/markdown  # 説明の形式
License-File: LICENSE           # ライセンスファイル
Requires-Dist: numpy           # 依存パッケージ
Requires-Dist: torch
Requires-Dist: h5py
Requires-Dist: dawgz
```

**長い説明セクション**:
- README.mdの内容が含まれる
- プロジェクトの詳細説明
- 論文情報と引用方法
- インストール手順

### 2. SOURCES.txt

パッケージに含まれるすべてのファイルのリスト。

```
LICENSE                         # ライセンスファイル
README.md                       # プロジェクト説明
pyproject.toml                  # 新形式の設定ファイル
setup.py                        # 従来の設定ファイル
experiments/__init__.py         # experimentsパッケージ
experiments/kolmogorov/*.py     # Kolmogorov実験
sda/__init__.py                 # メインパッケージ
sda/mcs.py                      # マルコフ連鎖システム
sda/nn.py                       # ニューラルネットワーク
sda/score.py                    # スコアモデル
sda/utils.py                    # ユーティリティ
sda.egg-info/*                  # メタデータファイル
```

### 3. requires.txt

実行時の依存パッケージリスト（最小構成）。

```
numpy       # 数値計算
torch       # 深層学習フレームワーク
h5py        # HDF5ファイル操作
dawgz       # Slurm ジョブスケジューラ
```

### 4. top_level.txt

インポート可能なトップレベルモジュール。

```
sda
```

これにより、`import sda` でパッケージをインポートできます。

### 5. dependency_links.txt

外部の依存関係リンク（通常は空）。カスタムパッケージリポジトリのURLなどを指定する際に使用。

## パッケージビルドシステム

### 二重構成の問題

現在、SDAパッケージは**2つの設定ファイル**が存在します：

#### 1. pyproject.toml (新形式 - PEP 518)
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sda"
version = "0.0.1"  # ← バージョン 0.0.1
dependencies = [
  "numpy",
  "torch",
  "h5py",
  "dawgz",
]
```

#### 2. setup.py (従来形式)
```python
setuptools.setup(
    name='sda',
    version='1.0.0',  # ← バージョン 1.0.0 (不整合!)
    packages=setuptools.find_packages(),
)
```

### 推奨される修正

バージョンの不整合を解消するため、以下のいずれかを推奨：

**オプション1: pyproject.toml のみを使用（推奨）**
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sda"
version = "1.0.0"  # 統一されたバージョン
description = "Score-based Data Assimilation"
readme = "README.md"
requires-python = ">=3.10"
license = {file = "LICENSE"}
authors = [
    {name = "François Rozet", email = "francois.rozet@uliege.be"},
    {name = "Gilles Louppe"}
]
dependencies = [
    "numpy",
    "torch",
    "h5py",
    "dawgz",
]

[project.optional-dependencies]
dev = [
    "jupyter",
    "matplotlib",
    "seaborn",
    "wandb",
]

[tool.setuptools.packages.find]
include = ["sda*", "experiments*"]
```

**オプション2: setup.pyを削除し、pyproject.tomlのみを使用**

## 依存関係管理

### コア依存関係（requires.txt）

最小限の実行に必要なパッケージ：

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| numpy | 任意 | 数値計算基盤 |
| torch | 任意 | 深層学習フレームワーク |
| h5py | 任意 | 大規模データセット管理 |
| dawgz | 任意 | Slurm統合 |

### 完全な依存関係（environment.yml）

開発・実験環境用の完全な依存関係：

```yaml
name: sda
channels:
  - pytorch      # PyTorchパッケージ
  - nvidia       # CUDA関連
  - conda-forge  # 一般パッケージ
dependencies:
  # Core dependencies
  - h5py=3.7.0
  - numpy=1.23.5
  - python=3.9.16    # 注意: pyproject.tomlは3.10以上を要求
  - pytorch=1.13.1
  - pytorch-cuda=11.7

  # Development tools
  - jupyter=1.0.0
  - pip=22.3.1

  # Pip packages
  - pip:
    - dawgz==0.4.1      # Slurm ジョブ管理
    - jax==0.4.4        # JAX (流体力学シミュレーション用)
    - jaxlib==0.4.4     # JAX バックエンド
    - matplotlib==3.6.2 # 可視化
    - POT==0.9.0        # 最適輸送（EMD計算）
    - seaborn==0.12.2   # 統計的可視化
    - tqdm==4.64.1      # プログレスバー
    - wandb==0.13.10    # 実験追跡
    - zuko==0.1.4       # 正規化フロー
```

### Python バージョンの不整合

**問題**:
- `pyproject.toml`: Python >= 3.10
- `environment.yml`: Python = 3.9.16

**解決策**:
```yaml
# environment.yml を修正
python=3.10.9  # または 3.10 以上の任意のバージョン
```

## バージョン管理

### セマンティックバージョニング

推奨される形式: `MAJOR.MINOR.PATCH`

- **MAJOR**: 後方互換性のない変更
- **MINOR**: 後方互換性のある機能追加
- **PATCH**: バグ修正

### バージョン管理のベストプラクティス

1. **単一の情報源**: バージョンを1箇所で定義

```python
# sda/__init__.py に追加
__version__ = "1.0.0"
```

2. **動的バージョン取得**:

```toml
# pyproject.toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "sda.__version__"}
```

3. **Git タグとの同期**:

```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## インストールと配布

### 開発モードインストール

```bash
# 編集可能インストール（開発用）
pip install -e .

# または
pip install --editable .

# pyproject.toml を明示的に使用
pip install -e . --config-settings editable_mode=strict
```

### 通常のインストール

```bash
# ローカルインストール
pip install .

# GitHubから直接
pip install git+https://github.com/francois-rozet/sda.git

# 特定のブランチから
pip install git+https://github.com/francois-rozet/sda.git@qg
```

### パッケージのビルド

```bash
# ビルドツールのインストール
pip install build

# ソース配布とホイールの作成
python -m build

# 生成されるファイル:
# dist/sda-1.0.0.tar.gz      (ソース配布)
# dist/sda-1.0.0-py3-none-any.whl (ホイール)
```

### PyPIへの公開

```bash
# テストPyPIへのアップロード
pip install twine
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# 本番PyPIへのアップロード
twine upload dist/*
```

## インストール検証

### パッケージの確認

```bash
# インストール済みパッケージの確認
pip show sda

# 出力例:
# Name: sda
# Version: 0.0.1
# Location: /path/to/sda
# Requires: numpy, torch, h5py, dawgz
```

### インポートテスト

```python
# Python REPLで確認
import sda
print(sda.__version__)  # エラーになる場合は __version__ が未定義

# 各モジュールの確認
from sda import score, nn, mcs, utils
print("All modules imported successfully")
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. egg-info が生成されない

**原因**: パッケージがインストールされていない

**解決**:
```bash
pip install -e .
```

#### 2. バージョン不整合エラー

**症状**: `setup.py` と `pyproject.toml` のバージョンが異なる

**解決**:
- `setup.py` を削除し、`pyproject.toml` のみを使用
- または、両方のバージョンを統一

#### 3. 依存関係の解決失敗

**症状**: 特定のパッケージがインストールできない

**解決**:
```bash
# 環境をクリーンアップ
pip uninstall sda
rm -rf sda.egg-info/

# 依存関係を個別にインストール
pip install numpy torch h5py dawgz

# パッケージを再インストール
pip install -e .
```

#### 4. ImportError: No module named 'sda'

**原因**: パッケージが正しくインストールされていない

**解決**:
```bash
# Pythonパスの確認
python -c "import sys; print(sys.path)"

# PYTHONPATH に追加
export PYTHONPATH=/workspace/sda:$PYTHONPATH

# または開発モードで再インストール
pip install -e /workspace/sda
```

#### 5. jax-cfd インストールエラー

**症状**: JAXとjax-cfdのバージョン不整合

**解決**:
```bash
# 特定のバージョンをインストール
pip install jax==0.4.4 jaxlib==0.4.4
pip install git+https://github.com/google/jax-cfd@v0.2.0
```

### egg-info の再生成

```bash
# egg-info を削除して再生成
rm -rf sda.egg-info/
pip install -e .

# または
python setup.py egg_info
```

### 開発環境のリセット

```bash
# 仮想環境の再作成
conda deactivate
conda env remove -n sda
conda env create -f environment.yml
conda activate sda
pip install -e .
```

## メタデータのカスタマイズ

### 追加のメタデータフィールド

```toml
[project]
name = "sda"
version = "1.0.0"
description = "Score-based Data Assimilation"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
keywords = ["diffusion", "data-assimilation", "machine-learning"]
authors = [
    {name = "François Rozet", email = "francois.rozet@uliege.be"},
]
maintainers = [
    {name = "François Rozet", email = "francois.rozet@uliege.be"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/francois-rozet/sda"
Documentation = "https://github.com/francois-rozet/sda/wiki"
Repository = "https://github.com/francois-rozet/sda"
"Bug Tracker" = "https://github.com/francois-rozet/sda/issues"
```

## ベストプラクティス

### 1. パッケージ構造の標準化

```
sda/
├── src/
│   └── sda/          # ソースコード
│       ├── __init__.py
│       ├── score.py
│       └── ...
├── tests/            # テストコード
├── docs/             # ドキュメント
├── examples/         # サンプルコード
├── pyproject.toml    # パッケージ設定
├── README.md         # プロジェクト説明
├── LICENSE           # ライセンス
└── .gitignore        # Git除外設定
```

### 2. .gitignore の設定

```gitignore
# Python egg metadata
*.egg-info/
*.egg

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
```

### 3. MANIFEST.in の使用

非Pythonファイルを含める場合：

```
include LICENSE
include README.md
recursive-include sda/data *.json *.yaml
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
```

## まとめ

`sda.egg-info` ディレクトリは、SDAパッケージのメタデータを管理する重要な要素です。適切な設定により：

1. **一貫性のあるバージョン管理**
2. **明確な依存関係の定義**
3. **スムーズなインストールプロセス**
4. **PyPIへの配布準備**

が実現できます。現在の設定には改善の余地があり、特に：
- バージョンの統一
- Pythonバージョン要件の整合
- pyproject.toml への一元化

を行うことで、より保守性の高いパッケージ構成となります。