# Python開発環境セットアップガイド

本リポジトリの開発環境構成と使い方をまとめたドキュメント。

## モダンPython開発スタック（2025年版）

| カテゴリ | 本リポジトリ | 代替/旧来ツール |
|----------|-------------|-----------------|
| パッケージ管理 | **uv** | pip, poetry, pipenv |
| Linter/Formatter | **Ruff** | flake8 + black + isort |
| 型チェッカー | **Pyright** | mypy |
| テスト | **pytest** | unittest |
| Pre-commit | **pre-commit** | 手動実行 |
| 設定管理 | **Pydantic** | dataclasses, 辞書 |
| ロギング | **structlog** | logging標準 |
| CLI出力 | **rich** | print |

---

## ツールの役割

### Ruff vs Pyright

両方使う必要がある。役割が異なる。

| ツール | 役割 | 検出例 |
|--------|------|--------|
| **Ruff** | Linter/Formatter | 未使用import、コードスタイル、一般的なバグパターン |
| **Pyright** | 型チェッカー | 型不一致、存在しないメソッド呼び出し、引数の型エラー |

```python
def add(a: int, b: int) -> int:
    return a + b

result = add("hello", 1)  # Pyright: 型エラー, Ruff: 検出しない
unused_var = 42           # Ruff: 未使用変数, Pyright: 検出しない
```

### structlog

構造化ロギング。開発時はrich console、本番ではJSON出力。

```python
from sda.logging import quick_setup, get_logger

quick_setup(debug=True)
log = get_logger(__name__)
log.info("training_started", epoch=1, lr=0.001)
```

### Pydantic設定

型安全な設定管理。バリデーション付き。

```python
from sda.config import ExperimentConfig

config = ExperimentConfig(
    training={"epochs": 100, "batch_size": 32},
    model={"embedding": 128}
)
```

### richコンソール

CLI出力の視認性向上。

```python
from sda.console import track, print_metrics_table

# 進捗バー
for batch in track(dataloader, description="Training..."):
    ...

# メトリクス表示
print_metrics_table({"loss": 0.05, "accuracy": 0.95})
```

---

## コマンドリファレンス

### 日常的な開発

```bash
# Lintチェック
uv run ruff check .

# Lint + 自動修正
uv run ruff check . --fix

# フォーマット
uv run ruff format .

# 型チェック
uv run pyright

# テスト実行
uv run pytest

# 全チェック（pre-commit経由）
uv run pre-commit run --all-files
```

### セットアップ

```bash
# 依存インストール
uv sync

# 開発用依存も含む
uv sync --dev

# pre-commitフック有効化
uv run pre-commit install
```

### パッケージ管理

```bash
# パッケージ追加
uv add <package>

# 開発用パッケージ追加
uv add --dev <package>

# パッケージ削除
uv remove <package>
```

---

## 設定ファイル

### pyproject.toml

全ツールの設定を一元管理。

```toml
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.pyright]
pythonVersion = "3.10"
typeCheckingMode = "basic"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### .pre-commit-config.yaml

コミット前の自動チェック設定。

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: local
    hooks:
      - id: pyright
```

---

## ディレクトリ構成

```
fluid-sbi/
├── sda/sda/           # コアパッケージ
│   ├── config.py      # Pydantic設定
│   ├── console.py     # rich出力
│   ├── logging.py     # structlogロギング
│   ├── nn.py          # ニューラルネット
│   ├── score.py       # スコアモデル
│   └── utils.py       # ユーティリティ
├── tests/             # pytestテスト
├── docs/              # ドキュメント
├── pyproject.toml     # プロジェクト設定
└── .pre-commit-config.yaml
```

---

## 選定理由

### uv（パッケージ管理）
- Rust製で pip比 10-100x高速
- lockfile対応で再現性確保
- Python自体のバージョン管理も可能

### Ruff（Linter/Formatter）
- Rust製で flake8比 100x高速
- flake8 + black + isort を1ツールに統合
- 自動修正機能付き

### Pyright（型チェッカー）
- mypy比 3-5x高速
- VSCode Pylance と同一エンジン
- 新しい型機能の早期サポート

### structlog（ロギング）
- 構造化ログでJSON出力対応
- 本番環境での解析が容易
- コンテキスト情報の自動付与

---

## 参考リソース

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pyright Documentation](https://microsoft.github.io/pyright/)
- [structlog Best Practices](https://www.structlog.org/en/stable/logging-best-practices.html)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [uv Documentation](https://docs.astral.sh/uv/)
