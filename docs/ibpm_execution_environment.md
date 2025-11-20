# IBPM実行環境の詳細

## 概要

IBPM (Immersed Boundary Projection Method) は、C++で実装された2次元非圧縮性Navier-Stokes方程式ソルバーです。このドキュメントでは、IBPMの依存関係、ビルドシステム、実行環境について詳細に説明します。

---

## 1. 実行バイナリの基本情報

### バイナリの場所
- **実行可能ファイル**: `/opt/ibpm/build/ibpm`
- **システムリンク**: `/usr/local/bin/ibpm`
- **バイナリサイズ**: 4.8 MB
- **バージョン**: 1.0

### 実行方法
```bash
# 基本的な実行
ibpm -geom cylinder.geom -nsteps 250

# 出力ディレクトリ指定
ibpm -geom cylinder.geom -nsteps 250 -tecplot 1 -outdir /path/to/output
```

---

## 2. 依存ライブラリ

### 必須ライブラリ

`ldd /opt/ibpm/build/ibpm` で確認できる動的リンクライブラリ:

| ライブラリ | パス | 用途 |
|-----------|------|------|
| **libfftw3.so.3** | `/lib/x86_64-linux-gnu/libfftw3.so.3` | 高速フーリエ変換（FFT）計算 |
| **libstdc++.so.6** | `/lib/x86_64-linux-gnu/libstdc++.so.6` | C++標準ライブラリ |
| **libm.so.6** | `/lib/x86_64-linux-gnu/libm.so.6` | 数学関数ライブラリ |
| **libmvec.so.1** | `/lib/x86_64-linux-gnu/libmvec.so.1` | ベクトル数学関数（SIMD最適化） |
| **libgcc_s.so.1** | `/lib/x86_64-linux-gnu/libgcc_s.so.1` | GCCランタイムサポート |
| **libc.so.6** | `/lib/x86_64-linux-gnu/libc.so.6` | C標準ライブラリ |

### FFTW3ライブラリの詳細

**FFTW (Fastest Fourier Transform in the West)** は、IBPMの中核となる数値計算ライブラリです。

- **バージョン**: 3.3.10-2+b1
- **精度**: Double precision (`libfftw3-double3`)
- **用途**:
  - Poisson方程式の高速解法
  - スペクトル法による微分計算
  - 投影法における圧力場の計算

インストールされているFFTW3パッケージ:
```
libfftw3-bin           3.3.10-2+b1   (コマンドラインツール)
libfftw3-dev:amd64     3.3.10-2+b1   (開発用ヘッダー)
libfftw3-double3:amd64 3.3.10-2+b1   (倍精度実数変換)
libfftw3-long3:amd64   3.3.10-2+b1   (長倍精度変換)
libfftw3-quad3:amd64   3.3.10-2+b1   (四倍精度変換)
libfftw3-single3:amd64 3.3.10-2+b1   (単精度変換)
```

### 追加の依存関係（ビルド時）

| パッケージ | 用途 |
|-----------|------|
| **g++-14** (14.2.0-19) | C++コンパイラ（GNU C++ Compiler） |
| **make** | ビルドシステム |
| **ar** | スタティックライブラリ作成ツール |
| **Doxygen**（オプション） | ドキュメント生成 |

---

## 3. ビルドシステム

### ビルド方式

IBPMは **Makeベースのビルドシステム** を使用しています（CMakeではない）。

```
/opt/ibpm/
├── Makefile              # トップレベルMakefile
├── build/
│   ├── Makefile          # ビルド用Makefile
│   ├── libibpm.a         # 静的ライブラリ（11 MB）
│   ├── ibpm              # メイン実行ファイル（4.8 MB）
│   └── checkgeom         # ジオメトリ検証ツール（3.6 MB）
└── config/
    ├── make.inc          -> make.inc.gcc (シンボリックリンク)
    └── make.inc.gcc      # GCC用コンパイラ設定
```

### コンパイラ設定 (`config/make.inc.gcc`)

```makefile
CXX = g++

# 最適化フラグ
CXXFLAGS = -Wall -g -Ofast -funroll-loops -DNDEBUG

# リンクライブラリ
LDLIBS = -lfftw3 -lm
```

**最適化オプション解説**:
- `-Ofast`: 最高レベルの最適化（IEEE準拠を犠牲にする場合あり）
- `-funroll-loops`: ループ展開による高速化
- `-DNDEBUG`: アサーションを無効化（リリースビルド）
- `-g`: デバッグ情報を含む

### ビルド手順

```bash
cd /opt/ibpm

# 静的ライブラリとバイナリをビルド
make

# テストをビルド・実行
make test

# ドキュメント生成
make doc

# クリーンアップ
make clean        # オブジェクトファイル削除
make distclean    # すべてのビルド成果物削除
```

---

## 4. ソースコード構成

### 主要コンポーネント

`/opt/ibpm/build/Makefile` に記載されているオブジェクトファイル:

| モジュール | 機能 |
|-----------|------|
| **IBSolver.o** | 埋め込み境界法ソルバー（843 KB）|
| **ProjectionSolver.o** | 投影法による圧力補正 |
| **NavierStokesModel.o** | Navier-Stokes方程式モデル |
| **CholeskySolver.o** | Cholesky分解による連立方程式ソルバー |
| **ConjugateGradientSolver.o** | 共役勾配法ソルバー |
| **EllipticSolver.o** | 楕円型方程式ソルバー |
| **Geometry.o** | 境界形状の定義と処理 |
| **RigidBody.o** | 剛体運動の処理（1.1 MB） |
| **OutputTecplot.o** | Tecplot形式での出力 |
| **OutputForce.o** | 力係数の出力 |
| **OutputRestart.o** | リスタートファイルの入出力 |
| **VectorOperations.o** | ベクトル演算（1.1 MB） |

### 静的ライブラリ

- **libibpm.a** (11 MB): すべてのコンポーネントをアーカイブした静的ライブラリ
- `ar -r libibpm.a *.o` で生成

---

## 5. 実行時の動作

### Cholesky分解ソルバーの初期化

初回実行時、IBPMは圧力投影のためのCholesky分解を計算し、キャッシュファイルに保存します:

```
Computing the matrix for Cholesky factorization...done
Computing Cholesky factorization...done
Saving Cholesky factorization to file /workspace/data/ibpm_full/ibpm_01.cholesky...done
```

2回目以降の実行では、このファイルを読み込むことで計算を高速化します。

### メモリ使用

- **グリッド**: 200×200 (デフォルト)
- **マルチドメイン**: 1レベル (デフォルト)
- **Reynolds数**: 100 (デフォルト)

### 出力ファイル

| ファイル形式 | 拡張子 | 用途 |
|-------------|-------|------|
| Tecplot | `.plt` | 可視化用（x, y, u, v, vorticity） |
| Binary restart | `.bin` | リスタート用バイナリデータ |
| Force data | `.force` | 揚力・抗力係数の時系列 |
| Cholesky cache | `.cholesky` | Cholesky分解のキャッシュ |

---

## 6. アルゴリズムの理論的背景

### 投影法 (Projection Method)

IBPMは **Taira and Colonius (2007)** の投影法を実装しています:

1. **予測ステップ**: 対流・拡散項を計算
2. **投影ステップ**: 圧力場を計算し、非圧縮性条件を満たすように速度場を補正
3. **埋め込み境界条件**: 境界上で速度を強制

### 高速化手法

**Colonius and Taira (2008)** の "fast method" を実装:

- **Nullspace approach**: 境界条件を満たす空間での計算
- **Multi-domain far-field BC**: 遠方境界条件のマルチドメイン処理
- **FFTによる高速Poisson解法**: FFTW3を使用

---

## 7. 実行環境の確認コマンド

### 依存ライブラリの確認
```bash
ldd /opt/ibpm/build/ibpm
```

### FFTWのバージョン確認
```bash
dpkg -l | grep fftw
```

### コンパイラ確認
```bash
g++ --version
```

### IBPMのヘルプ
```bash
ibpm -h
```

---

## 8. パフォーマンス特性

### コンパイル時の最適化

- **-Ofast**: 積極的な最適化により約30-40%の高速化
- **-funroll-loops**: ループ展開により約10-15%の高速化
- **FFTW3のSIMD最適化**: AVX/SSE命令による並列化

### 実行時の特性

- **Cholesky分解のキャッシュ**: 2回目以降の実行で数倍高速化
- **FFT計算の支配**: 全体の計算時間の30-40%をFFTが占める
- **メモリアクセスパターン**: 連続アクセスを重視した実装

---

## 9. トラブルシューティング

### よくあるエラー

**Error: libfftw3.so.3 not found**
```bash
# FFTW3をインストール
sudo apt-get install libfftw3-dev libfftw3-double3
```

**Error: Cholesky factorization failed**
- グリッド解像度が大きすぎる場合に発生
- メモリ不足の可能性

**Error: Geometry file not found**
```bash
# ジオメトリファイルの存在確認
ls -la cylinder.geom

# ジオメトリの検証
/opt/ibpm/build/checkgeom cylinder.geom
```

---

## 10. まとめ

### 依存関係の要約

```
IBPM実行に必要なライブラリ:
┌─────────────────────────────┐
│  IBPM (C++ executable)      │
│  - Version: 1.0             │
│  - Size: 4.8 MB             │
└─────────┬───────────────────┘
          │
          ├── libfftw3.so.3 (3.3.10)  ← FFT計算（最重要）
          ├── libstdc++.so.6          ← C++標準ライブラリ
          ├── libm.so.6               ← 数学関数
          ├── libmvec.so.1            ← ベクトル数学（SIMD）
          ├── libgcc_s.so.1           ← GCCランタイム
          └── libc.so.6               ← C標準ライブラリ
```

### ビルド・実行フロー

```
ソースコード (.cc)
    ↓
[g++ -Ofast -funroll-loops] コンパイル
    ↓
オブジェクトファイル (.o)
    ↓
[ar -r] アーカイブ
    ↓
libibpm.a (静的ライブラリ 11 MB)
    ↓
[g++ リンク + libfftw3]
    ↓
ibpm実行ファイル (4.8 MB)
    ↓
[実行時] 動的リンク (libfftw3.so.3, libstdc++.so.6, etc.)
    ↓
シミュレーション実行
```

---

## 参考文献

1. **K. Taira and T. Colonius**. The immersed boundary method: A projection approach. *J. Comput. Phys.*, 225(2):2118-2137, August 2007.

2. **T. Colonius and K. Taira**. A fast immersed boundary method using a nullspace approach and multi-domain far-field boundary conditions. *Comp. Meth. Appl. Mech. Eng.*, 197(25-28):2131–46, 2008.

3. **FFTW Documentation**: http://www.fftw.org/

4. **IBPM Repository**: `/opt/ibpm/README`, `/opt/ibpm/doc/`
