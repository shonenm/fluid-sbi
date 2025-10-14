# IBPM (Immersed Boundary Projection Method) 包括的解説

## 目次
1. [概要](#概要)
2. [理論背景](#理論背景)
3. [インストール](#インストール)
4. [プロジェクト構造](#プロジェクト構造)
5. [主要コンポーネント](#主要コンポーネント)
6. [使用方法](#使用方法)
7. [サンプル実行](#サンプル実行)
8. [出力ファイル](#出力ファイル)
9. [開発とテスト](#開発とテスト)
10. [参考文献](#参考文献)

---

## 概要

IBPMは、埋め込み境界法(Immersed Boundary Method)を用いて、複雑な形状周りの2次元非圧縮性Navier-Stokes方程式を解くC++ライブラリおよび数値計算ツールです。

### 主な特徴
- **埋め込み境界法**: 物体境界を直交格子に埋め込むことで、複雑形状周りの流れを効率的に計算
- **投影法**: Taira and Colonius (2007)で記述された投影法を使用
- **高速計算**: Colonius and Taira (2008)のSection 3.3で説明されている"fast method"を実装
- **多領域アプローチ**: 遠方境界条件のための多領域アプローチを採用
- **2D流体解析**: 円柱、翼型などの2次元物体周りの流れ解析に対応

### 適用例
- 円柱周りの流れ解析
- 翼型のピッチング・プランジング運動
- 複雑形状物体周りの非定常流れ
- 揚力・抗力の時間変化の計算

---

## 理論背景

### 基本方程式
IBPMは2次元非圧縮性Navier-Stokes方程式を解きます:

```
∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
∇·u = 0
```

ここで:
- `u`: 速度ベクトル
- `p`: 圧力
- `Re`: レイノルズ数

### 埋め込み境界法
- 物体境界を構造格子に埋め込む手法
- 直交格子を使用するため、格子生成が容易
- 移動物体の扱いが簡単

### 投影法
1. 対流項と拡散項から中間速度場を計算
2. 圧力ポアソン方程式を解く
3. 速度場を発散なし場に投影

### 主要論文
- **Taira and Colonius (2007)**: 投影アプローチとしての埋め込み境界法
- **Colonius and Taira (2008)**: ヌル空間アプローチと多領域遠方場境界条件を用いた高速埋め込み境界法

---

## インストール

### 必要要件

#### 必須
- **C++コンパイラ**: GCC推奨
- **FFTW library version 3**: 高速フーリエ変換ライブラリ
  - URL: http://www.fftw.org/

#### オプション
- **Doxygen**: ドキュメント生成用
  - URL: http://www.stack.nl/~dimitri/doxygen/

### ビルド手順

#### 基本ビルド
```bash
cd /workspace/ibpm
make
```

これにより、ライブラリと実行ファイルが`build`ディレクトリに生成されます。

#### カスタムビルド設定
システムに合わせてビルド設定をカスタマイズする場合:

```bash
# 設定ファイルをコピー
cp config/make.inc.gcc config/make.inc

# make.incを編集してFFTWのパスなどを設定
# その後ビルド
make
```

#### テストのビルドと実行
```bash
make test
```

#### ドキュメントのビルド
```bash
make doc
```

生成されたドキュメントは以下で確認できます:
- HTML版: `doc/html/index.html`
- LaTeX版: `doc/latex/refman.tex`

---

## プロジェクト構造

```
ibpm/
├── README                    # プロジェクト概要とクイックスタート
├── LICENSE                   # ライセンス情報
├── Makefile                  # トップレベルMakefile
├── ibpm.geom                 # サンプルジオメトリファイル
│
├── src/                      # ソースコード (49個のヘッダー, 31個の実装ファイル)
│   ├── ibpm.cc              # メインプログラム
│   ├── ibpm.h               # メインヘッダー
│   ├── Grid.h/cc            # 格子管理
│   ├── Geometry.h/cc        # 幾何形状定義
│   ├── IBSolver.h/cc        # 埋め込み境界法ソルバー
│   ├── ProjectionSolver.h/cc # 投影法ソルバー
│   ├── NavierStokesModel.h/cc # Navier-Stokesモデル
│   ├── RigidBody.h/cc       # 剛体運動
│   ├── Motion.h             # 運動定義基底クラス
│   ├── FixedPosition.h      # 固定位置
│   ├── FixedVelocity.h      # 固定速度
│   ├── PitchPlunge.h        # ピッチング・プランジング運動
│   ├── MotionFile.h         # ファイルからの運動読み込み
│   ├── State.h/cc           # 状態ベクトル
│   ├── Field.h/cc           # スカラー/ベクトル場
│   ├── Flux.h/cc            # フラックス計算
│   ├── Output*.h/cc         # 各種出力機能
│   ├── EllipticSolver*.h/cc # 楕円型方程式ソルバー
│   ├── Regularizer.h/cc     # 正則化関数
│   └── ...                  # その他のユーティリティ
│
├── build/                    # ビルド成果物
│   └── ibpm                 # メイン実行ファイル (ビルド後)
│
├── examples/                 # サンプルファイル
│   ├── cylinder.geom        # 円柱の幾何形状定義
│   ├── Oseen.cc             # Oseen流れのサンプル
│   ├── pitching.cc          # ピッチング運動のサンプル
│   ├── plunging.cc          # プランジング運動のサンプル
│   ├── bin2plt.cc           # バイナリからTecplotへの変換
│   └── ...
│
├── test/                     # ユニットテスト
│   ├── gtest-1.6.0/         # Google Testフレームワーク
│   ├── GridTest.cc          # 格子のテスト
│   ├── GeometryTest.cc      # 幾何形状のテスト
│   ├── FluxTest.cc          # フラックス計算のテスト
│   ├── ProjectionSolverTest.cc
│   ├── NavierStokesModelTest.cc
│   └── ...                  # 24個のテストファイル
│
├── doc/                      # ドキュメント
│   ├── ibpm_manual.tex      # ユーザーマニュアル (LaTeX)
│   ├── ibpm_design.tex      # 設計ドキュメント (LaTeX)
│   ├── Doxyfile             # Doxygen設定ファイル
│   ├── Overview.dox         # 概要ドキュメント
│   ├── Library.dox          # ライブラリAPIドキュメント
│   ├── References.dox       # 参考文献
│   ├── Figures/             # 図表
│   │   ├── grid.pdf
│   │   ├── IBPMDesign.pdf
│   │   └── IBFSDesign.pdf
│   ├── examples/            # ドキュメント用サンプル
│   └── snippets/            # コードスニペット
│
├── config/                   # ビルド設定
│   └── make.inc.gcc         # GCC用Makefile設定
│
├── benchmarking/            # ベンチマーク用スクリプト
└── xcode/                   # Xcodeプロジェクトファイル
```

---

## 主要コンポーネント

### 1. 格子管理 (Grid)
**ファイル**: `Grid.h/cc`

- 直交格子の生成と管理
- 格子点の座標計算
- 境界条件の設定

### 2. 幾何形状 (Geometry)
**ファイル**: `Geometry.h/cc`

- 物体形状の定義と読み込み
- `.geom`ファイルのパース
- 境界点の離散化

**形状定義例**:
```
body Cylinder
  circle_n 0 0 0.5 160
end
```

### 3. 埋め込み境界法ソルバー (IBSolver)
**ファイル**: `IBSolver.h/cc`

- 埋め込み境界条件の適用
- 境界上の力の計算
- 正則化とinterpolation

### 4. 投影法ソルバー (ProjectionSolver)
**ファイル**: `ProjectionSolver.h/cc`

- 圧力ポアソン方程式の解法
- 速度場の発散なし条件の強制
- 高速フーリエ変換(FFT)の利用

### 5. Navier-Stokesモデル (NavierStokesModel)
**ファイル**: `NavierStokesModel.h/cc`

- 非圧縮性Navier-Stokes方程式の実装
- 時間積分スキーム
- 対流項と拡散項の計算

### 6. 剛体運動 (RigidBody)
**ファイル**: `RigidBody.h/cc`

- 剛体の運動学
- 位置・速度・加速度の更新
- 運動パターンの適用

### 7. 運動パターン (Motion hierarchy)
- **FixedPosition**: 静止物体
- **FixedVelocity**: 一定速度運動
- **PitchPlunge**: ピッチング・プランジング運動
- **MotionFile**: ファイルからの運動データ読み込み

### 8. 出力機能
- **OutputForce**: 揚力・抗力の出力
- **OutputTecplot**: Tecplot形式での可視化
- **OutputRestart**: 再計算用のバイナリ出力
- **OutputProbes**: 指定点での物理量の計測
- **OutputEnergy**: エネルギーの時間変化

### 9. 数値計算ツール
- **EllipticSolver**: 楕円型方程式ソルバー
- **CholeskySolver**: Cholesky分解
- **ConjugateGradientSolver**: 共役勾配法
- **Regularizer**: デルタ関数の正則化

---

## 使用方法

### 基本的な実行コマンド

```bash
cd examples
../build/ibpm -geom cylinder.geom
```

### コマンドラインオプション

利用可能なオプションを表示:
```bash
../build/ibpm -h
```

主要オプション:
- `-geom <file>`: 幾何形状ファイルの指定
- `-nx <int>`: x方向の格子点数
- `-ny <int>`: y方向の格子点数
- `-ngrid <int>`: 格子細分化レベル
- `-length <float>`: 計算領域の長さ
- `-Re <float>`: レイノルズ数
- `-dt <float>`: 時間刻み幅
- `-nsteps <int>`: 計算ステップ数
- `-tecplot`: Tecplot形式で出力
- `-restart <file>`: リスタートファイルから継続

### 幾何形状ファイルの作成

#### 円柱の例
```
# Cylinder, diameter 1, with 160 points

body Cylinder
  circle_n 0 0 0.5 160
end
```

- `circle_n x y r n`: 中心(x,y), 半径r, n個の点で円を定義

#### 翼型やカスタム形状
```
body Airfoil
  points
    x1 y1
    x2 y2
    ...
    xn yn
  end
end
```

詳細は`doc/ibpm_manual.tex`を参照してください。

---

## サンプル実行

### 1. 円柱周りの流れ

```bash
cd examples
../build/ibpm -geom cylinder.geom -Re 100 -nsteps 1000
```

**パラメータ**:
- レイノルズ数: 100
- 計算ステップ数: 1000

**期待される挙動**:
- Re < 47: 定常流れ
- 47 < Re < 180: カルマン渦列 (周期的渦放出)
- Re > 180: 乱流遷移

### 2. ピッチング翼型

```bash
# examples/pitching.ccをビルドして実行
cd examples
make pitching
./pitching
```

### 3. プランジング翼型

```bash
# examples/plunging.ccをビルドして実行
cd examples
make plunging
./plunging
```

### 4. カスタムパラメータでの実行

```bash
../build/ibpm \
  -geom cylinder.geom \
  -nx 512 \
  -ny 512 \
  -Re 200 \
  -dt 0.01 \
  -nsteps 5000 \
  -tecplot
```

---

## 出力ファイル

計算実行後、以下のファイルが生成されます:

### 1. **ibpm.force**
- 揚力・抗力の時間変化
- ASCII形式
- 列: 時刻, x方向力, y方向力, その他

### 2. **ibpmXXXXX.bin** (リスタートファイル)
- 計算途中の状態をバイナリ形式で保存
- 計算の再開に使用
- 例: `ibpm00100.bin` (ステップ100の状態)

### 3. **ibpmXXXXX.plt** (Tecplotファイル)
- Tecplot可視化ソフトウェア用
- ASCII形式
- 速度場、圧力場、渦度などを含む

### 4. **ibpm.cmd**
- 実行されたコマンドの記録
- 計算の再現性のため

### 5. **ibpm.log** (オプション)
- 計算ログ
- 収束履歴や警告メッセージ

### バイナリからTecplotへの変換

```bash
cd examples
make bin2plt
./bin2plt ../path/to/ibpm00100.bin
```

---

## 開発とテスト

### ソースコードの構成

- **ヘッダーファイル**: 49個 (`*.h`)
- **実装ファイル**: 31個 (`*.cc`)
- **テストファイル**: 24個 (`*Test.cc`)

### テストフレームワーク

Google Test (gtest-1.6.0)を使用

### テストの実行

```bash
cd test
make
./runtest
```

主なテスト:
- `GridTest`: 格子生成と操作
- `GeometryTest`: 幾何形状の読み込みと処理
- `FluxTest`: フラックス計算の精度
- `ProjectionSolverTest`: 投影法の検証
- `NavierStokesModelTest`: Navier-Stokesソルバーの検証
- `VectorOperationsTest`: ベクトル演算

### コードの検証テスト

```bash
# 随伴演算子のチェック
./CheckAdjoint

# ラプラシアン演算子の精度検証
./CheckLaplacian

# 回転演算子の検証
./CheckCurl
```

### デバッグビルド

```makefile
# config/make.incで以下を設定
CXXFLAGS = -g -O0 -Wall
```

---

## 主要クラスとAPI

### State クラス
システムの状態(速度場、境界力など)を保持
```cpp
State q;
q.omega    // 渦度場
q.f        // 境界力
```

### IBSolver クラス
埋め込み境界法の主要ソルバー
```cpp
IBSolver solver(grid, geometry);
solver.solve(q);
```

### NavierStokesModel クラス
Navier-Stokes方程式の時間発展
```cpp
NavierStokesModel model(grid, geometry, Re);
model.step(q, dt);  // 1ステップ進める
```

### Geometry クラス
物体形状の管理
```cpp
Geometry geom;
geom.load("cylinder.geom");
int nPoints = geom.getNumPoints();
```

### RigidBody クラス
剛体運動の制御
```cpp
RigidBody body;
body.setMotion(new PitchPlunge(amplitude, frequency));
body.moveToTime(t);
```

---

## 数値計算の詳細

### 時間積分スキーム
- 対流項: Adams-Bashforth法 (2次精度)
- 拡散項: Crank-Nicolson法 (2次精度)

### 空間離散化
- 2次中心差分
- スタッガード格子 (MAC格子)

### ポアソン方程式の解法
- FFTベースの高速解法
- 多領域手法による遠方場処理

### 埋め込み境界の処理
- 正則化デルタ関数
- 離散化フーリエ変換との整合性

---

## トラブルシューティング

### ビルドエラー

**問題**: FFTWが見つからない
```
error: fftw3.h: No such file or directory
```

**解決**:
```bash
# FFTWをインストール
sudo apt-get install libfftw3-dev  # Ubuntu/Debian
# または
brew install fftw  # macOS

# config/make.incでパスを指定
FFTW_DIR = /usr/local
```

### 実行時エラー

**問題**: ジオメトリファイルが開けない
```
Error: Could not open geometry file
```

**解決**:
- ファイルパスを確認
- 相対パスまたは絶対パスを正しく指定

**問題**: 計算が不安定
```
Warning: CFL condition violated
```

**解決**:
- 時間刻み幅`-dt`を小さくする
- 格子点数`-nx`, `-ny`を増やす
- CFL条件: `dt < dx / max(|u|)`

---

## パフォーマンス最適化

### 並列化
- 現在の実装は単一コア
- FFTWは内部でマルチスレッド利用可能

### メモリ使用量
- 格子点数に比例: `O(nx * ny)`
- 大規模計算では格子サイズを調整

### 計算時間の見積もり
- 1ステップあたり: 数ms〜数秒 (格子サイズによる)
- 典型的な計算: 1000〜10000ステップ

---

## 今後の拡張可能性

### 実装可能な機能
- 3次元への拡張
- MPI並列化
- 適合格子細分化 (AMR)
- 複数物体の相互作用
- 流体構造連成 (FSI)
- 能動的流れ制御

### 現在のコードベースの活用
- モジュール設計により拡張が容易
- 抽象基底クラス(Motion, Output等)の継承により新機能追加が可能

---

## 参考文献

### 主要論文

1. **K. Taira and T. Colonius** (2007)
   "The immersed boundary method: A projection approach"
   *Journal of Computational Physics*, 225(2):2118-2137

2. **T. Colonius and K. Taira** (2008)
   "A fast immersed boundary method using a nullspace approach and multi-domain far-field boundary conditions"
   *Computer Methods in Applied Mechanics and Engineering*, 197(25-28):2131-2146

### 関連資料

- **ユーザーマニュアル**: `doc/ibpm_manual.tex`
  - 幾何形状ファイルの詳細フォーマット
  - 全コマンドラインオプションの説明

- **設計ドキュメント**: `doc/ibpm_design.tex`
  - ソフトウェアアーキテクチャ
  - クラス設計の詳細

- **Doxygenドキュメント**: `doc/html/index.html` (ビルド後)
  - 全クラスのAPI詳細
  - 関数の説明とパラメータ

### オンラインリソース

- **FFTW**: http://www.fftw.org/
- **Doxygen**: http://www.stack.nl/~dimitri/doxygen/
- **埋め込み境界法の概要**: 計算流体力学の教科書を参照

---

## ライセンス

詳細は`LICENSE`ファイルを参照してください。

---

## 連絡先

プロジェクトの詳細や問題報告については、オリジナルリポジトリを参照:
- GitHub: https://github.com/cwrowley/ibpm

---

## まとめ

IBPMは、複雑形状周りの2D非圧縮性流れを効率的に計算するための強力なツールです。

**主な利点**:
- 直交格子による簡単な実装
- 移動物体の扱いが容易
- 高速な計算アルゴリズム
- 充実したドキュメントとテストスイート

**学習の進め方**:
1. `examples/cylinder.geom`から始める
2. `doc/ibpm_manual.tex`を読む
3. パラメータを変えて実験
4. カスタム形状ファイルを作成
5. ソースコードを読んで理解を深める

**推奨される使い方**:
- 流体力学の研究と教育
- 数値計算手法の学習
- 流れ制御の検証
- CFDコードの開発ベース

このドキュメントが、IBPMの理解と活用の助けになれば幸いです。
