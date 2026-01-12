#!/bin/bash
# 汎化テスト用IBPMシミュレーション実行スクリプト
#
# Usage:
#   bash scripts/run_ibpm_generalization.sh
#
# 環境変数:
#   IBPM_BIN: IBPMバイナリパス（デフォルト: ibpm）

set -e  # エラー時に停止

IBPM_BIN=${IBPM_BIN:-ibpm}
DATA_DIR=/home/devuser/fluid-sbi/data
OUTPUT_BASE=$DATA_DIR/ibpm_gen

# 共通パラメータ（学習データと同じ）
NX=400
NY=200
LENGTH=8
XOFFSET=-2
YOFFSET=-2
RE=100
DT=0.02
NSTEPS=300

echo "=============================================="
echo "IBPM Generalization Data Generation"
echo "=============================================="
echo "IBPM binary: $IBPM_BIN"
echo "Output base: $OUTPUT_BASE"
echo ""

# IBPMが利用可能か確認
if ! command -v $IBPM_BIN &> /dev/null; then
    echo "ERROR: IBPM binary not found: $IBPM_BIN"
    echo "Set IBPM_BIN environment variable to the correct path."
    exit 1
fi

mkdir -p $OUTPUT_BASE

# --- 1. 円柱位置バリエーション ---
echo ""
echo "=== Cylinder Position Variations ==="
for GEOM in cylinder_y_m02 cylinder_y_m01 cylinder_y_p02; do
    GEOM_FILE=$DATA_DIR/${GEOM}.geom
    OUTDIR=$OUTPUT_BASE/${GEOM}

    if [ ! -f "$GEOM_FILE" ]; then
        echo "WARNING: Geometry file not found: $GEOM_FILE"
        continue
    fi

    mkdir -p $OUTDIR
    echo ""
    echo "Running: $GEOM"
    echo "  Geometry: $GEOM_FILE"
    echo "  Output: $OUTDIR"

    $IBPM_BIN -nx $NX -ny $NY -length $LENGTH \
              -xoffset $XOFFSET -yoffset $YOFFSET \
              -Re $RE -dt $DT -nsteps $NSTEPS \
              -geom $GEOM_FILE \
              -tecplot 1 -outdir $OUTDIR

    echo "  Done: $(ls $OUTDIR/*.plt 2>/dev/null | wc -l) timesteps generated"
done

# --- 2. 円柱サイズバリエーション ---
echo ""
echo "=== Cylinder Size Variations ==="
for GEOM in cylinder_r04 cylinder_r06; do
    GEOM_FILE=$DATA_DIR/${GEOM}.geom
    OUTDIR=$OUTPUT_BASE/${GEOM}

    if [ ! -f "$GEOM_FILE" ]; then
        echo "WARNING: Geometry file not found: $GEOM_FILE"
        continue
    fi

    mkdir -p $OUTDIR
    echo ""
    echo "Running: $GEOM"
    echo "  Geometry: $GEOM_FILE"
    echo "  Output: $OUTDIR"

    $IBPM_BIN -nx $NX -ny $NY -length $LENGTH \
              -xoffset $XOFFSET -yoffset $YOFFSET \
              -Re $RE -dt $DT -nsteps $NSTEPS \
              -geom $GEOM_FILE \
              -tecplot 1 -outdir $OUTDIR

    echo "  Done: $(ls $OUTDIR/*.plt 2>/dev/null | wc -l) timesteps generated"
done

echo ""
echo "=============================================="
echo "All simulations completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Convert to HDF5:"
echo "   for DIR in $OUTPUT_BASE/cylinder_*; do"
echo "     NAME=\$(basename \$DIR)"
echo "     python scripts/convert_ibpm_to_sda.py \\"
echo "       --input \$DIR \\"
echo "       --output data/ibpm_h5_gen_\${NAME} \\"
echo "       --window 42 --stride 8"
echo "   done"
echo ""
echo "2. Run generalization tests:"
echo "   cd sda && python experiments/ibpm/evaluate.py \\"
echo "     --run-dir runs/ibpm/ibpm_vpsde_w16_lr1e-04_bs2_wd1e-03_seed0_j4939kd1 \\"
echo "     --mode generalization"
