#!/bin/bash

# Iris Evaluation Script
# Usage: ./run_eval.sh [baseline] [mode]
#
# Arguments:
#   baseline: lotus | e2e | marigold
#   mode: For lotus, use 'd' (discriminative) or 'g' (generative). Ignored for other baselines.
#
# Examples:
#   ./run_eval.sh lotus d    # Evaluate Lotus Discriminative
#   ./run_eval.sh lotus g    # Evaluate Lotus Generative
#   ./run_eval.sh e2e        # Evaluate E2E
#   ./run_eval.sh marigold   # Evaluate Marigold

set -e

BASELINE=${1:-""}
MODE=${2:-""}


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$BASELINE" in
    lotus)
        if [ -z "$MODE" ]; then
            echo "Error: Lotus requires a mode ('d' for discriminative or 'g' for generative)"
            echo "Usage: ./run_eval.sh lotus [d|g]"
            exit 1
        fi

        if [ "$MODE" != "d" ] && [ "$MODE" != "g" ]; then
            echo "Error: Invalid mode '$MODE'. Must be 'd' (discriminative) or 'g' (generative)"
            exit 1
        fi

        echo "=========================================="
        echo "Evaluating Lotus ($MODE mode)"
        echo "=========================================="
        cd lotus
        if [ "$MODE" == "d" ]; then
            bash eval_scripts/eval-depth-d.sh
        else
            bash eval_scripts/eval-depth-g.sh
        fi
        ;;

    e2e)
        echo "=========================================="
        echo "Evaluating E2E (all datasets)"
        echo "=========================================="
        cd diffusion-e2e-ft

        EVAL_DIR="experiments/depth/eval_args/stable_diffusion_e2e_ft"

        echo "--- [1/5] NYUv2 ---"
        bash "$EVAL_DIR/11_infer_nyu.sh"
        bash "$EVAL_DIR/12_eval_nyu.sh"

        echo "--- [2/5] KITTI ---"
        bash "$EVAL_DIR/21_infer_kitti.sh"
        bash "$EVAL_DIR/22_eval_kitti.sh"

        echo "--- [3/5] ETH3D ---"
        bash "$EVAL_DIR/31_infer_eth3d.sh"
        bash "$EVAL_DIR/32_eval_eth3d.sh"

        echo "--- [4/5] ScanNet ---"
        bash "$EVAL_DIR/41_infer_scannet.sh"
        bash "$EVAL_DIR/42_eval_scannet.sh"

        echo "--- [5/5] DIODE ---"
        bash "$EVAL_DIR/51_infer_diode.sh"
        bash "$EVAL_DIR/52_eval_diode.sh"
        ;;

    marigold)
        echo "=========================================="
        echo "Evaluating Marigold"
        echo "=========================================="
        cd marigold
        bash eval.sh
        ;;

    *)
        echo "Error: Unknown baseline '$BASELINE'"
        echo "Available baselines: lotus, e2e, marigold"
        exit 1
        ;;
esac

echo ""
echo "Evaluation completed!"
