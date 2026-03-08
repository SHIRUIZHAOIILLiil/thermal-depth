#!/bin/bash

# Iris Training Script
# Usage: ./run_train.sh [baseline] [mode] [options]
# 
# Arguments:
#   baseline: lotus | e2e | marigold
#   mode: For lotus, use 'd' (discriminative) or 'g' (generative). Ignored for other baselines.
#
# Examples:
#   ./run_train.sh lotus d    # Train Lotus Discriminative
#   ./run_train.sh lotus g    # Train Lotus Generative
#   ./run_train.sh e2e        # Train E2E
#   ./run_train.sh marigold   # Train Marigold

set -e

# Get baseline and mode
BASELINE=${1:-""}
MODE=${2:-""}


# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run based on baseline
case "$BASELINE" in
    lotus)
        if [ -z "$MODE" ]; then
            echo "Error: Lotus requires a mode ('d' for discriminative or 'g' for generative)"
            echo "Usage: ./run_train.sh lotus [d|g]"
            exit 1
        fi
        
        if [ "$MODE" != "d" ] && [ "$MODE" != "g" ]; then
            echo "Error: Invalid mode '$MODE'. Must be 'd' (discriminative) or 'g' (generative)"
            exit 1
        fi
        
        echo "=========================================="
        echo "Training Lotus ($MODE mode)"
        echo "=========================================="
        cd lotus
        if [ "$MODE" == "d" ]; then
            bash train_scripts/train_iris_d_depth.sh
        else
            bash train_scripts/train_iris_g_depth.sh
        fi
        ;;
        
    e2e)
        echo "=========================================="
        echo "Training E2E"
        echo "=========================================="
        cd diffusion-e2e-ft
        bash training/scripts/train_stable_diffusion_e2e_ft_depth.sh
        ;;
        
    marigold)
        echo "=========================================="
        echo "Training Marigold"
        echo "=========================================="
        cd marigold
        bash run.sh
        ;;
        
    *)
        echo "Error: Unknown baseline '$BASELINE'"
        echo "Available baselines: lotus, e2e, marigold"
        exit 1
        ;;
esac

echo ""
echo "Training completed!"
