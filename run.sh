#!/bin/bash
# ============================================================================
# Run script for Abusive Tamil Text Detection
# DravidianLangTech@ACL 2026
# ============================================================================

# ----------------------------------------------------------------------------
# Project paths (must be set first for SSL config)
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
SRC_DIR="$PROJECT_DIR/src"
DATA_DIR="$PROJECT_DIR/data"
CONFIG_DIR="$PROJECT_DIR/configs"
OUTPUT_DIR="$PROJECT_DIR/outputs"

# ----------------------------------------------------------------------------
# SSL Certificate Configuration (Corporate Network)
# ----------------------------------------------------------------------------
export SSL_CERT_FILE="${SCRIPT_DIR}/micron-ca-bundle.crt"
export REQUESTS_CA_BUNDLE="${SCRIPT_DIR}/micron-ca-bundle.crt"
export CURL_CA_BUNDLE="${SCRIPT_DIR}/micron-ca-bundle.crt"

# ----------------------------------------------------------------------------
# HuggingFace Token (for gated models)
# Set your token here or export HF_TOKEN before running
# ----------------------------------------------------------------------------
# export HF_TOKEN="your_token_here"

# Verify HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Some models may not be accessible."
    echo "Export HF_TOKEN or set it in this script."
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_DIR/.venv" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
    echo "Virtual environment activated"
fi

# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

function show_help() {
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train-transformer   Train a transformer model"
    echo "  predict            Generate predictions on test data"
    echo "  analyze-model      Analyze model performance with confusion matrix"
    echo "  setup              Set up the virtual environment"
    echo ""
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh train-transformer --config configs/muril_base.json"
    echo "  ./run.sh predict --model outputs/models/muril_base"
    echo "  ./run.sh analyze-model --model outputs/models/muril_base --data data/dev.csv"
}

function setup_env() {
    echo "Setting up virtual environment with uv..."
    cd "$PROJECT_DIR"

    # Create venv with Python 3.11
    uv venv --python 3.11 .venv

    # Activate
    source .venv/bin/activate

    # Install dependencies (use --native-tls for corporate network)
    uv pip install -r requirements.txt --native-tls

    echo ""
    echo "Setup complete! Activate with: source .venv/bin/activate"
}

function train_transformer() {
    # Default values
    CONFIG="$CONFIG_DIR/muril_base.json"
    TRAIN_DATA="$DATA_DIR/train.csv"
    DEV_DATA="$DATA_DIR/dev.csv"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG="$2"
                shift 2
                ;;
            --train)
                TRAIN_DATA="$2"
                shift 2
                ;;
            --dev)
                DEV_DATA="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo "Training transformer model..."
    echo "  Config: $CONFIG"
    echo "  Train: $TRAIN_DATA"
    echo "  Dev: $DEV_DATA"
    echo ""

    python "$SRC_DIR/train_transformer.py" \
        --config "$CONFIG" \
        --train "$TRAIN_DATA" \
        --dev "$DEV_DATA"
}

function predict() {
    # Default values
    MODEL_DIR="$OUTPUT_DIR/models/muril_base"
    TEST_DATA="$DATA_DIR/test.csv"
    PRED_OUTPUT="$OUTPUT_DIR/predictions"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL_DIR="$2"
                shift 2
                ;;
            --test)
                TEST_DATA="$2"
                shift 2
                ;;
            --output)
                PRED_OUTPUT="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo "Generating predictions..."
    echo "  Model: $MODEL_DIR"
    echo "  Test: $TEST_DATA"
    echo "  Output: $PRED_OUTPUT"
    echo ""

    python "$SRC_DIR/inference.py" \
        --model "$MODEL_DIR" \
        --test "$TEST_DATA" \
        --output "$PRED_OUTPUT"
}

function batch_predict() {
    # Default values
    TEST_DATA="$DATA_DIR/TestV2 - testV2.csv"
    PRED_OUTPUT="$OUTPUT_DIR/predictions"
    TEAM_NAME="CHMOD_777"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --test)
                TEST_DATA="$2"
                shift 2
                ;;
            --output)
                PRED_OUTPUT="$2"
                shift 2
                ;;
            --team)
                TEAM_NAME="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo "Running batch inference for multi-run submission..."
    echo "  Test: $TEST_DATA"
    echo "  Output: $PRED_OUTPUT"
    echo "  Team: $TEAM_NAME"
    echo ""

    python "$SRC_DIR/batch_inference.py" \
        --test "$TEST_DATA" \
        --output "$PRED_OUTPUT" \
        --team "$TEAM_NAME"
}

function analyze_model() {
    # Default values
    MODEL_DIR="$OUTPUT_DIR/models/muril_base"
    DATA_PATH="$DATA_DIR/dev.csv"
    ANALYSIS_OUTPUT="$OUTPUT_DIR"
    SPLIT="dev"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL_DIR="$2"
                shift 2
                ;;
            --data)
                DATA_PATH="$2"
                shift 2
                ;;
            --output)
                ANALYSIS_OUTPUT="$2"
                shift 2
                ;;
            --split)
                SPLIT="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo "Analyzing model..."
    echo "  Model: $MODEL_DIR"
    echo "  Data: $DATA_PATH"
    echo "  Split: $SPLIT"
    echo ""

    python "$SRC_DIR/analyze.py" \
        --model "$MODEL_DIR" \
        --data "$DATA_PATH" \
        --output "$ANALYSIS_OUTPUT" \
        --split "$SPLIT"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
case "${1:-}" in
    train-transformer)
        shift
        train_transformer "$@"
        ;;
    predict)
        shift
        predict "$@"
        ;;
    batch-predict)
        shift
        batch_predict "$@"
        ;;
    analyze-model)
        shift
        analyze_model "$@"
        ;;
    setup)
        setup_env
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
