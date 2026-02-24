#!/bin/bash
# ============================================================
# Organ-Aware MedGemma 1.5 Pipeline — Setup & Execution
# ============================================================
# Uses official TotalSegmentator test CT data
#
# Recommended usage (inside tmux):
#   tmux new -s medgemma
#   cd ~/organ_aware_medgemma && bash setup_and_run.sh
#
# Options:
#   --setup-only   Setup environment only (do not run)
#   --run-only     Run only (environment already configured)
#   --organ NAME   Specify target organ (default: liver)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
TEST_DATA_DIR="${SCRIPT_DIR}/test_data"
LOG_FILE="${SCRIPT_DIR}/pipeline_run.log"

SETUP_ONLY=false
RUN_ONLY=false
ORGAN="liver"

for arg in "$@"; do
    case $arg in
        --setup-only)  SETUP_ONLY=true ;;
        --run-only)    RUN_ONLY=true ;;
        --organ=*)     ORGAN="${arg#*=}" ;;
    esac
done

echo "============================================================"
echo "  Organ-Aware MedGemma 1.5 — Setup & Run"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# ---- Activate environment ----
activate_env() {
    source "${VENV_DIR}/bin/activate"
    echo "  Python: $(python --version) @ $(which python)"
}

# ---- SETUP ----
do_setup() {

    # --- 1. Download official TotalSegmentator test data ---
    echo ""
    echo "[1/4] Downloading TotalSegmentator test data..."
    mkdir -p "${TEST_DATA_DIR}"

    BASE_URL="https://github.com/wasserth/TotalSegmentator/raw/master/tests/reference_files"

    for f in example_ct_sm.nii.gz example_ct.nii.gz example_seg.nii.gz; do
        if [ -f "${TEST_DATA_DIR}/${f}" ]; then
            echo "  [Exists] ${f}"
        else
            echo "  [Downloading] ${f}..."
            wget -q --show-progress "${BASE_URL}/${f}" -O "${TEST_DATA_DIR}/${f}"
        fi
    done

    echo "  Verifying test data:"
    python3 -c "
import nibabel as nib, numpy as np
ct = nib.load('${TEST_DATA_DIR}/example_ct_sm.nii.gz')
seg = nib.load('${TEST_DATA_DIR}/example_seg.nii.gz')
print(f'    CT:  shape={ct.shape}, spacing={ct.header.get_zooms()}, HU=[{ct.get_fdata().min():.0f}, {ct.get_fdata().max():.0f}]')
print(f'    Seg: shape={seg.shape}, labels={len(np.unique(seg.get_fdata()))} classes')
" 2>/dev/null || echo "    (nibabel not installed yet — verification will be available after installation)"

    # --- 2. Create Python environment ---
    echo ""
    echo "[2/4] Creating Python virtual environment..."
    if [ -d "${VENV_DIR}" ]; then
        echo "  Virtual environment already exists: ${VENV_DIR}"
    else
        PYTHON_BIN=""
        for p in python3.10 python3.11 python3.12 python3; do
            if command -v "$p" &>/dev/null; then
                ver=$($p --version 2>&1 | awk '{print $2}')
                major=$(echo "$ver" | cut -d. -f1)
                minor=$(echo "$ver" | cut -d. -f2)
                if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                    PYTHON_BIN="$p"
                    break
                fi
            fi
        done

        if [ -z "$PYTHON_BIN" ]; then
            echo "  [Error] Python >= 3.10 not found"
            exit 1
        fi

        echo "  Using Python: $($PYTHON_BIN --version)"
        $PYTHON_BIN -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    echo "  Environment Python: $(python --version) @ $(which python)"

    # --- 3. Install dependencies ---
    echo ""
    echo "[3/4] Installing dependencies..."
    pip install --upgrade pip

    echo "  -> Installing PyTorch + CUDA 11.8..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    echo "  -> Installing Transformers & Accelerate..."
    pip install "transformers>=4.52" accelerate bitsandbytes

    echo "  -> Installing medical imaging libraries..."
    pip install nibabel scikit-image pillow matplotlib numpy

    echo "  -> Installing TotalSegmentator..."
    pip install TotalSegmentator

    pip install huggingface-hub

    echo ""
    echo "  Verifying installation:"
    python -c "
import torch
print(f'    PyTorch:      {torch.__version__}')
print(f'    CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU:          {torch.cuda.get_device_name(0)}')
    print(f'    VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
import transformers; print(f'    Transformers: {transformers.__version__}')
import nibabel;      print(f'    Nibabel:      {nibabel.__version__}')
"

    # --- 4. Hugging Face authentication ---
    echo ""
    echo "[4/4] Checking Hugging Face authentication..."
    if [ -f ~/.cache/huggingface/token ]; then
        echo "  Hugging Face token found"
        python -c "
from huggingface_hub import HfApi
try:
    info = HfApi().whoami()
    print(f'    Logged in as: {info[\"name\"]}')
except Exception as e:
    print(f'    [Warning] Verification failed: {e}')
" || true
    else
        echo "  [Important] Hugging Face token not found!"
        echo "    1. Accept model terms: https://huggingface.co/google/medgemma-1.5-4b-it"
        echo "    2. Login: huggingface-cli login"
        read -p "  Login now? (y/n): " yn
        [[ "$yn" =~ ^[Yy]$ ]] && huggingface-cli login
    fi

    echo ""
    echo "============================================================"
    echo "  Environment setup complete!"
    echo "============================================================"
}

# ---- RUN ----
do_run() {
    activate_env

    echo ""
    echo "[Run] Executing ablation test using official TotalSegmentator data"
    echo "  Target organ: ${ORGAN}"
    echo "  Log file: ${LOG_FILE}"
    echo ""

    python "${SCRIPT_DIR}/organ_aware_pipeline.py" \
        --organ "${ORGAN}" \
        --output "${SCRIPT_DIR}/ablation_results" \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "============================================================"
        echo "  Execution completed successfully!"
        echo "  Comparison image: ${SCRIPT_DIR}/ablation_results/ablation_comparison.png"
        echo "  Report JSON: ${SCRIPT_DIR}/ablation_results/ablation_report.json"
        echo "============================================================"
    else
        echo "  [Error] Exit code: ${EXIT_CODE} — please check ${LOG_FILE}"
    fi

    return $EXIT_CODE
}

# ---- Main flow ----
if $RUN_ONLY; then
    do_run
elif $SETUP_ONLY; then
    do_setup
else
    do_setup
    do_run
fi
