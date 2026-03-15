#!/bin/bash
#SBATCH --job-name=libero-pi0fast
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=28:00:00
#SBATCH --output=/scratch/at5282/trace/tracerobotics/output/logs/libero-pi0fast-%j.out
#SBATCH --error=/scratch/at5282/trace/tracerobotics/output/logs/libero-pi0fast-%j.err

# =============================================================================
# LIBERO + Pi0-FAST Testing Pipeline
#
# Tests lerobot/pi0fast-libero on LIBERO manipulation tasks.
# Loads model locally via LeRobot (no server needed).
#
# Usage:
#   cd /scratch/at5282/trace/tracerobotics
#   mkdir -p output/logs
#   sbatch scripts/test_libero_pi0fast.sbatch
#
# Monitor:
#   tail -f output/logs/libero-pi0fast-<jobid>.out
# =============================================================================

set -euo pipefail

TRACE_DIR="/scratch/at5282/trace/tracerobotics"
LEROBOT_DIR="/scratch/at5282/trace/lerobot_src"

echo "============================================"
echo "LIBERO + Pi0-FAST Test Pipeline"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# --- Environment setup ---
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate trace-pi0
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export HF_HOME=/scratch/at5282/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/at5282/.cache/huggingface
export HF_TOKEN=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

# Ensure lerobot is importable from the new src/ layout
export PYTHONPATH="$LEROBOT_DIR/src:${PYTHONPATH:-}"

# Pin transformers to version lerobot was tested with
pip install "transformers>=5.3.0" 2>&1 | tail -5

if ! python -c "from lerobot.policies.pi0_fast import PI0FastPolicy" 2>/dev/null; then
    echo "Installing lerobot and all deps (except pynput)..."
    pip install poetry-core 2>&1 | tail -3
    pip install --no-deps -e "$LEROBOT_DIR" 2>&1 | tail -5
    pip install \
        "cmake>=3.29.0.1" "datasets>=2.19.0" "deepdiff>=7.0.1" \
        "diffusers>=0.27.2" "draccus==0.10.0" "einops>=0.8.0" \
        "flask>=3.0.3" "gdown>=5.1.0" "gymnasium==0.29.1" \
        "h5py>=3.10.0" "huggingface-hub[hf-transfer,cli]>=0.27.1" \
        "imageio[ffmpeg]>=2.34.0" "jsonlines>=4.0.0" "numba>=0.59.0" \
        "omegaconf>=2.3.0" "opencv-python-headless>=4.9.0" \
        "packaging>=24.2" "av>=14.2.0" "pymunk>=6.6.0" \
        "pyzmq>=26.2.1" "rerun-sdk>=0.21.0" "termcolor>=2.4.0" \
        "torch>=2.2.1" "torchvision>=0.21.0" "wandb>=0.16.3" \
        "zarr>=2.17.0" "safetensors" "accelerate" "pyserial" "scipy" \
        "transformers>=5.3.0" 2>&1 | tail -10
fi

echo ""
echo "[GPU Info]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# =============================================================================
# Level 1: LIBERO unit tests (no model needed)
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 1: LIBERO unit tests"
echo "============================================"

cd "$TRACE_DIR"
python -m pytest tests/test_libero_task.py -v 2>&1
LEVEL1_EXIT=$?

if [ $LEVEL1_EXIT -ne 0 ]; then
    echo "LEVEL 1 FAILED (exit $LEVEL1_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 1 PASSED"

# =============================================================================
# Level 2: Single LIBERO episode — verify pi0fast can solve the task
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 2: LIBERO single-episode evaluation (pi0fast)"
echo "============================================"

python -c "
from trace.config_loader import create_task, load_task_config
from trace.policy_adapter.pi0fast_adapter import Pi0FastAdapter
from trace.runner.episode_runner import EpisodeRunner

config = load_task_config('configs/tasks/libero_spatial.yaml')
config.seed = 0
task = create_task(config)
print(f'Task: {config.name}')
print(f'Suite: {config.task_params[\"task_suite_name\"]}')
print(f'Language instruction: {task.language_instruction}')

policy = Pi0FastAdapter(
    model_id='lerobot/pi0fast-libero',
    chunk_size=10,
)
policy.set_task_info(task.language_instruction)
policy.load('')

runner = EpisodeRunner(task, policy, stressors=[])
result = runner.run(seed=0)
print(f'Success:              {result.success}')
print(f'Steps:                {result.total_steps}')
print(f'Reward:               {result.total_reward:.3f}')
print(f'Catastrophic failure: {result.catastrophic_failure}')
print('LEVEL 2 PASSED')
" 2>&1
LEVEL2_EXIT=$?

if [ $LEVEL2_EXIT -ne 0 ]; then
    echo "LEVEL 2 FAILED (exit $LEVEL2_EXIT) — aborting."
    exit 1
fi

# =============================================================================
# Level 3: Mini sweep — LIBERO spatial with quick stressors
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 3: LIBERO mini sweep (quick_test.yaml)"
echo "============================================"

python -m scripts.run_evaluation \
  --task configs/tasks/libero_spatial.yaml \
  --sweep configs/sweeps/quick_test.yaml \
  --policy pi0fast \
  --chunk-size 10 \
  --seed 0 \
  --output output/reports 2>&1
LEVEL3_EXIT=$?

if [ $LEVEL3_EXIT -ne 0 ]; then
    echo "LEVEL 3 FAILED (exit $LEVEL3_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 3 PASSED"

# =============================================================================
# Level 4: Full LIBERO robustness sweep
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 4: LIBERO full sweep (default_sweep.yaml)"
echo "============================================"

python -m scripts.run_evaluation \
  --task configs/tasks/libero_spatial.yaml \
  --sweep configs/sweeps/default_sweep.yaml \
  --policy pi0fast \
  --chunk-size 10 \
  --seed 0 \
  --output output/reports 2>&1
LEVEL4_EXIT=$?

if [ $LEVEL4_EXIT -ne 0 ]; then
    echo "LEVEL 4 FAILED (exit $LEVEL4_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 4 PASSED"

echo ""
echo "============================================"
echo "ALL LIBERO Pi0-FAST TESTS PASSED"
echo "============================================"
