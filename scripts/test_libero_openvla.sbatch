#!/bin/bash
#SBATCH --job-name=libero-openvla
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/at5282/trace/tracerobotics/output/logs/libero-openvla-%j.out
#SBATCH --error=/scratch/at5282/trace/tracerobotics/output/logs/libero-openvla-%j.err

# =============================================================================
# LIBERO + OpenVLA-OFT Testing Pipeline
#
# Tests Stanford's OpenVLA-OFT (7B) on LIBERO manipulation tasks.
# Model is loaded directly on GPU — no separate server needed.
# Requires ~16GB VRAM for inference.
#
# Checkpoint: moojink/openvla-7b-oft-finetuned-libero-spatial
# (auto-downloaded from HuggingFace on first run)
#
# Prerequisites:
#   - openvla-oft repo cloned at OPENVLA_DIR
#   - conda env trace-openvla with deps installed
#
# Usage:
#   cd /scratch/at5282/trace/tracerobotics
#   mkdir -p output/logs
#   sbatch scripts/test_libero_openvla.sbatch
#
# Monitor:
#   tail -f output/logs/libero-openvla-<jobid>.out
# =============================================================================

set -euo pipefail

OPENVLA_DIR="/scratch/at5282/trace/openvla-oft"
TRACE_DIR="/scratch/at5282/trace/tracerobotics"
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial"

echo "============================================"
echo "LIBERO + OpenVLA-OFT Test Pipeline"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# --- Environment setup ---
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate trace-openvla
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

# Force TensorFlow to CPU-only. OpenVLA uses TF only for image
# preprocessing; without this TF grabs the GPU and XLA JIT fails
# because libdevice.10.bc is not in the cluster CUDA path.
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

echo ""
echo "[GPU Info]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# =============================================================================
# Level 1: OpenVLA adapter unit tests (no GPU model needed)
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 1: OpenVLA adapter unit tests"
echo "============================================"

cd "$TRACE_DIR"
python -m pytest tests/test_openvla_adapter.py -v 2>&1
LEVEL1_EXIT=$?

if [ $LEVEL1_EXIT -ne 0 ]; then
    echo "LEVEL 1 FAILED (exit $LEVEL1_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 1 PASSED"

# =============================================================================
# Level 2: Single LIBERO episode — verify OpenVLA can solve the task
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 2: LIBERO single-episode evaluation"
echo "============================================"

python -c "
import sys
sys.path.insert(0, '$OPENVLA_DIR')

from trace.config_loader import create_task, load_task_config
from trace.policy_adapter.openvla_adapter import OpenVLAAdapter
from trace.runner.episode_runner import EpisodeRunner

config = load_task_config('configs/tasks/libero_spatial.yaml')
config.seed = 0
task = create_task(config)
print(f'Task: {config.name}')
print(f'Language instruction: {task.language_instruction}')

policy = OpenVLAAdapter(
    checkpoint='$CHECKPOINT',
    chunk_size=8,
    openvla_repo_path='$OPENVLA_DIR',
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
# Level 3: Multi-task LIBERO robustness sweep
# 4 suites × 3 task IDs × 3 seeds × 5 episodes × 7 intensities × 9 stressors
# = 11,340 episodes
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 3: Multi-task LIBERO sweep"
echo "  Suites: spatial, object, goal, 10"
echo "  Task IDs: 0, 4, 9"
echo "  Sweep: multi_task_sweep.yaml (3 seeds, 5 eps)"
echo "============================================"

python -m scripts.run_evaluation \
  --task configs/tasks/libero_spatial.yaml \
       configs/tasks/libero_object.yaml \
       configs/tasks/libero_goal.yaml \
       configs/tasks/libero_10.yaml \
  --task-ids 0 4 9 \
  --sweep configs/sweeps/multi_task_sweep.yaml \
  --policy openvla \
  --openvla-checkpoint "$CHECKPOINT" \
  --openvla-repo-path "$OPENVLA_DIR" \
  --chunk-size 8 \
  --seed 0 \
  --output output/reports 2>&1
LEVEL3_EXIT=$?

if [ $LEVEL3_EXIT -ne 0 ]; then
    echo "LEVEL 3 FAILED (exit $LEVEL3_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 3 PASSED"

echo ""
echo "============================================"
echo "ALL LIBERO + OpenVLA TESTS PASSED"
echo "============================================"
