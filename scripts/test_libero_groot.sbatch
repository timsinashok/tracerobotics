#!/bin/bash
#SBATCH --job-name=libero-groot
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=28:00:00
#SBATCH --output=/scratch/at5282/trace/tracerobotics/output/logs/libero-groot-%j.out
#SBATCH --error=/scratch/at5282/trace/tracerobotics/output/logs/libero-groot-%j.err

# =============================================================================
# LIBERO + GR00T N1 Testing Pipeline
#
# Tests NVIDIA GR00T N1 on LIBERO manipulation tasks.
# Starts GR00T ZeroMQ server, runs LIBERO-specific tests incrementally.
#
# Prerequisites:
#   - Isaac-GR00T repo cloned and installed
#   - LIBERO finetuned checkpoint downloaded
#   - conda env with gr00t + libero deps
#
# Usage:
#   cd /scratch/at5282/trace/tracerobotics
#   mkdir -p output/logs
#   sbatch scripts/test_libero_groot.sbatch
#
# Monitor:
#   tail -f output/logs/libero-groot-<jobid>.out
# =============================================================================

set -euo pipefail

GROOT_DIR="/scratch/at5282/trace/Isaac-GR00T"
TRACE_DIR="/scratch/at5282/trace/tracerobotics"
GROOT_PORT=5555
# Update this path after downloading the finetuned checkpoint
GROOT_CHECKPOINT="/scratch/at5282/trace/groot_checkpoints/libero_spatial"

echo "============================================"
echo "LIBERO + GR00T N1 Test Pipeline"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# --- Environment setup ---
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate trace-pi0
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

echo ""
echo "[GPU Info]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# =============================================================================
# Level 1: GR00T adapter unit tests (no server needed)
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 1: GR00T adapter unit tests"
echo "============================================"

cd "$TRACE_DIR"
python -m pytest tests/test_groot_adapter.py -v 2>&1
LEVEL1_EXIT=$?

if [ $LEVEL1_EXIT -ne 0 ]; then
    echo "LEVEL 1 FAILED (exit $LEVEL1_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 1 PASSED"

# =============================================================================
# Start the GR00T server (needed for levels 2+)
# =============================================================================
echo ""
echo "============================================"
echo "Starting GR00T server on port $GROOT_PORT..."
echo "============================================"

cd "$GROOT_DIR"
python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag LIBERO_PANDA \
  --model-path "$GROOT_CHECKPOINT" \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port $GROOT_PORT \
  --use-sim-policy-wrapper &
SERVER_PID=$!

# Wait for the server to be ready (check port every 5s, timeout after 10min)
echo "Waiting for server (PID $SERVER_PID) to start listening on port $GROOT_PORT..."
TIMEOUT=600
ELAPSED=0
while ! ss -tlnp | grep -q ":${GROOT_PORT} " 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died before becoming ready."
        exit 1
    fi
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Server did not start within ${TIMEOUT}s."
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... waiting (${ELAPSED}s)"
done

echo "Server is ready (took ${ELAPSED}s)."
echo ""

# Cleanup: kill the server when the script exits
cleanup() {
    echo ""
    echo "Shutting down server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

cd "$TRACE_DIR"

# =============================================================================
# Level 2: Single LIBERO episode — verify GR00T can solve the task
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 2: LIBERO single-episode evaluation"
echo "============================================"

python -c "
from trace.config_loader import create_task, load_task_config
from trace.policy_adapter.groot_adapter import GR00TAdapter
from trace.runner.episode_runner import EpisodeRunner

config = load_task_config('configs/tasks/libero_spatial.yaml')
config.seed = 0
task = create_task(config)
print(f'Task: {config.name}')
print(f'Suite: {config.task_params[\"task_suite_name\"]}')
print(f'Language instruction: {task.language_instruction}')

policy = GR00TAdapter(
    host='localhost', port=$GROOT_PORT, chunk_size=8,
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
  --policy groot \
  --groot-host localhost --groot-port $GROOT_PORT \
  --chunk-size 8 \
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
  --policy groot \
  --groot-host localhost --groot-port $GROOT_PORT \
  --chunk-size 8 \
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
echo "ALL LIBERO + GR00T TESTS PASSED"
echo "============================================"
