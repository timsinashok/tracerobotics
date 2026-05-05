#!/bin/bash
#SBATCH --job-name=libero-pi0-5t
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/at5282/trace/tracerobotics/output/logs/libero-pi0-5t-%j.out
#SBATCH --error=/scratch/at5282/trace/tracerobotics/output/logs/libero-pi0-5t-%j.err

# =============================================================================
# LIBERO + Pi0 Testing Pipeline (5 tasks, 4 suites in parallel)
#
# Runs all 4 LIBERO suites simultaneously against the same pi0 server.
# Each suite runs as a separate process, all sharing one GPU server.
#
# 4 suites × 5 task IDs × 3 seeds × 5 episodes × 7 intensities × 9 stressors
# = 18,900 episodes (~10 hours with 4-way parallelism at ~7s/ep)
# =============================================================================

set -euo pipefail

OPENPI_DIR="/scratch/at5282/trace/openpi"
TRACE_DIR="/scratch/at5282/trace/tracerobotics"
PI0_PORT=8000

echo "============================================"
echo "LIBERO + Pi0 Test Pipeline (5 tasks, parallel)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# --- Environment setup ---
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate trace-pi0
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export OPENPI_DATA_HOME=/scratch/at5282/.cache/openpi

echo ""
echo "[GPU Info]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- GPU keepalive ---
(while true; do nvidia-smi > /dev/null 2>&1; sleep 30; done) &
KEEPALIVE_PID=$!

# =============================================================================
# Level 1: LIBERO unit tests (no server needed)
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
# Start the pi0 server
# =============================================================================
echo ""
echo "============================================"
echo "Starting pi0 server on port $PI0_PORT..."
echo "============================================"

cd "$OPENPI_DIR"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero &
SERVER_PID=$!

echo "Waiting for server (PID $SERVER_PID) to start listening on port $PI0_PORT..."
TIMEOUT=600
ELAPSED=0
while ! ss -tlnp | grep -q ":${PI0_PORT} " 2>/dev/null; do
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

cleanup() {
    echo ""
    echo "Shutting down server and workers..."
    kill $SERVER_PID 2>/dev/null || true
    kill $KEEPALIVE_PID 2>/dev/null || true
    kill $PID_SPATIAL $PID_OBJECT $PID_GOAL $PID_LONG 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    wait $KEEPALIVE_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

cd "$TRACE_DIR"

# =============================================================================
# Level 2: Quick sanity check
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 2: LIBERO single-episode evaluation"
echo "============================================"

python -c "
from trace.config_loader import create_task, load_task_config
from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter
from trace.runner.episode_runner import EpisodeRunner

config = load_task_config('configs/tasks/libero_spatial.yaml')
config.seed = 0
task = create_task(config)
print(f'Task: {config.name}')
print(f'Suite: {config.task_params[\"task_suite_name\"]}')
print(f'Language instruction: {task.language_instruction}')

policy = Pi0PolicyAdapter(
    host='localhost', port=$PI0_PORT, chunk_size=5,
    action_mode='passthrough', state_format='axis_angle',
)
policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())
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
# Level 3: 4 suites in PARALLEL, each with 5 task IDs
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 3: Multi-task LIBERO sweep (4 suites in parallel)"
echo "  Suites: spatial, object, goal, 10"
echo "  Task IDs: 0, 2, 4, 7, 9"
echo "  Sweep: multi_task_sweep.yaml (3 seeds, 5 eps)"
echo "============================================"

COMMON_ARGS="--sweep configs/sweeps/multi_task_sweep.yaml \
  --policy pi0 --action-mode passthrough \
  --pi0-host localhost --pi0-port $PI0_PORT \
  --chunk-size 5 --seed 0"

python -m scripts.run_evaluation \
  --task configs/tasks/libero_spatial.yaml \
  --task-ids 0 2 4 7 9 \
  $COMMON_ARGS \
  --output output/reports/spatial 2>&1 | sed 's/^/[spatial] /' &
PID_SPATIAL=$!

python -m scripts.run_evaluation \
  --task configs/tasks/libero_object.yaml \
  --task-ids 0 2 4 7 9 \
  $COMMON_ARGS \
  --output output/reports/object 2>&1 | sed 's/^/[object] /' &
PID_OBJECT=$!

python -m scripts.run_evaluation \
  --task configs/tasks/libero_goal.yaml \
  --task-ids 0 2 4 7 9 \
  $COMMON_ARGS \
  --output output/reports/goal 2>&1 | sed 's/^/[goal] /' &
PID_GOAL=$!

python -m scripts.run_evaluation \
  --task configs/tasks/libero_10.yaml \
  --task-ids 0 2 4 7 9 \
  $COMMON_ARGS \
  --output output/reports/long 2>&1 | sed 's/^/[long] /' &
PID_LONG=$!

echo "Launched 4 parallel evaluation workers:"
echo "  spatial: PID $PID_SPATIAL"
echo "  object:  PID $PID_OBJECT"
echo "  goal:    PID $PID_GOAL"
echo "  long:    PID $PID_LONG"
echo ""

# Wait for all to finish
FAILED=0
for SUITE_PID in $PID_SPATIAL $PID_OBJECT $PID_GOAL $PID_LONG; do
    wait $SUITE_PID || FAILED=$((FAILED + 1))
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED suite(s) failed."
fi

echo ""
echo "============================================"
echo "ALL SUITES COMPLETE ($FAILED failures)"
echo "Reports in: output/reports/{spatial,object,goal,long}/"
echo "============================================"
