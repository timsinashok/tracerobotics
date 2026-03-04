#!/bin/bash
#SBATCH --job-name=pi0-test
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=28:00:00
#SBATCH --output=/scratch/at5282/trace/tracerobotics/output/logs/pi0-test-%j.out
#SBATCH --error=/scratch/at5282/trace/tracerobotics/output/logs/pi0-test-%j.err

# =============================================================================
# Pi0 Testing Pipeline (sbatch)
#
# Starts the openpi server, waits for it to be ready, then runs tests
# incrementally. Uncomment test levels as you go.
#
# Usage:
#   cd /scratch/at5282/trace/tracerobotics
#   mkdir -p at5282/scratch/trace/tracerobotics/output/logs
#   sbatch scripts/test_pi0.sbatch
#
# Monitor:
#   tail -f at5282/scratch/trace/tracerobotics/output/logs/pi0-test-<jobid>.out
# =============================================================================

set -euo pipefail

OPENPI_DIR="/scratch/at5282/trace/openpi"
TRACE_DIR="/scratch/at5282/trace/tracerobotics"
PI0_PORT=8000

echo "============================================"
echo "Pi0 Test Pipeline"
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
# Level 1: Unit tests (no server needed)
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 1: Unit tests"
echo "============================================"

cd "$TRACE_DIR"
python -m pytest tests/test_pi0_adapter.py -v 2>&1
LEVEL1_EXIT=$?

if [ $LEVEL1_EXIT -ne 0 ]; then
    echo "LEVEL 1 FAILED (exit $LEVEL1_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 1 PASSED"

# =============================================================================
# Start the pi0 server (needed for levels 2+)
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

# Wait for the server to be ready (check port every 5s, timeout after 10min)
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

# Cleanup: kill the server when the script exits (success or failure)
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
# Level 2: Smoke test — verify server responds
# =============================================================================
echo "============================================"
echo "LEVEL 2: Smoke test (server connection)"
echo "============================================"

python -c "
import websockets.sync.client
from openpi_client import msgpack_numpy
import numpy as np
import time

# Connect with large ping_timeout — first inference triggers JAX JIT compilation
# which can take 60-120s. Default 20s timeout causes keepalive ping failure.
uri = 'ws://localhost:$PI0_PORT'
print('Connecting to server...')
ws = websockets.sync.client.connect(uri, compression=None, max_size=None, ping_timeout=300)
metadata = msgpack_numpy.unpackb(ws.recv())
print(f'Server metadata: {list(metadata.keys())}')

obs = {
    'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/state': np.zeros(8, dtype=np.float32),
    'prompt': 'reach the target',
}

packer = msgpack_numpy.Packer()

# First inference is slow (JAX JIT warmup)
print('Running first inference (JIT warmup, may take 1-2 min)...')
t0 = time.time()
ws.send(packer.pack(obs))
response = msgpack_numpy.unpackb(ws.recv())
t1 = time.time()
actions = response['actions']
print(f'Warmup inference took {t1-t0:.1f}s')
print(f'Action shape: {actions.shape}')
print(f'Action sample: {actions[0][:4]}')
assert actions.ndim == 2, f'Expected 2D actions, got {actions.ndim}D'

# Second inference should be fast
print('Running second inference...')
t0 = time.time()
ws.send(packer.pack(obs))
response = msgpack_numpy.unpackb(ws.recv())
t1 = time.time()
print(f'Second inference took {t1-t0:.1f}s')

ws.close()
print('LEVEL 2 PASSED')
" 2>&1
LEVEL2_EXIT=$?

if [ $LEVEL2_EXIT -ne 0 ]; then
    echo "LEVEL 2 FAILED (exit $LEVEL2_EXIT) — aborting."
    exit 1
fi

# =============================================================================
# Level 3: Single-episode evaluation (~30 seconds)
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 3: Single-episode evaluation"
echo "============================================"

python -c "
from trace.config_loader import create_task, load_task_config
from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter
from trace.runner.episode_runner import EpisodeRunner

config = load_task_config('configs/tasks/reach_pi0.yaml')
config.seed = 0
task = create_task(config)
print(f'Task initialized: {config.name}')

policy = Pi0PolicyAdapter(host='localhost', port=$PI0_PORT, chunk_size=5)
policy.load('')
policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())
policy.set_task_info(config.task_params.get('language_instruction', 'reach the target'))

runner = EpisodeRunner(task, policy, stressors=[])
result = runner.run(seed=0)
print(f'Success:              {result.success}')
print(f'Steps:                {result.total_steps}')
print(f'Reward:               {result.total_reward:.3f}')
print(f'Catastrophic failure: {result.catastrophic_failure}')
print('LEVEL 3 PASSED')
" 2>&1
LEVEL3_EXIT=$?

if [ $LEVEL3_EXIT -ne 0 ]; then
    echo "LEVEL 3 FAILED (exit $LEVEL3_EXIT) — aborting."
    exit 1
fi

# =============================================================================
# Level 4: Mini sweep (few minutes)
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 4: Mini sweep (quick_test.yaml)"
echo "============================================"

python -m scripts.run_evaluation \
  --task configs/tasks/reach_pi0.yaml \
  --sweep configs/sweeps/quick_test.yaml \
  --policy pi0 \
  --pi0-host localhost --pi0-port $PI0_PORT \
  --chunk-size 5 \
  --seed 0 \
  --output output/reports 2>&1
LEVEL4_EXIT=$?

if [ $LEVEL4_EXIT -ne 0 ]; then
    echo "LEVEL 4 FAILED (exit $LEVEL4_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 4 PASSED"

# =============================================================================
# Level 5: Full robustness sweep (~26 hours)
# =============================================================================
echo ""
echo "============================================"
echo "LEVEL 5: Full sweep (default_sweep.yaml)"
echo "============================================"

python -m scripts.run_evaluation \
  --task configs/tasks/reach_pi0.yaml \
  --sweep configs/sweeps/default_sweep.yaml \
  --policy pi0 \
  --pi0-host localhost --pi0-port $PI0_PORT \
  --chunk-size 5 \
  --seed 0 \
  --output output/reports 2>&1
LEVEL5_EXIT=$?

if [ $LEVEL5_EXIT -ne 0 ]; then
    echo "LEVEL 5 FAILED (exit $LEVEL5_EXIT) — aborting."
    exit 1
fi
echo "LEVEL 5 PASSED"

echo ""
echo "============================================"
echo "ALL ACTIVE TESTS PASSED"
echo "============================================"
