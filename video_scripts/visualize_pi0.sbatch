#!/bin/bash
#SBATCH --job-name=viz-pi0
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/at5282/trace/tracerobotics/video_scripts/logs/viz-pi0-%j.out
#SBATCH --error=/scratch/at5282/trace/tracerobotics/video_scripts/logs/viz-pi0-%j.err

# =============================================================================
# Visualize pi0.5 episodes on LIBERO tasks
#
# Records videos of individual episodes with optional stressors.
# Edit the EPISODES section below to choose what to visualize.
#
# Usage:
#   cd /scratch/at5282/trace/tracerobotics
#   mkdir -p video_scripts/logs
#   sbatch video_scripts/visualize_pi0.sbatch
# =============================================================================

set -euo pipefail

OPENPI_DIR="/scratch/at5282/trace/openpi"
TRACE_DIR="/scratch/at5282/trace/tracerobotics"
VIDEO_DIR="$TRACE_DIR/output/videos"
PI0_PORT=8000

echo "============================================"
echo "Pi0.5 Episode Visualization"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# --- Environment setup ---
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate trace-pi0
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export OPENPI_DATA_HOME=/scratch/at5282/.cache/openpi

mkdir -p "$VIDEO_DIR"

echo ""
echo "[GPU Info]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# =============================================================================
# Start the pi0 server
# =============================================================================
echo ""
echo "Starting pi0 server on port $PI0_PORT..."

cd "$OPENPI_DIR"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero &
SERVER_PID=$!

# Wait for server
echo "Waiting for server (PID $SERVER_PID)..."
TIMEOUT=600
ELAPSED=0
while ! ss -tlnp | grep -q ":${PI0_PORT} " 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died."
        exit 1
    fi
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Server timeout after ${TIMEOUT}s."
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "Server ready (${ELAPSED}s)."

cleanup() {
    echo "Shutting down server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

cd "$TRACE_DIR"

# =============================================================================
# EPISODES — edit this section to choose what to visualize
#
# Each call records one episode as a video.
# Format:
#   python -m scripts.visualize_episode \
#     --task <config> [--task-id <id>] \
#     --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
#     [--stressor <name> --intensity <0.0-1.0>] \
#     [--seed <n>] [--resolution <px>] \
#     --output "$VIDEO_DIR"
# =============================================================================

echo ""
echo "============================================"
echo "Recording episodes..."
echo "============================================"

# --- Baseline episodes (no stressor) ---

# Spatial task 0 — baseline
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 4 — baseline
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 4 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Object task 0 — baseline
python -m scripts.visualize_episode \
  --task configs/tasks/libero_object.yaml --task-id 0 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# --- Stressor episodes (showing degradation) ---

# Spatial task 0 — latency at breakpoint (0.3)
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --stressor LatencyStressor --intensity 0.3 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 0 — latency at high intensity (0.7)
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --stressor LatencyStressor --intensity 0.7 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 0 — dropout at breakpoint (0.3)
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --stressor DropoutStressor --intensity 0.3 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 0 — image noise at 0.5
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --stressor ImageNoiseStressor --intensity 0.5 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 0 — occlusion at 0.5
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy pi0 --pi0-host localhost --pi0-port $PI0_PORT \
  --stressor OcclusionStressor --intensity 0.5 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

echo ""
echo "============================================"
echo "All videos saved to: $VIDEO_DIR"
echo "============================================"
ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || ls -lh "$VIDEO_DIR"/*.gif 2>/dev/null || echo "No videos found"
