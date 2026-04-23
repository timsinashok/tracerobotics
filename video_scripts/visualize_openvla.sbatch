#!/bin/bash
#SBATCH --job-name=viz-openvla
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/at5282/trace/tracerobotics/video_scripts/logs/viz-openvla-%j.out
#SBATCH --error=/scratch/at5282/trace/tracerobotics/video_scripts/logs/viz-openvla-%j.err

# =============================================================================
# Visualize OpenVLA episodes on LIBERO tasks
#
# No server needed — OpenVLA runs locally on GPU.
# Edit the EPISODES section below to choose what to visualize.
#
# Usage:
#   cd /scratch/at5282/trace/tracerobotics
#   mkdir -p video_scripts/logs
#   sbatch video_scripts/visualize_openvla.sbatch
# =============================================================================

set -euo pipefail

TRACE_DIR="/scratch/at5282/trace/tracerobotics"
VIDEO_DIR="$TRACE_DIR/output/videos"

echo "============================================"
echo "OpenVLA Episode Visualization"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# --- Environment setup ---
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate trace-pi0
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

mkdir -p "$VIDEO_DIR"

echo ""
echo "[GPU Info]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$TRACE_DIR"

# =============================================================================
# EPISODES — edit this section to choose what to visualize
#
# Note: OpenVLA checkpoint is spatial-only, so only libero_spatial tasks work.
# =============================================================================

echo ""
echo "============================================"
echo "Recording episodes..."
echo "============================================"

# --- Baseline episodes ---

# Spatial task 0 — baseline
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy openvla \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 4 — baseline
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 4 \
  --policy openvla \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# --- Stressor episodes ---

# Spatial task 0 — latency at 0.3 (where OpenVLA breaks)
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy openvla \
  --stressor LatencyStressor --intensity 0.3 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 0 — image noise at 0.5
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy openvla \
  --stressor ImageNoiseStressor --intensity 0.5 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

# Spatial task 0 — dropout at 0.3
python -m scripts.visualize_episode \
  --task configs/tasks/libero_spatial.yaml --task-id 0 \
  --policy openvla \
  --stressor DropoutStressor --intensity 0.3 \
  --seed 0 --resolution 512 \
  --output "$VIDEO_DIR"

echo ""
echo "============================================"
echo "All videos saved to: $VIDEO_DIR"
echo "============================================"
ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || ls -lh "$VIDEO_DIR"/*.gif 2>/dev/null || echo "No videos found"
