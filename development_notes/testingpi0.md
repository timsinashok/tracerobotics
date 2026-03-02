# Testing Pi0 on HPC

**Last updated:** 2026-03-03

---

## Current State

- HPC node `cn002` with **A100 40GB GPU** (interactive SLURM job)
- `trace-pi0` conda env active with Python 3.11
- `trace-robotics` installed in editable mode
- The `openpi` repo is **not cloned yet** — needed for server + `openpi_client`

---

## Step-by-Step Setup

### 1. Clone openpi and install the client (on HPC)

```bash
cd /scratch/at5282/trace
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
pip install -e packages/openpi-client/
```

### 2. Run the pi0 server (on HPC — needs GPU)

The server **must** run on HPC because it needs the A100 GPU for inference. You need `uv` and the server's own dependencies:

```bash
# Install uv if not already available
pip install uv

# From the openpi repo, sync server dependencies and start serving
cd /scratch/at5282/trace/openpi
uv sync
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

### 3. Run the evaluation

Two options:

#### Option A: Run evals on HPC too (recommended — simplest)

Open a second terminal/tmux pane on the same node, or submit a second SLURM job:

```bash
conda activate trace-pi0
cd /scratch/at5282/trace/tracerobotics
python -m scripts.run_evaluation \
  --task configs/tasks/reach_pi0.yaml \
  --sweep configs/sweeps/default_sweep.yaml \
  --policy pi0 \
  --pi0-host localhost --pi0-port 8000
```

#### Option B: Run evals from Mac (connecting to HPC)

This works because the pi0 adapter connects via WebSocket.

**1. SSH tunnel from Mac to HPC node:**

```bash
# On your Mac:
ssh -L 8000:cn002:8000 at5282@<hpc-login-host>
```

**2. Install trace-robotics on Mac (needs MuJoCo + Python 3.11):**

```bash
conda create -n trace-pi0 python=3.11 -y
conda activate trace-pi0
cd /path/to/tracerobotics
pip install -e ".[dev]"
pip install -e /path/to/openpi/packages/openpi-client/
```

**3. Run the eval pointing to localhost:**

```bash
python -m scripts.run_evaluation \
  --task configs/tasks/reach_pi0.yaml \
  --sweep configs/sweeps/default_sweep.yaml \
  --policy pi0 \
  --pi0-host localhost --pi0-port 8000
```

---

## Notes

- Option A is recommended — running everything on HPC avoids network latency in WebSocket calls
- The `pi05_libero` checkpoint is already trained on MuJoCo simulation (LIBERO benchmark)
- Server needs ~22 GB VRAM; the A100 40GB has plenty of headroom
- Action chunking amortizes inference latency (~73ms/chunk, 5 actions per call)
