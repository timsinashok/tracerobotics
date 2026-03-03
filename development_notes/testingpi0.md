# Testing Pi0 on HPC

**Last updated:** 2026-03-03

---

## Current State

- HPC node with **A100 40GB GPU** (interactive SLURM job)
- `trace-pi0` conda env active with Python 3.11
- `trace-robotics` installed in editable mode
- `openpi` repo cloned at `/scratch/at5282/trace/openpi`

---

## Step-by-Step Setup

### 1. Get a GPU node

```bash
# Request an interactive session with an A100
srun --gres=gpu:a100:1 --mem=64G --time=04:00:00 --pty bash
conda activate trace-pi0
```

### 2. Clone openpi and install the client

```bash
cd /scratch/at5282/trace
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git submodule update --init --recursive   # if you forgot --recurse-submodules

# Install the client into the trace-pi0 conda env
pip install -e packages/openpi-client/
```

### 3. Install server dependencies

```bash
pip install uv

cd /scratch/at5282/trace/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Install lerobot transitive deps** (the openpi lockfile intentionally skips lerobot's
heavy dependencies via `sys_platform == 'never'` override, but the server imports need them):

```bash
# Install torchvision from PyTorch index (matches torch==2.7.1+cu126)
cd /tmp
uv pip install --python /scratch/at5282/trace/openpi/.venv/bin/python \
  torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# Install remaining lerobot deps (skip pynput/evdev/rerun-sdk — not needed, no C compiler on compute nodes)
uv pip install --python /scratch/at5282/trace/openpi/.venv/bin/python \
  draccus deepdiff diffusers flask gdown gymnasium h5py av pymunk pyzmq torchcodec cmake packaging
```

### 4. Start the pi0 server

The uv-managed Python can't find system CA certs, so set `SSL_CERT_FILE` for the GCS download:

```bash
cd /scratch/at5282/trace/openpi
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

First run downloads the checkpoint (~11.6 GB) to `~/.cache/openpi/`. Subsequent runs load from cache.

**Expected output when ready:**
- Model loaded onto GPU (~31 GB VRAM)
- Server listening on `0.0.0.0:8000`

**Verify the server is up:**

```bash
# In another terminal on the same node:
ss -tlnp | grep 8000
nvidia-smi   # should show ~31 GB used
```

---

## Running Tests

### Level 1: Unit tests (no server needed)

Run these first to verify the codebase is healthy:

```bash
cd /scratch/at5282/trace/tracerobotics
conda activate trace-pi0

# Pi0 adapter unit tests (uses mocked WebSocket client)
pytest tests/test_pi0_adapter.py -v

# All unit tests
pytest tests/ -v
```

### Level 2: Smoke test (server must be running)

Quick sanity check that the server responds and actions flow end-to-end.
Open a second terminal on the same GPU node:

```bash
conda activate trace-pi0
cd /scratch/at5282/trace/tracerobotics

python -c "
from openpi_client import websocket_client_policy as wcp
client = wcp.WebsocketClientPolicy('localhost', 8000)
import numpy as np
obs = {
    'observation/image': np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/wrist_image': np.zeros((224, 224, 3), dtype=np.uint8),
    'observation/state': np.zeros(8, dtype=np.float32),
    'prompt': 'reach the target',
}
action = client.infer(obs)
print(f'Action shape: {action[\"actions\"].shape}')
print(f'Action sample: {action[\"actions\"][0][:4]}')
print('Server connection OK')
"
```

### Level 3: Single-episode evaluation (quick, ~30 seconds)

Run one episode with no stressors to verify the full pipeline:

```bash
cd /scratch/at5282/trace/tracerobotics

python -c "
from trace.task_spec.reach_task import ReachTask
from trace.policy_adapter.pi0_adapter import Pi0PolicyAdapter
from trace.runner.episode_runner import EpisodeRunner
from trace.task_spec.base_task import TaskConfig

config = TaskConfig(
    name='reach',
    max_episode_steps=200,
    success_threshold=0.95,
    seed=0,
    task_params={
        'success_radius': 0.05,
        'target_x_range': [0.2, 0.6],
        'target_y_range': [-0.3, 0.3],
        'target_z_range': [0.1, 0.5],
        'catastrophic_vel_threshold': 50.0,
        'render': {
            'width': 224, 'height': 224,
            'cameras': {
                'image': 'third_person',
                'wrist_image': 'wrist_camera',
            }
        }
    },
)
task = ReachTask(config)
task.initialize()

policy = Pi0PolicyAdapter(host='localhost', port=8000, chunk_size=5)
policy.load(None)
policy.set_env(task.get_mujoco_model(), task.get_mujoco_data())
policy.set_task_info('reach the target')

runner = EpisodeRunner(task, policy, stressors=[])
result = runner.run(seed=0)
print(f'Success: {result.success}')
print(f'Steps: {result.total_steps}')
print(f'Reward: {result.total_reward:.3f}')
print(f'Catastrophic failure: {result.catastrophic_failure}')
"
```

### Level 4: Mini sweep (few minutes)

Run a small sweep with one stressor to validate the sweep pipeline before the full run:

```bash
cd /scratch/at5282/trace/tracerobotics

python -m scripts.run_evaluation \
  --task configs/tasks/reach_pi0.yaml \
  --sweep configs/sweeps/default_sweep.yaml \
  --policy pi0 \
  --pi0-host localhost --pi0-port 8000 \
  --chunk-size 5 \
  --seed 0 \
  --output output/reports
```

To run faster for validation, create a minimal sweep config:

```yaml
# configs/sweeps/quick_test.yaml
seeds: [0]
episodes_per_config: 2
stressors:
  - type: LatencyStressor
    intensities: [0.0, 0.5, 1.0]
    params:
      max_delay_steps: 10
```

Then: `--sweep configs/sweeps/quick_test.yaml`

### Level 5: Full robustness sweep

The default sweep runs 9 stressors x 7 intensities x 5 seeds x 10 episodes = **3,150 episodes**.
At ~30s per episode, this takes approximately **26 hours**. Use tmux or a batch SLURM job.

```bash
# In tmux (so it survives SSH disconnects):
tmux new -s pi0-eval

# Terminal 1: server
cd /scratch/at5282/trace/openpi
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

# Terminal 2 (Ctrl-b c): evaluation
conda activate trace-pi0
cd /scratch/at5282/trace/tracerobotics
python -m scripts.run_evaluation \
  --task configs/tasks/reach_pi0.yaml \
  --sweep configs/sweeps/default_sweep.yaml \
  --policy pi0 \
  --pi0-host localhost --pi0-port 8000 \
  --output output/reports
```

Report output goes to `output/reports/`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'datasets'` | `uv pip install datasets jsonlines` from the openpi dir |
| `ModuleNotFoundError: No module named 'torchvision'` | Install from PyTorch index: `uv pip install --python .venv/bin/python torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126` |
| `ModuleNotFoundError: No module named 'draccus'` | Install lerobot transitive deps (see step 3 above) |
| `SSL: CERTIFICATE_VERIFY_FAILED` on GCS download | `export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt` |
| `evdev` build fails (no `cc` compiler) | Ignore — openpi intentionally excludes evdev/pynput. Install lerobot deps individually, not via `pip install lerobot` |
| Server exits immediately | Check `nvidia-smi` — needs ~31 GB VRAM. A100 40GB works; smaller GPUs may not |
| `Unable to initialize backend 'rocm'` / `'tpu'` | Harmless warnings, ignore |
| Server hangs at "Downloading..." | First-run checkpoint download is 11.6 GB. Check `du -sh ~/.cache/openpi/` for progress |
| Actions are all zeros | Verify server is running (`ss -tlnp \| grep 8000`) and policy loaded successfully |

---

## Notes

- The `pi05_libero` checkpoint is trained on the LIBERO benchmark (MuJoCo simulation)
- Server uses ~31 GB VRAM; A100 40GB has headroom for one model instance
- Action chunking: each server call returns `chunk_size` (default 5) actions, amortizing ~73ms inference latency
- Images are rotated 180 degrees before sending to match LIBERO convention
- The adapter uses Jacobian transpose control to convert Cartesian deltas to joint-space actions
- Checkpoint is cached at `~/.cache/openpi/` after first download
