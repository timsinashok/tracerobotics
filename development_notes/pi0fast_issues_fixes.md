# Pi0-FAST Integration: Issues & Fixes

## Overview

Integrating `lerobot/pi0fast-libero` (HuggingFace) into the Trace evaluation
framework required solving a chain of dependency and compatibility issues.
The model uses the LeRobot library (PyTorch) instead of OpenPI (JAX), so it
runs locally on GPU with no server needed.

---

## Issue 1: `evdev` build failure during `pip install -e lerobot`

**Error:** `evdev` (a Linux input device library) fails to compile on HPC nodes
that lack `/dev/input`.

**Cause:** `pynput` is a core lerobot dependency, and it pulls in `evdev` on
Linux. `pynput` is only needed for real robot teleoperation, not inference.

**Fix:** Install lerobot with `--no-deps`, then install all dependencies
manually *except* `pynput`:

```bash
pip install poetry-core
pip install --no-deps -e "$LEROBOT_DIR"
pip install "transformers>=5.3.0" "einops>=0.8.0" ...  # all deps except pynput
```

---

## Issue 2: `poetry.core.masonry.api` not found

**Error:** `BackendUnavailable: Cannot import 'poetry.core.masonry.api'`

**Cause:** lerobot uses `poetry-core` as its build backend but it wasn't
installed in the conda environment.

**Fix:** `pip install poetry-core` before installing lerobot.

---

## Issue 3: `No module named 'lerobot.policies'`

**Error:** Import fails because Python finds the wrong `lerobot/` directory.

**Cause:** The lerobot repo has two package layouts — an old `lerobot/` at the
root and the real code under `src/lerobot/`. The `-e` install finds the old one.

**Fix:** Add `src/` to PYTHONPATH:
```bash
export PYTHONPATH="$LEROBOT_DIR/src:${PYTHONPATH:-}"
```

---

## Issue 4: Missing transitive dependencies (`serial`, `av`, `accelerate`, etc.)

**Error:** Chain of `ModuleNotFoundError` for `serial`, `accelerate`, `av`, etc.

**Cause:** lerobot's `__init__.py` eagerly imports ALL policies at module load
time, pulling in hardware/dataset dependencies not needed for inference.
Installing with `--no-deps` skips all of these.

**Fix:** Install the full dependency list from lerobot's `pyproject.toml`:
```bash
pip install "av>=14.2.0" "pyserial" "accelerate" "gymnasium==0.29.1" "scipy" ...
```

---

## Issue 5: Gated HuggingFace model (PaliGemma)

**Error:** `GatedRepoError: Cannot access gated repo google/paligemma-3b-pt-224`

**Cause:** PaliGemma is a gated model requiring:
1. A HuggingFace account with accepted license terms
2. An authentication token

**Fix:**
1. Accept terms at https://huggingface.co/google/paligemma-3b-pt-224
2. Set `export HF_TOKEN=<your_token>` in the sbatch script

**Security note:** Move the token to `~/.cache/huggingface/token` after
initial setup so it's not in version-controlled scripts.

---

## Issue 6: `transformers` version incompatibility (THE BIG ONE)

This was the most complex issue, with multiple sub-problems:

### 6a: `siglip.check` module not found

**Error:**
```
ImportError: cannot import name 'check' from 'transformers.models.siglip'
ValueError: An incorrect transformer version is used
```

**Cause:** The old lerobot code (pre-PR #2964) expected a custom
`transformers.models.siglip.check` module that doesn't exist in any stock
`transformers` release. This was a validation check for a planned
"transformers_replace" mechanism that was never fully implemented.

### 6b: `PaliGemmaForConditionalGeneration` has no `language_model` attribute

**Error:**
```
AttributeError: 'PaliGemmaForConditionalGeneration' object has no attribute 'language_model'
```

**Cause:** The old lerobot code assumed PaliGemma had a `.language_model`
attribute. In transformers v5, the API changed. The latest lerobot (post-PR
#2964) fixes this by monkey-patching the attribute:
```python
self.paligemma.model.language_model = PiGemmaModel(text_config)
```

### 6c: `bfloat16` vs `float32` dtype mismatch

**Error:**
```
RuntimeError: expected scalar type Float but found BFloat16
```

**Cause:** Model weights are saved in `bfloat16` but SigLIP's `LayerNorm`
needs matching dtypes between weights and inputs.

**Fix:** Use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around
inference calls. Do NOT cast the model to float32 — this corrupts the model
and causes all-`<bos>` token output.

### 6d: Model produces all `<bos>` tokens

**Error:**
```
AssertionError: Token sequence does not start with ['Action', ':']: ['<bos>', '<bos>', ...]
```

**Cause:** This was the symptom of using old lerobot code with stock
`transformers`. The vision tower (SigLIP) produced garbage features because
the old code didn't properly initialize the model for the transformers version
in use.

### SOLUTION

**Update lerobot to commit `3e451202` or later** (post-PR #2964, pre-Python
3.12 requirement) and use **`transformers>=5.3.0`**.

```bash
cd /path/to/lerobot_src
git reset --hard 3e451202  # has transformers v5 fix, supports Python 3.11
```

Key commits:
- `f0d2b37b` — `chore(dependencies): bump transformers v5 (#2964)` — THE FIX
- `e489ba24` — `feat(dependencies): require Python 3.12+` — avoid this if on 3.11

References:
- https://github.com/huggingface/lerobot/issues/2319
- https://github.com/huggingface/lerobot/issues/2641
- https://github.com/huggingface/lerobot/pull/2964

---

## Issue 7: Image feature key mismatch

**Error:**
```
ValueError: All image features are missing from the batch.
(image_features: {'observation.images.image': ..., 'observation.images.image2': ...})
```

**Cause:** The model config uses `observation.images.image` and
`observation.images.image2` as camera keys. OpenPI uses `base_0_rgb` and
`left_wrist_0_rgb`. The lerobot eval docs use `--rename_map` for this.

**Fix:** Use the model's native keys in the adapter's `_build_batch()`:
```python
return {
    "observation.images.image": base_tensor,      # not base_0_rgb
    "observation.images.image2": wrist_tensor,     # not left_wrist_0_rgb
    "observation.state": state_tensor,
    "task": self._prompt,
}
```

---

## Issue 8: Missing `dataset_stats` / preprocessor pipeline

**Observation:** The model config specifies `"STATE": "MEAN_STD"` normalization
but there is no `dataset_stats.json` in the HF model repo. Manually building
the preprocessor with `make_pi0_fast_pre_post_processors()` without stats
caused the model to produce garbage (0% success) because the state was not
normalized before discretization.

**Root cause:** The model repo ships the **full preprocessor pipeline** as
saved artifacts, not raw stats:

```
policy_preprocessor.json
policy_preprocessor_step_2_normalizer_processor.safetensors   # <-- the stats!
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

**Fix:** Load the saved pipelines directly from HF instead of building them:
```python
from lerobot.processor.pipeline import DataProcessorPipeline

self._preprocess = DataProcessorPipeline.from_pretrained(
    self._model_id,
    config_filename="policy_preprocessor.json",
    overrides={"device_processor": {"device": self._device}},
)
self._postprocess = DataProcessorPipeline.from_pretrained(
    self._model_id,
    config_filename="policy_postprocessor.json",
    overrides={"device_processor": {"device": "cpu"}},
)
```

---

## Issue 9: Postprocessor expects dict, not Tensor

**Error:**
```
ValueError: EnvTransition must be a dictionary. Got Tensor
```

**Cause:** `predict_action_chunk()` returns a raw `torch.Tensor`, but the
`DataProcessorPipeline` postprocessor expects a dict (the `EnvTransition`
format).

**Fix:** Wrap the action tensor in a dict before postprocessing:
```python
action_dict = {"action": actions}
result = self._postprocess(action_dict)
actions = result["action"]
```

---

## Issue 10: `PYTHONPATH` unbound variable

**Error:** `/var/spool/slurmd/.../slurm_script: line 46: PYTHONPATH: unbound variable`

**Cause:** `set -euo pipefail` in the sbatch script treats unset variables as
errors.

**Fix:** Use `${PYTHONPATH:-}` instead of `$PYTHONPATH`.

---

## Issue 11: Latest lerobot requires Python 3.12+

**Error:** `ERROR: Package 'lerobot' requires a different Python: 3.11.14 not in '>=3.12'`

**Cause:** lerobot's latest `main` branch (post-commit `e489ba24`) bumped the
minimum Python version to 3.12. Our HPC conda env uses 3.11.

**Fix:** Pin lerobot to commit `3e451202` — the last commit that has both the
transformers v5 fix (PR #2964) and Python 3.11 support:

```bash
cd /path/to/lerobot_src
git reset --hard 3e451202
```

---

## Summary: Required Environment

```bash
# Python 3.11 compatible
# lerobot at commit 3e451202 (post-transformers-v5, pre-Python-3.12)
# transformers >= 5.3.0
# All lerobot deps installed EXCEPT pynput
# HF_TOKEN set for gated model access
# PYTHONPATH includes lerobot_src/src/
```

## Files

- `trace/policy_adapter/pi0fast_adapter.py` — The adapter
- `scripts/test_libero_pi0fast.sbatch` — SLURM job script
- `scripts/run_evaluation.py` — `pi0fast` registered in POLICY_REGISTRY
