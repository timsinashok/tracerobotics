# 🎯 MVP GOAL (very precise)

> **Input:** a multimodal robot policy checkpoint
> **Output:** a short, brutal report that shows **non-obvious failure modes under stress**

If you can do that **once**, convincingly, you have a company.

---

## 🔹 What to build (MVP scope)

### 1️⃣ Supported policy type (pick ONE)

**Multimodal manipulation policy**

* Vision (+ optional language)
* Continuous control output
* Runs in simulation loop

Do **not** support:

* locomotion
* humanoids
* navigation
* language-only policies

👉 Narrow = faster + credible.

---

### 2️⃣ Simulator (pick ONE first)

Choose **one**, abstract the rest.

Best choice:

* **MuJoCo**

  * Fast
  * Deterministic
  * Widely trusted in research
  * Easy CI-style execution

(Isaac comes later, not now.)

---

### 3️⃣ Tasks (3–5 total, that’s enough)

Pick **canonical manipulation tasks**:

* Pick & place
* Reach + grasp
* Drawer open / door open
* Simple tool use (optional)

Each task should have:

* clear success metric
* fixed episode length
* deterministic reset (before stressors)

---

### 4️⃣ Stressors (THIS IS YOUR MOAT)

Implement **5 stressors**, cleanly:

Must-have:

1. **Action latency** (e.g. 50–200ms)
2. **Observation dropout** (camera blackout / noise)
3. **Physics shift** (friction, mass, damping)
4. **Embodiment mismatch** (slight arm length / joint limit change)
5. **Long-horizon drift** (success decay over time)

Each stressor should be:

* parameterized
* sweepable
* reproducible

---

### 5️⃣ Metrics (simple but sharp)

Do **not** invent fancy metrics yet.

Track:

* success rate
* time-to-success
* catastrophic failure rate
* recovery rate (after perturbation)
* variance across seeds

The key is **comparative degradation**, not absolute numbers.

---

### 6️⃣ Output = the product

Your MVP is **not code**, it’s the **report**.

Each run produces:

* baseline vs stressed performance
* failure curves
* “breakpoints” (e.g. fails after 120ms latency)
* short written interpretation

Think:

> “Here is exactly where your policy stops being reliable.”

---

## 🧱 How to build it (architecture)

Minimal but correct:

```
policy_adapter/
  └── wraps checkpoint → act(obs)

task_spec/
  └── env + success condition

stressor_engine/
  ├── latency.py
  ├── dropout.py
  ├── physics_shift.py
  └── embodiment.py

runner/
  └── sweeps + seeds

metrics/
  └── aggregate + compare

report/
  └── auto-generated PDF / markdown
```

You want **headless execution**, no UI.

---

## 🗓️ 3-MONTH EXECUTION PLAN (weekly)

### Month 1 — Core system

* Week 1: task + policy adapter
* Week 2: MuJoCo env + runner
* Week 3: 2 stressors implemented
* Week 4: metrics + basic plots

🎯 End of month 1:
You can break *your own* dummy policy.

---

### Month 2 — Credible evaluation

* Add remaining stressors
* Add 2–3 tasks
* Clean failure attribution
* Make report readable & sharp

🎯 End of month 2:
You can evaluate a **real research checkpoint**.

---

### Month 3 — Design partner ready

* Run with 1 real external policy
* Tighten results
* Remove anything flaky
* Polish report (not UI)

🎯 End of month 3:
You have:

> “We evaluated X. Here’s what broke. They didn’t expect it.”

That’s YC-ready.

---

## 🗺️ 6-MONTH ROADMAP (after MVP)

### Months 4–6

* Add second simulator (**Isaac Sim**)
* Cross-embodiment tests
* CI-style regression mode
* Private dashboard (optional)
* Start charging

Do **not** add:

* public benchmarks
* leaderboards
* many robots
* hardware

---

## ⚠️ Common traps to avoid

* ❌ Building a simulator
* ❌ Supporting many policy formats
* ❌ Too many tasks
* ❌ Fancy UI
* ❌ Public shaming (“your model fails”)


