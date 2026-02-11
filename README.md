# Video-to-Constraint Engine (V2C)
### Extracting Latent Physical Structure from Passive Video

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Stage](https://img.shields.io/badge/Current_Stage-Phase_1:_2D_Spatial_Logic-orange) ![License](https://img.shields.io/badge/License-MIT-green)

> **The "Scale AI" for Physics.**  
> Transforming raw RGB video into structured, time-indexed **Constraint Graphs** for robotics pre-training and world-model generation.

---

## 🏗 The Problem
Current robotic learning pipelines suffer from a **Physical Semantic Gap**:
*   **Raw Video (YouTube/Ego4D)** contains pixels, but no explicit physical information (forces, friction, connection).
*   **Teleoperation** is accurate but unscalable.
*   **Simulation** suffers from the sim-to-real gap.

**V2C** bridges this gap by inferring a **Constraint Graph ($G_t$)** from passive video. We don't just track objects; we track the *rules* governing their interaction (e.g., `Rigid`, `Revolute`, `Planar Sliding`).

## 📊 The Data Product
Instead of pixels, V2C outputs a temporal graph usable for Behavior Cloning (BC) or VLA training:

```json
{
  "frame": 145,
  "timestamp": "4.2s",
  "entities": ["hand_right", "knife", "bread"],
  "constraints": [
    {
      "type": "RIGID_GRASP",
      "actors": ["hand_right", "knife"],
      "confidence": 0.92
    },
    {
      "type": "PLANAR_SLIDE",
      "actors": ["knife", "bread"],
      "normal_vector": [0, 0, 1]
    }
  ]
}
```

---

## 🗺️ Roadmap & Development Phases

We are building this engine in 5 distinct layers of abstraction.

### ✅ Phase 1: The "Touch" Detector (2D Spatial Logic)
**Current Status: Active MVP**
*   **Goal:** Establish spatial proximity logic and contact onset detection.
*   **Tech:** YOLO-World (Open-Vocabulary Object Detection) + MediaPipe (Hand Tracking) + IoU Heuristics.
*   **Output:** Boolean Contact State (Contact / No Contact).
*   **Why YOLO-World?** Standard YOLO only detects 80 predefined objects. YOLO-World uses text prompts to detect ANY physical object, making it ideal for general-purpose physics.

### 🚧 Phase 2: The "Geometry" Layer (3D Lift)
*   **Goal:** Move from 2D pixels to 3D metric space to solve occlusion and hovering errors.
*   **Tech:** Monocular Depth Estimation (ZoeDepth/Metric3D) + Camera Motion Cancellation.
*   **Output:** Relative 3D Trajectories $(x, y, z)$.

### 🔮 Phase 3: The "Constraint" Classifier (Physics Inference)
*   **Goal:** Classify the *type* of interaction based on relative motion gradients.
*   **Logic:**
    *   `v_obj == 0` → Static
    *   `v_obj == v_hand` → Rigid Grasp
    *   `v_obj | v_hand` (constrained axis) → Hinge/Slide
*   **Output:** Labeled Constraint Types.

### ⚙️ Phase 4: The Data Factory (Scale)
*   **Goal:** Automated batch processing of large datasets (Ego4D/Epic-Kitchens).
*   **Tech:** Temporal Smoothing (Kalman Filters) + JSON/Parquet Serialization.
*   **Output:** The "100-Grasp Dataset" (Structured training data).

### 🚀 Phase 5: The Interface (Product MVP)
*   **Goal:** Interactive Dashboard for robotics engineers.
*   **Features:** Video overlay + Scrolling Constraint Timeline + Sim-Preview.
*   **Pitch:** "The ImageNet for Physical Interaction."

---

## ⚡ Quick Start (Phase 1)

### Prerequisites
*   Python 3.8+
*   Webcam or Video File

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-constraint-engine.git
cd video-constraint-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe ultralytics numpy
```

### Running the Detector
This will launch the Phase 1 Real-time Contact Detector using your webcam.

```bash
python detector.py
```

*   **Green Box:** Free Motion (Hand is moving freely).
*   **Red Box:** Contact Detected (Hand is interacting with an object).

---

## 🧠 Technical Approach (Constraint Theory)

We model the scene as a dynamic graph $G_t = (V, E)$.
*   **Nodes ($V$):** Rigid bodies (Hands, Tools, Objects, Support Surfaces).
*   **Edges ($E$):** Physical constraints active at time $t$.

Unlike standard Computer Vision which asks *"What is this object?"*, V2C asks *"How is this object constrained?"*

| Constraint Type | Physical Meaning | DoF Removed |
| :--- | :--- | :--- |
| **None** | Free fall / Ballistic | 0 |
| **Point Contact** | Touching but not grasping | 1 (Normal) |
| **Planar Slide** | Object on table | 3 (z, roll, pitch) |
| **Hinge** | Door / Laptop | 5 (All except rotation axis) |
| **Rigid** | Grasping | 6 (Fully locked to hand) |

---

## 🤝 Contributing
We are currently focused on **Phase 2 (Depth Integration)**.
If you have experience with Monocular Depth or 3D Trajectory Optimization, please open a PR.

## 📄 License
MIT License. Open for research and commercial use.