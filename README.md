---
title: Medical Triage OpenEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - healthcare
  - triage
  - reinforcement-learning
  - agent-evaluation
---

# 🏥 Medical Triage OpenEnv

An **OpenEnv** environment where AI agents learn to triage emergency department patients — a real, high-stakes clinical task performed thousands of times daily in hospitals worldwide.

---

## 🎯 Why This Environment?

Emergency triage errors cause preventable deaths and waste critical resources. Current AI systems lack structured environments to train and evaluate triage decision-making. This environment fills that gap: a rigorous, reproducible benchmark for clinical reasoning agents.

- **Under-triage a heart attack as "routine"?** Heavy penalty.
- **Correctly identify a stroke within the thrombolysis window?** Full reward.
- **Overcome anchoring bias on an anxious patient with NSTEMI?** Bonus points.

---

## 📋 Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `task_easy` | 🟢 Easy | 5 | Triage a single patient — assign priority, conditions, actions |
| `task_medium` | 🟡 Medium | 7 | Batch triage 5 simultaneous patients + rank by urgency |
| `task_hard` | 🔴 Hard | 10 | Ambiguous NSTEMI/panic disorder case — request tests, override anchoring bias |

---

## 🔌 API (OpenEnv Spec)

### `POST /reset`
Start a new episode.
```json
{
  "task_id": "task_easy",
  "session_id": "my_session"
}
```

### `POST /step`
Submit a triage action.
```json
{
  "session_id": "my_session",
  "action": {
    "patient_id": "P001",
    "priority": "CRITICAL",
    "suspected_conditions": ["Acute Myocardial Infarction"],
    "immediate_actions": ["ECG", "IV access", "aspirin", "oxygen"],
    "reasoning": "67M with chest pain radiating to left arm, diaphoresis, elevated BP, reduced SpO2. Classic STEMI presentation. Immediate cath lab activation required.",
    "request_more_info": null,
    "updated_priority": null
  }
}
```

### `GET /state`
Get current episode state without acting.
```
GET /state?session_id=my_session
```

---

## 🎬 Action Space

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | string | Patient ID to triage |
| `priority` | enum | `CRITICAL` / `URGENT` / `SEMI_URGENT` / `ROUTINE` |
| `suspected_conditions` | list[str] | Suspected diagnoses |
| `immediate_actions` | list[str] | Recommended interventions |
| `reasoning` | string | Clinical reasoning (scored) |
| `request_more_info` | list[str] \| null | Tests to request (hard task) |
| `updated_priority` | enum \| null | Priority update after new results (hard task) |

**Priority Levels:**
- `CRITICAL` — Immediately life-threatening, resuscitation bay
- `URGENT` — Serious, seen within 15 minutes
- `SEMI_URGENT` — Stable, seen within 30 minutes
- `ROUTINE` — Non-urgent

---

## 👁️ Observation Space

```json
{
  "task_id": "task_easy",
  "task_name": "Single Patient Triage",
  "step_count": 1,
  "max_steps": 5,
  "patients": [
    {
      "patient_id": "P001",
      "age": 67,
      "sex": "Male",
      "chief_complaint": "Chest pain radiating to left arm, sweating",
      "vitals": {
        "heart_rate": 112,
        "blood_pressure": "160/95",
        "oxygen_saturation": 94,
        "temperature": 37.1,
        "respiratory_rate": 22,
        "gcs": 15
      },
      "medical_history": ["Hypertension", "Hypercholesterolemia"],
      "current_medications": ["Amlodipine 5mg"],
      "additional_notes": "Diaphoretic on arrival. Onset 30 minutes ago.",
      "arrival_mode": "ambulance"
    }
  ],
  "feedback": "Step feedback from previous action",
  "hints": ["Hint 1", "Hint 2"],
  "additional_info": null,
  "best_score_so_far": 0.0,
  "done": false
}
```

---

## 🏆 Reward Function

The reward (0.0–1.0) is a weighted composite with partial progress at every step:

| Component | Weight | Description |
|-----------|--------|-------------|
| Priority correctness | 40% | Correct level = 1.0; under-triage penalised 0.4/level; over-triage 0.15/level |
| Condition identification | 25% | Keyword overlap with true diagnosis |
| Immediate actions | 25% | Coverage of critical interventions |
| Clinical reasoning | 10% | Substantive reasoning required |
| Efficiency bonus | +5% | Bonus for solving in fewer steps |

**Key design: asymmetric penalty.** Missing a CRITICAL patient costs 0.4 reward per severity level — far more than over-triaging. This mirrors real medical ethics where under-triage is more dangerous than caution.

---

## 🚀 Setup

### Run locally
```bash
git clone <your-repo>
cd medical-triage-openenv
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t medical-triage-openenv .
docker run -p 7860:7860 medical-triage-openenv
```

### Run baseline inference
```bash
export HF_TOKEN=your_api_key
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

---

## 📊 Baseline Scores

Tested with `gpt-4o-mini`:

| Task | Score | Notes |
|------|-------|-------|
| `task_easy` | ~0.82 | Strong on clear-cut cases like STEMI |
| `task_medium` | ~0.68 | Struggles with correct ranking order |
| `task_hard` | ~0.55 | Anchoring bias is a genuine challenge |

The hard task genuinely challenges frontier models — anchoring on "anxiety disorder" and missing NSTEMI is a common failure mode.

---

## 🗂️ Project Structure

```
medical-triage-openenv/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI server (OpenEnv endpoints)
│   ├── models.py        # Pydantic typed models (Action, Observation, Reward)
│   ├── environment.py   # Core step()/reset()/state() logic
│   ├── cases.py         # Patient case definitions (all 3 tasks)
│   └── graders.py       # Deterministic task graders (0.0–1.0)
├── inference.py         # Baseline agent (OpenAI client)
├── openenv.yaml         # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔬 OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

All three tasks pass automated validation with deterministic, reproducible scores.
