"""
Typed Pydantic models for the Medical Triage OpenEnv environment.
Implements the full OpenEnv spec: Action, Observation, Reward.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Enums ─────────────────────────────────────────────────────────────────────

class PriorityLevel(str, Enum):
    CRITICAL = "CRITICAL"   # Level 1 - Immediate, life-threatening
    URGENT   = "URGENT"     # Level 2 - Serious, seen within 15 min
    SEMI     = "SEMI_URGENT"# Level 3 - Stable, seen within 30 min
    ROUTINE  = "ROUTINE"    # Level 4 - Non-urgent


class TaskID(str, Enum):
    EASY   = "task_easy"
    MEDIUM = "task_medium"
    HARD   = "task_hard"


# ─── Patient Data ───────────────────────────────────────────────────────────────

class Vitals(BaseModel):
    heart_rate: Optional[int]   = Field(None, description="bpm")
    blood_pressure: Optional[str] = Field(None, description="systolic/diastolic mmHg")
    oxygen_saturation: Optional[int] = Field(None, description="SpO2 %")
    temperature: Optional[float] = Field(None, description="°C")
    respiratory_rate: Optional[int] = Field(None, description="breaths/min")
    gcs: Optional[int] = Field(None, description="Glasgow Coma Scale 3-15")


class PatientCase(BaseModel):
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    vitals: Vitals
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    additional_notes: Optional[str] = None
    arrival_mode: str = "walk-in"   # walk-in, ambulance, transfer
    true_priority: PriorityLevel    # ground truth, hidden from agent
    true_condition: str             # ground truth, hidden from agent


# ─── OpenEnv Core Models ────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    Action submitted by the agent each step.
    The agent reads a patient case and decides how to triage them.
    """
    patient_id: str = Field(..., description="ID of the patient being triaged")
    priority: PriorityLevel = Field(..., description="Assigned triage priority level")
    suspected_conditions: List[str] = Field(
        ..., description="List of suspected conditions/diagnoses"
    )
    immediate_actions: List[str] = Field(
        ..., description="Immediate interventions recommended (e.g. ECG, IV access, oxygen)"
    )
    reasoning: str = Field(
        ..., description="Clinical reasoning behind the triage decision"
    )
    request_more_info: Optional[List[str]] = Field(
        default=None,
        description="For hard task only: list of additional tests or info needed"
    )
    updated_priority: Optional[PriorityLevel] = Field(
        default=None,
        description="For hard task: updated priority after receiving new information"
    )


class TriageObservation(BaseModel):
    """
    Observation returned to the agent after each step.
    """
    task_id: str
    task_name: str
    step_count: int
    max_steps: int
    patients: List[PatientCase]
    feedback: str
    hints: List[str] = Field(default_factory=list)
    additional_info: Optional[Dict[str, Any]] = None   # revealed in hard task
    best_score_so_far: float = 0.0
    done: bool = False


class TriageReward(BaseModel):
    """
    Reward signal with partial progress breakdown.
    """
    value: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


class StepResult(BaseModel):
    """Result returned by step()."""
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Result returned by state()."""
    task_id: str
    step_count: int
    max_steps: int
    current_patients: List[PatientCase]
    best_score: float
    done: bool
    episode_rewards: List[float]
