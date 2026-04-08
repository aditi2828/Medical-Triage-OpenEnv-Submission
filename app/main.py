"""
FastAPI server for the Medical Triage OpenEnv environment.
Exposes standard OpenEnv endpoints: /reset, /step, /state
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.environment import MedicalTriageEnv
from app.models import TriageAction, StepResult, StateResult, TriageObservation

# ─── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Triage OpenEnv",
    description=(
        "An OpenEnv environment where AI agents learn to triage patients "
        "across three tasks of increasing difficulty."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store (single-user for hackathon) ──────────────────────
_sessions: Dict[str, MedicalTriageEnv] = {}


def _get_env(session_id: str) -> MedicalTriageEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


# ─── Request/Response Models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    session_id: str = "default"


class StepRequest(BaseModel):
    session_id: str = "default"
    action: TriageAction


class StateRequest(BaseModel):
    session_id: str = "default"


class ResetResponse(BaseModel):
    session_id: str
    task_id: str
    observation: TriageObservation


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Medical Triage OpenEnv",
        "version": "1.0.0",
        "tasks": ["task_easy", "task_medium", "task_hard"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Single Patient Triage",
                "difficulty": "easy",
                "description": "Triage a single patient. Assign priority, suspected conditions, and immediate actions.",
                "max_steps": 5,
            },
            {
                "id": "task_medium",
                "name": "Batch Triage (5 Patients)",
                "difficulty": "medium",
                "description": "Triage 5 simultaneous patients and rank them by priority order.",
                "max_steps": 7,
            },
            {
                "id": "task_hard",
                "name": "Ambiguous Case — Anchoring Bias",
                "difficulty": "hard",
                "description": (
                    "Handle an ambiguous presentation. Request diagnostic tests, "
                    "overcome anchoring bias, and update your priority when results arrive."
                ),
                "max_steps": 10,
            },
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = Body(default=None)):
    """Reset the environment for the given task. Returns initial observation."""
    if req is None:
        req = ResetRequest()
    valid_tasks = ["task_easy", "task_medium", "task_hard"]
    if req.task_id not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task_id. Must be one of: {valid_tasks}")

    env = MedicalTriageEnv(task_id=req.task_id)
    _sessions[req.session_id] = env
    obs = env.reset()

    return ResetResponse(
        session_id=req.session_id,
        task_id=req.task_id,
        observation=obs,
    )


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """Submit a triage action and receive the next observation and reward."""
    env = _get_env(req.session_id)
    result = env.step(req.action)
    return result


@app.get("/state", response_model=StateResult)
def state(session_id: str = "default"):
    """Return the current environment state without taking an action."""
    env = _get_env(session_id)
    return env.state()


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session."""
    _sessions.pop(session_id, None)
    return {"deleted": session_id}
