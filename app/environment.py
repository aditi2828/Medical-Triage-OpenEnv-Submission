"""
Medical Triage OpenEnv Environment - Core Logic
Implements step() / reset() / state() per the OpenEnv spec.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models import (
    PriorityLevel, TaskID, TriageAction, TriageObservation,
    TriageReward, StepResult, StateResult, PatientCase,
)
from app.cases import (
    EASY_CASES, MEDIUM_CASES,
    HARD_CASE_INITIAL, HARD_CASE_ADDITIONAL_INFO, HARD_CASE_HINTS,
)
from app.graders import grade_easy, grade_medium, grade_hard


MAX_STEPS = {
    TaskID.EASY:   5,
    TaskID.MEDIUM: 7,
    TaskID.HARD:   10,
}

TASK_NAMES = {
    TaskID.EASY:   "Single Patient Triage",
    TaskID.MEDIUM: "Batch Triage (5 Patients)",
    TaskID.HARD:   "Ambiguous Case — Anchoring Bias",
}


class MedicalTriageEnv:
    """
    Medical Triage OpenEnv Environment.

    Three tasks of increasing difficulty:
      task_easy   — triage a single patient (3 variants)
      task_medium — triage 5 simultaneous patients, rank by priority
      task_hard   — handle an ambiguous case; request tests; update priority
    """

    def __init__(self, task_id: str = "task_easy"):
        self.task_id = TaskID(task_id)
        self._step_count: int = 0
        self._done: bool = False
        self._episode_rewards: List[float] = []
        self._best_score: float = 0.0
        self._current_patients: List[PatientCase] = []
        self._easy_case_index: int = 0
        self._hard_revealed_info: Dict[str, str] = {}
        self._hard_step_actions: List[TriageAction] = []

    # ─── OpenEnv API ────────────────────────────────────────────────────────────

    def reset(self) -> TriageObservation:
        """Reset the environment and return the initial observation."""
        self._step_count = 0
        self._done = False
        self._episode_rewards = []
        self._best_score = 0.0
        self._hard_revealed_info = {}
        self._hard_step_actions = []

        self._current_patients = self._load_patients()

        return self._build_observation(
            feedback="Welcome to Medical Triage. Assess the patient(s) and submit your triage decision.",
            hints=self._get_initial_hints(),
        )

    def step(self, action: TriageAction) -> StepResult:
        """Process an agent action and return observation, reward, done, info."""
        if self._done:
            return StepResult(
                observation=self._build_observation(feedback="Episode already complete. Call reset() to start a new episode."),
                reward=0.0,
                done=True,
                info={"warning": "Episode already done"},
            )

        self._step_count += 1

        # ── Route to task-specific grader ──
        if self.task_id == TaskID.EASY:
            score, breakdown, feedback = self._step_easy(action)
        elif self.task_id == TaskID.MEDIUM:
            score, breakdown, feedback = self._step_medium(action)
        else:
            score, breakdown, feedback = self._step_hard(action)

        self._episode_rewards.append(score)
        self._best_score = max(self._best_score, score)

        # Episode is done if: max steps reached, score is perfect, or hard task resolved
        max_s = MAX_STEPS[self.task_id]
        done_by_steps = self._step_count >= max_s
        done_by_score = score >= 0.95
        done_by_hard_resolution = (
            self.task_id == TaskID.HARD
            and action.updated_priority is not None
            and self._step_count >= 2
        )

        self._done = done_by_steps or done_by_score or done_by_hard_resolution

        if self._done:
            feedback += f"\n\n🏁 Episode complete. Final score: {score:.3f}"
            if score >= 0.85:
                feedback += " — Excellent triage performance!"
            elif score >= 0.65:
                feedback += " — Good, but review missed items above."
            else:
                feedback += " — Significant improvements needed. Review the hints."

        obs = self._build_observation(
            feedback=feedback,
            hints=self._get_contextual_hints(action, score),
            additional_info=self._hard_revealed_info if self.task_id == TaskID.HARD else None,
        )

        return StepResult(
            observation=obs,
            reward=score,
            done=self._done,
            info={"breakdown": breakdown, "step": self._step_count},
        )

    def state(self) -> StateResult:
        """Return the current environment state."""
        return StateResult(
            task_id=self.task_id.value,
            step_count=self._step_count,
            max_steps=MAX_STEPS[self.task_id],
            current_patients=self._current_patients,
            best_score=self._best_score,
            done=self._done,
            episode_rewards=self._episode_rewards,
        )

    # ─── Task-specific step logic ────────────────────────────────────────────────

    def _step_easy(self, action: TriageAction) -> tuple[float, dict, str]:
        patient = self._current_patients[0]
        return grade_easy(action, patient, self._step_count)

    def _step_medium(self, action: TriageAction) -> tuple[float, dict, str]:
        # Medium task expects one action per patient; wrap single action into list
        # Agent may submit one action at a time or all at once
        self._hard_step_actions.append(action)  # reuse accumulator field
        actions_so_far = self._hard_step_actions

        # If all 5 patients have been addressed (or max steps) — grade now
        addressed_pids = {a.patient_id for a in actions_so_far}
        all_addressed = all(p.patient_id in addressed_pids for p in self._current_patients)

        if all_addressed or self._step_count >= MAX_STEPS[TaskID.MEDIUM]:
            return grade_medium(actions_so_far, self._current_patients, self._step_count)
        else:
            remaining = [p.patient_id for p in self._current_patients if p.patient_id not in addressed_pids]
            return 0.0, {}, (
                f"✅ Received triage for patient {action.patient_id}. "
                f"Still waiting for: {', '.join(remaining)}. "
                f"Submit a decision for each remaining patient."
            )

    def _step_hard(self, action: TriageAction) -> tuple[float, dict, str]:
        # Reveal information if agent requested tests
        if action.request_more_info:
            for test in action.request_more_info:
                test_key = test.lower().replace(" ", "_").replace("-", "_")
                for key, value in HARD_CASE_ADDITIONAL_INFO.items():
                    if key.lower() in test.lower() or test.lower() in key.lower():
                        self._hard_revealed_info[key] = value

        is_final = action.updated_priority is not None or self._step_count >= MAX_STEPS[TaskID.HARD]
        return grade_hard(action, self._step_count, self._hard_revealed_info, is_final)

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def _load_patients(self) -> List[PatientCase]:
        if self.task_id == TaskID.EASY:
            # Rotate through cases on each reset
            case = EASY_CASES[self._easy_case_index % len(EASY_CASES)]
            self._easy_case_index += 1
            return [case]
        elif self.task_id == TaskID.MEDIUM:
            return list(MEDIUM_CASES)
        else:
            return [HARD_CASE_INITIAL]

    def _get_initial_hints(self) -> List[str]:
        if self.task_id == TaskID.EASY:
            return [
                "Assign a priority level: CRITICAL, URGENT, SEMI_URGENT, or ROUTINE",
                "List your suspected conditions and immediate actions",
                "Provide clinical reasoning for your decision",
            ]
        elif self.task_id == TaskID.MEDIUM:
            return [
                "You must triage all 5 patients — submit one TriageAction per patient",
                "Consider the relative priority order between patients",
                "CRITICAL patients must be identified correctly to pass",
            ]
        else:
            return HARD_CASE_HINTS

    def _get_contextual_hints(self, action: TriageAction, score: float) -> List[str]:
        hints = []
        if score < 0.5:
            hints.append("💡 Score below 0.5 — review your priority assignment and clinical reasoning.")
        if not action.reasoning or len(action.reasoning) < 50:
            hints.append("💡 Provide more detailed clinical reasoning to earn full marks.")
        if self.task_id == TaskID.HARD and not action.request_more_info:
            hints.append("💡 In ambiguous cases, request clarifying tests using request_more_info.")
        if self.task_id == TaskID.HARD and self._hard_revealed_info and action.updated_priority is None:
            hints.append("💡 You have new test results — update your priority using updated_priority.")
        return hints

    def _build_observation(
        self,
        feedback: str,
        hints: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> TriageObservation:
        # Strip ground-truth fields before exposing to agent
        safe_patients = []
        for p in self._current_patients:
            safe = p.model_copy(update={"true_priority": PriorityLevel.ROUTINE, "true_condition": "[hidden]"})
            safe_patients.append(safe)

        return TriageObservation(
            task_id=self.task_id.value,
            task_name=TASK_NAMES[self.task_id],
            step_count=self._step_count,
            max_steps=MAX_STEPS[self.task_id],
            patients=safe_patients,
            feedback=feedback,
            hints=hints or [],
            additional_info=additional_info,
            best_score_so_far=self._best_score,
            done=self._done,
        )
