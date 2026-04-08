"""
Task graders for the Medical Triage environment.
Each grader returns a score 0.0–1.0 with a full breakdown for interpretability.
All graders are deterministic and reproducible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.models import PriorityLevel, TriageAction, PatientCase


# ─── Priority Severity Map ──────────────────────────────────────────────────────
# Used to compute how "wrong" a priority assignment is
PRIORITY_SEVERITY = {
    PriorityLevel.CRITICAL: 4,
    PriorityLevel.URGENT:   3,
    PriorityLevel.SEMI:     2,
    PriorityLevel.ROUTINE:  1,
}


def _priority_score(assigned: PriorityLevel, true: PriorityLevel) -> Tuple[float, str]:
    """
    Score the priority assignment.
    Under-triaging a critical patient is penalised heavily.
    Over-triaging is penalised lightly.
    """
    a = PRIORITY_SEVERITY[assigned]
    t = PRIORITY_SEVERITY[true]

    if a == t:
        return 1.0, "✅ Correct priority level assigned."

    diff = t - a  # positive = under-triage, negative = over-triage
    if diff > 0:
        # Under-triage: dangerous
        penalty = 0.4 * diff
        score = max(0.0, 1.0 - penalty)
        msg = f"⚠️ UNDER-TRIAGE: assigned {assigned.value} but true level is {true.value}. This is dangerous."
    else:
        # Over-triage: wasteful but safe
        penalty = 0.15 * abs(diff)
        score = max(0.0, 1.0 - penalty)
        msg = f"ℹ️ Over-triage: assigned {assigned.value} but true level is {true.value}. Safe but resource-wasteful."

    return round(score, 3), msg


def _condition_score(suspected: List[str], true_condition: str) -> Tuple[float, str]:
    """Check if the agent's suspected conditions overlap with the true condition."""
    true_keywords = set(true_condition.lower().replace("(", "").replace(")", "").split())
    # Remove common stop words
    stop = {"of", "the", "a", "an", "and", "or", "with", "in", "on", "at", "to"}
    true_keywords -= stop

    best_overlap = 0.0
    for condition in suspected:
        cond_words = set(condition.lower().replace("(", "").replace(")", "").split()) - stop
        if not true_keywords:
            continue
        overlap = len(true_keywords & cond_words) / len(true_keywords)
        best_overlap = max(best_overlap, overlap)

    if best_overlap >= 0.6:
        return 1.0, f"✅ Correctly identified condition: {true_condition}"
    elif best_overlap >= 0.3:
        return 0.5, f"⚠️ Partially identified condition. True diagnosis: {true_condition}"
    else:
        return 0.0, f"❌ Missed condition. True diagnosis: {true_condition}"


def _actions_score(actions: List[str], critical_actions: List[str]) -> Tuple[float, str]:
    """Check how many critical actions the agent recommended."""
    if not critical_actions:
        return 1.0, "✅ No specific actions required."

    actions_lower = [a.lower() for a in actions]
    matched = 0
    for req in critical_actions:
        req_lower = req.lower()
        if any(req_lower in a or any(w in a for w in req_lower.split()) for a in actions_lower):
            matched += 1

    score = matched / len(critical_actions)
    if score == 1.0:
        return 1.0, f"✅ All critical actions recommended: {', '.join(critical_actions)}"
    elif score > 0:
        missing = [r for r in critical_actions if not any(r.lower() in a.lower() for a in actions)]
        return round(score, 3), f"⚠️ Missed actions: {', '.join(missing)}"
    else:
        return 0.0, f"❌ No critical actions matched. Expected: {', '.join(critical_actions)}"


def _reasoning_score(reasoning: str) -> Tuple[float, str]:
    """Lightweight check that the agent provided substantive reasoning."""
    if not reasoning or len(reasoning.strip()) < 20:
        return 0.0, "❌ No reasoning provided."
    if len(reasoning.strip()) < 80:
        return 0.5, "⚠️ Reasoning is too brief — needs more clinical detail."
    return 1.0, "✅ Substantive reasoning provided."


# ─── TASK EASY GRADER ──────────────────────────────────────────────────────────

EASY_CRITICAL_ACTIONS = {
    "P001": ["ECG", "IV access", "oxygen", "aspirin"],
    "P002": ["X-ray"],
    "P003": ["oxygen", "IV access", "ENT", "airway"],
}


def grade_easy(action: TriageAction, patient: PatientCase, step: int) -> Tuple[float, Dict, str]:
    """
    Grade a single-patient triage decision.
    Weights: priority 40%, condition 25%, actions 25%, reasoning 10%
    """
    breakdown: Dict[str, float] = {}
    feedback: List[str] = []

    # Priority (40%)
    p_score, p_msg = _priority_score(action.priority, patient.true_priority)
    breakdown["priority"] = round(p_score * 0.40, 4)
    feedback.append(p_msg)

    # Condition (25%)
    c_score, c_msg = _condition_score(action.suspected_conditions, patient.true_condition)
    breakdown["condition"] = round(c_score * 0.25, 4)
    feedback.append(c_msg)

    # Actions (25%)
    required = EASY_CRITICAL_ACTIONS.get(patient.patient_id, [])
    a_score, a_msg = _actions_score(action.immediate_actions, required)
    breakdown["actions"] = round(a_score * 0.25, 4)
    feedback.append(a_msg)

    # Reasoning (10%)
    r_score, r_msg = _reasoning_score(action.reasoning)
    breakdown["reasoning"] = round(r_score * 0.10, 4)
    feedback.append(r_msg)

    # Step efficiency bonus (5% max, decays with steps)
    step_bonus = max(0.0, 0.05 * (1 - (step - 1) / 5))
    breakdown["efficiency"] = round(step_bonus, 4)

    total = min(1.0, sum(breakdown.values()))
    return round(total, 4), breakdown, "\n".join(feedback)


# ─── TASK MEDIUM GRADER ────────────────────────────────────────────────────────

MEDIUM_TRUE_RANKING = ["B001", "B003", "B005", "B002", "B004"]  # correct priority order
MEDIUM_TRUE_PRIORITIES = {
    "B001": PriorityLevel.CRITICAL,
    "B002": PriorityLevel.SEMI,
    "B003": PriorityLevel.CRITICAL,
    "B004": PriorityLevel.ROUTINE,
    "B005": PriorityLevel.URGENT,
}
MEDIUM_CRITICAL_ACTIONS = {
    "B001": ["CT head", "neurosurgery", "IV access"],
    "B002": ["suture", "wound clean"],
    "B003": ["CT head", "thrombolysis", "stroke team", "ECG"],
    "B004": ["throat swab"],
    "B005": ["IV access", "surgery", "IV fluids", "CT abdomen"],
}


def grade_medium(actions: List[TriageAction], patients: List[PatientCase], step: int) -> Tuple[float, Dict, str]:
    """
    Grade batch triage of 5 patients.
    Weights: individual priority avg 35%, ranking order 25%, conditions 20%, actions 15%, reasoning 5%
    """
    breakdown: Dict[str, float] = {}
    feedback: List[str] = []
    action_map = {a.patient_id: a for a in actions}

    # Individual priority scores (35%)
    prio_scores = []
    for patient in patients:
        act = action_map.get(patient.patient_id)
        if not act:
            prio_scores.append(0.0)
            feedback.append(f"❌ No triage decision submitted for patient {patient.patient_id}")
            continue
        ps, pm = _priority_score(act.priority, patient.true_priority)
        prio_scores.append(ps)
        feedback.append(f"Patient {patient.patient_id}: {pm}")
    breakdown["individual_priorities"] = round((sum(prio_scores) / len(prio_scores)) * 0.35, 4)

    # Ranking order (25%) — Kendall tau-style
    agent_ranking = sorted(
        [a.patient_id for a in actions],
        key=lambda pid: PRIORITY_SEVERITY.get(
            action_map[pid].priority, 0) if pid in action_map else 0,
        reverse=True
    )
    correct_ranking = MEDIUM_TRUE_RANKING
    concordant = sum(
        1 for i in range(len(correct_ranking))
        if i < len(agent_ranking) and agent_ranking[i] == correct_ranking[i]
    )
    ranking_score = concordant / len(correct_ranking)
    breakdown["ranking_order"] = round(ranking_score * 0.25, 4)
    feedback.append(
        f"{'✅' if ranking_score >= 0.8 else '⚠️'} Priority ranking: {concordant}/{len(correct_ranking)} correct positions. "
        f"Correct order: {' > '.join(correct_ranking)}"
    )

    # Conditions (20%)
    cond_scores = []
    for patient in patients:
        act = action_map.get(patient.patient_id)
        if not act:
            cond_scores.append(0.0)
            continue
        cs, _ = _condition_score(act.suspected_conditions, patient.true_condition)
        cond_scores.append(cs)
    breakdown["conditions"] = round((sum(cond_scores) / len(cond_scores)) * 0.20, 4)

    # Actions (15%)
    act_scores = []
    for patient in patients:
        act = action_map.get(patient.patient_id)
        if not act:
            act_scores.append(0.0)
            continue
        required = MEDIUM_CRITICAL_ACTIONS.get(patient.patient_id, [])
        as_, _ = _actions_score(act.immediate_actions, required)
        act_scores.append(as_)
    breakdown["actions"] = round((sum(act_scores) / len(act_scores)) * 0.15, 4)

    # Reasoning (5%)
    reasoning_scores = [_reasoning_score(a.reasoning)[0] for a in actions]
    breakdown["reasoning"] = round((sum(reasoning_scores) / max(len(reasoning_scores), 1)) * 0.05, 4)

    total = min(1.0, sum(breakdown.values()))
    return round(total, 4), breakdown, "\n".join(feedback)


# ─── TASK HARD GRADER ──────────────────────────────────────────────────────────

HARD_REQUIRED_TESTS = ["ECG", "troponin"]
HARD_TRUE_INITIAL_PRIORITY = PriorityLevel.URGENT
HARD_TRUE_FINAL_PRIORITY = PriorityLevel.CRITICAL
HARD_TRUE_CONDITION = "NSTEMI Non-ST-Elevation Myocardial Infarction"


def grade_hard(
    action: TriageAction,
    step: int,
    revealed_info: Dict[str, str],
    is_final: bool,
) -> Tuple[float, Dict, str]:
    """
    Grade the ambiguous multi-step triage case.
    The agent must:
    1. Request appropriate tests (ECG, troponin)
    2. Correctly re-prioritise after seeing results
    3. Identify NSTEMI despite psychiatric history
    Weights: initial priority 15%, test requests 25%, updated priority 30%, condition 20%, reasoning 10%
    """
    breakdown: Dict[str, float] = {}
    feedback: List[str] = []

    # Initial priority (15%) — URGENT is correct initially (ambiguous case)
    init_priority = action.priority
    ip_score, ip_msg = _priority_score(init_priority, HARD_TRUE_INITIAL_PRIORITY)
    breakdown["initial_priority"] = round(ip_score * 0.15, 4)
    feedback.append(f"Initial priority: {ip_msg}")

    # Test requests (25%) — must ask for ECG and troponin
    requested = [r.lower() for r in (action.request_more_info or [])]
    test_hits = sum(1 for t in HARD_REQUIRED_TESTS if any(t.lower() in r for r in requested))
    test_score = test_hits / len(HARD_REQUIRED_TESTS)
    breakdown["test_requests"] = round(test_score * 0.25, 4)
    feedback.append(
        f"{'✅' if test_score == 1.0 else '⚠️'} Tests requested: {test_hits}/{len(HARD_REQUIRED_TESTS)} critical tests. "
        f"Required: ECG, troponin."
    )

    # Updated priority (30%) — after seeing ECG + troponin, must upgrade to CRITICAL
    updated = action.updated_priority
    if updated is not None:
        up_score, up_msg = _priority_score(updated, HARD_TRUE_FINAL_PRIORITY)
        breakdown["updated_priority"] = round(up_score * 0.30, 4)
        feedback.append(f"Updated priority: {up_msg}")
    else:
        breakdown["updated_priority"] = 0.0
        if is_final:
            feedback.append("❌ No updated priority provided after receiving test results.")

    # Condition identification (20%)
    c_score, c_msg = _condition_score(action.suspected_conditions, HARD_TRUE_CONDITION)
    # Bonus if agent overcame anchoring bias (mentioned anxiety but still considered cardiac)
    reasoning_lower = action.reasoning.lower()
    overcame_anchoring = (
        "anxiety" in reasoning_lower or "panic" in reasoning_lower
    ) and (
        "cardiac" in reasoning_lower or "nstemi" in reasoning_lower or "troponin" in reasoning_lower
    )
    if overcame_anchoring:
        c_score = min(1.0, c_score + 0.2)
        feedback.append("✅ Bonus: agent correctly considered cardiac cause despite psychiatric history.")
    breakdown["condition"] = round(c_score * 0.20, 4)
    feedback.append(c_msg)

    # Reasoning (10%)
    r_score, r_msg = _reasoning_score(action.reasoning)
    breakdown["reasoning"] = round(r_score * 0.10, 4)
    feedback.append(r_msg)

    total = min(1.0, sum(breakdown.values()))
    return round(total, 4), breakdown, "\n".join(feedback)
