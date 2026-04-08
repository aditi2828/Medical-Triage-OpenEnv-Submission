"""
Patient case generator for the Medical Triage environment.
All cases are deterministic and reproducible — no randomness unless seeded.
"""
from __future__ import annotations

from app.models import PatientCase, PriorityLevel, Vitals


# ─── Task Easy: Single Patient Cases ───────────────────────────────────────────

EASY_CASES: list[PatientCase] = [
    PatientCase(
        patient_id="P001",
        age=67,
        sex="Male",
        chief_complaint="Chest pain radiating to left arm, sweating profusely",
        vitals=Vitals(
            heart_rate=112,
            blood_pressure="160/95",
            oxygen_saturation=94,
            temperature=37.1,
            respiratory_rate=22,
            gcs=15,
        ),
        medical_history=["Hypertension", "Hypercholesterolemia"],
        current_medications=["Amlodipine 5mg", "Atorvastatin 40mg"],
        additional_notes="Onset 30 minutes ago. Diaphoretic on arrival.",
        arrival_mode="ambulance",
        true_priority=PriorityLevel.CRITICAL,
        true_condition="Acute Myocardial Infarction (STEMI)",
    ),
    PatientCase(
        patient_id="P002",
        age=28,
        sex="Female",
        chief_complaint="Sprained ankle after fall, moderate pain",
        vitals=Vitals(
            heart_rate=82,
            blood_pressure="118/74",
            oxygen_saturation=99,
            temperature=36.8,
            respiratory_rate=16,
            gcs=15,
        ),
        medical_history=[],
        current_medications=[],
        additional_notes="Right ankle swollen, bruised. Weight-bearing painful but possible.",
        arrival_mode="walk-in",
        true_priority=PriorityLevel.ROUTINE,
        true_condition="Ankle sprain (Grade II)",
    ),
    PatientCase(
        patient_id="P003",
        age=5,
        sex="Male",
        chief_complaint="High fever and difficulty breathing",
        vitals=Vitals(
            heart_rate=140,
            blood_pressure="90/60",
            oxygen_saturation=91,
            temperature=39.8,
            respiratory_rate=38,
            gcs=14,
        ),
        medical_history=["Asthma"],
        current_medications=["Salbutamol inhaler"],
        additional_notes="Stridor present. Drooling. Refuses to lie down.",
        arrival_mode="ambulance",
        true_priority=PriorityLevel.CRITICAL,
        true_condition="Epiglottitis",
    ),
]


# ─── Task Medium: Batch of 5 Patients ──────────────────────────────────────────

MEDIUM_CASES: list[PatientCase] = [
    PatientCase(
        patient_id="B001",
        age=42,
        sex="Female",
        chief_complaint="Sudden severe headache, worst of her life",
        vitals=Vitals(
            heart_rate=88,
            blood_pressure="178/102",
            oxygen_saturation=97,
            temperature=37.4,
            respiratory_rate=18,
            gcs=14,
        ),
        medical_history=["Migraines"],
        current_medications=["Sumatriptan"],
        additional_notes="Describes it as thunderclap onset. Neck stiffness noted.",
        arrival_mode="walk-in",
        true_priority=PriorityLevel.CRITICAL,
        true_condition="Subarachnoid Haemorrhage",
    ),
    PatientCase(
        patient_id="B002",
        age=19,
        sex="Male",
        chief_complaint="Laceration to forearm from broken glass",
        vitals=Vitals(
            heart_rate=90,
            blood_pressure="122/78",
            oxygen_saturation=99,
            temperature=36.9,
            respiratory_rate=16,
            gcs=15,
        ),
        medical_history=[],
        current_medications=[],
        additional_notes="5cm laceration, bleeding controlled with pressure. Neurovascular intact.",
        arrival_mode="walk-in",
        true_priority=PriorityLevel.SEMI,
        true_condition="Laceration requiring sutures",
    ),
    PatientCase(
        patient_id="B003",
        age=74,
        sex="Male",
        chief_complaint="Sudden weakness in right arm and leg, slurred speech",
        vitals=Vitals(
            heart_rate=94,
            blood_pressure="188/108",
            oxygen_saturation=96,
            temperature=37.0,
            respiratory_rate=19,
            gcs=13,
        ),
        medical_history=["Atrial Fibrillation", "Type 2 Diabetes"],
        current_medications=["Warfarin", "Metformin"],
        additional_notes="Symptom onset ~45 minutes ago. Facial droop present. FAST positive.",
        arrival_mode="ambulance",
        true_priority=PriorityLevel.CRITICAL,
        true_condition="Acute Ischaemic Stroke (within thrombolysis window)",
    ),
    PatientCase(
        patient_id="B004",
        age=33,
        sex="Female",
        chief_complaint="Sore throat and mild fever for 2 days",
        vitals=Vitals(
            heart_rate=86,
            blood_pressure="114/72",
            oxygen_saturation=99,
            temperature=38.2,
            respiratory_rate=16,
            gcs=15,
        ),
        medical_history=[],
        current_medications=[],
        additional_notes="No stridor. Tonsils mildly enlarged. Eating and drinking normally.",
        arrival_mode="walk-in",
        true_priority=PriorityLevel.ROUTINE,
        true_condition="Viral Pharyngitis",
    ),
    PatientCase(
        patient_id="B005",
        age=55,
        sex="Male",
        chief_complaint="Severe abdominal pain, unable to walk upright",
        vitals=Vitals(
            heart_rate=118,
            blood_pressure="102/68",
            oxygen_saturation=97,
            temperature=38.9,
            respiratory_rate=24,
            gcs=15,
        ),
        medical_history=["Appendectomy 20 years ago", "Hypertension"],
        current_medications=["Lisinopril"],
        additional_notes="Guarding and rigidity on palpation. Rebound tenderness positive.",
        arrival_mode="ambulance",
        true_priority=PriorityLevel.URGENT,
        true_condition="Bowel Perforation / Peritonitis",
    ),
]


# ─── Task Hard: Ambiguous & Multi-step Case ─────────────────────────────────────

HARD_CASE_INITIAL = PatientCase(
    patient_id="H001",
    age=45,
    sex="Female",
    chief_complaint="Chest tightness and shortness of breath",
    vitals=Vitals(
        heart_rate=98,
        blood_pressure="132/84",
        oxygen_saturation=96,
        temperature=37.2,
        respiratory_rate=20,
        gcs=15,
    ),
    medical_history=["Generalised Anxiety Disorder", "GERD"],
    current_medications=["Sertraline 50mg", "Omeprazole 20mg"],
    additional_notes=(
        "Patient appears anxious. Reports similar episodes in the past diagnosed as panic attacks. "
        "However, this episode started at rest, not during a stressful event. "
        "Mild diaphoresis noted. Pain is 6/10, non-radiating."
    ),
    arrival_mode="walk-in",
    true_priority=PriorityLevel.URGENT,
    true_condition="Non-ST-Elevation Myocardial Infarction (NSTEMI)",
)

# Information revealed after agent requests tests
HARD_CASE_ADDITIONAL_INFO = {
    "ECG": "ST depression in leads V4-V6. T-wave inversion in V5, V6.",
    "troponin": "High-sensitivity troponin T: 52 ng/L (normal < 14 ng/L). Elevated.",
    "chest_xray": "No pneumothorax. Mild cardiomegaly.",
    "d_dimer": "D-dimer: 0.3 mg/L (normal < 0.5 mg/L). Not suggestive of PE.",
    "blood_glucose": "Blood glucose: 6.8 mmol/L. Normal.",
    "full_blood_count": "WBC 9.2, Hb 13.4, platelets 234. Normal.",
    "history_clarification": (
        "On further questioning: patient reports the pain started 2 hours ago, "
        "not during anxiety. She has never had diaphoresis with previous panic attacks."
    ),
}

HARD_CASE_HINTS = [
    "Anxiety and cardiac events can present identically — don't anchor on psychiatric history",
    "Ask for ECG and troponin to differentiate cardiac from non-cardiac chest pain",
    "Diaphoresis at rest is a red flag — not typical of panic disorder",
    "NSTEMI does not always show ST elevation — look for ST depression or T-wave changes",
    "Update your priority once new test results are available",
]
