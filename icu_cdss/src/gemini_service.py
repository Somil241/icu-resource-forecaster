"""Gemini AI integration for clinical care report and summary generation.

Uses the Google Gemini REST API (Gemini 3 Flash) to produce:
1. Structured, patient-specific care reports (medication review, warnings, etc.)
2. Clinician-facing care highlights and narrative summaries for the dashboard.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
import urllib.error


_SSL_CTX = ssl._create_unverified_context()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "Key")
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
    f":generateContent?key={GEMINI_API_KEY}"
)




def _call_gemini(prompt: str, temperature: float = 0.4, max_tokens: int = 4096) -> str:
    """Call the Gemini REST API.  Returns the raw text content."""
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        GEMINI_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        candidates = body.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API {e.code}: {error_body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini API network error: {e}") from e


def _parse_json_response(raw: str) -> dict | None:
    """Try to parse JSON from Gemini response, stripping markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None




def _format_meds(medications: list[dict]) -> str:
    if not medications:
        return "None reported"
    lines = []
    for m in medications:
        name = m.get("name", "Unknown")
        dosage = m.get("dosage", "")
        freq = m.get("frequency", "")
        route = m.get("route", "")
        lines.append(f"  - {name} {dosage} {freq} ({route})")
    return "\n".join(lines)




def generate_care_report(
    patient: dict,
    predictions: dict | None = None,
    medications: list[dict] | None = None,
) -> dict:
    """Generate a comprehensive AI care report using Gemini.

    Parameters
    ----------
    patient : dict
        Patient object (vitals, labs, scores, demographics).
    predictions : dict | None
        Optional extra prediction block (sepsis, vent, los).
    medications : list[dict] | None
        Doctor-entered current medications.

    Returns
    -------
    dict with keys: overall_assessment, risk_analysis, care_recommendations,
    medication_review, monitoring_plan, warnings, diet_and_nutrition
    """
    vitals = patient.get("vitals", {})
    labs = patient.get("labs", {})
    scores = patient.get("clinicalScores", {})
    meds = medications or patient.get("medications", [])
    preds = predictions or {}

    sepsis_risk = preds.get("sepsisRisk", patient.get("sepsisRisk", 0))
    vent_prob = preds.get("ventProbability", preds.get("_vent_prob", 0))
    los = preds.get("predictedLOS", patient.get("predictedLOS", 0))
    diagnosis = preds.get("diagnosis", patient.get("diagnosis", "N/A"))

    prompt = f"""You are an experienced ICU attending physician and clinical decision support expert.
Generate a comprehensive, actionable patient care report based on the following data.

PATIENT DEMOGRAPHICS:
  Age: {patient.get('age', 'Unknown')} | Gender: {patient.get('gender', 'Unknown')}
  Diagnosis: {diagnosis}
  Acuity Level: {patient.get('acuityLevel', 'Unknown')}

CURRENT VITALS:
  Heart Rate: {vitals.get('heartRate', 'N/A')} bpm
  Systolic BP: {vitals.get('systolicBP', 'N/A')} mmHg
  Temperature: {vitals.get('temperature', 'N/A')} °C
  Respiratory Rate: {vitals.get('respiratoryRate', 'N/A')} breaths/min
  SpO2: {vitals.get('oxygenSaturation', 'N/A')}%

LABORATORY VALUES:
  WBC: {labs.get('wbc', 'N/A')} k/μL
  Lactate: {labs.get('lactate', 'N/A')} mmol/L
  Creatinine: {labs.get('creatinine', 'N/A')} mg/dL
  CRP: {labs.get('crp', 'N/A')} mg/L
  Platelets: {labs.get('platelets', 'N/A')} k/μL
  Bilirubin: {labs.get('bilirubin', 'N/A')} mg/dL

CLINICAL SCORES:
  SOFA: {scores.get('sofa', 'N/A')} | GCS: {scores.get('gcs', 'N/A')} | APACHE II: {scores.get('apacheII', 'N/A')}

ML-BASED PREDICTIONS:
  Sepsis Risk: {round(float(sepsis_risk) * 100, 1) if sepsis_risk else 'N/A'}%
  Ventilator Need (24h): {round(float(vent_prob) * 100, 1) if vent_prob else 'N/A'}%
  Estimated Remaining ICU Stay: {los} days

CURRENT MEDICATIONS THE PATIENT IS TAKING:
{_format_meds(meds)}

ALLERGIES: {', '.join(patient.get('allergies', [])) or 'None known'}
PRE-EXISTING CONDITIONS: {', '.join(patient.get('preExistingConditions', [])) or 'None documented'}

---

Provide a detailed clinical care report as valid JSON with exactly these keys:

{{
  "overall_assessment": "A 2-3 sentence overall clinical assessment of the patient's current status",
  "risk_analysis": "Detailed analysis of sepsis risk, organ dysfunction, and clinical trajectory based on the vitals, labs, and ML predictions",
  "care_recommendations": ["recommendation 1", "recommendation 2", "...up to 6 specific actionable recommendations"],
  "medication_review": "Review of current medications. Flag any potential drug interactions, suggest adjustments based on renal/hepatic function, and recommend additional medications if clinically indicated. If no medications are listed, recommend an appropriate empiric regimen based on the patient's condition.",
  "monitoring_plan": "Specific monitoring recommendations including frequency of vitals, labs to recheck, and clinical milestones to watch for",
  "warnings": ["critical warning 1 if any", "..."],
  "diet_and_nutrition": "Nutritional recommendations considering the patient's condition and medications"
}}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no code fences, no extra text."""

    raw = _call_gemini(prompt)
    report = _parse_json_response(raw)

    if report is None:
        report = {
            "overall_assessment": raw[:500] if raw else "Unable to generate assessment.",
            "risk_analysis": "Please retry — the AI returned a non-structured response.",
            "care_recommendations": ["Review patient data manually and consult attending."],
            "medication_review": "Unable to parse AI response for medication review.",
            "monitoring_plan": "Standard ICU monitoring recommended.",
            "warnings": [],
            "diet_and_nutrition": "Consult nutrition team.",
        }

    # Ensure all keys exist with defaults.
    defaults = {
        "overall_assessment": "",
        "risk_analysis": "",
        "care_recommendations": [],
        "medication_review": "",
        "monitoring_plan": "",
        "warnings": [],
        "diet_and_nutrition": "",
    }
    for k, v in defaults.items():
        report.setdefault(k, v)

    return report


# ---------------------------------------------------------------------------
# 2) AI Care Highlights + Narrative Summary (for dashboard panel)
# ---------------------------------------------------------------------------

def generate_ai_highlights(patient_data: dict) -> dict:
    """Generate AI-powered care highlights and a narrative clinical summary.

    Parameters
    ----------
    patient_data : dict
        A patient object with vitals, labs, clinicalScores, sepsisRisk, etc.

    Returns
    -------
    dict with keys: highlights (list[str]), fullSummary (str)
    """
    vitals = patient_data.get("vitals", {})
    labs = patient_data.get("labs", {})
    scores = patient_data.get("clinicalScores", {})
    sepsis = patient_data.get("sepsisRisk", 0)
    vent = patient_data.get("_vent_prob", 0)
    los_days = patient_data.get("predictedLOS", 0)
    diagnosis = patient_data.get("diagnosis", "ICU Observation")
    acuity = patient_data.get("acuityLevel", "Unknown")
    meds = patient_data.get("medications", [])

    prompt = f"""You are a senior ICU physician writing a concise clinical dashboard summary.

PATIENT:
  Age: {patient_data.get('age', '?')} | Gender: {patient_data.get('gender', '?')}
  Diagnosis: {diagnosis} | Acuity: {acuity}

VITALS:
  HR: {vitals.get('heartRate', '?')} bpm | SBP: {vitals.get('systolicBP', '?')} mmHg
  Temp: {vitals.get('temperature', '?')} °C | RR: {vitals.get('respiratoryRate', '?')} /min
  SpO2: {vitals.get('oxygenSaturation', '?')}%

LABS:
  WBC: {labs.get('wbc', '?')} | Lactate: {labs.get('lactate', '?')} mmol/L
  Creatinine: {labs.get('creatinine', '?')} mg/dL | Platelets: {labs.get('platelets', '?')}
  Bilirubin: {labs.get('bilirubin', '?')} | CRP: {labs.get('crp', '?')}

SCORES: SOFA {scores.get('sofa', '?')} | GCS {scores.get('gcs', '?')} | APACHE II {scores.get('apacheII', '?')}

ML PREDICTIONS:
  Sepsis Risk: {round(float(sepsis) * 100, 1) if sepsis else 0}%
  Ventilator Need (24h): {round(float(vent) * 100, 1) if vent else 0}%
  Estimated Remaining ICU Stay: {los_days} days ({round(float(los_days) * 24)} hours)

CURRENT MEDICATIONS: {_format_meds(meds) if meds else 'None documented'}

ALLERGIES: {', '.join(patient_data.get('allergies', [])) or 'None known'}
PRE-EXISTING CONDITIONS: {', '.join(patient_data.get('preExistingConditions', [])) or 'None documented'}

---

Generate a JSON object with exactly these keys:
{{
  "highlights": [
    "4-6 concise bullet points for the dashboard. Each should be a short (1 sentence) clinical observation. The FIRST highlight must state the exact sepsis risk percentage ({round(float(sepsis)*100,1)}%). The SECOND must state the exact ventilator need percentage. The THIRD must state the exact estimated remaining ICU stay. The remaining highlights should cover the most important clinical concerns (abnormal labs, organ dysfunction, medication concerns, etc.)."
  ],
  "fullSummary": "A 4-6 sentence clinical narrative summarizing the patient's current condition, key risks, and recommended immediate priorities. Write in professional medical language. Must reference the exact ML prediction percentages."
}}

IMPORTANT: Return ONLY the JSON object. No markdown, no code fences, no extra text.
The highlights MUST use the EXACT prediction numbers provided above — do NOT round or change them."""

    raw = _call_gemini(prompt, temperature=0.3, max_tokens=2048)
    parsed = _parse_json_response(raw)

    if parsed and "highlights" in parsed and "fullSummary" in parsed:
        return parsed

    # Fallback: build highlights from raw data if Gemini fails to return JSON.
    sep_pct = round(float(sepsis) * 100, 1) if sepsis else 0
    vent_pct = round(float(vent) * 100, 1) if vent else 0
    los_h = round(float(los_days) * 24) if los_days else 0

    return {
        "highlights": [
            f"Sepsis risk {sep_pct}% — {'ALERT' if sepsis and float(sepsis) >= 0.5 else 'monitor'}.",
            f"Ventilator need within 24 h: {vent_pct}%.",
            f"Estimated remaining ICU stay: {los_h} h ({los_days} d).",
            f"SOFA: {scores.get('sofa', '?')} | GCS: {scores.get('gcs', '?')} | APACHE II: {scores.get('apacheII', '?')}.",
        ],
        "fullSummary": raw[:1000] if raw else f"Patient presents with {diagnosis}. ML models predict {sep_pct}% sepsis risk. Further clinical assessment recommended.",
    }
