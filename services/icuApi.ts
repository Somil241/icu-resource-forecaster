import { Patient, BedDemandData, ResourceNeed, XAIFactor, Medication } from "../types";

// In dev, Vite proxies /api → http://127.0.0.1:8765 (see vite.config.ts).
// Override at build/runtime by setting VITE_ICU_API_BASE.
const BASE: string = (import.meta as any).env?.VITE_ICU_API_BASE ?? "/api";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status} ${path}: ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<{ status: string; time: string }> {
  return getJSON("/health");
}

export async function fetchPatients(limit = 12): Promise<Patient[]> {
  const raw = await getJSON<any[]>(`/patients?limit=${limit}`);
  return raw.map(normalizePatient);
}

export async function fetchPatient(stayId: string): Promise<Patient> {
  const raw = await getJSON<any>(`/patients/${stayId}`);
  return normalizePatient(raw);
}

export interface ClinicalSummary {
  payload: any;
  summary_text: string;
  highlights: string[];
  fullSummary: string;
}

export async function fetchSummary(stayId: string): Promise<ClinicalSummary> {
  return getJSON<ClinicalSummary>(`/summary/${stayId}`);
}

export async function fetchBedForecast(unit?: string, days = 7): Promise<BedDemandData[]> {
  const q = new URLSearchParams();
  if (unit) q.set("unit", unit);
  q.set("days", String(days));
  return getJSON<BedDemandData[]>(`/bed_forecast?${q.toString()}`);
}

export async function fetchResources(): Promise<ResourceNeed[]> {
  return getJSON<ResourceNeed[]>("/resources");
}

export async function fetchXAI(stayId: string): Promise<XAIFactor[]> {
  return getJSON<XAIFactor[]>(`/xai/${stayId}`);
}

// Backwards-compatible API used by App.tsx and PatientAnalysis.
// runMLInference originally took a CSV row and returned a Patient via Groq;
// we now ignore the CSV row and pull a fresh batch of real patients from the
// CDSS backend so the UI flow ("Upload Records" -> analyze) still works.
export async function runMLInference(_rawRow: unknown): Promise<Patient> {
  const list = await fetchPatients(1);
  if (!list.length) throw new Error("ICU API returned no patients");
  return list[0];
}

export async function getClinicalSummary(patient: Patient): Promise<{ highlights: string[]; fullSummary: string }> {
  // Custom / CSV patients have IDs like "CUSTOM-XXXXX" — the backend summary
  // endpoint only knows about real MIMIC stay_ids (integers).  For custom
  // patients, call the Gemini-powered /api/ai_highlights endpoint instead.
  const isCustom = patient.id.startsWith("CUSTOM-");
  if (isCustom) {
    try {
      const res = await fetch(`${BASE}/ai_highlights`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      });
      if (res.ok) {
        const data = await res.json();
        return { highlights: data.highlights, fullSummary: data.fullSummary };
      }
    } catch (err) {
      console.warn("Gemini highlights call failed, using fallback:", err);
    }

    // Fallback: build highlights from patient object if Gemini is unavailable.
    const sepPct  = Math.round(patient.sepsisRisk * 100);
    const losDays = patient.predictedLOS;
    const losH    = Math.round(losDays * 24);
    const ventPct = Math.round(((patient as any)._vent_prob ?? 0) * 100);

    return {
      highlights: [
        `Sepsis risk ${sepPct}% — ${patient.sepsisRisk >= 0.5 ? 'ALERT' : 'monitor'}.`,
        `Ventilator need within 24 h: ${ventPct}%.`,
        `Estimated remaining ICU stay: ${losH} h (${losDays} d).`,
      ],
      fullSummary: `${patient.name} is a ${patient.age}-year-old ${patient.gender === 'M' ? 'male' : 'female'} admitted with ${patient.diagnosis}. ML models predict ${sepPct}% sepsis risk. Acuity: ${patient.acuityLevel}.`,
    };
  }

  const s = await fetchSummary(patient.id);
  return { highlights: s.highlights, fullSummary: s.fullSummary };
}

export async function getXAIExplanation(patient: Patient): Promise<XAIFactor[]> {
  return fetchXAI(patient.id);
}

function normalizePatient(p: any): Patient {
  return {
    id: String(p.id),
    name: p.name ?? `Patient ${p.id}`,
    age: Number(p.age ?? 60),
    gender: String(p.gender ?? "M"),
    admissionTime: String(p.admissionTime ?? new Date().toISOString()),
    vitals: {
      heartRate: Number(p.vitals?.heartRate ?? 80),
      systolicBP: Number(p.vitals?.systolicBP ?? 120),
      temperature: Number(p.vitals?.temperature ?? 37),
      respiratoryRate: Number(p.vitals?.respiratoryRate ?? 16),
      oxygenSaturation: Number(p.vitals?.oxygenSaturation ?? 97),
    },
    labs: {
      wbc: Number(p.labs?.wbc ?? 9),
      lactate: Number(p.labs?.lactate ?? 1.5),
      creatinine: Number(p.labs?.creatinine ?? 1),
      crp: Number(p.labs?.crp ?? 5),
      platelets: Number(p.labs?.platelets ?? 220),
      bilirubin: Number(p.labs?.bilirubin ?? 0.7),
    },
    clinicalScores: {
      sofa: Number(p.clinicalScores?.sofa ?? 0),
      gcs: Number(p.clinicalScores?.gcs ?? 15),
      apacheII: Number(p.clinicalScores?.apacheII ?? 8),
    },
    allergies: Array.isArray(p.allergies) ? p.allergies : ["None Documented"],
    preExistingConditions: Array.isArray(p.preExistingConditions) ? p.preExistingConditions : [],
    medications: Array.isArray(p.medications) ? p.medications : [],
    sepsisRisk: Number(p.sepsisRisk ?? 0),
    predictedLOS: Number(p.predictedLOS ?? 0),
    _vent_prob: Number(p._vent_prob ?? 0),
    diagnosis: String(p.diagnosis ?? "ICU Observation"),
    acuityLevel: (p.acuityLevel ?? "Low") as Patient["acuityLevel"],
  };
}

// ---------------------------------------------------------------------------
// Custom patient prediction
// ---------------------------------------------------------------------------

export interface PatientInput {
  name?: string;
  age: number;
  gender: string;
  vitals: {
    heartRate: number;
    systolicBP: number;
    diastolicBP?: number;
    temperature: number;
    respiratoryRate: number;
    oxygenSaturation: number;
  };
  labs: {
    wbc: number;
    lactate: number;
    creatinine: number;
    platelets: number;
    bilirubin: number;
    crp?: number;
    pao2?: number;
    fio2?: number;
  };
  gcs: { eye: number; verbal: number; motor: number };
  medications?: Medication[];
  allergies?: string[];
  preExistingConditions?: string[];
}

export async function predictPatient(input: PatientInput): Promise<Patient> {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Predict ${res.status}: ${body.slice(0, 200)}`);
  }
  const raw = await res.json();
  return normalizePatient(raw);
}

export async function predictFromCSV(csvText: string): Promise<Patient[]> {
  const res = await fetch(`${BASE}/predict/csv`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ csv_text: csvText }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`CSV Predict ${res.status}: ${body.slice(0, 200)}`);
  }
  const raw: any[] = await res.json();
  return raw.filter((r) => !r.error).map(normalizePatient);
}

// ---------------------------------------------------------------------------
// Gemini AI care report
// ---------------------------------------------------------------------------

export interface AIReport {
  overall_assessment: string;
  risk_analysis: string;
  care_recommendations: string[];
  medication_review: string;
  monitoring_plan: string;
  warnings: string[];
  diet_and_nutrition: string;
}

export async function generateAIReport(
  patient: Patient,
  medications?: Medication[]
): Promise<AIReport> {
  const res = await fetch(`${BASE}/ai_report`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      patient,
      medications: medications ?? patient.medications,
    }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`AI Report ${res.status}: ${body.slice(0, 200)}`);
  }
  const data = await res.json();
  return data.report as AIReport;
}

