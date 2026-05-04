import React, { useState, useRef } from 'react';
import { Medication } from '../types';
import { PatientInput, predictPatient, predictFromCSV } from '../services/icuApi';
import type { Patient } from '../types';

interface Props {
  open: boolean;
  onClose: () => void;
  onPatientCreated: (patient: Patient) => void;
}

type Tab = 'manual' | 'csv';

const defaultVitals = { heartRate: 80, systolicBP: 120, diastolicBP: 80, temperature: 37, respiratoryRate: 16, oxygenSaturation: 97 };
const defaultLabs = { wbc: 9.0, lactate: 1.5, creatinine: 1.0, platelets: 220, bilirubin: 0.7, crp: 5 };
const defaultGCS = { eye: 4, verbal: 5, motor: 6 };

const PatientInputModal: React.FC<Props> = ({ open, onClose, onPatientCreated }) => {
  const [tab, setTab] = useState<Tab>('manual');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  // Manual form state
  const [name, setName] = useState('');
  const [age, setAge] = useState(60);
  const [gender, setGender] = useState('M');
  const [vitals, setVitals] = useState(defaultVitals);
  const [labs, setLabs] = useState(defaultLabs);
  const [gcs, setGcs] = useState(defaultGCS);
  const [allergies, setAllergies] = useState('');
  const [conditions, setConditions] = useState('');
  const [meds, setMeds] = useState<Medication[]>([]);
  const [medForm, setMedForm] = useState({ name: '', dosage: '', frequency: '', route: 'Oral' });

  // CSV state
  const [csvText, setCsvText] = useState('');
  const [csvFile, setCsvFile] = useState<File | null>(null);

  if (!open) return null;

  const addMed = () => {
    if (!medForm.name.trim()) return;
    setMeds([...meds, { ...medForm, status: 'Active' as const, durationDays: 0 }]);
    setMedForm({ name: '', dosage: '', frequency: '', route: 'Oral' });
  };
  const removeMed = (i: number) => setMeds(meds.filter((_, idx) => idx !== i));

  const handleManualSubmit = async () => {
    setLoading(true);
    setError('');
    try {
      const input: PatientInput = {
        name: name || 'Custom Patient',
        age,
        gender,
        vitals,
        labs,
        gcs,
        medications: meds,
        allergies: allergies ? allergies.split(',').map(a => a.trim()) : [],
        preExistingConditions: conditions ? conditions.split(',').map(c => c.trim()) : [],
      };
      const patient = await predictPatient(input);
      onPatientCreated(patient);
      onClose();
    } catch (e: any) {
      setError(e?.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCSVSubmit = async () => {
    setLoading(true);
    setError('');
    try {
      let text = csvText;
      if (csvFile && !text) {
        text = await csvFile.text();
      }
      if (!text.trim()) {
        setError('Please paste CSV data or upload a file.');
        setLoading(false);
        return;
      }
      const patients = await predictFromCSV(text);
      if (patients.length === 0) {
        setError('No valid patients found in CSV.');
      } else {
        patients.forEach(p => onPatientCreated(p));
        onClose();
      }
    } catch (e: any) {
      setError(e?.message || 'CSV processing failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) setCsvFile(file);
  };

  const field = (label: string, value: number, onChange: (v: number) => void, unit: string, step = 1) => (
    <div className="flex-1 min-w-[120px]">
      <label className="block text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">{label}</label>
      <div className="flex items-center gap-1">
        <input
          type="number"
          step={step}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-bold focus:border-indigo-500 focus:outline-none transition-colors"
        />
        <span className="text-[8px] text-slate-600 font-bold whitespace-nowrap">{unit}</span>
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-[#0b0e1f] border border-white/10 rounded-[2rem] shadow-2xl w-full max-w-[800px] max-h-[90vh] overflow-hidden flex flex-col m-4">
        {/* Header */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-white/5 bg-white/[0.02]">
          <div>
            <h2 className="text-lg font-black text-white tracking-tight">Analyze New Patient</h2>
            <p className="text-[9px] text-slate-500 font-bold uppercase tracking-widest mt-0.5">Enter data or upload CSV for ML predictions</p>
          </div>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center bg-white/5 hover:bg-white/10 rounded-lg transition-colors text-slate-400 hover:text-white">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-8 pt-5">
          {(['manual', 'csv'] as Tab[]).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-5 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${
                tab === t
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white'
              }`}
            >
              {t === 'manual' ? '📝 Manual Entry' : '📄 Upload CSV'}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6 custom-scrollbar">
          {tab === 'manual' ? (
            <>
              {/* Demographics */}
              <div>
                <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-3">Patient Demographics</h3>
                <div className="flex flex-wrap gap-3">
                  <div className="flex-1 min-w-[160px]">
                    <label className="block text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Patient Name</label>
                    <input value={name} onChange={e => setName(e.target.value)} placeholder="John Doe" className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-bold focus:border-indigo-500 focus:outline-none" />
                  </div>
                  {field('Age', age, setAge, 'yrs')}
                  <div className="flex-1 min-w-[100px]">
                    <label className="block text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Gender</label>
                    <select value={gender} onChange={e => setGender(e.target.value)} className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-bold focus:border-indigo-500 focus:outline-none">
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Vitals */}
              <div>
                <h3 className="text-[10px] font-black text-rose-400 uppercase tracking-[0.2em] mb-3">Vital Signs</h3>
                <div className="flex flex-wrap gap-3">
                  {field('Heart Rate', vitals.heartRate, v => setVitals({ ...vitals, heartRate: v }), 'bpm')}
                  {field('Systolic BP', vitals.systolicBP, v => setVitals({ ...vitals, systolicBP: v }), 'mmHg')}
                  {field('Diastolic BP', vitals.diastolicBP, v => setVitals({ ...vitals, diastolicBP: v }), 'mmHg')}
                  {field('Temperature', vitals.temperature, v => setVitals({ ...vitals, temperature: v }), '°C', 0.1)}
                  {field('Resp Rate', vitals.respiratoryRate, v => setVitals({ ...vitals, respiratoryRate: v }), '/min')}
                  {field('SpO2', vitals.oxygenSaturation, v => setVitals({ ...vitals, oxygenSaturation: v }), '%')}
                </div>
              </div>

              {/* Labs */}
              <div>
                <h3 className="text-[10px] font-black text-purple-400 uppercase tracking-[0.2em] mb-3">Laboratory Values</h3>
                <div className="flex flex-wrap gap-3">
                  {field('WBC', labs.wbc, v => setLabs({ ...labs, wbc: v }), 'k/μL', 0.1)}
                  {field('Lactate', labs.lactate, v => setLabs({ ...labs, lactate: v }), 'mmol/L', 0.1)}
                  {field('Creatinine', labs.creatinine, v => setLabs({ ...labs, creatinine: v }), 'mg/dL', 0.1)}
                  {field('Platelets', labs.platelets, v => setLabs({ ...labs, platelets: v }), 'k/μL')}
                  {field('Bilirubin', labs.bilirubin, v => setLabs({ ...labs, bilirubin: v }), 'mg/dL', 0.1)}
                  {field('CRP', labs.crp, v => setLabs({ ...labs, crp: v }), 'mg/L', 0.1)}
                </div>
              </div>

              {/* GCS */}
              <div>
                <h3 className="text-[10px] font-black text-emerald-400 uppercase tracking-[0.2em] mb-3">Glasgow Coma Scale</h3>
                <div className="flex flex-wrap gap-3">
                  {field('Eye (1-4)', gcs.eye, v => setGcs({ ...gcs, eye: Math.min(4, Math.max(1, v)) }), '')}
                  {field('Verbal (1-5)', gcs.verbal, v => setGcs({ ...gcs, verbal: Math.min(5, Math.max(1, v)) }), '')}
                  {field('Motor (1-6)', gcs.motor, v => setGcs({ ...gcs, motor: Math.min(6, Math.max(1, v)) }), '')}
                  <div className="flex-1 min-w-[120px] flex items-end pb-1">
                    <span className="text-sm font-black text-white">Total: <span className="text-indigo-400">{gcs.eye + gcs.verbal + gcs.motor}</span>/15</span>
                  </div>
                </div>
              </div>

              {/* Allergies & Conditions */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-[8px] font-black text-red-400 uppercase tracking-widest mb-1">Allergies (comma-separated)</label>
                  <input value={allergies} onChange={e => setAllergies(e.target.value)} placeholder="Penicillin, Sulfa" className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-bold focus:border-indigo-500 focus:outline-none" />
                </div>
                <div>
                  <label className="block text-[8px] font-black text-amber-400 uppercase tracking-widest mb-1">Pre-existing Conditions</label>
                  <input value={conditions} onChange={e => setConditions(e.target.value)} placeholder="Diabetes, HTN" className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-bold focus:border-indigo-500 focus:outline-none" />
                </div>
              </div>

              {/* Medications */}
              <div>
                <h3 className="text-[10px] font-black text-sky-400 uppercase tracking-[0.2em] mb-3">Current Medications</h3>
                {meds.length > 0 && (
                  <div className="space-y-2 mb-3">
                    {meds.map((m, i) => (
                      <div key={i} className="flex items-center gap-3 bg-slate-950/60 border border-white/5 rounded-xl px-4 py-2">
                        <span className="flex-1 text-xs text-white font-bold">{m.name}</span>
                        <span className="text-[10px] text-slate-400">{m.dosage}</span>
                        <span className="text-[10px] text-slate-400">{m.frequency}</span>
                        <span className="text-[10px] text-indigo-300">{m.route}</span>
                        <button onClick={() => removeMed(i)} className="text-rose-400 hover:text-rose-300 transition-colors">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="flex flex-wrap gap-2 items-end">
                  <div className="flex-1 min-w-[120px]">
                    <label className="block text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1">Drug Name</label>
                    <input value={medForm.name} onChange={e => setMedForm({ ...medForm, name: e.target.value })} placeholder="Meropenem" className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white font-bold focus:border-indigo-500 focus:outline-none" />
                  </div>
                  <div className="w-20">
                    <label className="block text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1">Dosage</label>
                    <input value={medForm.dosage} onChange={e => setMedForm({ ...medForm, dosage: e.target.value })} placeholder="1g" className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white font-bold focus:border-indigo-500 focus:outline-none" />
                  </div>
                  <div className="w-20">
                    <label className="block text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1">Frequency</label>
                    <input value={medForm.frequency} onChange={e => setMedForm({ ...medForm, frequency: e.target.value })} placeholder="q8h" className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white font-bold focus:border-indigo-500 focus:outline-none" />
                  </div>
                  <div className="w-20">
                    <label className="block text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1">Route</label>
                    <select value={medForm.route} onChange={e => setMedForm({ ...medForm, route: e.target.value })} className="w-full bg-slate-950 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white font-bold focus:border-indigo-500 focus:outline-none">
                      <option>Oral</option><option>IV</option><option>SubQ</option><option>IM</option><option>Inhaled</option><option>Topical</option>
                    </select>
                  </div>
                  <button onClick={addMed} className="bg-sky-600 hover:bg-sky-500 text-white text-[9px] font-black uppercase tracking-widest px-3 py-1.5 rounded-lg transition-all shadow-md">+ Add</button>
                </div>
              </div>
            </>
          ) : (
            /* CSV Tab */
            <div className="space-y-6">
              <div
                onDrop={handleFileDrop}
                onDragOver={e => e.preventDefault()}
                onClick={() => fileRef.current?.click()}
                className="border-2 border-dashed border-white/10 hover:border-indigo-500/40 rounded-2xl p-10 text-center cursor-pointer transition-colors group"
              >
                <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={e => setCsvFile(e.target.files?.[0] || null)} />
                <svg className="w-10 h-10 text-slate-600 group-hover:text-indigo-400 mx-auto mb-4 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                <p className="text-xs font-black text-slate-400 uppercase tracking-widest">
                  {csvFile ? csvFile.name : 'Drop CSV file here or click to browse'}
                </p>
              </div>

              <div>
                <label className="block text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">Or paste CSV data directly</label>
                <textarea
                  value={csvText}
                  onChange={e => setCsvText(e.target.value)}
                  rows={8}
                  placeholder={"name,age,gender,heart_rate,sbp,temperature,resp_rate,spo2,wbc,lactate,creatinine,platelets,bilirubin\nJohn Doe,65,M,110,90,38.5,24,91,18,4.2,2.1,95,3.2"}
                  className="w-full bg-slate-950 border border-white/10 rounded-xl px-4 py-3 text-xs text-white font-mono font-bold focus:border-indigo-500 focus:outline-none resize-none"
                />
              </div>

              <div className="bg-indigo-500/5 border border-indigo-500/20 rounded-xl p-4">
                <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-2">Supported Columns</p>
                <p className="text-[10px] text-slate-400 font-bold leading-relaxed">
                  <span className="text-white">Demographics:</span> name, age, gender &nbsp;|&nbsp;
                  <span className="text-white">Vitals:</span> heart_rate, sbp, dbp, temperature, resp_rate, spo2 &nbsp;|&nbsp;
                  <span className="text-white">Labs:</span> wbc, lactate, creatinine, platelets, bilirubin, crp &nbsp;|&nbsp;
                  <span className="text-white">GCS:</span> gcs_eye, gcs_verbal, gcs_motor
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-8 py-5 border-t border-white/5 bg-white/[0.02] flex items-center justify-between gap-4">
          {error && <p className="text-xs font-bold text-rose-400 flex-1 truncate">{error}</p>}
          <div className="flex gap-3 ml-auto">
            <button onClick={onClose} className="px-5 py-2.5 bg-white/5 hover:bg-white/10 text-slate-300 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all">Cancel</button>
            <button
              onClick={tab === 'manual' ? handleManualSubmit : handleCSVSubmit}
              disabled={loading}
              className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-black uppercase tracking-widest rounded-xl transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-60 flex items-center gap-2"
            >
              {loading && <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
              {loading ? 'Analyzing…' : tab === 'manual' ? 'Run ML Analysis' : 'Process CSV'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientInputModal;
