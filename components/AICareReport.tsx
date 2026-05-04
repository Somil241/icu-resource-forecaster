import React, { useState, useEffect } from 'react';
import { Patient, Medication } from '../types';
import { generateAIReport, AIReport } from '../services/icuApi';

interface Props {
  patient: Patient;
  medications: Medication[];
}

const AICareReport: React.FC<Props> = ({ patient, medications }) => {
  const [report, setReport] = useState<AIReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);

  const generate = async () => {
    setLoading(true);
    setError('');
    try {
      const r = await generateAIReport(patient, medications);
      setReport(r);
      setExpanded(true);
    } catch (e: any) {
      setError(e?.message || 'Failed to generate report');
    } finally {
      setLoading(false);
    }
  };

  // Reset report when patient changes.
  useEffect(() => {
    setReport(null);
    setError('');
    setExpanded(false);
  }, [patient.id]);

  return (
    <section className="bg-gradient-to-br from-slate-900/60 to-indigo-950/30 rounded-[2rem] border border-indigo-500/20 shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-8 py-5 border-b border-white/5 bg-white/[0.02]">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-lg shadow-indigo-500/20">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h3 className="text-sm font-black text-white tracking-tight uppercase">AI Clinical Care Report</h3>
            <p className="text-[8px] font-bold text-slate-500 uppercase tracking-widest">Powered by Google Gemini • Considers patient medications</p>
          </div>
        </div>
        <button
          onClick={generate}
          disabled={loading}
          className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-[9px] font-black uppercase tracking-widest rounded-xl transition-all shadow-lg shadow-indigo-600/30 disabled:opacity-60 flex items-center gap-2"
        >
          {loading && <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
          {loading ? 'Generating…' : report ? '↻ Regenerate' : '✦ Generate Report'}
        </button>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="p-12 flex flex-col items-center justify-center">
          <div className="w-12 h-12 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin mb-4" />
          <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest animate-pulse">Gemini is analyzing patient data…</p>
          <p className="text-[9px] text-slate-600 font-bold mt-2">Considering vitals, labs, predictions & {medications.length} medication{medications.length !== 1 ? 's' : ''}</p>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="p-6 m-6 bg-rose-500/10 border border-rose-500/20 rounded-xl">
          <p className="text-xs font-bold text-rose-400">{error}</p>
        </div>
      )}

      {/* Report Content */}
      {report && !loading && (
        <div className={`transition-all duration-500 ${expanded ? 'max-h-[5000px]' : 'max-h-0'} overflow-hidden`}>
          <div className="p-8 space-y-6">

            {/* Warnings */}
            {report.warnings && report.warnings.length > 0 && (
              <div className="bg-rose-500/10 border border-rose-500/20 rounded-2xl p-5">
                <h4 className="text-[10px] font-black text-rose-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>
                  Critical Warnings
                </h4>
                <div className="space-y-2">
                  {report.warnings.map((w, i) => (
                    <p key={i} className="text-xs font-bold text-rose-200 flex items-start gap-2">
                      <span className="text-rose-400 mt-0.5">⚠</span> {w}
                    </p>
                  ))}
                </div>
              </div>
            )}

            {/* Overall Assessment */}
            <div className="bg-slate-950/40 rounded-2xl p-5 border border-white/5">
              <h4 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-indigo-500 shadow-lg shadow-indigo-500/40" /> Overall Assessment
              </h4>
              <p className="text-sm text-slate-200 font-bold leading-relaxed">{report.overall_assessment}</p>
            </div>

            {/* Risk Analysis */}
            <div className="bg-slate-950/40 rounded-2xl p-5 border border-white/5">
              <h4 className="text-[10px] font-black text-amber-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-500 shadow-lg shadow-amber-500/40" /> Risk Analysis
              </h4>
              <p className="text-sm text-slate-200 font-bold leading-relaxed">{report.risk_analysis}</p>
            </div>

            {/* Care Recommendations */}
            <div className="bg-slate-950/40 rounded-2xl p-5 border border-white/5">
              <h4 className="text-[10px] font-black text-emerald-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/40" /> Care Recommendations
              </h4>
              <div className="space-y-3">
                {report.care_recommendations.map((r, i) => (
                  <div key={i} className="flex gap-3 items-start">
                    <div className="flex-shrink-0 w-5 h-5 bg-emerald-500/10 border border-emerald-500/20 rounded-md flex items-center justify-center text-emerald-400 font-black text-[9px]">{i + 1}</div>
                    <p className="text-xs text-slate-300 font-bold leading-relaxed">{r}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Medication Review */}
            <div className="bg-gradient-to-br from-sky-950/30 to-slate-950/40 rounded-2xl p-5 border border-sky-500/20">
              <h4 className="text-[10px] font-black text-sky-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                Medication Review
              </h4>
              <p className="text-sm text-slate-200 font-bold leading-relaxed">{report.medication_review}</p>
            </div>

            {/* Monitoring Plan */}
            <div className="bg-slate-950/40 rounded-2xl p-5 border border-white/5">
              <h4 className="text-[10px] font-black text-purple-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-500 shadow-lg shadow-purple-500/40" /> Monitoring Plan
              </h4>
              <p className="text-sm text-slate-200 font-bold leading-relaxed">{report.monitoring_plan}</p>
            </div>

            {/* Diet & Nutrition */}
            {report.diet_and_nutrition && (
              <div className="bg-slate-950/40 rounded-2xl p-5 border border-white/5">
                <h4 className="text-[10px] font-black text-orange-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-orange-500 shadow-lg shadow-orange-500/40" /> Diet & Nutrition
                </h4>
                <p className="text-sm text-slate-200 font-bold leading-relaxed">{report.diet_and_nutrition}</p>
              </div>
            )}

          </div>
        </div>
      )}

      {/* Collapsed prompt */}
      {!report && !loading && !error && (
        <div className="p-8 text-center">
          <p className="text-xs text-slate-500 font-bold">
            Click <span className="text-indigo-400">Generate Report</span> to create an AI-powered clinical care plan.
            {medications.length > 0 && (
              <span className="text-sky-400"> The report will consider the {medications.length} medication{medications.length !== 1 ? 's' : ''} you've added.</span>
            )}
          </p>
        </div>
      )}
    </section>
  );
};

export default AICareReport;
