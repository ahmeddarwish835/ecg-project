/**
 * ECG Test Lab – Interactive ECG Filtering Dashboard
 * Frontend entry point — React + Tailwind CSS + Plotly.js
 *
 * Architecture:
 *  - Single-page dashboard, no router needed
 *  - Controls → POST /analyze → display results in tabs
 *  - All DSP computation happens on the FastAPI backend
 */

import React, { useState, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import Plot from 'react-plotly.js';
import {
  Activity, HeartPulse, RadioTower, BarChart3, Cpu,
  Waves, Zap, ChevronDown, AlertCircle, TrendingUp,
  GitCompare, Info, Clock, Wifi, ArrowUpRight, Filter,
} from 'lucide-react';
import './index.css';
import { analyzeEcg } from './lib/api';

/* ─── Static config ─────────────────────────────────────────────── */

const ECG_TESTS = [
  { id: 'test1', label: 'Test 1', hint: 'Normal ECG',                tag: 'CLEAN' },
  { id: 'test2', label: 'Test 2', hint: 'Baseline wander noise',     tag: 'BW'    },
  { id: 'test3', label: 'Test 3', hint: '50 Hz power-line noise',    tag: 'PLI'   },
  { id: 'test4', label: 'Test 4', hint: 'EMG / muscle noise',        tag: 'EMG'   },
  { id: 'test5', label: 'Test 5', hint: 'Mixed noise',               tag: 'MIX'   },
];

const FILTER_TYPES = [
  {
    id: 'fir',
    label: 'FIR Filter',
    desc: 'Finite Impulse Response – linear phase, no feedback.',
    detail: 'Uses a window-based FIR design. High-pass (0.5 Hz, 301 taps), 50 Hz IIR notch, low-pass (40 Hz, 201 taps). Linear phase means no phase distortion — important for QRS morphology analysis.',
    color: '#22d3ee',
  },
  {
    id: 'butterworth',
    label: 'Butterworth Filter',
    desc: 'Maximally flat magnitude response in passband.',
    detail: 'Order-3 high-pass (0.5 Hz) + 50 Hz notch + order-4 low-pass (40 Hz). Butterworth filters have the flattest possible passband with no ripple, ideal for general ECG denoising without signal distortion.',
    color: '#818cf8',
  },
  {
    id: 'chebyshev',
    label: 'Chebyshev Type I',
    desc: 'Sharper roll-off at the cost of passband ripple.',
    detail: 'Order-3 high-pass + 50 Hz notch + order-4 low-pass. Chebyshev filters achieve a steeper transition band than Butterworth at the same order, at the cost of controlled passband ripple (0.5 dB).',
    color: '#f472b6',
  },
];

const TABS = [
  { id: 'time',   label: 'Time Domain',       icon: Activity    },
  { id: 'freq',   label: 'Frequency Domain',  icon: Waves       },
  { id: 'heart',  label: 'Heart Analysis',    icon: HeartPulse  },
  { id: 'filter', label: 'Filter Info',       icon: Filter      },
];

/* ─── Plotly shared helpers ──────────────────────────────────────── */

const PLOT_COLORS = {
  original:  '#94a3b8',
  filtered:  '#00c8f0',
  before:    '#f87171',
  after:     '#34d399',
  rpeaks:    '#fbbf24',
  bg:        'rgba(0,0,0,0)',
  plotbg:    'rgba(8, 14, 28, 0.65)',
  grid:      'rgba(148,163,184,0.10)',
  text:      '#cbd5e1',
  title:     '#e2e8f0',
};

function makePlotLayout(title, yTitle = 'Amplitude (mV)', extras = {}) {
  return {
    title: { text: title, font: { color: PLOT_COLORS.title, size: 15, family: 'Inter' } },
    paper_bgcolor: PLOT_COLORS.bg,
    plot_bgcolor:  PLOT_COLORS.plotbg,
    font: { color: PLOT_COLORS.text, family: 'Inter', size: 12 },
    margin: { l: 60, r: 24, t: 52, b: 50 },
    xaxis: {
      gridcolor: PLOT_COLORS.grid,
      zeroline: false,
      linecolor: 'rgba(148,163,184,0.2)',
      tickfont: { size: 11 },
      ...extras.xaxis,
    },
    yaxis: {
      title: yTitle,
      gridcolor: PLOT_COLORS.grid,
      zeroline: false,
      linecolor: 'rgba(148,163,184,0.2)',
      tickfont: { size: 11 },
      ...extras.yaxis,
    },
    legend: {
      orientation: 'h',
      y: -0.22,
      x: 0.5,
      xanchor: 'center',
      bgcolor: 'rgba(0,0,0,0)',
      bordercolor: 'rgba(0,0,0,0)',
      font: { size: 12 },
    },
    autosize: true,
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: 'rgba(8,14,28,0.92)',
      bordercolor: 'rgba(0,200,240,0.4)',
      font: { color: '#e2e8f0', size: 12 },
    },
    ...extras,
  };
}

const PLOT_CONFIG = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'select2d', 'lasso2d'],
  modeBarButtonsToAdd: [],
  scrollZoom: true,
};

/* ─── Sub-components ─────────────────────────────────────────────── */

/** A glassmorphism stat card */
function StatCard({ title, value, icon: Icon, suffix = '', accent = '#00c8f0', highlight = false }) {
  return (
    <div className={`glass-hover rounded-2xl p-5 animate-slide-up ${highlight ? 'ring-1 ring-cyan-400/30' : ''}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">{title}</span>
        <span className="p-2 rounded-xl" style={{ background: `${accent}18`, color: accent }}>
          <Icon size={15} />
        </span>
      </div>
      <div className="flex items-end gap-1.5">
        <span className="text-2xl font-bold text-white font-mono">{value}</span>
        {suffix && <span className="text-sm text-slate-400 mb-0.5">{suffix}</span>}
      </div>
    </div>
  );
}

/** Wrapper card for plot sections */
function PlotCard({ children, className = '' }) {
  return (
    <div className={`glass rounded-2xl p-4 animate-fade-in ${className}`}>
      {children}
    </div>
  );
}

/** Tab navigation pill */
function TabBar({ tabs, active, onChange }) {
  return (
    <div className="flex flex-wrap gap-2 p-1.5 glass rounded-2xl w-fit">
      {tabs.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
            active === id
              ? 'bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-400/20'
              : 'text-slate-300 hover:text-white hover:bg-white/5'
          }`}
        >
          <Icon size={15} />
          {label}
        </button>
      ))}
    </div>
  );
}

/** Custom styled select dropdown */
function StyledSelect({ label, value, onChange, options }) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</span>
      <div className="select-wrapper">
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-4 py-3 pr-10 text-sm text-slate-100
                     focus:border-cyan-400/60 focus:ring-1 focus:ring-cyan-400/30 transition-colors"
        >
          {options.map(o => (
            <option key={o.id} value={o.id}>{o.label} — {o.hint || o.desc}</option>
          ))}
        </select>
      </div>
    </label>
  );
}

/** Animated ECG icon in header */
function EcgIcon() {
  return (
    <svg width="120" height="40" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg" className="opacity-70">
      <path
        className="ecg-line"
        d="M0 20 L20 20 L28 8 L34 32 L42 4 L48 36 L54 20 L62 20 L70 20 L78 14 L82 26 L86 20 L120 20"
        stroke="#00c8f0"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

/** SNR improvement badge */
function SnrBadge({ value }) {
  const isPositive = parseFloat(value) >= 0;
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-bold px-2 py-0.5 rounded-full ${
      isPositive ? 'bg-emerald-400/15 text-emerald-300' : 'bg-red-400/15 text-red-300'
    }`}>
      <ArrowUpRight size={11} className={isPositive ? '' : 'rotate-90'} />
      {isPositive ? '+' : ''}{value} dB
    </span>
  );
}

/** Full loading state overlay */
function LoadingOverlay() {
  return (
    <div className="glass rounded-3xl p-14 text-center animate-fade-in">
      <div className="relative inline-flex mb-6">
        <div className="w-16 h-16 rounded-full border-4 border-cyan-400/20 border-t-cyan-400 animate-spin" />
        <HeartPulse className="absolute inset-0 m-auto text-cyan-300" size={24} />
      </div>
      <h2 className="text-xl font-bold text-white">Running DSP Analysis</h2>
      <p className="text-slate-400 mt-2 text-sm">Applying filter pipeline and computing metrics…</p>
    </div>
  );
}

/** Empty state */
function EmptyState() {
  return (
    <div className="glass rounded-3xl p-14 text-center animate-fade-in">
      <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl badge-glow mb-6">
        <HeartPulse className="text-cyan-300" size={38} />
      </div>
      <h2 className="text-2xl font-bold text-white">Ready for ECG Analysis</h2>
      <p className="text-slate-400 mt-3 max-w-md mx-auto">
        Select an ECG test file and a filter family, then click <strong className="text-cyan-300">Run Analysis</strong> to
        process the signal and view all DSP metrics and graphs.
      </p>
      <div className="mt-8 grid sm:grid-cols-3 gap-3 max-w-lg mx-auto text-left">
        {[
          { icon: Activity,   label: 'Choose ECG',   sub: '5 test signals available' },
          { icon: Filter,     label: 'Pick Filter',  sub: 'FIR, Butterworth, Chebyshev' },
          { icon: BarChart3,  label: 'View Results', sub: 'Metrics + interactive graphs' },
        ].map(({ icon: Icon, label, sub }) => (
          <div key={label} className="glass rounded-xl p-4 text-center">
            <Icon size={22} className="text-cyan-400 mx-auto mb-2" />
            <div className="text-sm font-semibold text-white">{label}</div>
            <div className="text-xs text-slate-400 mt-1">{sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Tab: Time Domain ───────────────────────────────────────────── */
function TimeDomainTab({ data }) {
  const lineBase = { type: 'scatter', mode: 'lines', line: { width: 1.5 } };

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="grid lg:grid-cols-2 gap-5">
        <PlotCard>
          <Plot
            useResizeHandler
            className="w-full"
            style={{ height: 380 }}
            data={[{
              ...lineBase,
              x: data.time,
              y: data.original_signal,
              name: 'Original ECG',
              line: { ...lineBase.line, color: PLOT_COLORS.original },
            }]}
            layout={makePlotLayout('Original ECG Signal', 'Amplitude (mV)', {
              xaxis: { title: 'Time (s)' },
            })}
            config={PLOT_CONFIG}
          />
        </PlotCard>

        <PlotCard>
          <Plot
            useResizeHandler
            className="w-full"
            style={{ height: 380 }}
            data={[{
              ...lineBase,
              x: data.time,
              y: data.filtered_signal,
              name: 'Filtered ECG',
              line: { ...lineBase.line, color: PLOT_COLORS.filtered },
            }]}
            layout={makePlotLayout('Filtered ECG Signal', 'Amplitude (mV)', {
              xaxis: { title: 'Time (s)' },
            })}
            config={PLOT_CONFIG}
          />
        </PlotCard>
      </div>

      <PlotCard>
        <Plot
          useResizeHandler
          className="w-full"
          style={{ height: 400 }}
          data={[
            {
              ...lineBase,
              x: data.time,
              y: data.original_signal,
              name: 'Original',
              line: { ...lineBase.line, color: PLOT_COLORS.original, dash: 'dot' },
              opacity: 0.75,
            },
            {
              ...lineBase,
              x: data.time,
              y: data.filtered_signal,
              name: 'Filtered',
              line: { ...lineBase.line, color: PLOT_COLORS.filtered },
            },
          ]}
          layout={makePlotLayout('Original vs Filtered Comparison', 'Amplitude (mV)', {
            xaxis: { title: 'Time (s)' },
          })}
          config={PLOT_CONFIG}
        />
      </PlotCard>
    </div>
  );
}

/* ─── Tab: Frequency Domain ──────────────────────────────────────── */
function FrequencyDomainTab({ data }) {
  const lineBase = { type: 'scatter', mode: 'lines', line: { width: 1.8 } };

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="grid lg:grid-cols-2 gap-5">
        <PlotCard>
          <Plot
            useResizeHandler
            className="w-full"
            style={{ height: 390 }}
            data={[
              {
                ...lineBase,
                x: data.psd.frequency,
                y: data.psd.before,
                name: 'PSD Before',
                line: { ...lineBase.line, color: PLOT_COLORS.before },
                fill: 'tozeroy',
                fillcolor: `${PLOT_COLORS.before}18`,
              },
              {
                ...lineBase,
                x: data.psd.frequency,
                y: data.psd.after,
                name: 'PSD After',
                line: { ...lineBase.line, color: PLOT_COLORS.after },
                fill: 'tozeroy',
                fillcolor: `${PLOT_COLORS.after}18`,
              },
            ]}
            layout={makePlotLayout('Power Spectral Density — Welch Method', 'Power/Hz', {
              xaxis: { title: 'Frequency (Hz)', range: [0, 120] },
              yaxis: { type: 'log', title: 'Power/Hz (log)' },
            })}
            config={PLOT_CONFIG}
          />
        </PlotCard>

        <PlotCard>
          <Plot
            useResizeHandler
            className="w-full"
            style={{ height: 390 }}
            data={[
              {
                ...lineBase,
                x: data.fft.frequency,
                y: data.fft.before,
                name: 'FFT Before',
                line: { ...lineBase.line, color: PLOT_COLORS.before },
              },
              {
                ...lineBase,
                x: data.fft.frequency,
                y: data.fft.after,
                name: 'FFT After',
                line: { ...lineBase.line, color: PLOT_COLORS.after },
              },
            ]}
            layout={makePlotLayout('FFT Frequency Spectrum', 'Magnitude', {
              xaxis: { title: 'Frequency (Hz)', range: [0, 120] },
            })}
            config={PLOT_CONFIG}
          />
        </PlotCard>
      </div>

      {/* Spectrogram if available */}
      {data.spectrogram && (
        <PlotCard>
          <Plot
            useResizeHandler
            className="w-full"
            style={{ height: 380 }}
            data={[{
              type: 'heatmap',
              x: data.spectrogram.times,
              y: data.spectrogram.frequencies,
              z: data.spectrogram.power,
              colorscale: 'Viridis',
              showscale: true,
              colorbar: {
                title: 'dB',
                titlefont: { color: '#cbd5e1' },
                tickfont: { color: '#cbd5e1' },
              },
            }]}
            layout={makePlotLayout('Spectrogram — Short-Time Fourier Transform', 'Frequency (Hz)', {
              xaxis: { title: 'Time (s)' },
            })}
            config={PLOT_CONFIG}
          />
        </PlotCard>
      )}
    </div>
  );
}

/* ─── Tab: Heart Analysis ────────────────────────────────────────── */
function HeartAnalysisTab({ data }) {
  return (
    <div className="space-y-5 animate-fade-in">
      <PlotCard>
        <Plot
          useResizeHandler
          className="w-full"
          style={{ height: 440 }}
          data={[
            {
              type: 'scatter', mode: 'lines',
              x: data.time,
              y: data.filtered_signal,
              name: 'Filtered ECG',
              line: { color: PLOT_COLORS.filtered, width: 1.5 },
            },
            {
              type: 'scatter', mode: 'markers',
              x: data.r_peaks.time,
              y: data.r_peaks.amplitude,
              name: 'Detected R-Peaks',
              marker: {
                symbol: 'triangle-down',
                size: 12,
                color: PLOT_COLORS.rpeaks,
                line: { width: 1.5, color: 'rgba(0,0,0,0.5)' },
              },
            },
          ]}
          layout={makePlotLayout('R-Peak Detection — Heartbeat Visualization', 'Amplitude (mV)', {
            xaxis: { title: 'Time (s)' },
          })}
          config={PLOT_CONFIG}
        />
      </PlotCard>

      {/* Heart stats summary row */}
      <div className="grid sm:grid-cols-3 gap-4">
        <div className="glass rounded-2xl p-5 text-center">
          <div className="text-4xl font-black font-mono text-cyan-300">{data.heart_rate_bpm}</div>
          <div className="text-sm text-slate-400 mt-1">Beats per Minute</div>
          <div className="text-xs text-slate-500 mt-1">Normal: 60–100 BPM</div>
        </div>
        <div className="glass rounded-2xl p-5 text-center">
          <div className="text-4xl font-black font-mono text-indigo-300">{data.detected_heartbeats}</div>
          <div className="text-sm text-slate-400 mt-1">Detected R-Peaks</div>
          <div className="text-xs text-slate-500 mt-1">Total heartbeats in signal</div>
        </div>
        <div className="glass rounded-2xl p-5 text-center">
          <div className="text-4xl font-black font-mono text-emerald-300">
            {data.detected_heartbeats > 0 ? (data.time[data.time.length - 1] / data.detected_heartbeats).toFixed(2) : '—'}
          </div>
          <div className="text-sm text-slate-400 mt-1">Mean RR Interval (s)</div>
          <div className="text-xs text-slate-500 mt-1">Average beat spacing</div>
        </div>
      </div>
    </div>
  );
}

/* ─── Tab: Filter Info ───────────────────────────────────────────── */
function FilterInfoTab({ data, selectedFilter }) {
  const filterMeta = FILTER_TYPES.find(f => f.id === selectedFilter) || FILTER_TYPES[1];

  const pipeline = [
    {
      step: '1',
      name: 'High-Pass Filter',
      desc: 'Removes slow baseline wander (< 0.5 Hz). Essential to eliminate DC drift and respiration artefacts.',
      freq: '0.5 Hz cutoff',
      icon: TrendingUp,
      color: '#34d399',
    },
    {
      step: '2',
      name: '50 Hz Notch Filter',
      desc: 'Suppresses power-line interference at exactly 50 Hz (IIR notch, Q=30). Narrow band rejection with minimal signal distortion.',
      freq: '50 Hz notch',
      icon: Zap,
      color: '#fbbf24',
    },
    {
      step: '3',
      name: 'Low-Pass Filter',
      desc: 'Removes high-frequency muscle (EMG) noise above 40 Hz. Preserves diagnostically relevant ECG content (< 40 Hz).',
      freq: '40 Hz cutoff',
      icon: Waves,
      color: '#818cf8',
    },
  ];

  return (
    <div className="space-y-5 animate-fade-in">

      {/* Selected filter description */}
      <div className="glass rounded-2xl p-6" style={{ borderColor: `${filterMeta.color}30` }}>
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl flex-shrink-0" style={{ background: `${filterMeta.color}18`, color: filterMeta.color }}>
            <Cpu size={22} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">{filterMeta.label}</h2>
            <p className="text-slate-400 mt-1 text-sm">{filterMeta.desc}</p>
            <p className="text-slate-300 mt-3 leading-relaxed">{filterMeta.detail}</p>
          </div>
        </div>
      </div>

      {/* Pipeline steps */}
      <div className="glass rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-5 flex items-center gap-2">
          <GitCompare size={18} className="text-cyan-400" />
          Automatic DSP Pipeline
        </h3>
        <div className="space-y-4">
          {pipeline.map((stage) => (
            <div key={stage.step} className="flex items-start gap-4 p-4 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center font-bold text-sm"
                   style={{ background: `${stage.color}20`, color: stage.color }}>
                {stage.step}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="font-semibold text-white">{stage.name}</span>
                  <span className="text-xs font-mono px-2 py-0.5 rounded-md"
                        style={{ background: `${stage.color}15`, color: stage.color }}>
                    {stage.freq}
                  </span>
                </div>
                <p className="text-slate-400 text-sm mt-1 leading-relaxed">{stage.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* SNR improvement explanation */}
      <div className="glass rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Info size={18} className="text-cyan-400" />
          SNR Metrics Explained
        </h3>
        <div className="grid sm:grid-cols-3 gap-4 text-sm">
          {[
            { label: 'SNR Before', value: `${data.snr_before_db} dB`, color: PLOT_COLORS.before, desc: 'Signal-to-noise ratio of the raw ECG' },
            { label: 'SNR After',  value: `${data.snr_after_db} dB`,  color: PLOT_COLORS.after,  desc: 'SNR after applying the filter pipeline' },
            { label: 'Improvement', value: `${data.snr_improvement_db} dB`, color: '#fbbf24', desc: 'Net SNR gain achieved by the filter' },
          ].map(item => (
            <div key={item.label} className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.05]">
              <div className="font-mono text-2xl font-bold" style={{ color: item.color }}>{item.value}</div>
              <div className="font-semibold text-white mt-1">{item.label}</div>
              <div className="text-slate-400 mt-1 text-xs">{item.desc}</div>
            </div>
          ))}
        </div>
        <p className="text-slate-500 text-xs mt-4">
          SNR is estimated using a Butterworth-filtered reference signal. Higher SNR indicates a cleaner signal.
          A positive improvement confirms successful noise suppression.
        </p>
      </div>
    </div>
  );
}

/* ─── Main App ───────────────────────────────────────────────────── */
function App() {
  const [testId,     setTestId]     = useState('test1');
  const [filterType, setFilterType] = useState('butterworth');
  const [activeTab,  setActiveTab]  = useState('time');
  const [data,       setData]       = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState('');

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const result = await analyzeEcg(testId, filterType);
      setData(result);
      setActiveTab('time');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [testId, filterType]);

  const selectedTest   = ECG_TESTS.find(t => t.id === testId);
  const selectedFilter = FILTER_TYPES.find(f => f.id === filterType);

  return (
    <main className="min-h-screen p-4 md:p-8"
      style={{
        background: `
          radial-gradient(ellipse 80% 50% at 10% -10%, rgba(0, 100, 130, 0.35), transparent),
          radial-gradient(ellipse 60% 40% at 90% 5%,  rgba(79, 70, 229, 0.25), transparent),
          radial-gradient(ellipse 50% 60% at 50% 100%, rgba(0, 30, 60, 0.5), transparent),
          #060d1a
        `,
      }}
    >
      <div className="max-w-7xl mx-auto space-y-6">

        {/* ── Header ── */}
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full badge-glow text-cyan-200 text-xs font-semibold uppercase tracking-wider">
              <span className="pulse-dot w-1.5 h-1.5 rounded-full bg-cyan-400 text-cyan-400" />
              DSP Biomedical Dashboard
            </div>
            <h1 className="text-4xl md:text-5xl font-black mt-3 tracking-tight">
              ECG <span className="text-gradient">Test Lab</span>
            </h1>
            <p className="text-slate-400 mt-2 max-w-xl text-sm leading-relaxed">
              Interactive ECG Filtering Dashboard — FastAPI · React · Plotly · NumPy · SciPy
            </p>
            <EcgIcon />
          </div>

          {/* Status badge */}
          <div className="glass rounded-2xl p-4 text-sm space-y-2 flex-shrink-0">
            <div className="flex items-center gap-2 text-slate-300">
              <Wifi size={14} className="text-emerald-400" />
              <span className="text-xs">Backend</span>
              <code className="font-mono text-xs text-emerald-300 ml-auto">localhost:8000</code>
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              <Activity size={14} className="text-cyan-400" />
              <span className="text-xs">Frontend</span>
              <code className="font-mono text-xs text-cyan-300 ml-auto">localhost:5173</code>
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              <Clock size={14} className="text-slate-400" />
              <span className="text-xs">Sample rate</span>
              <code className="font-mono text-xs text-slate-300 ml-auto">360 Hz</code>
            </div>
          </div>
        </header>

        {/* ── Controls ── */}
        <section className="glass rounded-3xl p-5">
          <div className="grid md:grid-cols-[1fr_1fr_auto] gap-4 items-end">
            <StyledSelect
              label="ECG Test Signal"
              value={testId}
              onChange={setTestId}
              options={ECG_TESTS}
            />
            <StyledSelect
              label="Filter Family"
              value={filterType}
              onChange={setFilterType}
              options={FILTER_TYPES}
            />
            <button
              onClick={runAnalysis}
              disabled={loading}
              className="flex items-center justify-center gap-2.5 rounded-xl px-7 py-3 font-bold text-slate-950
                         bg-gradient-to-r from-cyan-400 to-cyan-300
                         hover:from-cyan-300 hover:to-cyan-200
                         disabled:opacity-60 disabled:cursor-not-allowed
                         shadow-lg shadow-cyan-400/20 transition-all duration-200
                         active:scale-95"
            >
              <Zap size={18} />
              {loading ? 'Analyzing…' : 'Run Analysis'}
            </button>
          </div>

          {/* Selected context pills */}
          {!loading && (
            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="px-3 py-1 rounded-full bg-slate-800/60 text-slate-300 border border-slate-700/60">
                Signal: <strong className="text-cyan-300">{selectedTest?.label}</strong> — {selectedTest?.hint}
              </span>
              <span className="px-3 py-1 rounded-full bg-slate-800/60 text-slate-300 border border-slate-700/60">
                Filter: <strong style={{ color: selectedFilter?.color }}>{selectedFilter?.label}</strong>
              </span>
            </div>
          )}
        </section>

        {/* ── Error ── */}
        {error && (
          <div className="flex items-start gap-3 rounded-2xl bg-red-500/10 border border-red-400/25 p-4 text-red-200 animate-fade-in">
            <AlertCircle size={18} className="flex-shrink-0 mt-0.5 text-red-400" />
            <div>
              <div className="font-semibold">Analysis failed</div>
              <div className="text-sm text-red-300 mt-0.5">{error}</div>
            </div>
          </div>
        )}

        {/* ── Loading ── */}
        {loading && <LoadingOverlay />}

        {/* ── Results ── */}
        {!loading && data && (
          <>
            {/* Stat cards */}
            <section className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard title="Test Signal"       value={data.test_name}           icon={Activity}   accent="#00c8f0" />
              <StatCard title="Noise Type"        value={data.noise_type}          icon={RadioTower} accent="#f472b6" />
              <StatCard title="Heart Rate"        value={data.heart_rate_bpm}      icon={HeartPulse} accent="#f87171" suffix="BPM" highlight />
              <StatCard title="Detected Beats"   value={data.detected_heartbeats} icon={BarChart3}  accent="#fbbf24" />
              <StatCard title="Filter Applied"   value={data.filter_type}         icon={Cpu}        accent="#818cf8" />
              <StatCard title="SNR Before"       value={`${data.snr_before_db}`}  icon={Waves}      accent="#f87171" suffix="dB" />
              <StatCard title="SNR After"        value={`${data.snr_after_db}`}   icon={Waves}      accent="#34d399" suffix="dB" />
              <StatCard title="SNR Improvement"  value={`${data.snr_improvement_db}`} icon={Zap}    accent="#fbbf24" suffix="dB" highlight />
            </section>

            {/* Tabs */}
            <TabBar tabs={TABS} active={activeTab} onChange={setActiveTab} />

            {/* Tab content */}
            {activeTab === 'time'   && <TimeDomainTab      data={data} />}
            {activeTab === 'freq'   && <FrequencyDomainTab data={data} />}
            {activeTab === 'heart'  && <HeartAnalysisTab   data={data} />}
            {activeTab === 'filter' && <FilterInfoTab      data={data} selectedFilter={filterType} />}
          </>
        )}

        {/* ── Empty state ── */}
        {!loading && !data && !error && <EmptyState />}

        {/* ── Footer ── */}
        <footer className="text-center text-xs text-slate-600 py-4">
          ECG Test Lab · DSP Course Project · FastAPI + React + Plotly + NumPy + SciPy
        </footer>
      </div>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
