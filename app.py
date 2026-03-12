"""
OptiFlow — Production Line Optimizer
Smart machine allocation for manufacturing efficiency
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core_algorithm import (
    cookie_recipes,
    scaling_factors,
    station_names,
    cultural_algorithm,
    calculate_station_queue_metrics,
    run_monte_carlo,
    run_stress_test,
)
from templates import INDUSTRY_TEMPLATES

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="OptiFlow — Production Line Optimizer",
    page_icon="🏭",
    layout="wide",
)

# ============================================================================
# CUSTOM CSS — modern, interactive input fields
# ============================================================================

st.markdown("""
<style>
/* ─── Global ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ─── Input fields: clear border + focus glow ─── */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    border: 2px solid #333 !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    font-size: 16px !important;
    background: #1a1f2e !important;
    color: #fafafa !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.2) !important;
    outline: none !important;
}
div[data-testid="stNumberInput"] input:hover,
div[data-testid="stTextInput"] input:hover {
    border-color: #00d4aa80 !important;
}

/* ─── Input labels: bigger + bolder ─── */
div[data-testid="stNumberInput"] label p,
div[data-testid="stTextInput"] label p,
div[data-testid="stSlider"] label p {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #fafafa !important;
    margin-bottom: 4px !important;
}

/* ─── Slider: teal track ─── */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #00d4aa !important;
}
div[data-testid="stSlider"] div[data-testid="stTickBar"] {
    background: linear-gradient(to right, #00d4aa, #00d4aa) !important;
}

/* ─── Buttons ─── */
button[kind="primary"] {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    color: #0e1117 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    border: none !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 212, 170, 0.35) !important;
}
button[kind="secondary"] {
    border: 2px solid #00d4aa !important;
    color: #00d4aa !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
button[kind="secondary"]:hover {
    background: rgba(0, 212, 170, 0.1) !important;
}

/* ─── Data editor: styled table ─── */
div[data-testid="stDataEditor"] {
    border: 2px solid #333 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ─── Expander ─── */
div[data-testid="stExpander"] {
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
}

/* ─── Metrics cards ─── */
div[data-testid="stMetric"] {
    background: #1a1f2e !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
div[data-testid="stMetric"]:hover {
    border-color: #00d4aa60 !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stMetric"] label {
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: #888 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #00d4aa !important;
}

/* ─── Section headers ─── */
h2 {
    color: #fafafa !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}

/* ─── Dividers ─── */
hr {
    border-color: #2a2f3e !important;
    margin: 2rem 0 !important;
}

/* ─── Download button ─── */
div[data-testid="stDownloadButton"] button {
    border: 2px solid #00d4aa !important;
    color: #00d4aa !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ─── Help tooltips ─── */
div[data-testid="stTooltipIcon"] {
    color: #00d4aa !important;
}

/* ─── Hero badge pills ─── */
.feature-badge {
    display: inline-block;
    background: rgba(0, 212, 170, 0.12);
    color: #00d4aa;
    border: 1px solid rgba(0, 212, 170, 0.3);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px;
}
.hero-subtitle {
    color: #aaa;
    font-size: 18px;
    text-align: center;
    margin-top: 4px;
    margin-bottom: 12px;
}
.feature-badges-row {
    text-align: center;
    margin-bottom: 8px;
}

/* ─── Section description text ─── */
.section-desc {
    color: #999;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 16px;
}

/* ─── Input hint ─── */
.input-hint {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 212, 170, 0.08);
    border-left: 3px solid #00d4aa;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
    color: #bbb;
    font-size: 14px;
}

/* ─── Template selector buttons ─── */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    border: 1px solid #333 !important;
    font-size: 13px !important;
}
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    border: 2px solid #00d4aa !important;
    box-shadow: 0 0 12px rgba(0, 212, 170, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# COLORS & HELPERS
# ============================================================================

TEAL = "#00d4aa"
GREEN = "#2ecc71"
AMBER = "#f39c12"
RED = "#e74c3c"


def load_color(u):
    """Color based on how busy a station is."""
    if u < 0.70:
        return GREEN
    elif u < 0.85:
        return AMBER
    return RED


def load_label(u):
    """Plain-language label for station load."""
    if u < 0.50:
        return "Light"
    elif u < 0.70:
        return "Normal"
    elif u < 0.85:
        return "Heavy"
    return "Overloaded"


def status_icon(status):
    if status == "OK":
        return "🟢"
    elif status == "Warning":
        return "🟡"
    return "🔴"


# ============================================================================
# INTERACTIVE FACTORY SIMULATION (HTML/JS)
# ============================================================================

FACTORY_SIM_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0e1117; font-family:Inter,system-ui,sans-serif; overflow:hidden; }
  #controls {
    display:flex; gap:10px; align-items:center; padding:8px 14px;
    background:#1a1f2e; border-radius:8px; margin:8px 8px 6px 8px;
  }
  .sim-btn {
    background:#1a1f2e; color:#00d4aa; border:1px solid #00d4aa;
    border-radius:4px; padding:5px 14px; cursor:pointer; font-size:12px;
    font-family:Inter,system-ui,sans-serif; transition:all .15s;
  }
  .sim-btn:hover { background:#00d4aa; color:#0e1117; }
  .sim-btn.active { background:#00d4aa; color:#0e1117; }
  .sim-select {
    background:#1a1f2e; color:#fafafa; border:1px solid #333;
    border-radius:4px; padding:5px 8px; font-size:12px;
  }
  #clockDisplay { color:#00d4aa; font-family:monospace; font-size:14px; }
  .speed-label { color:#888; font-size:12px; }
  canvas { display:block; margin:0 8px; border-radius:6px; }
  #legend {
    display:flex; gap:18px; padding:6px 14px; margin:0 8px;
    color:#888; font-size:11px; align-items:center;
  }
  .leg-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:4px; vertical-align:middle; }

  /* ── Live Dashboard Panel ── */
  #dashboard {
    margin:8px; padding:0; background:#0e1117;
    font-family:Inter,system-ui,sans-serif;
  }
  .dash-section { margin-bottom:8px; }
  .dash-title {
    font-size:12px; font-weight:700; color:#00d4aa;
    text-transform:uppercase; letter-spacing:1px;
    padding:8px 12px; background:#1a1f2e;
    border-radius:8px 8px 0 0; border-bottom:1px solid #333;
  }
  .dash-grid {
    display:grid; grid-template-columns:repeat(auto-fit, minmax(110px, 1fr));
    gap:1px; background:#1a1f2e; border-radius:0 0 8px 8px; overflow:hidden;
  }
  .dash-card {
    background:#0e1117; padding:8px 10px; text-align:center;
  }
  .dash-card .label { font-size:9px; color:#888; text-transform:uppercase; letter-spacing:0.5px; }
  .dash-card .value { font-size:16px; font-weight:700; color:#00d4aa; margin-top:2px; }
  .dash-card .sub { font-size:9px; color:#666; margin-top:1px; }

  /* Per-station metrics cards */
  .st-metric-card {
    flex:1; min-width:0;
    background:#131720; border:1px solid #1a1f2e; border-radius:6px;
    padding:8px 10px; text-align:center;
  }
  .st-metric-card.bottleneck { border-color:#f39c12; }
  .st-metric-card .sm-name {
    font-size:10px; font-weight:700; color:#fafafa;
    margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .st-metric-card.bottleneck .sm-name { color:#f39c12; }
  .st-metric-card .sm-row {
    display:flex; justify-content:space-between; padding:2px 0;
    font-size:9px;
  }
  .st-metric-card .sm-label { color:#888; text-transform:uppercase; letter-spacing:0.3px; }
  .st-metric-card .sm-val { font-weight:600; color:#ccc; }
  .util-bar-bg {
    width:100%; height:5px; background:#1a1f2e; border-radius:3px; margin-top:2px;
  }
  .util-bar-fg { height:100%; border-radius:3px; transition: width 0.3s; }
  .status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }

  /* Event log */
  #eventLog {
    background:#0e1117; border:1px solid #1a1f2e; border-radius:0 0 8px 8px;
    max-height:80px; overflow-y:auto; padding:4px 8px; font-size:10px;
    font-family:monospace; color:#888;
  }
  #eventLog .evt { padding:1px 0; }
  #eventLog .evt-time { color:#00d4aa; }
  #eventLog .evt-warn { color:#f39c12; }
  #eventLog .evt-err { color:#e74c3c; }
</style>
</head>
<body>

<div id="controls">
  <button class="sim-btn active" id="playBtn" onclick="togglePlay()">Play</button>
  <button class="sim-btn" id="resetBtn" onclick="resetSim()">Reset</button>
  <span class="speed-label">Speed:</span>
  <select class="sim-select" id="speedSel" onchange="setSpeed(this.value)">
    <option value="1" selected>1x</option>
    <option value="2">2x</option>
    <option value="5">5x</option>
    <option value="10">10x</option>
  </select>
  <div style="flex:1"></div>
  <div id="clockDisplay">00:00 / 12:00</div>
</div>

<div id="legend">
  <span><span class="leg-dot" style="background:#2ecc71"></span> Light load</span>
  <span><span class="leg-dot" style="background:#f39c12"></span> Heavy load</span>
  <span><span class="leg-dot" style="background:#e74c3c"></span> Overloaded</span>
  <span style="margin-left:auto; color:#555;">Click station = breakdown &nbsp;|&nbsp; +/- = adjust machines</span>
</div>

<canvas id="simCanvas"></canvas>

<!-- ══════ LIVE DASHBOARD ══════ -->
<div id="dashboard">

  <!-- System Metrics (3 key cards) -->
  <div class="dash-section">
    <div class="dash-title">System Overview</div>
    <div class="dash-grid" id="sysMetrics"></div>
  </div>

  <!-- Per-Station Individual Metrics -->
  <div class="dash-section">
    <div class="dash-title">Station Metrics (Individual)</div>
    <div id="stationMetricsGrid" style="display:flex; gap:4px; padding:6px 8px; background:#0e1117; border-radius:0 0 8px 8px; overflow-x:auto;"></div>
  </div>

  <!-- Utilization History Chart -->
  <div class="dash-section">
    <div class="dash-title">Station Utilization Over Time</div>
    <div style="background:#0e1117; border-radius:0 0 8px 8px; padding:8px;">
      <canvas id="utilChart" style="width:100%; height:180px;"></canvas>
      <div id="utilLegend" style="display:flex; flex-wrap:wrap; gap:8px 16px; padding:6px 4px 2px; font-size:10px;"></div>
    </div>
  </div>

  <!-- Event Log -->
  <div class="dash-section">
    <div class="dash-title">Event Log</div>
    <div id="eventLog"></div>
  </div>

</div>

<script>
// ── CONFIG (injected from Python) ──
const CONFIG = __SIM_CONFIG__;

// ── CONSTANTS ──
const BG = '#0e1117';
const TEAL = '#00d4aa';
const GREEN = '#2ecc71';
const AMBER = '#f39c12';
const RED = '#e74c3c';
const TEXT = '#fafafa';
const DIM = '#888';
const PANEL = '#1a1f2e';

function utilColor(u) { return u >= 0.85 ? RED : u >= 0.70 ? AMBER : GREEN; }
function utilLabel(u) { return u < 0.50 ? 'Light' : u < 0.70 ? 'Normal' : u < 0.85 ? 'Heavy' : 'Overloaded'; }

// ── STATION CLASS ──
class Station {
  constructor(cfg, idx) {
    this.name = cfg.name;
    this.index = idx;
    this.machineCount = cfg.machineCount;
    this.outputQty = cfg.outputQty;
    this.cycleTime = cfg.cycleTime;
    this.scalingFactor = cfg.scalingFactor;
    this.ratePerMachine = (this.outputQty * this.scalingFactor) / this.cycleTime;
    this.queue = [];
    this.totalProcessed = 0;
    this.broken = false;
    this.brokenIdx = -1;
    this.repairTimer = null;
    this.machines = [];
    // Queuing theory reference values from optimization
    this.qtUtilization = cfg.utilization || 0;
    this.qtServiceRate = cfg.serviceRate || 0;
    this.qtArrivalRate = cfg.arrivalRate || 0;
    // Time-averaged utilization tracking
    this._busyTimeSum = 0;   // total busy-machine-minutes
    this._totalTimeSum = 0;  // total active-machine-minutes
    this._avgUtil = 0;
    this._initMachines();
  }
  _initMachines() {
    this.machines = [];
    for (let i = 0; i < this.machineCount; i++)
      this.machines.push({ busy: false, progress: 0, timeLeft: 0, item: null });
  }
  get activeMachines() {
    return this.machines.filter((_, i) => !(this.broken && i === this.brokenIdx)).length;
  }
  get effectiveRate() { return this.ratePerMachine * this.activeMachines; }
  // Instantaneous utilization (for canvas animation visuals)
  get instantUtil() {
    const active = this.activeMachines;
    if (active === 0) return 1;
    let busy = 0;
    this.machines.forEach((m, i) => {
      if (m.busy && !(this.broken && i === this.brokenIdx)) busy++;
    });
    return busy / active;
  }
  // Time-averaged utilization (matches queuing theory ρ = λ/cμ)
  get utilization() {
    return this._avgUtil;
  }
  updateUtilTracking(dt) {
    const active = this.activeMachines;
    if (active === 0) return;
    let busy = 0;
    this.machines.forEach((m, i) => {
      if (m.busy && !(this.broken && i === this.brokenIdx)) busy++;
    });
    this._busyTimeSum += busy * dt;
    this._totalTimeSum += active * dt;
    this._avgUtil = this._totalTimeSum > 0 ? this._busyTimeSum / this._totalTimeSum : 0;
  }
  addMachine() {
    this.machineCount++;
    this.machines.push({ busy: false, progress: 0, timeLeft: 0, item: null });
  }
  removeMachine() {
    if (this.machineCount <= 1) return;
    for (let i = this.machines.length - 1; i >= 0; i--) {
      if (!this.machines[i].busy) { this.machines.splice(i, 1); this.machineCount--; return; }
    }
    this.machines.pop(); this.machineCount--;
  }
  triggerBreakdown() {
    if (this.broken) return;
    this.broken = true;
    const candidates = this.machines.map((_, i) => i).filter(i => i !== this.brokenIdx);
    this.brokenIdx = candidates[Math.floor(Math.random() * candidates.length)];
    if (this.machines[this.brokenIdx]) {
      this.machines[this.brokenIdx].busy = false;
      this.machines[this.brokenIdx].item = null;
    }
    clearTimeout(this.repairTimer);
    this.repairTimer = setTimeout(() => {
      this.broken = false; this.brokenIdx = -1;
      showToast(this.name + ' repaired!');
    }, 5000);
  }
}

// ── SIMULATION ENGINE ──
class SimEngine {
  constructor(config) {
    this.stations = config.stations.map((s, i) => new Station(s, i));
    this.requiredRate = config.requiredRate;
    this.arrivalRate = config.systemCapacity || config.requiredRate;
    this.shiftMin = config.shiftMinutes;
    this.target = config.productionTarget;
    this.simTime = 0;
    this.produced = 0;
    this.running = false;
    this.speed = 1;
    this.arrivalAccum = 0;
    this.itemId = 0;
    this.items = [];
  }
  reset() {
    this.simTime = 0; this.produced = 0; this.arrivalAccum = 0;
    this.itemId = 0; this.items = [];
    this.stations.forEach(s => {
      s.queue = []; s.totalProcessed = 0; s.broken = false; s.brokenIdx = -1;
      s._busyTimeSum = 0; s._totalTimeSum = 0; s._avgUtil = 0;
      clearTimeout(s.repairTimer); s._initMachines();
    });
  }
  tick() {
    if (!this.running) return;
    const dt = (0.5 * this.speed) / 60; // sim-minutes per frame
    this.simTime += dt;
    if (this.simTime >= this.shiftMin) { this.running = false; showToast('Shift complete!'); return; }

    // Arrivals — feed at system capacity (bottleneck rate) so production
    // finishes ahead of schedule, matching Step 5 business impact math.
    // Stop feeding once target is reached (drain remaining items).
    if (this.produced < this.target) {
      this.arrivalAccum += this.arrivalRate * dt;
      while (this.arrivalAccum >= 1) {
        this.arrivalAccum -= 1;
        const item = { id: this.itemId++, si: 0, state: 'q' };
        this.stations[0].queue.push(item);
        this.items.push(item);
      }
    }

    // Process stations (with carry-over so fast machines aren't throttled)
    for (let si = 0; si < this.stations.length; si++) {
      const st = this.stations[si];
      const cyc = 1 / st.ratePerMachine; // time per item per machine
      st.machines.forEach((m, mi) => {
        if (st.broken && mi === st.brokenIdx) { m.busy = false; m.progress = 0; return; }
        if (m.busy) {
          m.timeLeft -= dt;
          if (m.timeLeft <= 0) {
            let leftover = -m.timeLeft; // unused time this tick
            st.totalProcessed++;
            if (m.item) { m.item.state = 'mv'; m.item.targetSi = si + 1; m.item = null; }
            // Process extra items that fit in the leftover time
            while (leftover >= cyc && st.queue.length > 0) {
              const extra = st.queue.shift();
              st.totalProcessed++;
              extra.state = 'mv'; extra.targetSi = si + 1;
              leftover -= cyc;
            }
            // Start next item with remaining leftover carried over
            if (st.queue.length > 0) {
              const next = st.queue.shift();
              m.busy = true; m.timeLeft = cyc - leftover; m.progress = leftover / cyc;
              m.item = next; next.state = 'p'; next.mi = mi;
            } else {
              m.busy = false; m.progress = 0;
            }
          } else {
            m.progress = Math.min(1, 1 - m.timeLeft / cyc);
          }
        }
        if (!m.busy && st.queue.length > 0) {
          const item = st.queue.shift();
          m.busy = true; m.timeLeft = cyc; m.progress = 0;
          m.item = item; item.state = 'p'; item.mi = mi;
        }
      });
    }

    // Update time-averaged utilization for each station
    for (let si = 0; si < this.stations.length; si++) {
      this.stations[si].updateUtilTracking(dt);
    }

    // Advance items
    for (let i = this.items.length - 1; i >= 0; i--) {
      const it = this.items[i];
      if (it.state === 'mv') {
        if (it.targetSi >= this.stations.length) {
          this.produced++; this.items.splice(i, 1);
        } else {
          it.si = it.targetSi; it.state = 'q';
          this.stations[it.targetSi].queue.push(it);
        }
      }
    }

    // Performance: cap items
    if (this.items.length > 600) {
      const excess = this.items.length - 400;
      // Remove oldest queued items, count as produced
      let removed = 0;
      for (let i = 0; i < this.items.length && removed < excess; i++) {
        if (this.items[i].state === 'q') {
          const it = this.items[i];
          const st = this.stations[it.si];
          const qi = st.queue.indexOf(it);
          if (qi >= 0) st.queue.splice(qi, 1);
          this.items.splice(i, 1);
          this.produced++;
          removed++; i--;
        }
      }
    }
  }
  get rate() { return this.simTime > 0 ? this.produced / this.simTime : 0; }
  get bnIdx() {
    let mx = 0, idx = 0;
    this.stations.forEach((s, i) => { const u = s.utilization; if (u > mx) { mx = u; idx = i; } });
    // Use QT bottleneck initially until sim has enough data
    if (this.simTime < 1) {
      let qtMx = 0;
      this.stations.forEach((s, i) => { if (s.qtUtilization > qtMx) { qtMx = s.qtUtilization; idx = i; } });
    }
    return idx;
  }
}

// ── TOASTS ──
const toasts = [];
function showToast(msg) { toasts.push({ msg, t: Date.now() }); }

// ── RENDERER ──
class Renderer {
  constructor(canvas, engine) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.e = engine;
    this.pos = [];
    this.hover = -1;
    this.W = 0; this.H = 0;
    this.resize();
  }
  resize() {
    const dpr = window.devicePixelRatio || 1;
    const parent = this.c.parentElement;
    const w = parent.clientWidth - 16;
    const h = 220;
    this.c.style.width = w + 'px';
    this.c.style.height = h + 'px';
    this.c.width = w * dpr;
    this.c.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = h;
    this._layout();
  }
  _layout() {
    const n = this.e.stations.length;
    const mx = 50, totalW = this.W - 2 * mx;
    const gap = totalW / n;
    const sw = Math.min(gap * 0.6, 110), sh = 120;
    const sy = 90;
    this.pos = [];
    for (let i = 0; i < n; i++) {
      const cx = mx + gap * i + gap / 2;
      this.pos.push({ x: cx - sw / 2, y: sy - sh / 2, w: sw, h: sh, cx, cy: sy });
    }
  }
  roundRect(x, y, w, h, r, fill, alpha) {
    const ctx = this.ctx;
    ctx.globalAlpha = alpha || 1;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.globalAlpha = 1;
  }
  draw() {
    const ctx = this.ctx;
    ctx.fillStyle = BG; ctx.fillRect(0, 0, this.W, this.H);
    this._drawProgressBar();
    this._drawConnections();
    this._drawQueueParticles();
    const bn = this.e.bnIdx;
    this.e.stations.forEach((s, i) => this._drawStation(s, this.pos[i], i === bn));
    this._drawPlusMinus();
    this._drawTooltip();
    this._drawToasts();
  }
  _drawProgressBar() {
    const ctx = this.ctx, pct = Math.min(1, this.e.simTime / this.e.shiftMin);
    ctx.fillStyle = PANEL; ctx.fillRect(0, 0, this.W, 4);
    if (pct > 0) {
      const g = ctx.createLinearGradient(0, 0, this.W * pct, 0);
      g.addColorStop(0, TEAL); g.addColorStop(1, '#00b894');
      ctx.fillStyle = g; ctx.fillRect(0, 0, this.W * pct, 4);
    }
  }
  _drawConnections() {
    const ctx = this.ctx;
    for (let i = 0; i < this.pos.length - 1; i++) {
      const a = this.pos[i], b = this.pos[i + 1];
      const x1 = a.x + a.w + 4, x2 = b.x - 4, y = a.cy;
      ctx.strokeStyle = '#333'; ctx.lineWidth = 2; ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke();
      ctx.setLineDash([]);
      // arrowhead
      ctx.fillStyle = '#555'; ctx.beginPath();
      ctx.moveTo(x2 - 6, y - 4); ctx.lineTo(x2, y); ctx.lineTo(x2 - 6, y + 4); ctx.fill();
    }
  }
  _drawStation(station, p, isBn) {
    const ctx = this.ctx, u = station.utilization, col = utilColor(u);
    // glow
    if (isBn) { ctx.shadowColor = col; ctx.shadowBlur = 12; }
    this.roundRect(p.x, p.y, p.w, p.h, 8, col, 0.85);
    ctx.shadowBlur = 0;
    // border
    ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 1;
    this.roundRect(p.x, p.y, p.w, p.h, 8); ctx.stroke();
    // name
    ctx.fillStyle = TEXT; ctx.font = 'bold 11px Inter,system-ui,sans-serif'; ctx.textAlign = 'center';
    const dname = station.name.length > 12 ? station.name.slice(0, 11) + '..' : station.name;
    ctx.fillText(dname, p.cx, p.y + 20);
    // machine count
    ctx.font = '10px Inter,system-ui,sans-serif';
    ctx.fillText(station.activeMachines + '/' + station.machineCount + ' machines', p.cx, p.y + 35);
    // machine dots
    this._drawMachineDots(station, p);
    // util bar
    const barY = p.y + p.h - 22, barW = p.w - 20, barH = 6;
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    this.roundRect(p.x + 10, barY, barW, barH, 3, 'rgba(0,0,0,0.3)');
    const fillW = Math.min(barW, barW * u);
    if (fillW > 0) this.roundRect(p.x + 10, barY, fillW, barH, 3, col);
    ctx.fillStyle = TEXT; ctx.font = '9px Inter,system-ui,sans-serif';
    ctx.fillText(Math.round(u * 100) + '%', p.cx, barY + barH + 12);
    // queue badge
    const qLen = station.queue.length;
    if (qLen > 0) {
      const bx = p.x + p.w - 2, by = p.y - 2, br = 10;
      ctx.fillStyle = qLen > 10 ? RED : AMBER;
      ctx.beginPath(); ctx.arc(bx, by, br, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = TEXT; ctx.font = 'bold 8px sans-serif'; ctx.textBaseline = 'middle';
      ctx.fillText(qLen > 99 ? '99+' : qLen, bx, by); ctx.textBaseline = 'alphabetic';
    }
    // breakdown overlay
    if (station.broken) {
      const flash = 0.3 + 0.2 * Math.sin(Date.now() / 150);
      this.roundRect(p.x, p.y, p.w, p.h, 8, RED, flash);
      ctx.fillStyle = TEXT; ctx.font = 'bold 12px Inter,system-ui,sans-serif';
      ctx.globalAlpha = 0.6 + 0.4 * Math.sin(Date.now() / 200);
      ctx.fillText('BREAKDOWN', p.cx, p.cy + 2);
      ctx.globalAlpha = 1;
    }
  }
  _drawMachineDots(station, p) {
    const ctx = this.ctx, n = station.machineCount;
    const cols = Math.min(n, 5), rows = Math.ceil(n / cols);
    const dotR = 5, pad = 14;
    const gridW = cols * pad, gridH = rows * pad;
    const ox = p.cx - gridW / 2 + pad / 2, oy = p.y + 46;
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      const dx = ox + c * pad, dy = oy + r * pad;
      const m = station.machines[i];
      if (station.broken && i === station.brokenIdx) {
        ctx.fillStyle = RED; ctx.globalAlpha = 0.5;
        ctx.beginPath(); ctx.arc(dx, dy, dotR, 0, Math.PI * 2); ctx.fill();
        // X mark
        ctx.strokeStyle = TEXT; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.7;
        ctx.beginPath(); ctx.moveTo(dx-3, dy-3); ctx.lineTo(dx+3, dy+3); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(dx+3, dy-3); ctx.lineTo(dx-3, dy+3); ctx.stroke();
        ctx.globalAlpha = 1;
      } else if (m.busy) {
        const pulse = 0.7 + 0.3 * Math.sin(Date.now() / 200 + i);
        ctx.fillStyle = TEXT; ctx.globalAlpha = pulse;
        ctx.beginPath(); ctx.arc(dx, dy, dotR, 0, Math.PI * 2); ctx.fill();
        // progress ring
        ctx.strokeStyle = TEAL; ctx.lineWidth = 2; ctx.globalAlpha = 0.8;
        ctx.beginPath(); ctx.arc(dx, dy, dotR + 2, -Math.PI/2, -Math.PI/2 + Math.PI*2*m.progress); ctx.stroke();
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = '#555'; ctx.globalAlpha = 0.6;
        ctx.beginPath(); ctx.arc(dx, dy, dotR, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
  }
  _drawQueueParticles() {
    const ctx = this.ctx;
    this.e.stations.forEach((st, si) => {
      const p = this.pos[si], maxVis = 25;
      const count = Math.min(st.queue.length, maxVis);
      for (let q = 0; q < count; q++) {
        const x = p.cx + (q % 2 === 0 ? -3 : 3);
        const y = p.y - 8 - q * 4;
        ctx.fillStyle = TEAL; ctx.globalAlpha = 0.7;
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
      }
    });
  }
  _drawPlusMinus() {
    const ctx = this.ctx;
    this.e.stations.forEach((st, i) => {
      const p = this.pos[i];
      const btnY = p.y + p.h + 6, btnS = 18;
      const mx = p.cx - 22, px = p.cx + 4;
      // minus
      this.roundRect(mx, btnY, btnS, btnS, 3, PANEL);
      ctx.strokeStyle = '#555'; ctx.lineWidth = 1; this.roundRect(mx, btnY, btnS, btnS, 3); ctx.stroke();
      ctx.fillStyle = DIM; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('-', mx + btnS/2, btnY + 14);
      // plus
      this.roundRect(px, btnY, btnS, btnS, 3, PANEL);
      ctx.strokeStyle = '#555'; ctx.lineWidth = 1; this.roundRect(px, btnY, btnS, btnS, 3); ctx.stroke();
      ctx.fillStyle = TEAL; ctx.fillText('+', px + btnS/2, btnY + 14);
    });
  }
  _drawHUD() {
    const ctx = this.ctx, hudH = 54, hudY = this.H - hudH;
    ctx.fillStyle = 'rgba(14,17,23,0.92)'; ctx.fillRect(0, hudY, this.W, hudH);
    ctx.fillStyle = TEAL; ctx.fillRect(0, hudY, this.W, 2);
    const metrics = [
      { l: 'Products Produced', v: this.e.produced.toLocaleString() + ' / ' + this.e.target.toLocaleString() },
      { l: 'Production Rate', v: this.e.rate.toFixed(1) + ' prod/min' },
      { l: 'Target Rate', v: this.e.requiredRate.toFixed(1) + ' prod/min' },
      { l: 'Bottleneck', v: this.e.stations[this.e.bnIdx].name },
      { l: 'Progress', v: Math.min(100, (this.e.produced / this.e.target * 100)).toFixed(1) + '%' },
    ];
    const sp = this.W / metrics.length;
    metrics.forEach((m, i) => {
      const x = sp * i + sp / 2;
      ctx.fillStyle = DIM; ctx.font = '9px Inter,system-ui,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(m.l, x, hudY + 20);
      ctx.fillStyle = TEAL; ctx.font = 'bold 13px Inter,system-ui,sans-serif';
      ctx.fillText(m.v, x, hudY + 40);
    });
  }
  _drawTooltip() {
    if (this.hover < 0) return;
    const ctx = this.ctx, st = this.e.stations[this.hover], p = this.pos[this.hover];
    const lines = [
      st.name,
      'Machines: ' + st.activeMachines + '/' + st.machineCount,
      'Rate: ' + st.effectiveRate.toFixed(1) + ' products/min',
      'Queue: ' + st.queue.length + ' items',
      'Processed: ' + st.totalProcessed.toLocaleString(),
      'Utilization: ' + Math.round(st.utilization * 100) + '%',
      st.broken ? 'STATUS: BREAKDOWN' : 'Click to trigger breakdown',
    ];
    const tw = 155, th = lines.length * 16 + 12;
    let tx = p.cx - tw / 2, ty = p.y + p.h + 32;
    if (tx < 4) tx = 4; if (tx + tw > this.W - 4) tx = this.W - tw - 4;
    if (ty + th > this.H - 60) ty = p.y - th - 8;
    this.roundRect(tx, ty, tw, th, 6, 'rgba(26,31,46,0.95)');
    ctx.strokeStyle = TEAL; ctx.lineWidth = 1; this.roundRect(tx, ty, tw, th, 6); ctx.stroke();
    ctx.font = '10px Inter,system-ui,sans-serif'; ctx.textAlign = 'left';
    lines.forEach((ln, j) => {
      ctx.fillStyle = j === 0 ? TEAL : ln.includes('BREAKDOWN') ? RED : '#ccc';
      if (j === 0) ctx.font = 'bold 11px Inter,system-ui,sans-serif';
      else ctx.font = '10px Inter,system-ui,sans-serif';
      ctx.fillText(ln, tx + 8, ty + 15 + j * 16);
    });
  }
  _drawToasts() {
    const ctx = this.ctx, now = Date.now();
    for (let i = toasts.length - 1; i >= 0; i--) {
      const t = toasts[i], age = now - t.t;
      if (age > 3000) { toasts.splice(i, 1); continue; }
      const alpha = age > 2500 ? 1 - (age - 2500) / 500 : 1;
      const y = 14 + i * 28;
      ctx.font = '11px Inter,system-ui,sans-serif'; ctx.textAlign = 'center';
      const tw = ctx.measureText(t.msg).width + 24;
      const tx = this.W / 2 - tw / 2;
      ctx.globalAlpha = alpha * 0.9;
      this.roundRect(tx, y, tw, 22, 4, PANEL);
      ctx.strokeStyle = TEAL; ctx.lineWidth = 1; this.roundRect(tx, y, tw, 22, 4); ctx.stroke();
      ctx.fillStyle = TEXT; ctx.globalAlpha = alpha;
      ctx.fillText(t.msg, this.W / 2, y + 15);
      ctx.globalAlpha = 1;
    }
  }
  fmtTime(min) {
    const h = Math.floor(min / 60), m = Math.floor(min % 60);
    return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0');
  }
}

// ── INIT ──
const canvas = document.getElementById('simCanvas');
const engine = new SimEngine(CONFIG);
const renderer = new Renderer(canvas, engine);
engine.speed = 1;
engine.running = true;

// Clock display
const clockEl = document.getElementById('clockDisplay');
const shiftStr = renderer.fmtTime(engine.shiftMin);

function togglePlay() {
  engine.running = !engine.running;
  document.getElementById('playBtn').textContent = engine.running ? 'Pause' : 'Play';
  document.getElementById('playBtn').classList.toggle('active', engine.running);
}
function resetSim() {
  engine.reset(); engine.running = true;
  utilHistory = []; lastHistoryUpdate = 0;
  utilLegendEl.innerHTML = '';
  document.getElementById('playBtn').textContent = 'Pause';
  document.getElementById('playBtn').classList.add('active');
  showToast('Simulation reset');
}
function setSpeed(v) { engine.speed = parseInt(v); }

// Click handling
canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = renderer.W / rect.width, scaleY = renderer.H / rect.height;
  const mx = (e.clientX - rect.left) * scaleX, my = (e.clientY - rect.top) * scaleY;
  for (let i = 0; i < renderer.pos.length; i++) {
    const p = renderer.pos[i], st = engine.stations[i];
    const btnY = p.y + p.h + 6, btnS = 18;
    const mnX = p.cx - 22, plX = p.cx + 4;
    if (my >= btnY && my <= btnY + btnS) {
      if (mx >= mnX && mx <= mnX + btnS) { st.removeMachine(); showToast(st.name + ': removed machine'); return; }
      if (mx >= plX && mx <= plX + btnS) { st.addMachine(); showToast(st.name + ': added machine'); return; }
    }
    if (mx >= p.x && mx <= p.x + p.w && my >= p.y && my <= p.y + p.h) {
      if (!st.broken) { st.triggerBreakdown(); showToast(st.name + ' breakdown! Repairing...'); }
      return;
    }
  }
});

// Hover
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = renderer.W / rect.width, scaleY = renderer.H / rect.height;
  const mx = (e.clientX - rect.left) * scaleX, my = (e.clientY - rect.top) * scaleY;
  renderer.hover = -1;
  for (let i = 0; i < renderer.pos.length; i++) {
    const p = renderer.pos[i];
    if (mx >= p.x && mx <= p.x + p.w && my >= p.y && my <= p.y + p.h) {
      renderer.hover = i; break;
    }
  }
  canvas.style.cursor = renderer.hover >= 0 ? 'pointer' : 'default';
});
canvas.addEventListener('mouseleave', () => { renderer.hover = -1; });

// Resize
window.addEventListener('resize', () => renderer.resize());

// ── DASHBOARD UPDATE ──
const sysMetricsEl = document.getElementById('sysMetrics');
const stationMetricsEl = document.getElementById('stationMetricsGrid');
const eventLogEl = document.getElementById('eventLog');
const eventEntries = [];
let lastDashUpdate = 0;

// Patch breakdown to log events
const origTriggerBreakdown = Station.prototype.triggerBreakdown;
Station.prototype.triggerBreakdown = function() {
  if (this.broken) return;
  logEvent('err', this.name + ' BREAKDOWN — machine offline, repairing...');
  const self = this;
  const origRepair = this.repairTimer;
  origTriggerBreakdown.call(this);
  // Patch repair callback
  clearTimeout(this.repairTimer);
  this.repairTimer = setTimeout(() => {
    self.broken = false; self.brokenIdx = -1;
    showToast(self.name + ' repaired!');
    logEvent('warn', self.name + ' repaired — back to full capacity');
  }, 5000);
};

function logEvent(type, msg) {
  const t = renderer.fmtTime(engine.simTime);
  eventEntries.push({ type, msg, t });
  if (eventEntries.length > 50) eventEntries.shift();
}

function utilColorCSS(u) { return u >= 0.85 ? RED : u >= 0.70 ? AMBER : GREEN; }

function updateDashboard() {
  const now = Date.now();
  if (now - lastDashUpdate < 500) return; // Throttle to 2 fps
  lastDashUpdate = now;

  const e = engine;
  const totalQueue = e.stations.reduce((s, st) => s + st.queue.length, 0);
  const totalMachines = e.stations.reduce((s, st) => s + st.machineCount, 0);
  const activeMachines = e.stations.reduce((s, st) => s + st.activeMachines, 0);
  const bn = e.stations[e.bnIdx];
  const pct = e.target > 0 ? Math.min(100, e.produced / e.target * 100) : 0;
  const shiftPct = e.shiftMin > 0 ? Math.min(100, e.simTime / e.shiftMin * 100) : 0;
  const avgUtil = e.stations.reduce((s, st) => s + st.utilization, 0) / e.stations.length;
  const maxUtil = Math.max(...e.stations.map(st => st.utilization));
  const avgQT = e.stations.reduce((s, st) => s + st.qtUtilization, 0) / e.stations.length;

  // System metrics — 3 key cards only
  sysMetricsEl.innerHTML =
    dashCard('Produced', e.produced.toLocaleString() + ' / ' + e.target.toLocaleString(), pct.toFixed(1) + '% of target') +
    dashCard('Shift Progress', shiftPct.toFixed(1) + '%', renderer.fmtTime(e.simTime) + ' / ' + shiftStr) +
    dashCard('Bottleneck', bn.name, (bn.utilization * 100).toFixed(0) + '% util');

  // Per-station individual metrics grid
  const bnIdx = e.bnIdx;
  let smHtml = '';
  e.stations.forEach((st, i) => {
    const u = st.utilization;
    const col = utilColorCSS(u);
    const isBn = i === bnIdx;
    const qt = st.qtUtilization;
    const qtCol = utilColorCSS(qt);
    const statusLabel = st.broken ? '<span style="color:' + RED + '">BREAKDOWN</span>'
      : u >= 0.85 ? '<span style="color:' + RED + '">Overloaded</span>'
      : u >= 0.70 ? '<span style="color:' + AMBER + '">Heavy</span>'
      : u >= 0.50 ? '<span style="color:' + GREEN + '">Normal</span>'
      : '<span style="color:#888">Light</span>';
    smHtml += '<div class="st-metric-card ' + (isBn ? 'bottleneck' : '') + '">' +
      '<div class="sm-name">' + (isBn ? '⚠ ' : '') + st.name + '</div>' +
      '<div class="sm-row"><span class="sm-label">Machines</span><span class="sm-val">' + st.activeMachines + '/' + st.machineCount + '</span></div>' +
      '<div class="sm-row"><span class="sm-label">Queue</span><span class="sm-val">' + st.queue.length + '</span></div>' +
      '<div class="sm-row"><span class="sm-label">Live Util</span><span class="sm-val" style="color:' + col + '">' + (u*100).toFixed(1) + '%</span></div>' +
      '<div class="util-bar-bg"><div class="util-bar-fg" style="width:' + Math.min(100, u*100) + '%;background:' + col + '"></div></div>' +
      '<div class="sm-row"><span class="sm-label">Model Util</span><span class="sm-val" style="color:' + qtCol + '">' + (qt*100).toFixed(1) + '%</span></div>' +
      '<div class="util-bar-bg"><div class="util-bar-fg" style="width:' + Math.min(100, qt*100) + '%;background:' + qtCol + '"></div></div>' +
      '<div class="sm-row"><span class="sm-label">Rate</span><span class="sm-val">' + st.effectiveRate.toFixed(1) + ' p/m</span></div>' +
      '<div class="sm-row"><span class="sm-label">Processed</span><span class="sm-val">' + st.totalProcessed.toLocaleString() + '</span></div>' +
      '<div style="text-align:center;font-size:9px;padding-top:3px;font-weight:600;">' + statusLabel + '</div>' +
      '</div>';
  });
  stationMetricsEl.innerHTML = smHtml;

  // Event log
  let logHtml = '';
  for (let i = eventEntries.length - 1; i >= Math.max(0, eventEntries.length - 20); i--) {
    const ev = eventEntries[i];
    const cls = ev.type === 'err' ? 'evt-err' : ev.type === 'warn' ? 'evt-warn' : '';
    logHtml += '<div class="evt"><span class="evt-time">[' + ev.t + ']</span> <span class="' + cls + '">' + ev.msg + '</span></div>';
  }
  eventLogEl.innerHTML = logHtml || '<div class="evt" style="color:#555">No events yet — click a station to trigger a breakdown</div>';
}

function dashCard(label, value, sub) {
  return '<div class="dash-card"><div class="label">' + label + '</div><div class="value">' + value + '</div><div class="sub">' + sub + '</div></div>';
}

// ── UTILIZATION HISTORY CHART ──
const utilChartCanvas = document.getElementById('utilChart');
const utilChartCtx = utilChartCanvas.getContext('2d');
const utilLegendEl = document.getElementById('utilLegend');

// Station colors — distinct palette for up to 8 stations
const stationColors = ['#00d4aa', '#f39c12', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#2ecc71'];

// History storage: array of { time, utils: [u0, u1, ...] }
let utilHistory = [];
const MAX_HISTORY = 300; // max data points
let lastHistoryUpdate = 0;

function recordUtilHistory() {
  const now = Date.now();
  if (now - lastHistoryUpdate < 1000) return; // sample every 1s
  lastHistoryUpdate = now;
  if (engine.simTime <= 0) return;
  const utils = engine.stations.map(st => st.utilization);
  utilHistory.push({ time: engine.simTime, utils });
  if (utilHistory.length > MAX_HISTORY) utilHistory.shift();
}

function drawUtilChart() {
  const c = utilChartCanvas;
  const dpr = window.devicePixelRatio || 1;
  const w = c.parentElement.clientWidth - 16;
  const h = 180;
  c.style.width = w + 'px';
  c.style.height = h + 'px';
  c.width = w * dpr;
  c.height = h * dpr;
  const ctx = utilChartCtx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Background
  ctx.fillStyle = '#0e1117';
  ctx.fillRect(0, 0, w, h);

  const pad = { l: 40, r: 12, t: 8, b: 24 };
  const cw = w - pad.l - pad.r;
  const ch = h - pad.t - pad.b;

  if (utilHistory.length < 2) {
    ctx.fillStyle = '#555';
    ctx.font = '11px Inter,system-ui,sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for data...', w / 2, h / 2);
    return;
  }

  const minT = utilHistory[0].time;
  const maxT = utilHistory[utilHistory.length - 1].time;
  const tRange = maxT - minT || 1;

  // Grid lines and Y axis labels (0%, 25%, 50%, 75%, 100%)
  ctx.strokeStyle = '#1a1f2e';
  ctx.lineWidth = 1;
  ctx.font = '9px Inter,system-ui,sans-serif';
  ctx.textAlign = 'right';
  for (let pct = 0; pct <= 100; pct += 25) {
    const y = pad.t + ch - (pct / 100) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.fillText(pct + '%', pad.l - 4, y + 3);
  }

  // Danger zone (85% line)
  const dangerY = pad.t + ch - (0.85) * ch;
  ctx.strokeStyle = 'rgba(231, 76, 60, 0.3)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, dangerY);
  ctx.lineTo(w - pad.r, dangerY);
  ctx.stroke();
  ctx.setLineDash([]);

  // X axis time labels
  ctx.textAlign = 'center';
  ctx.fillStyle = '#666';
  const nLabels = Math.min(6, Math.floor(cw / 60));
  for (let i = 0; i <= nLabels; i++) {
    const t = minT + (tRange * i / nLabels);
    const x = pad.l + (cw * i / nLabels);
    const hrs = Math.floor(t / 60);
    const mins = Math.floor(t % 60);
    ctx.fillText((hrs < 10 ? '0' : '') + hrs + ':' + (mins < 10 ? '0' : '') + mins, x, h - 4);
  }

  // Draw lines for each station
  const n = engine.stations.length;
  for (let si = 0; si < n; si++) {
    const col = stationColors[si % stationColors.length];
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    let started = false;
    for (let di = 0; di < utilHistory.length; di++) {
      const d = utilHistory[di];
      const x = pad.l + ((d.time - minT) / tRange) * cw;
      const y = pad.t + ch - Math.min(1, d.utils[si]) * ch;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Current values as dots at the right edge
  const last = utilHistory[utilHistory.length - 1];
  for (let si = 0; si < n; si++) {
    const col = stationColors[si % stationColors.length];
    const x = pad.l + cw;
    const y = pad.t + ch - Math.min(1, last.utils[si]) * ch;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateUtilLegend() {
  if (utilLegendEl.children.length > 0) return; // only build once
  engine.stations.forEach((st, i) => {
    const col = stationColors[i % stationColors.length];
    utilLegendEl.innerHTML += '<span style="color:' + col + '">● ' + st.name + '</span>';
  });
}

// Log initial event
logEvent('info', 'Simulation started — ' + engine.stations.length + ' stations, target: ' + engine.target.toLocaleString() + ' products');

// Track production milestones
let lastMilestone = 0;
function checkMilestones() {
  const pct = engine.target > 0 ? Math.floor(engine.produced / engine.target * 100) : 0;
  const milestone = Math.floor(pct / 25) * 25;
  if (milestone > lastMilestone && milestone > 0) {
    lastMilestone = milestone;
    logEvent('info', 'Production milestone: ' + milestone + '% of target reached (' + engine.produced.toLocaleString() + ' products)');
  }
}

// Animation loop
function animate() {
  engine.tick();
  renderer.draw();
  clockEl.textContent = renderer.fmtTime(engine.simTime) + ' / ' + shiftStr;
  updateDashboard();
  recordUtilHistory();
  drawUtilChart();
  updateUtilLegend();
  checkMilestones();
  requestAnimationFrame(animate);
}
animate();
</script>
</body>
</html>
"""


# ============================================================================
# FACTORY FLOW DIAGRAM
# ============================================================================

def render_factory_flow(names, machine_counts, station_metrics=None):
    """Render horizontal production pipeline."""
    fig = go.Figure()

    n = len(names)
    box_w = 0.9
    gap = 2.2
    box_h = 1.0
    yc = 0.5

    for i in range(n):
        xc = i * gap
        machines = machine_counts[i]

        if station_metrics:
            u = station_metrics[i]['utilization']
            color = load_color(u)
            label = load_label(u)
        else:
            color = TEAL
            label = ""

        # Station box
        fig.add_shape(
            type="rect",
            x0=xc - box_w / 2, y0=yc - box_h / 2,
            x1=xc + box_w / 2, y1=yc + box_h / 2,
            fillcolor=color, opacity=0.85,
            line=dict(color="white", width=1.5),
        )

        # Name above
        display_name = names[i][:10]  # Truncate long names
        fig.add_annotation(
            x=xc, y=yc + box_h / 2 + 0.25,
            text=f"<b>{display_name}</b>",
            showarrow=False, font=dict(size=10, color="white"),
        )

        # Machine count inside
        fig.add_annotation(
            x=xc, y=yc + 0.12,
            text=f"<b>{machines}</b> machine{'s' if machines != 1 else ''}",
            showarrow=False, font=dict(size=12, color="white"),
        )

        # Load label inside
        if label:
            fig.add_annotation(
                x=xc, y=yc - 0.18,
                text=label,
                showarrow=False, font=dict(size=10, color="white"),
            )

        # Arrow to next
        if i < n - 1:
            ax = xc + box_w / 2 + 0.05
            bx = (i + 1) * gap - box_w / 2 - 0.05
            fig.add_annotation(
                x=bx, y=yc, ax=ax, ay=yc,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=2, arrowcolor="#888",
            )

    fig.update_layout(
        height=230, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-0.8, (n - 1) * gap + 0.8]),
        yaxis=dict(visible=False, range=[-0.3, 1.5]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================================
# METRICS HELPER
# ============================================================================

def solution_metrics(sol, rr, recipes, scalings=None):
    sc = scalings or scaling_factors
    qm = calculate_station_queue_metrics(sol, recipes, sc, rr)
    total_Lq = sum(m['Lq'] for m in qm)
    max_Wq = max(m['Wq'] for m in qm)
    throughput = min(m['output_rate'] for m in qm)
    utils = [m['utilization'] for m in qm]
    bn_idx = utils.index(max(utils))
    return {
        'qm': qm, 'total_Lq': total_Lq, 'max_Wq': max_Wq,
        'throughput': throughput, 'utils': utils,
        'bn_idx': bn_idx, 'bn_util': max(utils),
    }


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🏭 OptiFlow")
    st.caption("Smart production line optimization")
    st.divider()

    if 'result' in st.session_state:
        sol_s = st.session_state['result']['results'][0]['solution']
        ms = st.session_state.get('metrics')
        cm_s = st.session_state.get('current_machines', [1] * len(sol_s))
        additional_s = sum(sol_s) - sum(cm_s)
        st.markdown("**📊 Results Summary**")
        st.metric("Machines You Have", sum(cm_s))
        st.metric("Machines Recommended", sum(sol_s))
        if additional_s > 0:
            st.metric("Machines to Buy", f"+{additional_s}")
        if ms:
            st.metric("Busiest Station",
                      f"{st.session_state.get('stage_names', station_names)[ms['bn_idx']]} ({ms['bn_util']*100:.0f}%)")
        if 'mc' in st.session_state:
            st.metric("Reliability Score",
                      f"{st.session_state['mc']['success_rate']:.0f}%")
        st.divider()

    st.markdown("**What OptiFlow Does**")
    st.markdown(
        "- **Smart Optimization** — Finds the best machine setup for your line\n"
        "- **Performance Modeling** — Analyzes queues, wait times & throughput\n"
        "- **Reliability Testing** — Stress-tests your setup with real-world variability"
    )
    st.divider()

    st.markdown("**📋 Your Workflow**")
    st.markdown(
        "1. Configure your production line\n"
        "2. Run the optimizer\n"
        "3. Review recommended setup\n"
        "4. Test reliability & risk\n"
        "5. Analyze business impact\n"
        "6. Watch the live simulation"
    )


# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>🏭 OptiFlow</h1>"
    "<p class='hero-subtitle'>"
    "Optimize your production line in minutes</p>"
    "<div class='feature-badges-row'>"
    "<span class='feature-badge'>Smart Optimization</span>"
    "<span class='feature-badge'>Performance Modeling</span>"
    "<span class='feature-badge'>Reliability Testing</span>"
    "<span class='feature-badge'>Live Simulation</span>"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================================
# HOW OPTIFLOW WORKS
# ============================================================================

with st.expander("How OptiFlow Works", expanded=False):
    algo_c1, algo_c2, algo_c3 = st.columns(3)
    with algo_c1:
        st.markdown("#### Smart Optimization")
        st.markdown(
            "OptiFlow tests **thousands of possible machine configurations** to find "
            "the one that best balances cost, speed, and efficiency. It learns from "
            "each test to focus on the most promising setups, converging on the "
            "optimal allocation for your line."
        )
    with algo_c2:
        st.markdown("#### Performance Modeling")
        st.markdown(
            "Each production stage is modeled as a real queue system. OptiFlow "
            "calculates realistic metrics like **average queue length**, "
            "**waiting time**, and **station utilization** to evaluate how well "
            "each configuration actually performs under load."
        )
    with algo_c3:
        st.markdown("#### Reliability Testing")
        st.markdown(
            "Once the best setup is found, OptiFlow stress-tests it by simulating "
            "**thousands of production days** with random machine breakdowns and "
            "speed variations. This gives you a **reliability score** showing how "
            "often the setup meets your target under real-world conditions."
        )


# ============================================================================
# SECTION 1: YOUR PRODUCTION LINE
# ============================================================================

st.markdown("---")
st.markdown("## Step 1: Configure Your Production Line")

# ── Industry Template Selector ──
st.markdown("##### Choose an Industry Template")
st.markdown(
    '<div class="input-hint">'
    'Pick a template to pre-fill your production line, or start from scratch. '
    'You can customize everything after selecting.'
    '</div>',
    unsafe_allow_html=True,
)

template_keys = list(INDUSTRY_TEMPLATES.keys())
# Two rows of 3 for better readability
row1_keys = template_keys[:3]
row2_keys = template_keys[3:]
for row_keys in [row1_keys, row2_keys]:
    tcols = st.columns(len(row_keys))
    for i, key in enumerate(row_keys):
        tmpl_info = INDUSTRY_TEMPLATES[key]
        with tcols[i]:
            is_selected = st.session_state.get('selected_template', 'food_manufacturing') == key
            if st.button(
                f"{tmpl_info['icon']} {tmpl_info['label']}",
                key=f"tmpl_{key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state['selected_template'] = key
                # Clear data editor state so it reloads with new template
                for k in list(st.session_state.keys()):
                    if 'station_editor' in k:
                        del st.session_state[k]
                st.rerun()

active_template_key = st.session_state.get('selected_template', 'food_manufacturing')
active_template = INDUSTRY_TEMPLATES[active_template_key]

st.markdown(
    f'<div class="input-hint">'
    f'Using <strong>{active_template["icon"]} {active_template["label"]}</strong> template — '
    f'{active_template["description"]}. '
    f'<strong>Edit the fields below</strong> to match your factory.'
    f'</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([0.35, 0.3, 0.35])
with c1:
    st.markdown("##### 📦 Production Target")
    production_target = st.number_input(
        "Units needed per shift",
        value=active_template['default_target'], min_value=100, step=500,
    )
    st.caption("Your total production target for one shift")
with c2:
    st.markdown("##### ⏱️ Shift Duration")
    shift_hours = st.number_input(
        "Hours per shift",
        value=active_template['default_shift_hours'], min_value=1, max_value=24,
    )
    st.caption("How many hours is one production shift? (1–24 hours)")
with c3:
    required_rate = production_target / (shift_hours * 60)
    st.markdown("##### 📊 Required Rate")
    st.metric("Output needed", f"{required_rate:.1f} u/min")
    st.caption("Auto-calculated from your target and shift duration")

st.markdown("##### ⚙️ Your Production Stages")
st.markdown(
    '<div class="input-hint">'
    '✏️ Click any cell to edit. Use the <strong>+ button</strong> below the table to add stages, '
    'or select a row and press <strong>Delete</strong> to remove it.'
    '</div>',
    unsafe_allow_html=True,
)
tc1, tc2, tc3, tc4 = st.columns(4)
tc1.caption("**Stage Name** — Name of each production stage (e.g., Mixer, Oven)")
tc2.caption("**Output Per Cycle** — Units produced per machine cycle")
tc3.caption("**Cycle Time** — Minutes for one cycle to complete")
tc4.caption("**Current Machines** — Machines you currently own at this stage")

default_data = []
for stage in active_template['stages']:
    default_data.append({
        "Stage Name": stage["name"],
        "Output Per Cycle": stage["output_per_cycle"],
        "Cycle Time (min)": stage["cycle_time"],
        "Current Machines": stage.get("machines", 1),
    })

station_df = pd.DataFrame(default_data)
edited_df = st.data_editor(
    station_df,
    use_container_width=True,
    key="station_editor",
    hide_index=True,
    num_rows="dynamic",
    column_config={
        "Stage Name": st.column_config.TextColumn(
            "Stage Name", width="medium",
            help="Name of this production stage (e.g., Mixer, Oven, Packager)"),
        "Output Per Cycle": st.column_config.NumberColumn(
            "Output Per Cycle", min_value=1,
            help="How many units this machine produces per cycle"),
        "Cycle Time (min)": st.column_config.NumberColumn(
            "Cycle Time (min)", min_value=1,
            help="How long one cycle takes in minutes"),
        "Current Machines": st.column_config.NumberColumn(
            "Current Machines", min_value=1, max_value=20,
            help="How many machines you currently have at this stage (optimizer will not go below this)"),
    },
)

# Build recipes from user input
n_stages = len(edited_df)
edited_recipes = {}
current_machines = []
user_stage_names = []
user_scalings = []

for idx in range(n_stages):
    row = edited_df.iloc[idx]
    name = str(row["Stage Name"]) if pd.notna(row["Stage Name"]) else f"Stage {idx+1}"
    user_stage_names.append(name)
    edited_recipes[idx] = {
        "output_qty": int(row["Output Per Cycle"]),
        "time": int(row["Cycle Time (min)"]),
        "machine_type": name.lower(),
        "queue_type": "FIFO",
    }
    cm_val = row["Current Machines"] if pd.notna(row.get("Current Machines")) else 1
    current_machines.append(int(cm_val))
    # Use scaling factor from template, else 1
    if idx < len(active_template['stages']):
        user_scalings.append(active_template['stages'][idx].get('scaling', 1))
    else:
        user_scalings.append(1)

st.markdown("**Your Current Production Flow**")
st.plotly_chart(
    render_factory_flow(user_stage_names, current_machines),
    use_container_width=True,
    config={'displayModeBar': False},
)


# ============================================================================
# SECTION 2: OPTIMIZE
# ============================================================================

st.markdown("---")
st.markdown("## Step 2: Optimize")
st.markdown(
    "OptiFlow will test thousands of machine configurations and evaluate "
    "queue performance at each stage. It finds the setup that hits your "
    "target with minimal waiting and maximum efficiency."
)

with st.expander("⚙️ Optimizer Settings (optional)", expanded=False):
    st.markdown(
        '<div class="input-hint">'
        '🔧 Adjust the optimizer settings. '
        'Default values work well for most cases.'
        '</div>',
        unsafe_allow_html=True,
    )
    a1, a2, a3 = st.columns(3)
    with a1:
        pop_size = st.slider("Configurations to Test", 20, 200, 50, step=10)
        st.caption("How many setups to evaluate each round")
    with a2:
        max_gen = st.slider("Optimization Rounds", 100, 2000, 1000, step=100)
        st.caption("More rounds may find better solutions")
    with a3:
        seed = st.number_input("Random Seed", value=1, min_value=0)
        st.caption("Same seed = same results (for reproducibility)")

optimize_btn = st.button(
    "🚀 Run Optimizer — Find the Best Setup",
    type="primary",
    use_container_width=True,
)

if optimize_btn:
    # Live visualization placeholders
    progress_bar = st.progress(0, text="Initializing optimizer...")
    live_metrics_ph = st.empty()
    chart_placeholder = st.empty()
    belief_chart_ph = st.empty()
    best_sol_ph = st.empty()

    fitness_history = []
    mean_history = []
    std_history = []
    gen_history = []
    result_data = None

    for msg in cultural_algorithm(
        edited_recipes, user_scalings, required_rate,
        pop_size=pop_size, max_gen=max_gen, seed=seed,
        use_queuing=True, min_machines=current_machines,
    ):
        if msg['type'] == 'progress':
            gen = msg['generation']
            pct = gen / msg['max_gen']
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Optimizing... Round {gen}/{msg['max_gen']}"
            )
            fitness_history.append(msg['best_fitness'])
            mean_history.append(msg['pop_fitness_mean'])
            std_history.append(msg['pop_fitness_std'])
            gen_history.append(gen)

            if gen % 10 == 0 or msg.get('early_stop'):
                # ── Live metric cards ──
                with live_metrics_ph.container():
                    lm1, lm2, lm3, lm4 = st.columns(4)
                    lm1.metric("Round", f"{gen}/{msg['max_gen']}")
                    lm2.metric("Best Score", f"{msg['best_fitness']:.2f}")
                    lm3.metric("Search Diversity", f"±{msg['pop_fitness_std']:.2f}")
                    lm4.metric("No-Improvement Streak", f"{msg['stagnation']}/50",
                               delta="converging" if msg['stagnation'] > 30 else "exploring",
                               delta_color="inverse" if msg['stagnation'] > 30 else "normal")

                # ── Convergence + Population spread chart ──
                fig_live = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Optimization Progress", "Configuration Explorer"),
                    horizontal_spacing=0.1,
                )
                # Best fitness line
                fig_live.add_trace(go.Scatter(
                    x=gen_history, y=fitness_history,
                    mode='lines', line=dict(color=TEAL, width=2),
                    fill='tozeroy', fillcolor='rgba(0,212,170,0.1)',
                    name='Best Score',
                ), row=1, col=1)
                # Mean score line
                fig_live.add_trace(go.Scatter(
                    x=gen_history, y=mean_history,
                    mode='lines', line=dict(color=AMBER, width=1, dash='dot'),
                    name='Average Score',
                ), row=1, col=1)

                # Per-station population spread (box-like: mean ± std)
                station_means = msg['pop_station_mean']
                station_stds = msg['pop_station_std']
                belief_lower = msg['belief_lower']
                belief_upper = msg['belief_upper']
                snames_short = [n[:8] for n in user_stage_names]

                fig_live.add_trace(go.Bar(
                    x=snames_short, y=station_means,
                    error_y=dict(type='data', array=station_stds, visible=True,
                                 color='rgba(0,212,170,0.5)'),
                    marker_color=TEAL, opacity=0.7,
                    name='Avg. Machines ± Range',
                ), row=1, col=2)
                # Belief space bounds as scatter markers
                fig_live.add_trace(go.Scatter(
                    x=snames_short, y=belief_lower,
                    mode='markers', marker=dict(color=GREEN, size=10, symbol='triangle-up'),
                    name='Search Min',
                ), row=1, col=2)
                fig_live.add_trace(go.Scatter(
                    x=snames_short, y=belief_upper,
                    mode='markers', marker=dict(color=RED, size=10, symbol='triangle-down'),
                    name='Search Max',
                ), row=1, col=2)

                fig_live.update_layout(
                    height=320,
                    margin=dict(l=40, r=20, t=35, b=40),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.25, x=0.5, xanchor='center',
                                font=dict(size=10)),
                )
                fig_live.update_yaxes(title_text="Score (lower = better)", row=1, col=1)
                fig_live.update_yaxes(title_text="Machines", row=1, col=2)
                chart_placeholder.plotly_chart(
                    fig_live, use_container_width=True,
                    config={'displayModeBar': False},
                    key=f"live_chart_{gen}",
                )

                # ── Search range visualization (advanced view) ──
                with belief_chart_ph.container():
                    with st.expander("Advanced View", expanded=False):
                        fig_belief = go.Figure()
                        for i, sn in enumerate(user_stage_names):
                            short = sn[:8]
                            fig_belief.add_trace(go.Bar(
                                x=[short], y=[belief_upper[i] - belief_lower[i]],
                                base=[belief_lower[i]],
                                marker_color=TEAL, opacity=0.4,
                                showlegend=(i == 0),
                                name='Search Range',
                            ))
                            fig_belief.add_trace(go.Scatter(
                                x=[short], y=[msg['best_solution'][i]],
                                mode='markers',
                                marker=dict(color='white', size=12, symbol='diamond',
                                            line=dict(color=TEAL, width=2)),
                                showlegend=(i == 0),
                                name='Current Best',
                            ))
                        fig_belief.update_layout(
                            title="Search Range — How the optimizer narrows its focus",
                            yaxis_title="Machine Count",
                            height=250,
                            margin=dict(l=40, r=20, t=40, b=40),
                            barmode='overlay',
                            legend=dict(orientation='h', yanchor='bottom', y=-0.3, x=0.5, xanchor='center'),
                        )
                        st.plotly_chart(
                            fig_belief, use_container_width=True,
                            config={'displayModeBar': False},
                            key=f"belief_chart_{gen}",
                        )

                # ── Current best solution table ──
                with best_sol_ph.container():
                    st.markdown("**Current Best Solution**")
                    best_cols = st.columns(n_stages)
                    for i, sn in enumerate(user_stage_names):
                        with best_cols[i]:
                            st.markdown(
                                f"<div style='text-align:center;padding:6px;background:#1a1f2e;"
                                f"border-radius:8px;border:1px solid #333;'>"
                                f"<div style='font-size:11px;color:#888;'>{sn[:10]}</div>"
                                f"<div style='font-size:22px;font-weight:700;color:#00d4aa;'>"
                                f"{msg['best_solution'][i]}</div>"
                                f"<div style='font-size:10px;color:#666;'>machines</div></div>",
                                unsafe_allow_html=True,
                            )

        elif msg['type'] == 'result':
            result_data = msg

    progress_bar.progress(1.0, text="Done! Scroll down to see your results.")

    if result_data:
        best = result_data['results'][0]
        st.session_state['result'] = result_data
        st.session_state['recipes'] = edited_recipes
        st.session_state['required_rate'] = required_rate
        st.session_state['shift_hours'] = shift_hours
        st.session_state['production_target'] = production_target
        st.session_state['stage_names'] = user_stage_names
        st.session_state['scalings'] = user_scalings
        st.session_state['n_stages'] = n_stages
        st.session_state['current_machines'] = current_machines

        m = solution_metrics(best['solution'], required_rate, edited_recipes, user_scalings)
        st.session_state['metrics'] = m

        st.success(
            f"Optimization complete in **{result_data['elapsed']:.1f} seconds** — "
            f"Optimal allocation: **{best['machines']} machines** across {len(edited_recipes)} stages."
        )


# ============================================================================
# SECTIONS 3-5: RESULTS (only shown after optimization)
# ============================================================================

if 'result' in st.session_state:
    result = st.session_state['result']
    rr = st.session_state['required_rate']
    recipes_used = st.session_state['recipes']
    best = result['results'][0]
    sol = best['solution']
    m = st.session_state['metrics']
    snames = st.session_state.get('stage_names', station_names)
    scalings_used = st.session_state.get('scalings', scaling_factors)
    ns = st.session_state.get('n_stages', 6)
    current_mach = st.session_state.get('current_machines', [1] * ns)

    # ==================================================================
    # SECTION 3: YOUR RECOMMENDED SETUP
    # ==================================================================

    st.markdown("---")
    st.markdown("## Step 3: Recommended Setup")
    st.markdown("Here is the optimal machine allocation for your production line. "
                "Performance metrics below show queue behavior and utilization at each stage.")

    # Hero metrics — plain language
    additional_total = sum(sol) - sum(current_mach)
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Machines You Have", sum(current_mach))
    h2.metric("Machines Recommended", sum(sol),
              delta=f"+{additional_total} to buy" if additional_total > 0 else "No change needed")
    h3.metric("Production Speed", f"{m['throughput']:.1f} units/min",
              delta=f"{'Meets' if m['throughput'] >= rr else 'Below'} your target")
    h4.metric("Busiest Stage", snames[m['bn_idx']])
    headroom = (1 - m['bn_util']) * 100
    h5.metric("Spare Capacity", f"{headroom:.0f}%",
              delta="Room to grow" if headroom > 15 else "Very tight")

    # Flow diagram with load colors
    st.markdown("**Recommended Production Flow**")
    st.markdown("🟢 Green = Running smoothly &nbsp;&nbsp; "
                "🟡 Yellow = Getting busy &nbsp;&nbsp; "
                "🔴 Red = Needs attention")
    st.plotly_chart(
        render_factory_flow(snames, sol, m['qm']),
        use_container_width=True,
        config={'displayModeBar': False},
    )

    # Station detail table — plain language
    detail_data = []
    for i in range(ns):
        qm = m['qm'][i]
        u = qm['utilization']
        additional = sol[i] - current_mach[i]
        detail_data.append({
            "Stage": snames[i],
            "Current": current_mach[i],
            "Recommended": sol[i],
            "To Buy": f"+{additional}" if additional > 0 else "—",
            "How Busy": f"{u*100:.0f}%",
            "Status": load_label(u),
            "Avg. Wait": f"{qm['Wq']*60:.1f} sec" if qm['Wq'] < 1 else f"{qm['Wq']:.1f} min",
        })
    st.dataframe(
        pd.DataFrame(detail_data),
        use_container_width=True, hide_index=True,
    )

    # Machine allocation chart — current vs recommended
    colors = [load_color(m['utils'][i]) for i in range(ns)]
    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Bar(
        x=snames, y=current_mach,
        marker_color='rgba(255,255,255,0.3)',
        text=[f"{c}" for c in current_mach], textposition='outside',
        name='Current',
    ))
    fig_alloc.add_trace(go.Bar(
        x=snames, y=sol,
        marker_color=colors,
        text=[f"{s}" for s in sol], textposition='outside',
        name='Recommended',
    ))
    fig_alloc.update_layout(
        title="Machines Per Stage: Current vs Recommended",
        yaxis_title="Number of Machines",
        height=360, barmode='group',
    )
    st.plotly_chart(fig_alloc, use_container_width=True, config={'displayModeBar': False})


    # ==================================================================
    # SECTION 4: STRESS TEST & RELIABILITY
    # ==================================================================

    st.markdown("---")
    st.markdown("## Step 4: Reliability & Risk Analysis")
    st.markdown("Real factories deal with machine breakdowns, speed variations, and demand changes. "
                "Let's test if this setup can handle the real world.")

    # --- 4A: Reliability ---
    st.markdown("### Reliability Test")
    st.markdown("We simulate thousands of production days with random "
                "machine breakdowns (5% failure rate) and speed variations (±10%) to see how "
                "often this setup meets your target.")

    mc_col1, mc_col2 = st.columns([0.3, 0.7])
    with mc_col1:
        n_sims = st.slider("Number of simulated days", 1000, 10000, 10000,
                          step=1000, key="mc_sims")
        st.caption("More days = more reliable results but takes longer")
        mc_btn = st.button("🎲 Run Reliability Test")

    if mc_btn:
        with st.spinner(f"Simulating {n_sims:,} production days..."):
            mc = run_monte_carlo(sol, n_simulations=n_sims, required_rate=rr,
                                recipes=recipes_used, scalings=scalings_used, names=snames)
        st.session_state['mc'] = mc

    if 'mc' in st.session_state:
        mc = st.session_state['mc']

        r1, r2, r3 = st.columns(3)
        r1.metric("Days You Hit Target",
                  f"{mc['success_rate']:.0f}%",
                  delta=f"out of {mc['n_simulations']:,} simulated days")
        r2.metric("Average Daily Output",
                  f"{mc['mean']:,.0f} units")
        r3.metric("Worst Day Output",
                  f"{mc['min']:,.0f} units",
                  delta="still above target" if mc['min'] >= mc['target'] else "below target",
                  delta_color="normal" if mc['min'] >= mc['target'] else "inverse")

        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=mc['productions'], nbinsx=50,
            marker_color=TEAL, opacity=0.8,
        ))
        fig_hist.add_vline(x=mc['target'], line_dash="dash", line_color=AMBER,
                           annotation_text=f"Your Target: {mc['target']:,.0f}")
        fig_hist.update_layout(
            title="How Much You Produce Each Day (simulated)",
            xaxis_title="Units Produced", yaxis_title="Number of Days",
            height=380, showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

        # Bottleneck frequency
        st.markdown("**Which stage slows you down most often?**")
        bn = mc['bottleneck_counts']
        bn_items = [(name, bn.get(name, 0) / mc['n_simulations'] * 100)
                    for name in snames if bn.get(name, 0) > 0]
        if bn_items:
            bn_items.sort(key=lambda x: -x[1])
            fig_bn = go.Figure(go.Bar(
                x=[b[0] for b in bn_items],
                y=[b[1] for b in bn_items],
                marker_color=TEAL,
                text=[f"{b[1]:.0f}%" for b in bn_items],
                textposition='outside',
            ))
            fig_bn.update_layout(
                yaxis_title="% of days as bottleneck",
                height=300, showlegend=False,
            )
            st.plotly_chart(fig_bn, use_container_width=True, config={'displayModeBar': False})

    # --- 4B: Demand Flexibility ---
    st.markdown("### Can You Handle More Orders?")
    st.markdown("What happens if demand goes up 10% or 20%? Will your line keep up?")

    stress = run_stress_test(sol, rr, recipes_used, scalings_used, names=snames)

    cols = st.columns(5)
    for i, s in enumerate(stress):
        with cols[i]:
            pct = s['bottleneck_util'] * 100
            st.markdown(
                f"**{s['scenario']} demand**  \n"
                f"{status_icon(s['status'])} **{s['status']}**  \n"
                f"Busiest: {s['bottleneck_station']}  \n"
                f"at {pct:.0f}% capacity"
            )

    # Headroom message
    max_ok = 0
    for s in stress:
        if s['status'] == 'OK':
            max_ok = max(max_ok, (s['demand_factor'] - 1) * 100)
    if max_ok > 0:
        st.success(f"Your line can handle up to **+{max_ok:.0f}% more orders** "
                   f"before any stage gets overloaded.")
    else:
        st.warning("Your line is running near full capacity. "
                   "Consider adding machines if you expect demand to grow.")

    # Stress chart
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(
        x=[s['scenario'] for s in stress],
        y=[s['bottleneck_util'] * 100 for s in stress],
        mode='lines+markers',
        line=dict(color=TEAL, width=3),
        marker=dict(size=10, color=[load_color(s['bottleneck_util']) for s in stress]),
    ))
    fig_stress.add_hline(y=85, line_dash="dash", line_color=AMBER,
                         annotation_text="Danger Zone (85%)")
    fig_stress.add_hline(y=100, line_dash="solid", line_color=RED,
                         annotation_text="Maximum Capacity")
    fig_stress.update_layout(
        title="How Busy Is Your Busiest Stage at Different Demand Levels?",
        xaxis_title="Demand Change", yaxis_title="Busiest Stage Load (%)",
        yaxis_range=[0, max(115, max(s['bottleneck_util'] * 100 for s in stress) + 15)],
        height=380, showlegend=False,
    )
    st.plotly_chart(fig_stress, use_container_width=True, config={'displayModeBar': False})


    # ==================================================================
    # SECTION 5: BUSINESS IMPACT
    # ==================================================================

    st.markdown("---")
    st.markdown("## Step 5: Business Impact")

    shift_h = st.session_state.get('shift_hours', 12)
    target_units = rr * shift_h * 60
    system_cap = min(
        (recipes_used[i]['output_qty'] / recipes_used[i]['time'])
        * scalings_used[i] * sol[i]
        for i in range(ns)
    )
    hours_to_target = (target_units / system_cap) / 60 if system_cap > 0 else shift_h
    buffer_hours = shift_h - hours_to_target
    total_capacity = system_cap * shift_h * 60

    additional_buy = sum(sol) - sum(current_mach)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric(
        "You'll Finish In",
        f"{hours_to_target:.1f} hours",
        delta=f"{buffer_hours:.1f} hours to spare",
    )
    e2.metric(
        "Maximum Daily Output",
        f"{total_capacity:,.0f} units",
        delta=f"+{(total_capacity / target_units - 1) * 100:.0f}% above your target",
    )
    e3.metric("Equipment You Have", f"{sum(current_mach)} machines")
    e4.metric(
        "Additional to Buy",
        f"{additional_buy} machine{'s' if additional_buy != 1 else ''}",
        delta="No purchase needed" if additional_buy == 0 else f"+{additional_buy} new",
    )

    # Timeline
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Bar(
        y=['Your Line'], x=[hours_to_target], orientation='h',
        marker_color=TEAL, text=[f"Production: {hours_to_target:.1f} hrs"],
        textposition='inside', textfont=dict(size=14, color='white'),
    ))
    fig_tl.add_trace(go.Bar(
        y=['Your Line'], x=[buffer_hours], orientation='h',
        marker_color='rgba(0,212,170,0.25)',
        text=[f"Buffer: {buffer_hours:.1f} hrs"],
        textposition='inside', textfont=dict(size=14, color='white'),
    ))
    fig_tl.update_layout(
        barmode='stack',
        title=f"Your {shift_h}-Hour Shift",
        xaxis_title="Hours", xaxis_range=[0, shift_h + 0.5],
        height=160, showlegend=False,
        margin=dict(l=20, r=20, t=40, b=40),
    )
    fig_tl.add_vline(x=shift_h, line_dash="dash", line_color="white",
                     annotation_text="Shift Ends")
    st.plotly_chart(fig_tl, use_container_width=True, config={'displayModeBar': False})

    # Cumulative curve
    hours = np.linspace(0, shift_h, 200)
    cumulative = system_cap * hours * 60

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=hours, y=cumulative, mode='lines',
        line=dict(color=TEAL, width=3),
        fill='tozeroy', fillcolor='rgba(0,212,170,0.1)',
    ))
    fig_cum.add_hline(y=target_units, line_dash="dash", line_color=AMBER,
                      annotation_text=f"Target: {target_units:,.0f} units")
    fig_cum.update_layout(
        title="Production Throughout the Day",
        xaxis_title="Hours Into Shift", yaxis_title="Total Units Produced",
        height=380,
    )
    st.plotly_chart(fig_cum, use_container_width=True, config={'displayModeBar': False})

    # Download
    st.markdown("---")
    st.markdown("### Export Your Recommended Setup")
    report = []
    for i in range(ns):
        qm = m['qm'][i]
        additional = sol[i] - current_mach[i]
        report.append({
            "Stage": snames[i],
            "Current Machines": current_mach[i],
            "Recommended Machines": sol[i],
            "Additional Needed": additional,
            "How Busy (%)": round(qm['utilization'] * 100, 1),
            "Status": load_label(qm['utilization']),
            "Output Speed (units/min)": round(qm['output_rate'], 2),
        })
    csv = pd.DataFrame(report).to_csv(index=False)
    st.download_button(
        "Download Report (CSV)",
        data=csv,
        file_name="recommended_setup.csv",
        mime="text/csv",
        use_container_width=True,
    )


    # ==================================================================
    # SECTION 6: LIVE FACTORY SIMULATION
    # ==================================================================

    st.markdown("---")
    st.markdown("## Step 6: Watch Your Factory in Action")
    st.markdown(
        "See your optimized production line running in real time. "
        "Items flow through each stage, queues build up at bottlenecks, "
        "and you can experiment with changes on the fly."
    )

    # System capacity = bottleneck station rate (matches Step 5 math)
    system_cap = min(
        (recipes_used[i]['output_qty'] / recipes_used[i]['time'])
        * scalings_used[i] * sol[i]
        for i in range(ns)
    )
    # Feed at 95% of system capacity so bottleneck stays under 100%
    # This still finishes well before the shift ends, matching Step 5
    sim_arrival = system_cap * 0.95
    sim_config = {
        "stations": [],
        "requiredRate": float(rr),
        "systemCapacity": float(sim_arrival),
        "shiftMinutes": int(st.session_state.get('shift_hours', 12) * 60),
        "productionTarget": int(rr * st.session_state.get('shift_hours', 12) * 60),
    }
    for i in range(ns):
        qm = m['qm'][i]
        recipe = recipes_used[i]
        sim_config["stations"].append({
            "name": snames[i],
            "machineCount": int(sol[i]),
            "outputQty": int(recipe['output_qty']),
            "cycleTime": float(recipe['time']),
            "scalingFactor": float(scalings_used[i]),
            "utilization": float(qm['utilization']),
            "serviceRate": float(qm['service_rate']),
            "arrivalRate": float(qm['arrival_rate']),
        })

    sim_json = json.dumps(sim_config)
    factory_html = FACTORY_SIM_HTML.replace("__SIM_CONFIG__", sim_json)
    st.components.v1.html(factory_html, height=1150, scrolling=False)


else:
    st.markdown("---")
    st.info("👆 Configure your production line above, then click **Run Optimizer** to get started.")
