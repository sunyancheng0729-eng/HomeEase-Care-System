/* 轻量 Canvas 图表（无外部依赖，离线可用） ----------------------------------
   三个图：三维加速度曲线 / 概率曲线 / 实时滚动波形
------------------------------------------------------------------------- */
const C_ = {
  ax: '#4aa8ff', ay: '#31d0aa', az: '#ffa53b', mag: '#a78bfa',
  red: '#ff4d5e', orange: '#ffa53b', blue: '#4aa8ff',
  grid: '#222c44', axis: '#1c2438', txt: '#8f9bb5', bg: '#0b101b'
};

function fit(cv, h) {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || 600;
  const H = h || cv.clientHeight || 260;
  cv.style.height = H + 'px';
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  return { g, W, H };
}

function axes(g, pad, W, H, yLo, yHi, xLo, xHi, yFmt, xFmt, ny = 4, nx = 6) {
  g.font = '11px system-ui'; g.lineWidth = 1;
  g.fillStyle = C_.txt; g.textAlign = 'right'; g.textBaseline = 'middle';
  for (let i = 0; i <= ny; i++) {
    const v = yLo + (yHi - yLo) * i / ny;
    const y = pad.t + (1 - i / ny) * (H - pad.t - pad.b);
    g.strokeStyle = C_.grid; g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillText(yFmt ? yFmt(v) : v.toFixed(2), pad.l - 8, y);
  }
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (let i = 0; i <= nx; i++) {
    const v = xLo + (xHi - xLo) * i / nx;
    const x = pad.l + (i / nx) * (W - pad.l - pad.r);
    g.strokeStyle = C_.axis; g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, H - pad.b); g.stroke();
    g.fillStyle = C_.txt; g.fillText(xFmt ? xFmt(v) : v.toFixed(1), x, H - pad.b + 7);
  }
  return {
    X: v => pad.l + ((v - xLo) / (xHi - xLo || 1)) * (W - pad.l - pad.r),
    Y: v => pad.t + (1 - (v - yLo) / (yHi - yLo || 1)) * (H - pad.t - pad.b),
    pw: W - pad.l - pad.r, ph: H - pad.t - pad.b
  };
}

function rangeOf(arrays, extra = 0.12) {
  let lo = Infinity, hi = -Infinity;
  for (const s of arrays) for (const v of s) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) { lo = -1; hi = 1; }
  const p = (hi - lo) * extra + 0.02;
  return [lo - p, hi + p];
}

function poly(g, T, S, X, Y, color, w) {
  g.strokeStyle = color; g.lineWidth = w; g.beginPath();
  for (let i = 0; i < T.length; i++) { const x = X(T[i]), y = Y(S[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
  g.stroke();
}

/* ---- 三维加速度曲线（报告 / 模拟结果）：t 为相对报警时刻的秒 ---- */
function drawAccelChart(cv, D, opts = {}) {
  const H = opts.height || 280;
  const { g, W } = fit(cv, H);
  const pad = { l: 54, r: 16, t: 16, b: 34 };
  if (!D || !D.t || !D.t.length) {
    g.fillStyle = C_.txt; g.font = '13px system-ui';
    g.fillText('无加速度数据', pad.l, 40); return;
  }
  const [lo, hi] = rangeOf([D.ax, D.ay, D.az, D.mag || []], .14);
  const a = axes(g, pad, W, H, lo, hi, D.t[0], D.t[D.t.length - 1], v => v.toFixed(2), v => v.toFixed(1) + 's');
  // 命中窗口底纹
  if (opts.winStart != null && opts.winEnd != null) {
    const x1 = a.X(Math.max(opts.winStart, D.t[0])), x2 = a.X(Math.min(opts.winEnd, D.t[D.t.length - 1]));
    g.fillStyle = 'rgba(255,77,94,.10)'; g.fillRect(x1, pad.t, Math.max(1, x2 - x1), a.ph);
    g.strokeStyle = 'rgba(255,77,94,.45)'; g.lineWidth = 1;
    g.strokeRect(x1, pad.t, Math.max(1, x2 - x1), a.ph);
  }
  poly(g, D.t, D.ax, a.X, a.Y, C_.ax, 1.4);
  poly(g, D.t, D.ay, a.X, a.Y, C_.ay, 1.4);
  poly(g, D.t, D.az, a.X, a.Y, C_.az, 1.4);
  if (D.mag) poly(g, D.t, D.mag, a.X, a.Y, C_.mag, 2.2);
  // 报警时刻
  const xa = a.X(0);
  if (xa >= pad.l && xa <= W - pad.r) {
    g.save(); g.setLineDash([5, 4]); g.strokeStyle = C_.red; g.lineWidth = 2;
    g.beginPath(); g.moveTo(xa, pad.t); g.lineTo(xa, pad.t + a.ph); g.stroke(); g.restore();
    g.fillStyle = C_.red; g.font = 'bold 11px system-ui'; g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText('报警时刻 t=0', xa + 6, pad.t + 4);
  }
}

/* ---- 概率曲线（模拟结果）：x 为窗结束时刻（秒） ---- */
function drawProbChart(cv, curve, threshold, opts = {}) {
  const H = opts.height || 170;
  const { g, W } = fit(cv, H);
  const pad = { l: 54, r: 16, t: 16, b: 34 };
  const T = curve && curve.t ? curve.t : [];
  const P = curve && curve.p ? curve.p : [];
  if (!T.length) { g.fillStyle = C_.txt; g.font = '13px system-ui'; g.fillText('无窗口数据', pad.l, 40); return; }
  const a = axes(g, pad, W, H, 0, 1, T[0], T[T.length - 1], v => v.toFixed(1), v => v.toFixed(1) + 's', 4, 6);
  // 命中柱
  for (let i = 0; i < T.length; i++) {
    const bw = Math.max(2, (a.pw / T.length) * 0.72);
    g.fillStyle = P[i] >= threshold ? 'rgba(255,77,94,.85)' : 'rgba(74,168,255,.30)';
    const x = a.X(T[i]) - bw / 2, y = a.Y(P[i]);
    g.fillRect(x, y, bw, a.Y(0) - y);
  }
  poly(g, T, P, a.X, a.Y, C_.blue, 1.8);
  // 阈值线
  g.strokeStyle = C_.orange; g.lineWidth = 1.6; g.save(); g.setLineDash([6, 4]);
  g.beginPath(); g.moveTo(pad.l, a.Y(threshold)); g.lineTo(W - pad.r, a.Y(threshold)); g.stroke(); g.restore();
  g.fillStyle = C_.orange; g.font = '11px system-ui'; g.textAlign = 'left'; g.textBaseline = 'bottom';
  g.fillText('报警阈值 ' + Number(threshold).toFixed(2), pad.l + 4, a.Y(threshold) - 3);
}

/* ---- 实时波形：滚动窗口 ---- */
function drawLiveWave(cv, samples, tEnd, span, opts = {}) {
  const H = opts.height || 150;
  const { g, W } = fit(cv, H);
  const pad = { l: 54, r: 14, t: 12, b: 28 };
  const t0 = tEnd - span;
  const vis = samples.filter(s => s.t >= t0);
  if (!vis.length) {
    g.fillStyle = C_.txt; g.font = '12px system-ui';
    g.fillText('等待传感器数据…', pad.l, 40); return;
  }
  const [lo, hi] = rangeOf([vis.map(s => s.mag)], .18);
  const a = axes(g, pad, W, H, lo, hi, t0, tEnd, v => v.toFixed(1), v => v.toFixed(1) + 's', 3, 6);
  poly(g, vis.map(s => s.t), vis.map(s => s.mag), a.X, a.Y, C_.mag, 2);
  poly(g, vis.map(s => s.t), vis.map(s => s.ax), a.X, a.Y, C_.ax, 1);
  poly(g, vis.map(s => s.t), vis.map(s => s.ay), a.X, a.Y, C_.ay, 1);
  poly(g, vis.map(s => s.t), vis.map(s => s.az), a.X, a.Y, C_.az, 1);
}

/* ---- 实时概率曲线（滑动） ---- */
function drawLiveProb(cv, hist, threshold, opts = {}) {
  const H = opts.height || 130;
  const { g, W } = fit(cv, H);
  const pad = { l: 54, r: 14, t: 12, b: 28 };
  if (!hist.length) {
    g.fillStyle = C_.txt; g.font = '12px system-ui';
    g.fillText('等待窗口评估…', pad.l, 34); return;
  }
  const t0 = hist[0].t, t1 = Math.max(hist[hist.length - 1].t, t0 + 1);
  const a = axes(g, pad, W, H, 0, 1, t0, t1, v => v.toFixed(1), v => v.toFixed(1) + 's', 2, 5);
  const bw = Math.max(2, (a.pw / hist.length) * 0.7);
  for (const h of hist) {
    g.fillStyle = h.p >= threshold ? 'rgba(255,77,94,.85)' : 'rgba(74,168,255,.32)';
    const x = a.X(h.t) - bw / 2, y = a.Y(h.p);
    g.fillRect(x, y, bw, a.Y(0) - y);
  }
  poly(g, hist.map(h => h.t), hist.map(h => h.p), a.X, a.Y, C_.blue, 1.6);
  g.strokeStyle = C_.orange; g.lineWidth = 1.5; g.save(); g.setLineDash([6, 4]);
  g.beginPath(); g.moveTo(pad.l, a.Y(threshold)); g.lineTo(W - pad.r, a.Y(threshold)); g.stroke(); g.restore();
  g.fillStyle = C_.orange; g.font = '11px system-ui'; g.textAlign = 'left'; g.textBaseline = 'bottom';
  g.fillText('阈值 ' + Number(threshold).toFixed(2), pad.l + 4, a.Y(threshold) - 3);
}
