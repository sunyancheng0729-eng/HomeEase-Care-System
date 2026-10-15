/* 跌倒检测系统 · 控制台前端
   - ① 传感器实时流：WebSocket 逐样本推入，服务端实时检测并回推状态/报警
   - ② 模拟数据输入：整段离线检测
   - ③ 跌倒记录与报告
------------------------------------------------------------------------- */
const $ = id => document.getElementById(id);
const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/stream';

const S = {
  threshold: 0.5,
  model: null,
  sequences: null,
  ws: null,
  wsOpen: false,
  replaying: false,
  raf: null,
  replay: null,        // {t:[], a:[[]]}
  speed: 5,
  live: [],            // 实时波形样本 {t,ax,ay,az,mag}
  curve: [],           // 窗口概率 {t,p}
  sessionFalls: [],
  nFall: 0,
};

/* ---------------- 基础工具 ---------------- */
async function api(url, opt) {
  const r = await fetch(url, opt);
  if (!r.ok) {
    let msg = r.status + ' ' + r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.headers.get('content-type')?.includes('json') ? r.json() : r.text();
}
function toast(msg, bad) {
  const d = document.createElement('div');
  if (bad) d.className = 'bad';
  d.textContent = msg;
  $('toast').appendChild(d);
  setTimeout(() => d.remove(), 4200);
}
const fmt = (v, n = 3) => (v === null || v === undefined) ? '—' : Number(v).toFixed(n);

/* ---------------- 启动 ---------------- */
(async function init() {
  try {
    const h = await api('/api/health');
    S.model = h.algorithm;
    S.threshold = h.threshold ?? 0.5;
    $('chipAlg').textContent = h.algorithm || '—';
    $('chipThr').textContent = fmt(S.threshold, 2);
    $('chipModel').textContent = h.config
      ? `${h.config.resample_fs | 0}Hz·butter${h.config.butter_cutoff | 0}·${h.config.win_s}s/${h.config.step_s}s`
      : '—';
    $('liveThr').value = S.threshold; $('simThr').value = S.threshold;
    if (h.status !== 'ok') toast('模型未加载：' + (h.error || ''), true);
    if (!h.data_available) toast('未找到 URFD 数据目录，模拟数据模式仅能上传 CSV', true);
    $('chipCount').textContent = h.db.total;
  } catch (e) { toast('无法连接后端：' + e.message, true); }

  try {
    const sq = await api('/api/sequences');
    S.sequences = sq;
    const mk = kind => sq[kind].map(s => `<option value="${s.seq}">${s.seq} · ${s.hint}</option>`).join('');
    $('liveSeq').innerHTML = mk('fall') + mk('adl');
    $('simSeq').innerHTML = mk('fall') + mk('adl');
    $('liveSeq').value = sq.fall[0]?.seq || '';
    $('simSeq').value = sq.fall[0]?.seq || '';
  } catch (e) { /* 忽略 */ }

  bindTabs(); bindLive(); bindSim(); bindRec();
  drawEmpty();
  loadRecords();
  window.addEventListener('resize', () => { redrawLive(); redrawSim(); });
})();

function drawEmpty() {
  drawLiveWave($('cvWave'), [], 0, 8);
  drawLiveProb($('cvProb'), [], S.threshold);
  drawProbChart($('cvSimProb'), { t: [], p: [] }, S.threshold);
}

/* ---------------- 标签页 ---------------- */
function bindTabs() {
  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
    document.querySelectorAll('.panel').forEach(p =>
      p.classList.toggle('active', p.id === 'panel-' + t.dataset.tab));
    if (t.dataset.tab === 'rec') loadRecords();
    redrawLive(); redrawSim();
  });
}

/* ==================== ① 传感器实时流 ==================== */
function bindLive() {
  $('btnStart').onclick = startReplay;
  $('btnStop').onclick = () => stopReplay(true);
  $('btnReset').onclick = () => {
    if (S.wsOpen) S.ws.send(JSON.stringify({ type: 'reset' }));
    S.curve = []; S.sessionFalls = []; S.live = []; S.nFall = 0;
    renderSession(); redrawLive();
    setStat($('stT'), 0, ' s'); setStat($('stP'), '—'); setStat($('stN'), 0);
    setStat($('stFall'), 0); setState('空闲', '');
  };
  $('liveThr').onchange = e => {
    const v = Math.min(1, Math.max(0, parseFloat(e.target.value) || 0));
    $('liveThr').value = v; S.threshold = v; $('chipThr').textContent = fmt(v, 2);
    if (S.wsOpen) S.ws.send(JSON.stringify({ type: 'config', threshold: v }));
    redrawLive();
  };
}

function setStat(el, v, unit) {
  el.innerHTML = v + (unit ? `<small>${unit}</small>` : '');
}
function setState(txt, cls) {
  $('stState').textContent = txt;
  $('stStateBox').className = 'stat' + (cls ? ' ' + cls : '');
}

function connectWS() {
  return new Promise((resolve, reject) => {
    if (S.wsOpen) return resolve();
    const ws = new WebSocket(WS_URL);
    S.ws = ws;
    ws.onopen = () => { S.wsOpen = true; $('liveConn').textContent = '已连接 /ws/stream'; resolve(); };
    ws.onerror = () => { if (!S.wsOpen) reject(new Error('WebSocket 连接失败')); };
    ws.onclose = () => { S.wsOpen = false; $('liveConn').textContent = '未连接';
      if (S.replaying) stopReplay(false); };
    ws.onmessage = ev => {
      const m = JSON.parse(ev.data);
      if (m.type === 'status') onStatus(m);
      else if (m.type === 'alarm') onAlarm(m);
      else if (m.type === 'fall') onFall(m);
      else if (m.type === 'flushed') { if (m.curve?.length) pushCurve(m.curve); renderSession(); }
      else if (m.type === 'error') toast(m.message, true);
    };
  });
}

function onStatus(m) {
  if (m.curve && m.curve.length) pushCurve(m.curve);
  setStat($('stT'), fmt(m.t_stream, 2), ' s');
  if (m.last_prob !== null && m.last_prob !== undefined) {
    setStat($('stP'), fmt(m.last_prob, 3));
    $('stProbBox').className = 'stat' + (m.last_prob >= (m.threshold ?? S.threshold) ? ' hot' : '');
  }
  setStat($('stN'), m.n_eval);
  setStat($('stFall'), m.n_fall ?? S.nFall);
  setState(m.alarming ? '⚠ 报警中' : '正常监测', m.alarming ? 'hot' : 'good');
  redrawLive();
}

function pushCurve(tail) {
  for (const r of tail) {
    if (!S.curve.length || r.t > S.curve[S.curve.length - 1].t) S.curve.push(r);
  }
  if (S.curve.length > 4000) S.curve = S.curve.slice(-4000);
}

/* 命中即刻报警（事件尚未闭合，记录稍后落库） */
function onAlarm(m) {
  setState('⚠ 报警中', 'hot');
  S.sessionFalls.unshift({
    pending: true, id: null,
    stream_alarm: m.t_stream, score: m.score, threshold: m.threshold,
    t_alarm: m.t_alarm, n_windows: 1, snippet: null,
  });
  renderSession();
  toast(`⚠ 检测到跌倒！流内时刻 ${fmt(m.t_stream, 2)}s · 概率 ${fmt(m.score, 3)}（正在确认并生成记录…）`, true);
}

function onFall(m) {
  S.nFall = m.n_fall ?? (S.nFall + 1);
  setStat($('stFall'), S.nFall);
  const e = m.event;
  const rec = { id: m.id, ...e, snippet: m.snippet, pending: false };
  // 用落库记录替换对应的「报警中」占位
  const k = S.sessionFalls.findIndex(f => f.pending && Math.abs(f.stream_alarm - e.stream_alarm) < 1.2);
  if (k >= 0) S.sessionFalls[k] = rec; else S.sessionFalls.unshift(rec);
  renderSession();
  toast(`跌倒记录 #${m.id} 已生成 · 报警时刻 ${fmt(e.stream_alarm, 2)}s · 概率 ${fmt(e.score, 3)}`, true);
  loadRecords();
}

function renderSession() {
  const box = $('liveList');
  $('liveCount').textContent = S.sessionFalls.length;
  if (!S.sessionFalls.length) { box.innerHTML = '<div class="empty">暂无检出</div>'; return; }
  box.innerHTML = S.sessionFalls.map(f => f.pending ? `
    <div class="item" style="border-left-color:var(--orange)">
      <div class="t" style="color:var(--orange)">⚠ 报警中… 流内 ${fmt(f.stream_alarm, 2)} s</div>
      <div class="m">
        <span>概率 <span class="score">${fmt(f.score, 3)}</span> / 阈值 ${fmt(f.threshold, 2)}</span>
        <span>正在确认并生成记录</span>
      </div>
    </div>` : `
    <div class="item" onclick="openReport(${f.id})">
      <div class="t">#${f.id} · 报警时刻 ${fmt(f.stream_alarm, 2)} s</div>
      <div class="m">
        <span>概率 <span class="score">${fmt(f.score, 3)}</span> / 阈值 ${fmt(f.threshold, 2)}</span>
        <span>命中窗 ${f.n_windows}</span>
        ${f.truncated ? '<span>流结束收尾</span>' : ''}
      </div>
    </div>`).join('');
}

async function startReplay() {
  const seq = $('liveSeq').value;
  if (!seq) return toast('请选择序列', true);
  S.speed = parseFloat($('liveSpeed').value) || 1;
  try {
    await connectWS();
    const d = await api('/api/sequence/' + encodeURIComponent(seq));
    S.replay = d.raw;
    S.replaying = true;
    S.curve = []; S.sessionFalls = []; S.live = []; S.nFall = 0;
    renderSession(); redrawLive();
    setStat($('stT'), 0, ' s'); setStat($('stP'), '—'); setStat($('stN'), 0);
    setStat($('stFall'), 0); setState('推流中', 'good');
    S.ws.send(JSON.stringify({ type: 'config', threshold: S.threshold }));
    S.ws.send(JSON.stringify({ type: 'reset' }));
    S.ws.send(JSON.stringify({ type: 'meta', seq }));
    $('btnStart').disabled = true; $('btnStop').disabled = false;
    toast(`开始推流：${seq}（${d.duration_s}s，${S.speed}× 倍速，${d.n} 个样本）`);
    const t0 = performance.now();
    const raw = S.replay;
    let i = 0;
    const step = () => {
      if (!S.replaying) return;
      const el = (performance.now() - t0) / 1000 * S.speed;
      while (i < raw.t.length && (raw.t[i] - raw.t[0]) <= el) {
        const t = raw.t[i], a = raw.a[i];
        S.ws.send(JSON.stringify({ type: 'sample', t, ax: a[0], ay: a[1], az: a[2] }));
        pushLive(t - raw.t[0], a);
        i++;
      }
      if (i >= raw.t.length) { finishReplay(); return; }
      S.raf = requestAnimationFrame(step);
    };
    step();
  } catch (e) { toast('推流失败：' + e.message, true); S.replaying = false; }
}

function pushLive(t, a) {
  S.live.push({ t, ax: a[0], ay: a[1], az: a[2], mag: Math.hypot(a[0], a[1], a[2]) });
  if (S.live.length > 3000) S.live = S.live.slice(-3000);
}

function finishReplay() {
  if (S.wsOpen) S.ws.send(JSON.stringify({ type: 'flush' }));
  stopReplay(false);
  setState('推流结束', 'good');
  toast(`推流结束：共检出 ${S.sessionFalls.length} 次跌倒，已全部存入数据库`);
}

function stopReplay(userAction) {
  S.replaying = false;
  if (S.raf) cancelAnimationFrame(S.raf), S.raf = null;
  $('btnStart').disabled = false; $('btnStop').disabled = true;
  if (!S.sessionFalls.length) setState('空闲', '');
  if (userAction) toast('已停止推流');
}

function redrawLive() {
  if (!document.getElementById('cvWave')) return;
  const tEnd = S.live.length ? S.live[S.live.length - 1].t : 0;
  const span = Math.max(6, Math.min(20, tEnd));
  drawLiveWave($('cvWave'), S.live, tEnd, span, { height: 150 });
  const tail = S.curve.slice(-120);
  drawLiveProb($('cvProb'), tail, S.threshold, { height: 130 });
}

/* ==================== ② 模拟数据输入 ==================== */
function bindSim() {
  $('btnSim').onclick = runSim;
  $('btnUpload').onclick = uploadSim;
  $('simThr').onchange = e => {
    const v = Math.min(1, Math.max(0, parseFloat(e.target.value) || 0));
    $('simThr').value = v;
  };
}

function applySimResult(r, title) {
  $('simTag').textContent = title;
  setStat($('simDur'), fmt(r.duration_s, 2), ' s');
  setStat($('simWin'), r.n_windows);
  setStat($('simHit'), r.n_hits);
  setStat($('simEv'), r.n_events);
  drawProbChart($('cvSimProb'), r.curve || { t: [], p: [] }, r.threshold, { height: 190 });
  const box = $('simList');
  if (!r.events.length) {
    box.innerHTML = '<div class="empty">未检出跌倒事件（全部窗口概率低于阈值）</div>';
    toast('检测完成：未检出跌倒', false);
    return;
  }
  box.innerHTML = r.events.map(e => `
    <div class="ev" onclick="openReport(${e.id})">
      <div class="h">
        <b>报警时刻 ${fmt(e.stream_alarm, 2)} s</b>
        <span class="badge alarm"><i class="dotled"></i>加速度报警</span>
      </div>
      <div class="m" style="margin-top:6px;font-size:12px;color:var(--dim);display:flex;gap:14px;flex-wrap:wrap">
        <span>记录 #${e.id}</span>
        <span>报警窗概率 <b style="color:var(--orange)">${fmt(e.score, 3)}</b></span>
        <span>最大概率 ${fmt(e.score_max, 3)}</span>
        <span>命中窗 ${e.n_windows}</span>
      </div>
    </div>`).join('');
  toast(`检测完成：检出 ${r.n_events} 次跌倒，已存入数据库`);
  loadRecords();
}

async function runSim() {
  const seq = $('simSeq').value;
  if (!seq) return toast('请选择序列', true);
  $('btnSim').disabled = true;
  try {
    const r = await api('/api/simulate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seq, threshold: parseFloat($('simThr').value) })
    });
    applySimResult(r, seq);
  } catch (e) { toast('检测失败：' + e.message, true); }
  finally { $('btnSim').disabled = false; }
}

async function uploadSim() {
  const f = $('simFile').files[0];
  if (!f) return toast('请先选择 CSV 文件', true);
  $('btnUpload').disabled = true;
  const fd = new FormData();
  fd.append('file', f);
  fd.append('threshold', parseFloat($('simThr').value));
  try {
    const r = await api('/api/simulate/upload', { method: 'POST', body: fd });
    applySimResult(r, f.name);
  } catch (e) { toast('上传检测失败：' + e.message, true); }
  finally { $('btnUpload').disabled = false; }
}

function redrawSim() { /* 概率图在结果到达时一次性绘制 */ }

/* ==================== ③ 跌倒记录 ==================== */
function bindRec() {
  $('btnRefresh').onclick = loadRecords;
  $('recSrc').onchange = loadRecords;
  $('btnClear').onclick = async () => {
    if (!confirm('确定清空全部跌倒记录？此操作不可恢复。')) return;
    try {
      const r = await api('/api/falls', { method: 'DELETE' });
      toast(`已清空 ${r.deleted} 条记录`); loadRecords();
    } catch (e) { toast('清空失败：' + e.message, true); }
  };
}

async function loadRecords() {
  const src = $('recSrc') ? $('recSrc').value : '';
  try {
    const r = await api('/api/falls?limit=300' + (src ? '&source=' + src : ''));
    $('recCount').textContent = `共 ${r.items.length} 条`;
    $('chipCount').textContent = r.items.length;
    const body = $('recBody');
    if (!r.items.length) {
      body.innerHTML = '<tr><td colspan="8" class="empty">暂无记录</td></tr>';
      return;
    }
    body.innerHTML = r.items.map(f => `
      <tr>
        <td>#${f.id}</td>
        <td class="mono">${f.alarm_time}</td>
        <td><span class="src ${f.source}">${f.source === 'sensor' ? '传感器流' : '模拟数据'}</span></td>
        <td>${f.seq || '—'}</td>
        <td style="color:var(--orange);font-weight:600">${fmt(f.score, 3)}</td>
        <td>${fmt(f.threshold, 2)}</td>
        <td>${f.n_windows ?? '—'}</td>
        <td>
          <button class="sm" onclick="openReport(${f.id})">报告</button>
          <button class="sm ghost" onclick="downloadReport(${f.id})">下载</button>
          <button class="sm danger" onclick="delRecord(${f.id})">删除</button>
        </td>
      </tr>`).join('');
  } catch (e) { toast('加载记录失败：' + e.message, true); }
}

function openReport(id) { window.open('/report/' + id, '_blank'); }
function downloadReport(id) { window.open('/api/falls/' + id + '/report.html', '_blank'); }
async function delRecord(id) {
  if (!confirm(`删除记录 #${id}？`)) return;
  try { await api('/api/falls/' + id, { method: 'DELETE' }); toast(`已删除 #${id}`); loadRecords(); }
  catch (e) { toast('删除失败：' + e.message, true); }
}
