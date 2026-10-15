/* ==========================================================================
   实时跌倒检测监控系统 —— 前端逻辑
   ========================================================================== */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

let curType = 'camera';
let audioCtx = null;
let lastRecordId = null;
let prevAlarm = false;
let demoMap = {};

/* ------------------------------- 工具 ------------------------------- */
function toast(msg, isErr = false, ms = 2600) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast on' + (isErr ? ' err' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.className = 'toast'; }, ms);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = 'HTTP ' + r.status;
    try {
      const j = await r.json();
      msg = j.detail || JSON.stringify(j);
    } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

function post(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  });
}

function beep() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const t0 = audioCtx.currentTime;
    [0, 0.28].forEach((off) => {
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.type = 'square';
      o.frequency.value = 880;
      g.gain.setValueAtTime(0.0001, t0 + off);
      g.gain.exponentialRampToValueAtTime(0.09, t0 + off + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + off + 0.22);
      o.connect(g); g.connect(audioCtx.destination);
      o.start(t0 + off); o.stop(t0 + off + 0.24);
    });
  } catch (e) { /* 浏览器可能不允许自动播放，忽略 */ }
}

/* ------------------------------- 初始化 ------------------------------- */
async function loadSources() {
  const d = await api('/api/sources');
  demoMap = d.demo || {};
  const dsSel = $('#demoDataset');
  dsSel.innerHTML = '';
  Object.keys(demoMap).forEach((k) => {
    const o = document.createElement('option');
    o.value = k; o.textContent = k;
    dsSel.appendChild(o);
  });
  fillSeqs();
}

function fillSeqs() {
  const ds = $('#demoDataset').value;
  const seqSel = $('#demoSeq');
  seqSel.innerHTML = '';
  (demoMap[ds] || []).forEach((s) => {
    const o = document.createElement('option');
    o.value = s; o.textContent = s;
    seqSel.appendChild(o);
  });
}

function bindUI() {
  $$('#segType button').forEach((b) => {
    b.addEventListener('click', () => {
      $$('#segType button').forEach((x) => x.classList.remove('on'));
      b.classList.add('on');
      curType = b.dataset.type;
      $('#paneCamera').classList.toggle('hidden', curType !== 'camera');
      $('#paneFile').classList.toggle('hidden', curType !== 'file');
      $('#paneDemo').classList.toggle('hidden', curType !== 'demo');
    });
  });

  $('#demoDataset').addEventListener('change', fillSeqs);

  $('#btnPick').addEventListener('click', () => $('#fileInput').click());
  $('#fileInput').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $('#uploadHint').textContent = `正在上传 ${f.name} …`;
    try {
      const fd = new FormData();
      fd.append('file', f);
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      if (!r.ok) throw new Error('上传失败');
      const d = await r.json();
      $('#filePath').value = d.path;
      $('#uploadHint').textContent = `已上传：${d.name}（${(d.size / 1048576).toFixed(1)} MB），可直接开始检测。`;
      toast('视频上传完成');
    } catch (err) {
      $('#uploadHint').textContent = '上传失败：' + err.message;
      toast('上传失败：' + err.message, true);
    }
  });

  $('#btnStart').addEventListener('click', startDetect);
  $('#btnStop').addEventListener('click', stopDetect);

  // 参数
  const thr = $('#thr'), cd = $('#cd'), pre = $('#pre');
  thr.addEventListener('input', () => { $('#thrOut').textContent = (+thr.value).toFixed(2); });
  thr.addEventListener('change', () => saveConfig({ threshold: +thr.value }));
  cd.addEventListener('input', () => { $('#cdOut').textContent = cd.value; });
  cd.addEventListener('change', () => saveConfig({ cooldown_s: +cd.value }));
  pre.addEventListener('input', () => {
    const v = +pre.value;
    $('#preOut').textContent = v.toFixed(1) + 's';
    $('#clipOut') && ($('#clipOut').textContent = '3s');
  });
  pre.addEventListener('change', () => {
    const v = +pre.value;
    saveConfig({ pre_s: v, post_s: Math.max(0.5, 3 - v) });
  });

  // 画面叠加开关（服务端烧录到视频上的内容，不影响模型输入）
  $('#swSkel').addEventListener('change', (e) => saveConfig({ show_skeleton: e.target.checked }));
  $('#swHud').addEventListener('change', (e) => saveConfig({ show_hud: e.target.checked }));
}

async function saveConfig(body) {
  try {
    await post('/api/config', body);
    toast('参数已更新');
  } catch (e) { toast('参数更新失败：' + e.message, true); }
}

/* ------------------------------- 启停 ------------------------------- */
function buildSpec() {
  if (curType === 'camera') {
    return { type: 'camera', index: parseInt($('#camIndex').value || '0', 10) };
  }
  if (curType === 'file') {
    const p = $('#filePath').value.trim();
    if (!p) throw new Error('请先填写或上传一个视频文件');
    return { type: 'file', path: p };
  }
  const ds = $('#demoDataset').value;
  const seq = $('#demoSeq').value;
  if (!ds || !seq) throw new Error('请选择示例数据集与序列');
  return { type: 'demo', dataset: ds, seq: seq };
}

async function startDetect() {
  let spec;
  try { spec = buildSpec(); } catch (e) { toast(e.message, true); return; }
  $('#btnStart').disabled = true;
  try {
    await post('/api/start', spec);
    toast('已开始检测');
    // 重启视频流，确保拿到最新画面
    $('#live').src = '/api/stream?t=' + Date.now();
    $('#btnStop').disabled = false;
    beepCtxUnlock();
  } catch (e) {
    $('#btnStart').disabled = false;
    toast('启动失败：' + e.message, true, 4200);
  }
}

function beepCtxUnlock() {
  // 借用户点击解锁音频上下文（浏览器自动播放策略）
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
  } catch (e) { /* ignore */ }
}

async function stopDetect() {
  try { await post('/api/stop'); toast('已停止检测'); }
  catch (e) { toast('停止失败：' + e.message, true); }
}

/* ------------------------------- 渲染 ------------------------------- */
const STATE_TEXT = { idle: '待机', running: '检测中', finished: '已结束', error: '错误' };

function renderStatus(s) {
  const alarm = !!s.alarm;
  const prob = s.prob || 0;

  // 顶部信息
  $('#mSource').textContent = s.source || '—';
  $('#mState').textContent = STATE_TEXT[s.state] || s.state;
  $('#mFrames').textContent = s.frames;
  $('#mFps').textContent = (s.fps || 0).toFixed(1) + ' fps';
  $('#mElapsed').textContent = fmtDur(s.elapsed);

  // 概率
  $('#pNum').textContent = prob.toFixed(2);
  const pct = Math.round(prob * 100);
  const color = alarm ? 'var(--danger)' : (prob > (s.threshold || 0.6) * 0.7 ? 'var(--warn)' : 'var(--ok)');
  $('#probRing').style.background =
    `conic-gradient(${color} ${pct * 3.6}deg, #eef2f7 ${pct * 3.6}deg)`;
  $('#pNum').style.color = alarm ? 'var(--danger)' : 'var(--text)';
  $('#pBar').style.width = pct + '%';
  $('#pBar').style.background = color;
  $('#pThr').style.left = Math.round((s.threshold || 0.6) * 100) + '%';

  const v = $('#verdict');
  v.textContent = alarm ? '检测到跌倒' : (s.state === 'running' ? '正常活动' : STATE_TEXT[s.state] || '待机');
  v.className = 'verdict' + (alarm ? ' alarm' : '');
  $('#pPeak').textContent = (s.peak || 0).toFixed(2);

  // 报警视觉
  $('#alarmFlash').classList.toggle('on', alarm);
  $('#alarmBadge').classList.toggle('on', alarm);
  if (alarm && !prevAlarm) beep();
  prevAlarm = alarm;

  // 计数
  $('#nFalls').textContent = s.n_falls || 0;
  $('#nRec').textContent = s.n_records || 0;

  // 示例数据的标注区间（参考）
  if (s.gt && s.gt.length) {
    $('#gtBox').classList.remove('hidden');
    $('#gtText').textContent = s.gt.map((p) => `${p[0].toFixed(1)}s ~ ${p[1].toFixed(1)}s`).join('，');
  } else {
    $('#gtBox').classList.add('hidden');
  }

  // 按钮状态
  const running = s.state === 'running';
  $('#btnStart').disabled = running;
  $('#btnStop').disabled = !running;
}

function fmtDur(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m > 0 ? `${m}m${String(s).padStart(2, '0')}s` : `${s}s`;
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function renderRecords(list) {
  const ul = $('#recList');
  if (!list.length) {
    ul.innerHTML = '<li class="empty">暂无记录</li>';
    return;
  }
  ul.innerHTML = list.slice(0, 12).map((r) => `
    <li>
      <div class="rec-head">
        <span class="rec-time">${esc(r.time)}</span>
        <span class="tag">报警</span>
        <button class="rec-play" data-clip="${esc(r.clip_url)}">播放</button>
      </div>
      <div class="rec-sub">
        峰值 ${(r.peak ?? 0).toFixed(2)} ｜ 录像 ${r.duration_s ?? '—'}s（${r.frames ?? 0} 帧）｜ 来源 ${esc(r.source || '—')}
      </div>
    </li>`).join('');

  ul.querySelectorAll('.rec-play').forEach((b) => {
    b.addEventListener('click', () => {
      const li = b.closest('li');
      const old = li.querySelector('video');
      if (old) { old.remove(); b.textContent = '播放'; return; }
      const v = document.createElement('video');
      v.controls = true;
      v.autoplay = true;
      v.className = 'rec-video';
      v.src = b.dataset.clip;
      li.appendChild(v);
      b.textContent = '收起';
    });
  });
}

function setConn(ok) {
  $('#connDot').className = 'dot ' + (ok ? 'on' : 'off');
  $('#connText').textContent = ok ? '已连接' : '连接断开';
}

/* ------------------------------- 轮询 ------------------------------- */
async function pollStatus() {
  try {
    const s = await api('/api/status');
    renderStatus(s);
    setConn(true);
    if (s.error) toast('引擎异常：' + s.error, true, 5000);

    if (s.last_record && s.last_record.id !== lastRecordId) {
      lastRecordId = s.last_record.id;
      toast(`已记录跌倒：${s.last_record.time}，3 秒录像已保存`, false, 3600);
      loadRecords();
    }
  } catch (e) {
    setConn(false);
  }
}

async function pollRecords() {
  try { await loadRecords(); } catch (e) { /* ignore */ }
}

async function loadRecords() {
  const d = await api('/api/records');
  renderRecords(d.records || []);
}

/* ------------------------------- 启动 ------------------------------- */
(async function init() {
  $('#live').src = '/api/stream?t=' + Date.now();
  bindUI();
  try { await loadSources(); } catch (e) { toast('读取输入源失败：' + e.message, true); }
  try {
    const c = await api('/api/config');
    $('#thr').value = c.threshold; $('#thrOut').textContent = (+c.threshold).toFixed(2);
    $('#cd').value = c.cooldown_s; $('#cdOut').textContent = c.cooldown_s;
    $('#pre').value = (c.pre_frames / c.clip_fps).toFixed(1);
    $('#preOut').textContent = (c.pre_frames / c.clip_fps).toFixed(1) + 's';
    $('#swSkel').checked = c.show_skeleton !== false;
    $('#swHud').checked = !!c.show_hud;
  } catch (e) { /* ignore */ }
  await pollStatus();
  await pollRecords();
  setInterval(pollStatus, 500);
  setInterval(pollRecords, 2500);
})();
