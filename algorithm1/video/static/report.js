/* ==========================================================================
   跌倒检测报告页 —— 前端逻辑
   ========================================================================== */
const $ = (s) => document.querySelector(s);

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
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function setConn(ok) {
  $('#connDot').className = 'dot ' + (ok ? 'on' : 'off');
  $('#connText').textContent = ok ? '已连接' : '连接断开';
}

function render(d) {
  const recs = d.records || [];
  $('#genTime').textContent = '报告时间：' + new Date().toLocaleString('zh-CN')
    + ' ｜ 记录总数：' + recs.length + ' 条';

  $('#sCount').textContent = recs.length;
  if (recs.length) {
    const times = recs.map((r) => r.time).sort();
    $('#sFirst').textContent = times[0];
    $('#sLast').textContent = times[times.length - 1];
    $('#sPeak').textContent = Math.max(...recs.map((r) => r.peak || 0)).toFixed(2);
  } else {
    $('#sFirst').textContent = '—';
    $('#sLast').textContent = '—';
    $('#sPeak').textContent = '—';
  }

  const box = $('#tableBox');
  if (!recs.length) {
    box.innerHTML = '<div class="empty-state">暂无跌倒记录。<br>'
      + '回到<a href="/">监控台</a>开始检测后，检测到的跌倒会自动出现在这里。</div>';
    return;
  }

  const rows = recs.map((r, i) => `
    <tr>
      <td class="mono">${i + 1}</td>
      <td class="mono">${esc(r.time)}</td>
      <td class="mono">${(r.prob ?? 0).toFixed(2)}</td>
      <td class="mono">${(r.peak ?? 0).toFixed(2)}</td>
      <td class="mono">${r.duration_s ?? '—'}s / ${r.frames ?? 0} 帧</td>
      <td>${esc(r.source || '—')}</td>
      <td><span class="tag">报警</span></td>
      <td><video controls preload="metadata" src="${esc(r.clip_url)}"></video></td>
      <td><button class="ghost" data-del="${esc(r.id)}">删除</button></td>
    </tr>`).join('');

  box.innerHTML = `
    <table class="report">
      <tr>
        <th>#</th><th>跌倒时间（系统时间）</th><th>触发概率</th><th>峰值概率</th>
        <th>录像时长</th><th>输入源</th><th>报警标记</th><th>3 秒录像</th><th>操作</th>
      </tr>
      ${rows}
    </table>`;

  box.querySelectorAll('[data-del]').forEach((b) => {
    b.addEventListener('click', async () => {
      if (!confirm('确定删除这条记录及其录像？')) return;
      try {
        const r = await api('/api/records/' + encodeURIComponent(b.dataset.del), { method: 'DELETE' });
        if (r && r.clip_removed === false) {
          // 记录已删，但录像文件被环境的批量删除保护拦下（成为孤儿文件）
          toast('记录已删除；录像文件暂未删除，下次点「清空记录」会一并清理', false, 5200);
        } else {
          toast('已删除');
        }
        load();
      } catch (e) { toast('删除失败：' + e.message, true); }
    });
  });
}

async function load() {
  try {
    const d = await api('/api/records');
    render(d);
    setConn(true);
  } catch (e) {
    setConn(false);
    toast('读取记录失败：' + e.message, true);
  }
}

$('#btnRefresh').addEventListener('click', () => { load(); toast('已刷新'); });

$('#btnExport').addEventListener('click', async () => {
  try {
    const r = await api('/api/report/export', { method: 'POST' });
    toast('报告已导出，正在打开…');
    window.open(r.url, '_blank');
  } catch (e) { toast('导出失败：' + e.message, true); }
});

$('#btnClear').addEventListener('click', async () => {
  if (!confirm('确定清空全部跌倒记录和录像？此操作不可撤销。')) return;
  try {
    const r = await api('/api/records/clear', { method: 'POST' });
    const left = (r && r.clips_left) || 0;
    if (left > 0) {
      // 录像分批删除，一轮有上限；剩下的录像文件再点一次即可清掉
      toast(`已清空 ${r.deleted || 0} 条记录；还有 ${left} 个录像文件，再点一次「清空记录」即可清除`, false, 5200);
    } else {
      toast(`已清空 ${r.deleted || 0} 条记录`);
    }
    load();
  } catch (e) { toast('清空失败：' + e.message, true); }
});

load();
setInterval(load, 5000);
