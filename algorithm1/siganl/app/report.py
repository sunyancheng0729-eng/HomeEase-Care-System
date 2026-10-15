# -*- coding: utf-8 -*-
"""生成**单文件、离线可看**的跌倒报告 HTML（可下载 / 分享 / 存档）。

报告内容：
  1. 报警时间（服务器系统时间，毫秒）
  2. 数据来源、报警依据（概率 / 阈值）、命中窗口数、报警时刻在数据流中的位置
  3. 报警时刻前后 3 s 的三轴加速度曲线（含模长）—— 报警时刻用红色竖线标记，
     命中窗口区间用红色带标记，阈值参考线在下方概率条上标出
  4. 报警窗口的算法上下文（最优组合配置、模型版本）
"""
import html
import json


_CSS = """
:root{--bg:#0f1420;--panel:#161d2e;--line:#26304a;--txt:#e8ecf5;--dim:#8f9bb5;
--red:#ff4d5e;--orange:#ffa53b;--blue:#4aa8ff;--green:#31d0aa;--purple:#a78bfa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;line-height:1.6}
.wrap{max-width:980px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:15px;margin:26px 0 10px;color:var(--dim);font-weight:600;
letter-spacing:.04em;text-transform:uppercase}
.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:16px}
.alarm{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.badge{background:rgba(255,77,94,.14);color:var(--red);border:1px solid rgba(255,77,94,.4);
padding:5px 12px;border-radius:999px;font-size:13px;font-weight:700}
.badge.ok{background:rgba(49,208,170,.12);color:var(--green);border-color:rgba(49,208,170,.4)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px 20px}
.kv .k{color:var(--dim);font-size:12px;letter-spacing:.03em}
.kv .v{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
.kv .v small{font-size:12px;color:var(--dim);font-weight:400}
canvas{width:100%;height:300px;display:block}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin-top:8px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;
vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);font-weight:500}
code{background:#0b101b;border:1px solid var(--line);border-radius:6px;padding:1px 6px;font-size:12px}
.foot{color:var(--dim);font-size:12px;margin-top:28px;border-top:1px solid var(--line);padding-top:14px}
"""

_JS = """
function drawChart(cv, D){
  const dpr = window.devicePixelRatio||1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W*dpr; cv.height = H*dpr;
  const g = cv.getContext('2d'); g.scale(dpr,dpr);
  const pad = {l:52,r:16,t:18,b:34};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  g.clearRect(0,0,W,H);
  const T = D.t, n = T.length;
  if(!n){ g.fillStyle='#8f9bb5'; g.fillText('无加速度数据', pad.l, pad.t+20); return; }
  const t0 = T[0], t1 = T[n-1];
  let lo = Infinity, hi = -Infinity;
  for(const s of [D.ax,D.ay,D.az,D.mag]) for(const v of s){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const padY = (hi-lo)*0.12+0.05; lo-=padY; hi+=padY;
  const X = t => pad.l + (t-t0)/(t1-t0||1)*pw;
  const Y = v => pad.t + (hi-v)/(hi-lo||1)*ph;
  // 命中窗口区间底纹
  if(D.win_start!=null && D.win_end!=null){
    g.fillStyle='rgba(255,77,94,.10)';
    g.fillRect(X(Math.max(D.win_start,t0)), pad.t, Math.max(1,X(Math.min(D.win_end,t1))-X(Math.max(D.win_start,t0))), ph);
  }
  // 网格
  g.strokeStyle='#25304a'; g.lineWidth=1; g.font='11px system-ui';
  g.fillStyle='#8f9bb5'; g.textAlign='right';
  for(let i=0;i<=4;i++){
    const v = lo + (hi-lo)*i/4, y = Y(v);
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(pad.l+pw,y); g.stroke();
    g.fillText(v.toFixed(2), pad.l-8, y+4);
  }
  g.textAlign='center';
  for(let i=0;i<=6;i++){
    const t = t0 + (t1-t0)*i/6, x = X(t);
    g.strokeStyle='#1c2438'; g.beginPath(); g.moveTo(x,pad.t); g.lineTo(x,pad.t+ph); g.stroke();
    g.fillStyle='#8f9bb5'; g.fillText(t.toFixed(1)+'s', x, pad.t+ph+18);
  }
  // 曲线
  const line=(s,c,w)=>{ g.strokeStyle=c; g.lineWidth=w; g.beginPath();
    for(let i=0;i<n;i++){ const x=X(T[i]), y=Y(s[i]); i?g.lineTo(x,y):g.moveTo(x,y);} g.stroke(); };
  line(D.ax,'#4aa8ff',1.4); line(D.ay,'#31d0aa',1.4); line(D.az,'#ffa53b',1.4);
  line(D.mag,'#a78bfa',2.2);
  // 报警时刻红线
  g.strokeStyle='#ff4d5e'; g.lineWidth=2; g.setLineDash([5,4]);
  g.beginPath(); g.moveTo(X(0),pad.t); g.lineTo(X(0),pad.t+ph); g.stroke(); g.setLineDash([]);
  g.fillStyle='#ff4d5e'; g.textAlign='left'; g.font='bold 11px system-ui';
  g.fillText('报警时刻 t=0', X(0)+6, pad.t+13);
  // 概率条
  if(D.probs && D.probs.length){
    const ph2=26, y0=pad.t+ph+22;
    g.fillStyle='#0b101b'; g.fillRect(pad.l,y0,pw,ph2);
    const m=D.probs.length;
    for(let i=0;i<m;i++){ const v=D.probs[i];
      g.fillStyle = v>=D.threshold ? 'rgba(255,77,94,.85)' : 'rgba(74,168,255,.35)';
      g.fillRect(pad.l + i/m*pw, y0+ph2*(1-v), Math.max(1,pw/m), ph2*v); }
    const yt=y0+ph2*(1-D.threshold);
    g.strokeStyle='#ffa53b'; g.lineWidth=1.5; g.beginPath();
    g.moveTo(pad.l,yt); g.lineTo(pad.l+pw,yt); g.stroke();
    g.fillStyle='#ffa53b'; g.textAlign='left'; g.font='11px system-ui';
    g.fillText('阈值 '+D.threshold, pad.l+4, yt-4);
  }
}
"""


def _f(v, nd=4, dash="—"):
    if v is None:
        return dash
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return dash


def render_report_html(ev, model_meta=None):
    """ev: db.get_fall() 返回的字典（含 snippet）。返回完整 HTML 字符串。"""
    snip = ev.get("snippet") or {}
    data = {
        "t": snip.get("t", []), "ax": snip.get("ax", []), "ay": snip.get("ay", []),
        "az": snip.get("az", []), "mag": snip.get("mag", []),
        "threshold": ev.get("threshold") or 0.5,
        "win_start": (ev.get("t_win_start") - ev.get("stream_alarm"))
        if ev.get("t_win_start") is not None and ev.get("stream_alarm") is not None else None,
        "win_end": (ev.get("t_win_end") - ev.get("stream_alarm"))
        if ev.get("t_win_end") is not None and ev.get("stream_alarm") is not None else None,
        "probs": [],
    }
    cfg = (model_meta or {}).get("config", {})
    src_txt = {"sensor": "传感器实时流", "sim": "模拟数据输入"}.get(ev.get("source"), ev.get("source"))
    rows = "".join(
        f"<tr><td>{html.escape(str(ev.get(k) if ev.get(k) is not None else '—'))}</td>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in [
            ("记录 ID", ev.get("id")),
            ("数据来源", src_txt),
            ("来源序列 / 文件", ev.get("seq") or "—"),
            ("报警时刻（数据流内）", f"{_f(ev.get('stream_alarm'),2)} s"),
            ("命中窗口数", ev.get("n_windows")),
            ("最小编号命中窗概率", _f(ev.get("score"))),
            ("命中窗最大概率", _f(ev.get("score_max"))),
            ("命中窗平均概率", _f(ev.get("score_mean"))),
            ("报警阈值", _f(ev.get("threshold"),2)),
            ("数据总时长", f"{_f(ev.get('duration_s'),2)} s" if ev.get("duration_s") else "—"),
            ("模型版本", ev.get("model") or "—"),
            ("记录写入时间", ev.get("created_at")),
        ])

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>跌倒检测报告 #{ev.get('id')}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>跌倒检测报告</h1>
  <div class="sub">加速度单通道跌倒检测系统 · 最优算法组合 A2（39 维特征 + RF/XGB 概率平均）</div>

  <div class="card">
    <div class="alarm">
      <span class="badge">⚠ 加速度报警</span>
      <span class="badge ok">已记录 # {ev.get('id')}</span>
    </div>
    <div class="grid" style="margin-top:16px">
      <div class="kv"><div class="k">跌倒报警时间（系统时间）</div>
        <div class="v">{html.escape(str(ev.get('alarm_time') or '—'))}</div></div>
      <div class="kv"><div class="k">报警窗口概率 / 阈值</div>
        <div class="v">{_f(ev.get('score'),3)} <small>/ {_f(ev.get('threshold'),2)}</small></div></div>
      <div class="kv"><div class="k">数据来源</div><div class="v">{html.escape(str(src_txt))}</div></div>
      <div class="kv"><div class="k">命中窗口数</div><div class="v">{ev.get('n_windows') or 0}</div></div>
    </div>
  </div>

  <h2>报警时刻前后 3 s 加速度</h2>
  <div class="card">
    <canvas id="cv"></canvas>
    <div class="legend">
      <span><i style="background:#4aa8ff"></i>ax</span>
      <span><i style="background:#31d0aa"></i>ay</span>
      <span><i style="background:#ffa53b"></i>az</span>
      <span><i style="background:#a78bfa"></i>|a| 模长</span>
      <span><i style="background:#ff4d5e;height:9px"></i>报警时刻 / 命中窗口</span>
    </div>
  </div>

  <h2>记录明细</h2>
  <div class="card"><table><tbody>{rows}</tbody></table></div>

  <h2>算法上下文</h2>
  <div class="card"><table><tbody>
    <tr><td>识别流</td><td>重采样 {_f(cfg.get('resample_fs'),0)} Hz →
      Butterworth 低通 {_f(cfg.get('butter_cutoff',5),0)} Hz（零相位）→
      {_f(cfg.get('win_s'),1)} s / {_f(cfg.get('step_s'),2)} s 滑窗 → 39 维统计特征 →
      RF(400)+XGB(400) 概率平均 → 事件聚合</td></tr>
    <tr><td>模型</td><td><code>{html.escape(str((model_meta or {}).get('algorithm', '—')))}</code>
      · seed {html.escape(str((model_meta or {}).get('seed','—')))}
      · 训练窗数 {html.escape(str((model_meta or {}).get('n_train_windows','—')))}
      · 训练时间 {html.escape(str((model_meta or {}).get('trained_at','—')))}</td></tr>
    <tr><td>参考性能</td><td>序列级 AUROC {_f((model_meta or {}).get('reference_metrics',{}).get('auroc_seq_mean'),4)}
      （3 seed × 分层 5 折 CV，见算法工程实验三）</td></tr>
  </tbody></table></div>

  <div class="foot">本报告由 加速度单通道跌倒检测系统 自动生成 ·
  报警时刻红线对应「最早命中窗口的结束时刻」，即系统最早可确认跌倒的时刻。</div>
</div>
<script>
const D = {json.dumps(data, ensure_ascii=False)};
{_JS}
drawChart(document.getElementById('cv'), D);
window.addEventListener('resize', ()=>drawChart(document.getElementById('cv'), D));
</script></body></html>"""
