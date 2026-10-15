# -*- coding: utf-8 -*-
"""跌倒检测系统 —— FastAPI 后端。

启动：
    cd systems
    E:/anaconda3/envs/animedev/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

能力总览（详见 systems/README.md）：
  · 传感器实时流：WS /ws/stream   —— 逐样本持续输入，实时检测、实时报警、实时落库
  · 模拟数据输入：POST /api/simulate（内置序列） / POST /api/simulate/upload（上传 CSV）
  · 记录与报告：GET /api/falls、GET /api/falls/{id}、GET /report/{id}、
                GET /api/falls/{id}/report.html（单文件离线报告）
"""
import os
import io
import sys
import csv
import time
import asyncio
import tempfile

from fastapi import (FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File,
                     Form, HTTPException, Query)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import config as C                       # noqa: E402
from core import preprocess as P                   # noqa: E402
from core import datasets as D                     # noqa: E402
from core.model import BestFallModel, feature_importance   # noqa: E402
from core.detector import detect_offline, StreamingDetector  # noqa: E402
from app import db                                 # noqa: E402
from app.report import render_report_html          # noqa: E402
from app.schemas import SimulateRequest, ThresholdUpdate     # noqa: E402

WEB_DIR = os.path.join(ROOT, "web")
MODEL_PATH = os.path.join(ROOT, "model", "best_model.pkl")
UPLOAD_DIR = os.path.join(ROOT, "runtime", "uploads")

STATE = {"model": None, "threshold": C.ALARM_THRESHOLD, "load_error": None}


def _load_model():
    if not os.path.isfile(MODEL_PATH):
        STATE["load_error"] = f"模型不存在：{MODEL_PATH}，请先运行 scripts/train_best_model.py"
        return
    try:
        STATE["model"] = BestFallModel.load(MODEL_PATH)
    except Exception as e:                                    # pragma: no cover
        STATE["load_error"] = f"{type(e).__name__}: {e}"


def _model():
    m = STATE["model"]
    if m is None:
        raise HTTPException(503, STATE["load_error"] or "模型未加载")
    return m


app = FastAPI(title="加速度单通道跌倒检测系统",
              description="最优算法组合 A2：39 维统计特征 + RF/XGB 概率平均",
              version="1.0.0")


@app.on_event("startup")
def _startup():
    db.init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _load_model()


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/report/{fid}", include_in_schema=False)
def report_page(fid: int):
    return FileResponse(os.path.join(WEB_DIR, "report.html"))


app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


# ---------------------------------------------------------------------------
# 系统 / 模型信息
# ---------------------------------------------------------------------------
@app.get("/api/health", summary="健康检查与运行配置")
def health():
    m = STATE["model"]
    return {
        "status": "ok" if m is not None else "model_missing",
        "error": STATE["load_error"],
        "algorithm": m.meta.get("algorithm") if m else None,
        "threshold": STATE["threshold"],
        "config": m.meta.get("config") if m else None,
        "data_available": D.is_available(),
        "data_dir": D.acc_csv_dir(),
        "db": db.stats(),
    }


@app.get("/api/model", summary="模型元信息与特征重要性")
def model_info(top: int = Query(20, ge=1, le=39)):
    m = _model()
    imp = [{"feature": k, "importance": round(v, 5)} for k, v in feature_importance(m)][:top]
    return {"meta": m.meta, "top_features": imp}


@app.get("/api/sequences", summary="可用内置序列列表")
def sequences():
    s = D.list_sequences()
    return {"fall": [D.sequence_meta(x) for x in s["fall"]],
            "adl": [D.sequence_meta(x) for x in s["adl"]],
            "available": D.is_available(), "dir": D.acc_csv_dir()}


@app.get("/api/sequence/{seq}", summary="读取某个内置序列的原始数据（预览 / 推流回放）")
def sequence_data(seq: str, stride: int = Query(1, ge=1, le=20),
                  max_points: int = Query(3000, ge=100, le=20000)):
    if seq not in D.list_sequences()["fall"] + D.list_sequences()["adl"]:
        raise HTTPException(404, f"序列不存在：{seq}")
    t, a = P.read_acc_csv(D.acc_csv(seq))
    n = len(t)
    step = max(stride, int((n + max_points - 1) // max_points))
    idx = list(range(0, n, step))
    meta = D.sequence_meta(seq)
    fs = float(1.0 / max((t[-1] - t[0]) / max(n - 1, 1), 1e-9))
    return {**meta, "n": n, "duration_s": round(float(t[-1] - t[0]), 3),
            "fs": round(fs, 2), "stride": step, "downsampled": step > 1,
            "t": [round(float(t[i]), 4) for i in idx],
            "ax": [round(float(a[i, 0]), 5) for i in idx],
            "ay": [round(float(a[i, 1]), 5) for i in idx],
            "az": [round(float(a[i, 2]), 5) for i in idx],
            "raw": {"t": [float(x) for x in t], "a": a.tolist()}}


@app.post("/api/threshold", summary="设置报警阈值（运行时）")
def set_threshold(body: ThresholdUpdate):
    STATE["threshold"] = float(body.threshold)
    return {"threshold": STATE["threshold"]}


# ---------------------------------------------------------------------------
# 模拟数据输入（整段离线检测）
# ---------------------------------------------------------------------------
def _run_and_save(t, a, threshold, source, seq, duration_s, save, include_curve, label=None):
    m = _model()
    events, info = detect_offline(t, a, m, threshold=threshold)
    ids = []
    if save:
        for ev in events:
            ids.append(db.insert_fall(ev, model_name=m.meta.get("algorithm", ""),
                                      seq=seq, duration_s=info["duration_s"]))
    out = {
        "duration_s": round(info["duration_s"], 3),
        "n_windows": info["n_windows"],
        "n_hits": info["n_hits"],
        "n_events": len(events),
        "threshold": threshold,
        "source": source, "seq": seq, "label": label,
        "saved_ids": ids,
        "events": [{"id": ids[i] if i < len(ids) else None,
                    "t_alarm": round(ev["t_alarm"], 3),
                    "stream_alarm": round(ev["t_alarm"], 3),
                    "score": round(ev["score"], 4),
                    "score_max": round(ev["score_max"], 4),
                    "n_windows": ev["n_windows"],
                    "snippet": ev["snippet"]} for i, ev in enumerate(events)],
    }
    if include_curve:
        out["curve"] = {"t": info["t_end"], "p": info["probs"]}
    return out


@app.post("/api/simulate", summary="模拟数据输入：对内置序列做整段检测")
def simulate(body: SimulateRequest):
    thr = STATE["threshold"] if body.threshold is None else float(body.threshold)
    seq = body.seq
    if not seq:
        raise HTTPException(400, "缺少 seq（内置序列名）")
    allseq = D.list_sequences()["fall"] + D.list_sequences()["adl"]
    if seq not in allseq:
        raise HTTPException(404, f"序列不存在：{seq}")
    t, a = P.read_acc_csv(D.acc_csv(seq))
    return _run_and_save(t, a, thr, "sim", seq, None, body.save, body.include_curve)


@app.post("/api/simulate/upload", summary="模拟数据输入：上传 CSV 做整段检测")
async def simulate_upload(file: UploadFile = File(...),
                          threshold: float = Form(None),
                          save: bool = Form(True),
                          include_curve: bool = Form(True)):
    thr = STATE["threshold"] if threshold is None else float(threshold)
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")
    name = file.filename or "upload.csv"
    path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{os.path.basename(name)}")
    with open(path, "wb") as f:
        f.write(raw)
    try:
        t, a = P.read_acc_csv(path)
    except Exception as e:
        raise HTTPException(400, f"CSV 解析失败：{e}（支持列序：t_ms,|a|,ax,ay,az 或 t,ax,ay,az）")
    return _run_and_save(t, a, thr, "sim", name, None, save, include_curve, label=name)


# ---------------------------------------------------------------------------
# 记录与报告
# ---------------------------------------------------------------------------
@app.get("/api/stats", summary="记录统计")
def stats():
    return db.stats()


@app.get("/api/falls", summary="跌倒记录列表")
def falls(limit: int = Query(200, ge=1, le=2000),
          source: str = Query(None, pattern="^(sensor|sim)$")):
    return {"items": db.list_falls(limit=limit, source=source)}


@app.get("/api/falls/{fid}", summary="跌倒记录详情（含 3 s 加速度切片）")
def fall_detail(fid: int):
    ev = db.get_fall(fid)
    if ev is None:
        raise HTTPException(404, f"记录不存在：{fid}")
    return ev


@app.delete("/api/falls/{fid}", summary="删除一条记录")
def fall_delete(fid: int):
    if not db.delete_fall(fid):
        raise HTTPException(404, f"记录不存在：{fid}")
    return {"deleted": fid}


@app.delete("/api/falls", summary="清空全部记录")
def falls_clear():
    return {"deleted": db.clear_falls()}


@app.get("/api/falls/{fid}/report.html", summary="下载单文件报告（离线可看）",
         response_class=HTMLResponse)
def fall_report_html(fid: int):
    ev = db.get_fall(fid)
    if ev is None:
        raise HTTPException(404, f"记录不存在：{fid}")
    m = STATE["model"]
    html = render_report_html(ev, m.meta if m else None)
    return HTMLResponse(content=html, headers={
        "Content-Disposition": f'inline; filename="fall_report_{fid}.html"'})


@app.get("/api/falls/{fid}/export.json", summary="导出记录 JSON")
def fall_export(fid: int):
    ev = db.get_fall(fid)
    if ev is None:
        raise HTTPException(404, f"记录不存在：{fid}")
    return JSONResponse(ev, headers={
        "Content-Disposition": f'attachment; filename="fall_{fid}.json"'})


# ---------------------------------------------------------------------------
# 传感器实时流（WebSocket）
# ---------------------------------------------------------------------------
def _compact(ev):
    return {k: ev.get(k) for k in ("source", "t_alarm", "t_win_start", "t_win_end",
                                   "n_windows", "score", "score_max", "score_mean",
                                   "threshold", "stream_alarm", "truncated")}


def _new_curve(det, sent_upto):
    """取出 det 中尚未下发的窗口概率（增量），返回 (list, 新的 sent_upto)。"""
    lo = max(0, sent_upto - det.log_base)
    tail = det.log[lo:]
    return tail, det.log_base + len(det.log)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """持续输入的加速度传感器流。

    客户端 → 服务端：
      {"type":"config","threshold":0.5}
      {"type":"meta","seq":"fall-01"}                       # 给后续记录打标签
      {"type":"sample","t":1.234,"ax":..,"ay":..,"az":..}   # 逐样本推入
      {"type":"flush"}                                       # 收尾（输出未闭合事件）
      {"type":"reset"}                                       # 重置检测状态
    服务端 → 客户端：
      {"type":"ready","threshold":..,"algorithm":..}
      {"type":"status","t":..,"p":..,"alarming":bool,"n_eval":..,"n_fall":..}
      {"type":"fall","id":..,"event":{...}}
      {"type":"flushed","ids":[...]} / {"type":"reset_ok"}
    """
    await ws.accept()
    m = STATE["model"]
    if m is None:
        await ws.send_json({"type": "error", "message": STATE["load_error"] or "模型未加载"})
        await ws.close()
        return

    threshold = STATE["threshold"]
    det = StreamingDetector(m, threshold=threshold)
    meta = {"seq": None}
    n_fall = 0
    last_status = 0.0
    sent_upto = 0          # det.recent 中已下发给前端的进度（保证概率曲线不丢点）

    await ws.send_json({"type": "ready", "threshold": threshold,
                        "algorithm": m.meta.get("algorithm"),
                        "config": m.meta.get("config")})
    try:
        while True:
            msg = await ws.receive_json()
            typ = msg.get("type")

            if typ == "config":
                if msg.get("threshold") is not None:
                    threshold = float(msg["threshold"])
                    det.threshold = threshold
                    STATE["threshold"] = threshold
                await ws.send_json({"type": "config_ok", "threshold": threshold})

            elif typ == "meta":
                meta["seq"] = msg.get("seq")

            elif typ == "sample":
                try:
                    t = float(msg["t"]); ax = float(msg["ax"])
                    ay = float(msg["ay"]); az = float(msg["az"])
                except (KeyError, TypeError, ValueError):
                    await ws.send_json({"type": "error", "message": "sample 字段非法"})
                    continue
                events = det.push(t, ax, ay, az)
                # 命中即刻的实时报警（早于落库，符合"最早可确认时刻"口径）
                for al in det.pop_alarms():
                    await ws.send_json({"type": "alarm", **al,
                                        "t_stream": round(al["t_alarm"] - det.t0, 3)})
                for ev in events:
                    fid = db.insert_fall(ev, model_name=m.meta.get("algorithm", ""),
                                         seq=meta["seq"], duration_s=det.status()["t_stream"])
                    n_fall += 1
                    await ws.send_json({"type": "fall", "id": fid, "n_fall": n_fall,
                                        "event": _compact(ev), "snippet": ev["snippet"]})
                now = time.time()
                if now - last_status > 0.12:
                    last_status = now
                    st = det.status()
                    tail, sent_upto = _new_curve(det, sent_upto)
                    await ws.send_json({"type": "status", **st, "n_fall": n_fall,
                                        "threshold": threshold, "curve": tail})

            elif typ == "flush":
                ids = []
                for ev in det.flush():
                    ids.append(db.insert_fall(ev, model_name=m.meta.get("algorithm", ""),
                                              seq=meta["seq"],
                                              duration_s=det.status()["t_stream"]))
                    n_fall += 1
                tail, sent_upto = _new_curve(det, sent_upto)
                await ws.send_json({"type": "flushed", "ids": ids, "n_fall": n_fall,
                                    "curve": tail})

            elif typ == "reset":
                det = StreamingDetector(m, threshold=threshold)
                n_fall = 0
                sent_upto = 0
                await ws.send_json({"type": "reset_ok"})

            else:
                await ws.send_json({"type": "error", "message": f"未知消息类型 {typ}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:                                     # pragma: no cover
        try:
            await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
