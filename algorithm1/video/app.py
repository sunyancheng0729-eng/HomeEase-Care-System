# -*- coding: utf-8 -*-
"""
app.py —— 实时跌倒检测系统（FastAPI 后端）
================================================================
把 engine.DetectionEngine 的能力暴露成 HTTP 接口 + 网页：

  GET  /                    实时监控页（浏览器打开这个）
  GET  /report              跌倒报告页
  GET  /api/stream          MJPEG 实时画面（含检测框、骨架、概率）
  GET  /api/status          当前状态（概率 / 报警 / fps / 命中数 ...）
  GET  /api/sources         可用输入源（摄像头 / 内置示例数据集）
  POST /api/start           开始检测   {type:camera|file|demo, ...}
  POST /api/stop            停止检测
  GET  /api/config          读取参数
  POST /api/config          修改参数（阈值 / 冷却 / 连续确认次数 / 前置后置秒数）
  GET  /api/records         跌倒记录列表
  POST /api/records/clear   清空记录（含录像）
  DELETE /api/records/{id}  删除单条记录
  GET  /api/clip/{file}     录像播放（支持 HTTP Range，可拖动进度条）
  POST /api/upload          上传本地视频文件
  POST /api/report/export   导出 HTML 报告
  GET  /report_file/{name}  查看导出的报告

启动：
  E:/anaconda3/envs/animedev/python.exe app.py
  然后浏览器打开 http://127.0.0.1:8000
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 性能：A4 混合集成的 7 个基学习器都很小，sklearn/joblib 默认会启用
# 「多进程 + OpenMP 多线程」，调度与结果回收的开销远大于计算本身。
# 这些线程池是在【首次导入 numpy/sklearn/torch 时】按环境变量初始化的，
# 所以必须在导入它们之前设置（因此这几行必须放在文件最前面）。
#
# 实测（RTX 3050 Ti，urfd/fall-01，detect_every=3 / prob_every=2）：
#   默认后端      68.5 ms/帧（14.6 fps）
#   下列设置      37.0 ms/帧（27.0 fps）
#
# 想恢复默认行为：启动前设 FVS_NO_TUNE=1。
if os.environ.get("FVS_NO_TUNE") != "1":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import io
import time
import json
import html
import argparse
import mimetypes
from datetime import datetime
from typing import Optional, Iterator

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import (HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from pydantic import BaseModel

import engine as eng

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
CLIPS_DIR = eng.CLIPS_DIR
RECORDS_DIR = eng.RECORDS_DIR
UPLOADS_DIR = eng.UPLOADS_DIR

app = FastAPI(title="实时跌倒检测系统", version="1.0")
ENGINE = eng.DetectionEngine()

# 首页缩略/占位
BOUNDARY = "fvs_frame"


# ===========================================================================
# 请求模型
# ===========================================================================
class StartReq(BaseModel):
    type: str                                   # camera | file | demo
    index: Optional[int] = 0
    path: Optional[str] = None
    dataset: Optional[str] = None
    seq: Optional[str] = None


class ConfigReq(BaseModel):
    threshold: Optional[float] = None
    cooldown_s: Optional[float] = None
    confirm: Optional[int] = None
    pre_s: Optional[float] = None
    post_s: Optional[float] = None
    show_hud: Optional[bool] = None        # 顶部信息条 + 文字 + 概率条（默认关）
    show_skeleton: Optional[bool] = None   # 检测框 + 骨架（默认开）


# ===========================================================================
# 静态页面
# ===========================================================================
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _html(path: str) -> HTMLResponse:
    if not os.path.isfile(path):
        raise HTTPException(404, f"页面不存在：{os.path.basename(path)}")
    return HTMLResponse(_read_text(path),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/", response_class=HTMLResponse)
def index():
    return _html(os.path.join(STATIC_DIR, "index.html"))


@app.get("/report", response_class=HTMLResponse)
def report_page():
    return _html(os.path.join(STATIC_DIR, "report.html"))


@app.get("/static/{name}")
def static_file(name: str):
    safe = os.path.basename(name)
    path = os.path.join(STATIC_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, "文件不存在")
    ctype = mimetypes.guess_type(safe)[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype in ("application/javascript",):
        ctype += "; charset=utf-8"
    with open(path, "rb") as f:
        return Response(f.read(), media_type=ctype,
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ===========================================================================
# 实时画面（MJPEG）
# ===========================================================================
@app.get("/api/stream")
def stream():
    def gen() -> Iterator[bytes]:
        while True:
            jpg = ENGINE.latest_jpeg()
            if jpg:
                yield (b"--" + BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                       + jpg + b"\r\n")
            time.sleep(0.033)
    return StreamingResponse(
        gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"})


# ===========================================================================
# 状态 / 参数 / 输入源
# ===========================================================================
@app.get("/api/health")
def health():
    return {"ok": True, "models_ready": ENGINE.models_ready}


@app.get("/api/status")
def status():
    return ENGINE.status()


@app.get("/api/config")
def get_config():
    return ENGINE.config()


@app.post("/api/config")
def set_config(req: ConfigReq):
    return ENGINE.set_config(threshold=req.threshold, cooldown_s=req.cooldown_s,
                             confirm=req.confirm, pre_s=req.pre_s, post_s=req.post_s,
                             show_hud=req.show_hud, show_skeleton=req.show_skeleton)


@app.get("/api/sources")
def sources():
    return {"demo": eng.list_demo_sources(),
            "camera_hint": "默认摄像头 index=0；笔记本内置摄像头通常是 0，外接设备依次为 1、2 ..."}


# ===========================================================================
# 启停
# ===========================================================================
@app.post("/api/start")
def start(req: StartReq):
    spec = {"type": req.type}
    if req.type == "camera":
        spec["index"] = int(req.index or 0)
    elif req.type == "file":
        spec["path"] = req.path
    elif req.type == "demo":
        spec["dataset"] = req.dataset
        spec["seq"] = req.seq
    else:
        raise HTTPException(400, "type 必须是 camera / file / demo")
    try:
        return ENGINE.start(spec)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/stop")
def stop():
    return ENGINE.stop()


# ===========================================================================
# 记录
# ===========================================================================
@app.get("/api/records")
def records(limit: Optional[int] = None):
    """跌倒记录（SQLite 读取，新的在前）。"""
    recs = ENGINE.records()
    if limit:
        recs = recs[:int(limit)]
    return {"count": len(recs), "records": recs}


@app.get("/api/records/stats")
def records_stats():
    return ENGINE.store.stats()


@app.post("/api/records/clear")
def clear_records():
    # 返回 {"ok", "count":0, "deleted":记录数, "clips_removed":已删录像数, "clips_left":剩余录像数}
    # 录像分批删除（单次上限见 engine.DetectionEngine.MAX_CLIP_DELETES），
    # 若仍有剩余，前端提示再点一次即可。
    res = ENGINE.clear_records(delete_clips=True)
    return {"ok": True, "count": 0, **res}


@app.delete("/api/records/{rid}")
def delete_record(rid: str):
    try:
        rid_int = int(rid)
    except ValueError:
        raise HTTPException(400, "记录 id 必须是整数")
    gone = ENGINE.delete_record(rid_int)
    if gone is None:
        raise HTTPException(404, "记录不存在")
    return {"ok": True, "deleted": gone["id"],
            "clip_removed": bool(gone.get("clip_removed"))}


# ===========================================================================
# 录像播放（支持 Range，浏览器可拖动进度条）
# ===========================================================================
def _stream_file(path: str, start: int, end: int, chunk: int = 256 * 1024) -> Iterator[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        remain = end - start + 1
        while remain > 0:
            data = f.read(min(chunk, remain))
            if not data:
                break
            remain -= len(data)
            yield data


def _clip_response(name: str, request: Request):
    safe = os.path.basename(name or "")
    path = os.path.join(CLIPS_DIR, safe)
    if not safe or not os.path.isfile(path):
        raise HTTPException(404, "录像不存在")
    size = os.path.getsize(path)
    rng = request.headers.get("range")
    if rng and rng.startswith("bytes="):
        try:
            s, e = rng[6:].split("-", 1)
            start = int(s) if s else 0
            end = int(e) if e else size - 1
        except ValueError:
            raise HTTPException(416, "Range 头格式错误")
        start = max(0, start)
        end = min(size - 1, end)
        if start > end:
            raise HTTPException(416, "Range 越界")
        headers = {"Content-Range": f"bytes {start}-{end}/{size}",
                   "Accept-Ranges": "bytes",
                   "Content-Length": str(end - start + 1)}
        return StreamingResponse(_stream_file(path, start, end), status_code=206,
                                 media_type="video/mp4", headers=headers)
    return StreamingResponse(_stream_file(path, 0, size - 1), media_type="video/mp4",
                             headers={"Accept-Ranges": "bytes",
                                      "Content-Length": str(size)})


@app.get("/api/clip/{name}")
def get_clip(name: str, request: Request):
    return _clip_response(name, request)


@app.get("/clips/{name}")
def get_clip_alias(name: str, request: Request):
    """给导出的 HTML 报告用（相对路径 clips/xxx.mp4）。"""
    return _clip_response(name, request)


# ===========================================================================
# 上传本地视频
# ===========================================================================
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "video.mp4")
    if not name:
        raise HTTPException(400, "文件名无效")
    dst = os.path.join(UPLOADS_DIR, name)
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(dst)
    if size == 0:
        raise HTTPException(400, "上传的文件为空")
    return {"path": dst, "name": name, "size": size}


# ===========================================================================
# 报告导出
# ===========================================================================
REPORT_TEMPLATE_HEAD = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>跌倒检测报告 · {gen}</title>
<style>
 body{{font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;background:#f5f7fa;
      color:#1f2937;margin:0;padding:32px}}
 .wrap{{max-width:1100px;margin:0 auto;background:#fff;border:1px solid #e3e8ef;
       border-radius:14px;padding:32px;box-shadow:0 2px 14px rgba(15,23,42,.05)}}
 h1{{font-size:22px;margin:0 0 6px}}
 .meta{{color:#6b7280;font-size:13px;margin-bottom:22px}}
 .stat{{display:flex;gap:28px;margin:18px 0 26px;flex-wrap:wrap}}
 .stat div{{background:#f8fafc;border:1px solid #e8edf3;border-radius:10px;
           padding:12px 20px;min-width:120px}}
 .stat b{{display:block;font-size:22px;color:#0f172a;margin-top:4px}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th,td{{border-bottom:1px solid #eef2f7;padding:10px 8px;text-align:left;vertical-align:top}}
 th{{background:#f8fafc;color:#475569;font-weight:600}}
 .badge{{display:inline-block;background:#fee2e2;color:#b91c1c;border-radius:6px;
        padding:2px 8px;font-size:12px;font-weight:600}}
 video{{width:240px;border-radius:8px;background:#0f172a}}
 .ok{{color:#059669}} .warn{{color:#b45309}}
 footer{{margin-top:26px;color:#94a3b8;font-size:12px}}
</style></head><body><div class="wrap">
<h1>跌倒检测报告</h1>
<div class="meta">生成时间：{gen} ｜ 数据来源：实时跌倒检测系统（YOLO 检测 → 自研 P1 姿态 → A4 混合集成）</div>
<div class="stat">
 <div>跌倒次数<b>{n}</b></div>
 <div>首次记录<b>{first}</b></div>
 <div>最近记录<b>{last}</b></div>
 <div>最高峰值概率<b>{peak}</b></div>
</div>
"""

REPORT_TEMPLATE_TAIL = """
<footer>本报告由系统自动生成，可直接分享给他人查看；录像文件与本报告同目录下的 clips/ 文件夹中。</footer>
</div></body></html>"""


def _build_report_html() -> str:
    recs = ENGINE.records()          # 已是新的在前
    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if recs:
        last = recs[0]["time"]
        first = recs[-1]["time"]
        peak = f"{max(r.get('peak', 0) for r in recs):.2f}"
    else:
        first = last = peak = "—"

    rows = []
    for i, r in enumerate(recs, 1):
        clip = html.escape(r.get("clip", ""))
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(r.get('time',''))}</td>"
            f"<td>{r.get('prob',0):.2f}</td>"
            f"<td>{r.get('peak',0):.2f}</td>"
            f"<td>{r.get('duration_s',0)}s / {r.get('frames',0)}帧</td>"
            f"<td>{html.escape(r.get('source',''))}</td>"
            f"<td><span class='badge'>报警</span></td>"
            f"<td><video controls preload='metadata' src='clips/{clip}'></video></td>"
            "</tr>")
    body = ("<table><tr><th>#</th><th>跌倒时间（系统时间）</th><th>触发概率</th>"
            "<th>峰值概率</th><th>录像时长</th><th>输入源</th><th>标记</th>"
            "<th>3 秒录像</th></tr>" + "".join(rows) + "</table>") if rows else \
        "<p class='warn'>暂无跌倒记录。</p>"

    head = REPORT_TEMPLATE_HEAD.format(gen=gen, n=len(recs), first=first,
                                       last=last, peak=peak)
    return head + body + REPORT_TEMPLATE_TAIL


@app.post("/api/report/export")
def export_report():
    gen = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"report_{gen}.html"
    path = os.path.join(RECORDS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_build_report_html())
    return {"ok": True, "name": name, "url": f"/report_file/{name}",
            "path": path, "n": len(ENGINE.records())}


@app.get("/report_file/{name}", response_class=HTMLResponse)
def report_file(name: str):
    safe = os.path.basename(name or "")
    path = os.path.join(RECORDS_DIR, safe)
    if not safe.endswith(".html") or not os.path.isfile(path):
        raise HTTPException(404, "报告不存在")
    return HTMLResponse(_read_text(path))


# ===========================================================================
# 入口
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="实时跌倒检测系统")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=None, help="跌倒判定阈值")
    ap.add_argument("--camera", type=int, default=None, help="启动时自动打开摄像头")
    ap.add_argument("--demo", nargs=2, metavar=("DATASET", "SEQ"), default=None,
                    help="启动时自动跑内置示例，如 --demo urfd fall-01")
    a = ap.parse_args()

    if a.threshold is not None:
        ENGINE.set_config(threshold=a.threshold)

    autostart = None
    if a.camera is not None:
        autostart = {"type": "camera", "index": a.camera}
    elif a.demo is not None:
        autostart = {"type": "demo", "dataset": a.demo[0], "seq": a.demo[1]}

    import uvicorn
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}"
    print("=" * 66)
    print("  实时跌倒检测系统已启动")
    print(f"  监控页：{url}")
    print(f"  报告页：{url}/report")
    print("  停止服务：Ctrl + C")
    print("=" * 66)
    if autostart:
        try:
            ENGINE.start(autostart)
            print(f"  已自动开始检测：{autostart}")
        except Exception as e:
            print(f"  [警告] 自动启动失败：{e}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
