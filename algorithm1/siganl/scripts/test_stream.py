# -*- coding: utf-8 -*-
"""WS 联调脚本：把某个序列通过 /ws/stream 逐样本推入，验证实时链路。

    E:/anaconda3/envs/animedev/python.exe scripts/test_stream.py fall-01 30   # 30 倍速
    E:/anaconda3/envs/animedev/python.exe scripts/test_stream.py fall-01 0    # 不限速

速度参数 speed<=0 表示尽快推完（用于功能联调）；speed>0 按真实时间轴的 1/speed 推进
（用于观察实时报警的"边推边报"行为）。

注意：服务端每个窗口要跑一次 RF+XGB（单窗 ≈23 ms），所以推流速度超过实时时，
`flushed` 会晚于最后一个 `sample` 到达 —— 本脚本会**等 flushed 或超时 20 s**。
"""
import sys
import json
import asyncio
import urllib.request

import websockets

PORT = 8011

# 本机回环请求必须绕过系统代理（本机环境设置了 HTTP_PROXY，否则 urllib 会走代理 → 502）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_seq(seq):
    url = f"http://127.0.0.1:{PORT}/api/sequence/{seq}"
    with _OPENER.open(url, timeout=10) as r:
        d = json.load(r)
    return d["raw"]["t"], d["raw"]["a"], d["duration_s"]


async def main(seq="fall-01", speed=30.0):
    t, a, dur = fetch_seq(seq)
    print(f"[推流] {seq}  n={len(t)}  dur={dur}s  speed={speed}x")
    uri = f"ws://127.0.0.1:{PORT}/ws/stream"
    falls, statuses, curve, alarms = [], 0, 0, 0
    done = asyncio.Event()
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"type": "config", "threshold": 0.5}))
        await ws.send(json.dumps({"type": "reset"}))
        await ws.send(json.dumps({"type": "meta", "seq": seq}))

        async def drain():
            nonlocal statuses, curve, alarms
            while True:
                try:
                    m = json.loads(await ws.recv())
                except Exception:
                    return
                if m["type"] == "alarm":
                    alarms += 1
                    print(f"  ⚠ 实时报警！流内 {m['t_stream']}s  score={m['score']:.3f}")
                elif m["type"] == "fall":
                    falls.append(m)
                    print(f"  ✔ 已记录 #{m['id']}  alarm={m['event']['stream_alarm']}s "
                          f"score={m['event']['score']:.3f} n_win={m['event']['n_windows']} "
                          f"snip={len(m['snippet']['t'])}")
                elif m["type"] == "status":
                    statuses += 1
                    curve += len(m.get("curve") or [])
                elif m["type"] == "flushed":
                    curve += len(m.get("curve") or [])
                    print(f"  [flush] 新增记录 ids={m['ids']} 曲线点累计={curve}")
                    done.set()
                elif m["type"] == "error":
                    print("  [error]", m["message"])

        task = asyncio.create_task(drain())
        for i in range(len(t)):
            await ws.send(json.dumps({"type": "sample", "t": t[i],
                                      "ax": a[i][0], "ay": a[i][1], "az": a[i][2]}))
            if speed > 0 and i + 1 < len(t):        # 按 1/speed 的真实时间推进
                await asyncio.sleep(max(0.0, (t[i + 1] - t[i]) / speed))
            elif speed <= 0 and i % 8 == 7:
                await asyncio.sleep(0)              # 让出事件循环，保证 drain 能收到消息
        print("  [推流完毕] 发送 flush，等待服务端收尾…")
        await ws.send(json.dumps({"type": "flush"}))
        try:
            await asyncio.wait_for(done.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            print("  [超时] 20 s 内未收到 flushed")
        await asyncio.sleep(0.2)
        task.cancel()

    print(f"[结果] 实时报警 {alarms} 次 / 落库 {len(falls)} 条；"
          f"收到 {statuses} 条状态、{curve} 个窗口概率点")


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "fall-01"
    sp = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    asyncio.run(main(seq, sp))
