import sys
import time
from datetime import datetime, timezone

from care.config import AUDIO
from care.storage import now

# Only the original pure policy functions run in the API process.
sys.path.append(str(AUDIO / "src"))
from algorithm3.policy import alarm_tag, inquiry_outcome


class Events:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def timeline(event, action, note, actor="系统", at=None):
        event["timeline"].append({"at": at or now(), "action": action, "note": note, "actor": actor})

    def notify(self, event, reason="新事件", contact_index=0):
        contacts = self.store.settings()["contacts"]
        contact = contacts[min(contact_index, len(contacts) - 1)] if contacts else None
        payload = {"title": "紧急求助" if event["level"] == "EMERGENCY" else "疑似跌倒待核实",
                   "message": f"{event['room']} · {reason}", "contact": contact, "mode": event["mode"], "level": event["level"]}
        self.store.notify(event["id"], payload)
        if self.store.settings().get("webhook_url") and event["mode"] == "live":
            self.store.notify(event["id"], payload, kind="webhook")

    def create(self, channel, evidence, mode="live", source="", dedup_key=None, at=None):
        settings = self.store.settings()
        timestamp = at or now()
        emergency = channel in {"manual", "audio"}
        markers = [{"channel": channel, "at": timestamp, "kind": "spoken_help" if channel == "audio" else channel}]
        tag = alarm_tag(markers)
        event = {
            "occurred_at": timestamp, "detected_at": now(), "mode": mode,
            "level": "EMERGENCY" if emergency else "PENDING", "state": "open", "channel": channel,
            "resident": settings["resident"]["name"] or "未命名长者", "room": settings["resident"]["room"],
            "source": source, "evidence": evidence, "markers": tag["markers"], "tag": tag["tag"],
            "channel_quality": [{k: row[k] for k in ("channel", "state", "reason", "started_at")}
                                for row in self.store.coverage() if row["ended_at"] is None],
            "channels": tag["channels"], "timeline": [], "outcome": None, "resolved_at": None,
            "acknowledged_at": None, "contact_index": 0, "next_escalation_at": time.time() + settings["escalation_seconds"],
            "inquiry": {"status": "not_started"}, "notes": "", "response_s": None,
        }
        self.timeline(event, "detected", "主动求助" if emergency else "原算法检测到疑似跌倒，等待照护者核实", at=timestamp)
        self.timeline(event, "notification_queued", "站内通知已排队")
        event, created = self.store.add_event(event, dedup_key)
        if created:
            self.notify(event)
        return event

    def action(self, event_id, action, outcome=None, note="", actor="照护者", revision=None):
        notification = []

        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("事件已经关闭")
            if action == "acknowledge":
                event["acknowledged_at"] = event["acknowledged_at"] or now()
                self.timeline(event, action, note or "已接手处理", actor)
            elif action == "contact":
                self.timeline(event, action, note or "已尝试联系长者", actor)
            elif action == "escalate":
                event["level"] = "EMERGENCY"
                event["contact_index"] += 1
                event["next_escalation_at"] = time.time() + self.store.settings()["escalation_seconds"]
                self.timeline(event, action, note or "已升级并通知下一位联系人", actor)
                notification.append(True)
            elif action == "resolve":
                if outcome not in {"confirmed_fall", "false_alarm", "safe", "unresolved"}:
                    raise ValueError("请选择处置结果")
                event.update(state="resolved", outcome=outcome, resolved_at=now(), notes=note)
                event["response_s"] = max(0, (datetime.fromisoformat(event["resolved_at"]) - datetime.fromisoformat(event["detected_at"])).total_seconds())
                self.timeline(event, action, note or outcome, actor)
            elif action == "safe":
                result = inquiry_outcome({"status": "RESPONDED", "intent": "OTHER", "prompt_played": True},
                                         decision_level=event["level"], button_ok=True)
                ongoing = event["inquiry"].get("status") in {"listening", "processing"}
                event["inquiry"].update(button_ok=True, result=result)
                if not ongoing:
                    event["inquiry"]["status"] = "button_safe"
                self.timeline(event, action, "长者表示安全，保留事件等待照护者核实", "长者")
            else:
                raise ValueError("未知操作")

        event = self.store.update_event(event_id, mutate, revision)
        if notification:
            self.notify(event, "事件升级", event["contact_index"])
        return event

    def inquiry_start(self, event_id, prompt_played, reason=None):
        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("事件已经关闭")
            if event["inquiry"].get("status") in {"listening", "processing"}:
                raise ValueError("询问正在进行")
            event["inquiry"] = {"status": "listening" if prompt_played else "audio_unavailable", "prompt_played": prompt_played,
                                "asked_at": now(), "deadline": time.time() + 10, "reason": reason}
            self.timeline(event, "inquiry_started", "语音询问已播放，等待回答" if prompt_played else "语音询问不可用", "长者端")
        return self.store.update_event(event_id, mutate)

    def inquiry_processing(self, event_id):
        def mutate(event):
            if event["state"] == "resolved" or event["inquiry"].get("status") != "listening":
                raise ValueError("当前没有正在等待回复的询问")
            event["inquiry"].update(status="processing", processing_deadline=time.time() + 125)
        return self.store.update_event(event_id, mutate)

    def inquiry_reply(self, event_id, record):
        notification = []

        existing = self.store.event(event_id)
        if existing["state"] == "resolved":
            decision = inquiry_outcome({**record, "prompt_played": existing["inquiry"].get("prompt_played")},
                                       decision_level=existing["level"], resolved=True)
            if decision["call"]:
                evidence = {k: v for k, v in record.items() if k not in {"transcript", "error"}}
                return self.create("audio", evidence, existing["mode"], "已处置事件后的语音求助")
            raise ValueError("事件已经关闭，回复未改变事件状态")

        def mutate(event):
            if event["state"] == "resolved":
                raise ValueError("事件已关闭，求助请建立新事件")
            if event["inquiry"].get("status") not in {"listening", "processing"}:
                raise ValueError("当前没有正在等待回复的询问")
            reply = {k: v for k, v in record.items() if k not in {"transcript", "error"}}
            reply["prompt_played"] = event["inquiry"].get("prompt_played", False)
            decision = inquiry_outcome(reply, decision_level=event["level"], button_ok=event["inquiry"].get("button_ok", False))
            if decision["marker"]:
                tag = alarm_tag([*event["markers"], {"channel": "audio", "kind": decision["marker"], "at": event["inquiry"]["asked_at"]}])
                event.update(markers=tag["markers"], channels=tag["channels"], tag=tag["tag"])
            if decision["escalate"]:
                event["level"] = "EMERGENCY"
            if decision["escalate"] or decision["notify"]:
                notification.append(True)
            event["inquiry"].update(status="completed", record=reply, result=decision, completed_at=now())
            self.timeline(event, "inquiry_completed", f"回复结果：{reply.get('intent') or reply.get('status')}；" + ("已升级" if decision["escalate"] else "保留待核实事件"))
        event = self.store.update_event(event_id, mutate)
        if notification:
            self.notify(event, "语音询问结果需要协助", event["contact_index"])
        return event

    def tick(self):
        settings = self.store.settings()
        for event in self.store.events(state="open", limit=10000):
            inquiry = event["inquiry"]
            # Reserve a short transport margin; the recording still covers exactly
            # the original 10 s response window. ASR has a separate bounded wait.
            if inquiry.get("status") == "listening" and time.time() >= inquiry["deadline"] + 2:
                self.inquiry_reply(event["id"], {"status": "NO_RESPONSE"})
            elif inquiry.get("status") == "processing" and time.time() >= inquiry["processing_deadline"]:
                self.inquiry_reply(event["id"], {"status": "audio_unavailable"})
            if (event["level"] == "EMERGENCY" and not event.get("acknowledged_at") and event["mode"] == "live"
                    and time.time() >= event["next_escalation_at"] and event["contact_index"] + 1 < len(settings["contacts"])):
                self.action(event["id"], "escalate", actor="通知调度")
