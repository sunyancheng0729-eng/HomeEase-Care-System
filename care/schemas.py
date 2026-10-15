from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Resident(Model):
    name: str = Field(max_length=60)
    room: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=1000)


class Contact(Model):
    name: str = Field(min_length=1, max_length=60)
    relation: str = Field(default="家属", max_length=30)
    phone: str = Field(min_length=3, max_length=40, pattern=r"^[+0-9 ()-]+$")


class Settings(Model):
    resident: Resident
    contacts: list[Contact] = Field(max_length=10)
    audio_consent: bool = False
    exercise_target: int = Field(default=10, ge=1, le=10000)
    escalation_seconds: int = Field(default=60, ge=15, le=3600)
    webhook_url: str = Field(default="", max_length=2000)
    notification_test_at: str | None = None

    @model_validator(mode="after")
    def valid_webhook(self):
        if self.webhook_url:
            from pydantic import TypeAdapter
            url = TypeAdapter(HttpUrl).validate_python(self.webhook_url)
            if url.username or url.password:
                raise ValueError("Webhook URL 不得含账户密码")
        return self


class VideoStart(Model):
    source: Literal["camera", "sample", "upload"]
    camera_index: int = Field(default=0, ge=0, le=10)
    upload_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}\.[a-z0-9]+$")


class SessionStart(Model):
    exercise: Literal["chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"]
    target_reps: int = Field(default=10, ge=1, le=10000)
    source: Literal["camera", "replay", "upload"] = "camera"
    camera_index: int = Field(default=0, ge=0, le=10)
    capture: Literal["shared", "client"] = "shared"
    upload_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}\.[a-z0-9]+$")


class SessionControl(Model):
    action: Literal["pause", "resume", "end"]


class EventAction(Model):
    action: Literal["acknowledge", "contact", "escalate", "resolve", "safe"]
    outcome: Literal["confirmed_fall", "false_alarm", "safe", "unresolved"] | None = None
    note: str = Field(default="", max_length=2000)
    revision: int | None = Field(default=None, ge=1)


class InquiryStart(Model):
    prompt_played: bool
    reason: str | None = Field(default=None, max_length=200)


class SampleAudio(Model):
    sample: str = Field(pattern=r"^[a-z0-9_]+$")
    task: Literal["monitor", "sounds", "help"] = "monitor"
