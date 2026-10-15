"""Versioned local HTTP contract. One worker owns in-memory session state."""
import hashlib
import json
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .engine import ExerciseSession, Predictor


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CreateSession(StrictModel):
    target_reps: int = Field(default=10, ge=1, le=10000)
    exercise: Literal["chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"] = "chair_sit_to_stand"


class Frame(StrictModel):
    timestamp_ms: float = Field(ge=0, le=1e15)
    person_id: str = Field(default="primary", min_length=1, max_length=128)
    keypoints: list[list[float]] = Field(min_length=17, max_length=17)

    @model_validator(mode="after")
    def check_points(self):
        for point in self.keypoints:
            if len(point) != 3 or not 0 <= point[2] <= 1 or abs(point[0]) > 1e7 or abs(point[1]) > 1e7:
                raise ValueError("each COCO point must be [x, y, confidence 0..1]")
        return self


class FrameBatch(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    coordinate_space: Literal["camera_xy"] = "camera_xy"
    schema_name: Literal["coco17"] = "coco17"
    frames: list[Frame] = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def check_timestamps(self):
        if any(b.timestamp_ms <= a.timestamp_ms for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("batch timestamps must strictly increase")
        return self


class Control(StrictModel):
    action: Literal["pause", "resume", "end"]


class CycleEvent(StrictModel):
    rep_index: int
    start_s: float
    end_s: float
    duration_s: float


class Interruption(StrictModel):
    reason: str
    timestamp_ms: float | None


class SessionState(StrictModel):
    api_version: Literal["v1"]
    session_id: str
    status: Literal["running", "paused", "ended"]
    exercise: Literal["chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"]
    recognized_exercise: str
    exercise_probabilities: dict[str, float]
    phase: str
    phase_confidence: float
    phase_probabilities: dict[str, float]
    count: int
    target_reps: int
    progress: float
    target_completed: bool
    elapsed_s: float
    active_duration_s: float
    valid_duration_s: float
    quality: str
    last_timestamp_ms: float | None
    interruptions: list[Interruption]
    cycles: list[CycleEvent]
    model_id: str
    model_validation: str


class BatchResult(SessionState):
    request_id: str
    accepted_frames: int
    new_cycles: list[CycleEvent]


class HealthState(StrictModel):
    status: Literal["ready"]
    api_version: Literal["v1"]
    model_id: str
    session_count: int


def create_app(predictor=None):
    sessions, caches = {}, {}
    registry_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor or Predictor()
        yield

    app = FastAPI(title="HomeEase Algorithm 2", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse(status_code=error.status_code, content={"error": {"code": str(error.detail)}, "api_version": "v1"})

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        # Buffer with an upper bound; protects chunked bodies as well as Content-Length.
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > 2_000_000:
                return JSONResponse(status_code=413, content={"error": {"code": "BODY_TOO_LARGE"}, "api_version": "v1"})
        request._body = bytes(payload)
        return await call_next(request)

    def get_session(session_id):
        with registry_lock:
            if session_id not in sessions:
                raise HTTPException(404, "SESSION_NOT_FOUND")
            return sessions[session_id]

    @app.get("/health", response_model=HealthState)
    def health():
        return {"status": "ready", "api_version": "v1", "model_id": app.state.predictor.selected["model"],
                "session_count": len(sessions)}

    @app.post("/v1/sessions", status_code=201, response_model=SessionState)
    def create(body: CreateSession):
        with registry_lock:
            if len(sessions) >= 100:
                raise HTTPException(429, "SESSION_LIMIT_REACHED")
            session_id = str(uuid.uuid4())
            session = ExerciseSession(app.state.predictor, session_id, body.target_reps, body.exercise)
            sessions[session_id], caches[session_id] = session, OrderedDict()
            return session.snapshot()

    @app.post("/v1/sessions/{session_id}/frames", response_model=BatchResult)
    def frames(session_id: str, body: FrameBatch):
        session = get_session(session_id)
        digest = hashlib.sha256(body.model_dump_json().encode()).hexdigest()
        with session.lock:
            cache = caches.get(session_id)
            if cache is None:
                raise HTTPException(404, "SESSION_NOT_FOUND")
            if body.request_id in cache:
                old_digest, old_result = cache[body.request_id]
                if digest != old_digest:
                    raise HTTPException(409, "REQUEST_ID_CONFLICT")
                return old_result
            if session.status != "running":
                raise HTTPException(409, "SESSION_NOT_RUNNING")
            if session.last_timestamp is not None and body.frames[0].timestamp_ms <= session.last_timestamp:
                raise HTTPException(409, "TIMESTAMP_NOT_INCREASING")
            before = session.counter.count
            for frame in body.frames:
                session.update(frame.timestamp_ms, frame.keypoints, frame.person_id)
            result = {**session.snapshot(), "request_id": body.request_id, "accepted_frames": len(body.frames),
                      "new_cycles": session.counter.events[before:]}
            cache[body.request_id] = (digest, result)
            while len(cache) > 128:
                cache.popitem(last=False)
            return result

    @app.get("/v1/sessions/{session_id}", response_model=SessionState)
    def get(session_id: str):
        session = get_session(session_id)
        with session.lock:
            return session.snapshot()

    @app.post("/v1/sessions/{session_id}/control", response_model=SessionState)
    def control(session_id: str, body: Control):
        session = get_session(session_id)
        try:
            return session.control(body.action)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete(session_id: str):
        session = get_session(session_id)
        with session.lock, registry_lock:
            session.control("end")
            sessions.pop(session_id, None)
            caches.pop(session_id, None)

    return app


app = create_app()
