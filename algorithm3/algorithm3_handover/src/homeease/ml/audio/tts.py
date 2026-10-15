"""Text-to-speech abstraction: macOS ``say``, Windows SAPI (System.Speech) or nothing.

Used for the spoken inquiry (`AudioInquiryService._speak`) and to render the reply phrases of
experiment A3. No extra pip packages and no network: SAPI is driven through Windows PowerShell,
``say`` through the macOS command line.

Security: the text to speak and the voice name are DATA. They never become part of a command string.
* SAPI: one constant script (passed with -EncodedCommand); the jobs (text, voice, output path) are
  written to a JSON file whose path travels in an environment variable. The script only parses the
  JSON and hands the fields to .NET method arguments, it never evaluates them. A voice name must
  match an installed voice before it is used at all.
* say: the text goes through stdin (``-f -``), so text starting with ``-`` cannot be read as an option.
* subprocess is always called with an argument list (shell=False).

The synthesized prompt is not user audio, but it is still only kept in a temporary directory for the
duration of one call.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import signal as sps

from homeease.core.util import get_logger
from homeease.ml.audio.features import SR, load_audio

log = get_logger(__name__)

MAX_TEXT_CHARS = 600
BACKENDS = ("sapi", "say", "none")
# names accepted in config / by callers: "system" = the platform's native engine ("system_say" is the historic alias)
NATIVE_ALIASES = ("auto", "system", "system_say")
# preferred voice per language when the caller does not name one (short names, matched against the installed voices)
PREFERRED = {"sapi": {"en": ("Zira", "David"), "zh": ("Huihui",)}, "say": {"en": ("Samantha", "Daniel"), "zh": ("Tingting",)}}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class TTSError(RuntimeError):
    """Synthesis or playback failed."""


class TTSUnavailable(TTSError):
    """No usable engine on this machine (or backend ``none``)."""


@dataclass(frozen=True)
class Voice:
    name: str  # engine name, e.g. "Microsoft Zira Desktop" or "Samantha"
    lang: str  # "en" | "zh" | other 2-letter code
    backend: str  # "sapi" | "say"
    gender: str | None = None

    @property
    def short(self) -> str:
        """Human label: "Zira" for "Microsoft Zira Desktop"."""
        toks = [t for t in self.name.split() if t.lower() not in ("microsoft", "desktop")]
        return " ".join(toks) or self.name


def norm_lang(lang: str | None) -> str:
    """"en-US" / "en_GB" / "EN" -> "en"; "zh-CN" / "zh_TW" -> "zh"."""
    return (lang or "en").replace("_", "-").split("-")[0].strip().lower() or "en"


def clean_text(text: str) -> str:
    """Whitespace-normalise and bound the text. Nothing here is a security filter: the text is data end to end."""
    if not isinstance(text, str):
        raise TTSError("text must be a string")
    t = re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()
    t = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    if not t:
        raise TTSError("empty text")
    if len(t) > MAX_TEXT_CHARS:
        raise TTSError(f"text longer than {MAX_TEXT_CHARS} characters")
    return t


def emulate_voice(x: np.ndarray, ratio: float, sr: int = SR) -> np.ndarray:
    """EMULATED second speaker: play ``x`` back ``ratio`` times as fast (resampling changes pitch AND speed together).

    ratio < 1 lowers the pitch and slows the speech, ratio > 1 does the opposite. This is not a real second
    voice (same vocal tract model, formants move with pitch); results must be labelled as emulated.
    """
    if not 0.5 <= ratio <= 2.0:
        raise ValueError("ratio must be within 0.5-2.0")
    f = Fraction(ratio).limit_denominator(100)
    if f == 1:
        return x.astype(np.float32)
    return sps.resample_poly(x.astype(np.float32), f.denominator, f.numerator).astype(np.float32)


# --------------------------------------------------------------------------- backends
class Backend:
    name = "none"

    def available(self) -> bool:
        return False

    def voices(self) -> list[Voice]:
        return []

    def render(self, jobs: Sequence[tuple[str, str]], out_dir: Path) -> list[Path]:
        """Write one 16-bit mono WAV/AIFF per (text, voice name) job and return the paths."""
        raise TTSUnavailable(f"TTS backend {self.name!r} is not available on this machine")

    def speak(self, text: str, voice: str) -> None:
        raise TTSUnavailable(f"TTS backend {self.name!r} is not available on this machine")


class NoneBackend(Backend):
    name = "none"


def _powershell() -> str | None:
    root = os.environ.get("SystemRoot", r"C:\Windows")
    cand = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if cand.is_file():
        return str(cand)
    return shutil.which("powershell.exe") or shutil.which("powershell")


def _encoded(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


# Constant scripts: they contain no caller data. Jobs arrive as JSON in the file named by $env:HOMEEASE_TTS_JOBS.
_SAPI_LIST = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $v = @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object {
    [pscustomobject]@{ name = [string]$_.VoiceInfo.Name; culture = [string]$_.VoiceInfo.Culture.Name; gender = [string]$_.VoiceInfo.Gender } })
  [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
  [Console]::Out.Write((ConvertTo-Json -InputObject $v -Compress))
} finally { $s.Dispose() }
"""
_SAPI_SYNTH = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$jobs = [IO.File]::ReadAllText($env:HOMEEASE_TTS_JOBS, [Text.Encoding]::UTF8) | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  foreach ($j in @($jobs)) {
    $synth.SelectVoice([string]$j.voice)
    $synth.SetOutputToWaveFile([string]$j.out, $fmt)
    $synth.Speak([string]$j.text)
    $synth.SetOutputToNull()
  }
} finally { $synth.Dispose() }
"""


class SapiBackend(Backend):
    """Windows 10/11 SAPI 5 voices (Zira, David, Huihui ...) through System.Speech in Windows PowerShell 5.1."""

    name = "sapi"

    def available(self) -> bool:
        return platform.system() == "Windows" and _powershell() is not None

    def _run(self, script: str, jobs_file: Path | None, timeout: float) -> str:
        ps = _powershell()
        if ps is None:
            raise TTSUnavailable("Windows PowerShell not found")
        env = dict(os.environ)
        if jobs_file is not None:
            env["HOMEEASE_TTS_JOBS"] = str(jobs_file)
        try:
            r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-EncodedCommand", _encoded(script)], capture_output=True, timeout=timeout, env=env, creationflags=_NO_WINDOW, shell=False)
        except (OSError, subprocess.SubprocessError) as e:
            raise TTSError(f"PowerShell call failed: {type(e).__name__}") from e
        if r.returncode != 0:
            err = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise TTSError("SAPI synthesis failed: " + (err[0][:160] if err else f"exit {r.returncode}"))
        return (r.stdout or b"").decode("utf-8-sig", "replace")

    def voices(self) -> list[Voice]:
        out = self._run(_SAPI_LIST, None, 30.0).strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        return [Voice(str(d["name"]), norm_lang(d.get("culture")), "sapi", (d.get("gender") or None)) for d in data if d.get("name")]

    def render(self, jobs: Sequence[tuple[str, str]], out_dir: Path) -> list[Path]:
        outs = [out_dir / f"tts_{i:04d}.wav" for i in range(len(jobs))]
        spec = [{"text": t, "voice": v, "out": str(p)} for (t, v), p in zip(jobs, outs)]
        jf = out_dir / "jobs.json"
        jf.write_text(json.dumps(spec, ensure_ascii=True), encoding="utf-8")  # ASCII-only JSON: no encoding ambiguity for any text
        self._run(_SAPI_SYNTH, jf, 30.0 + 6.0 * len(jobs))
        missing = [p for p in outs if not p.is_file() or p.stat().st_size < 100]
        if missing:
            raise TTSError(f"SAPI produced no audio for {len(missing)} of {len(outs)} jobs")
        return outs

    def speak(self, text: str, voice: str) -> None:
        with tempfile.TemporaryDirectory(prefix="he_tts_") as d:
            wav = self.render([(text, voice)], Path(d))[0]
            data = wav.read_bytes()
            try:
                import winsound

                winsound.PlaySound(data, winsound.SND_MEMORY)  # blocks until the prompt has finished (echo avoidance)
            except Exception:
                import sounddevice as sd

                sd.play(load_audio(str(wav)), SR)
                sd.wait()


_SAY_LINE = re.compile(r"^(?P<name>.+?)\s{2,}(?P<loc>[A-Za-z]{2,3}[_-][A-Za-z0-9]+)\s+#")


class SayBackend(Backend):
    """macOS ``say`` (existing behaviour)."""

    name = "say"

    def available(self) -> bool:
        return platform.system() == "Darwin" and shutil.which("say") is not None

    def voices(self) -> list[Voice]:
        try:
            r = subprocess.run(["say", "-v", "?"], capture_output=True, timeout=15, shell=False)
        except (OSError, subprocess.SubprocessError) as e:
            raise TTSError(f"say -v ? failed: {type(e).__name__}") from e
        out = []
        for line in (r.stdout or b"").decode("utf-8", "replace").splitlines():
            m = _SAY_LINE.match(line.rstrip())
            if m:
                out.append(Voice(m["name"].strip(), norm_lang(m["loc"]), "say"))
        return out

    def render(self, jobs: Sequence[tuple[str, str]], out_dir: Path) -> list[Path]:
        outs = []
        for i, (text, voice) in enumerate(jobs):
            p = out_dir / f"tts_{i:04d}.aiff"
            try:
                r = subprocess.run(["say", "-v", voice, "-o", str(p), "-f", "-"], input=text.encode("utf-8"), capture_output=True, timeout=30, shell=False)
            except (OSError, subprocess.SubprocessError) as e:
                raise TTSError(f"say failed: {type(e).__name__}") from e
            if r.returncode != 0 or not p.is_file():
                raise TTSError(f"say exited with {r.returncode}")
            outs.append(p)
        return outs

    def speak(self, text: str, voice: str) -> None:
        try:
            r = subprocess.run(["say", "-v", voice, "-f", "-"], input=text.encode("utf-8"), capture_output=True, timeout=30, shell=False)
        except (OSError, subprocess.SubprocessError) as e:
            raise TTSError(f"say failed: {type(e).__name__}") from e
        if r.returncode != 0:
            raise TTSError(f"say exited with {r.returncode}")


# --------------------------------------------------------------------------- selection
_VOICE_CACHE: dict[str, list[Voice]] = {}


def clear_caches() -> None:
    _VOICE_CACHE.clear()


def native_backend_name() -> str:
    system = platform.system()
    if system == "Windows" and SapiBackend().available():
        return "sapi"
    if system == "Darwin" and SayBackend().available():
        return "say"
    return "none"


def get_backend(name: str = "auto") -> Backend:
    """``auto`` / ``system`` / ``system_say``: the platform's native engine. ``sapi`` / ``say`` / ``none`` are explicit."""
    key = (name or "auto").strip().lower()
    if key in NATIVE_ALIASES:
        key = native_backend_name()
    if key == "sapi":
        return SapiBackend()
    if key == "say":
        return SayBackend()
    if key == "none":
        return NoneBackend()
    raise ValueError(f"unknown TTS backend {name!r}; use one of {NATIVE_ALIASES + BACKENDS}")


def available_voices(backend: str = "auto") -> list[Voice]:
    """Installed voices of the backend ([] when unavailable). The list is cached per process."""
    be = get_backend(backend)
    if not be.available():
        return []
    if be.name not in _VOICE_CACHE:
        try:
            _VOICE_CACHE[be.name] = be.voices()
        except TTSError as e:
            log.warning("listing %s voices failed: %s", be.name, e)
            return []
    return list(_VOICE_CACHE[be.name])


def resolve_voice(voice: str | None, lang: str, backend: str = "auto") -> Voice:
    """Map a requested name ("Zira", "Microsoft Zira Desktop") or a language to an INSTALLED voice.

    A name that is not installed is rejected here, before anything reaches an external program."""
    be = get_backend(backend)
    if not be.available():
        raise TTSUnavailable(f"TTS backend {be.name!r} is not available on this machine")
    installed = available_voices(be.name)
    if voice:
        v = voice.strip().lower()
        for cand in installed:
            if cand.name.lower() == v:
                return cand
        for cand in installed:
            if v in (t.lower() for t in cand.name.replace("(", " ").replace(")", " ").split()) or cand.short.lower() == v:
                return cand
        raise TTSError(f"voice {voice[:60]!r} is not installed for backend {be.name}")
    lg = norm_lang(lang)
    for pref in PREFERRED.get(be.name, {}).get(lg, ()):
        for cand in installed:
            if cand.lang == lg and pref.lower() in cand.name.lower():
                return cand
    for cand in installed:
        if cand.lang == lg:
            return cand
    raise TTSError(f"no {lg!r} voice installed for backend {be.name}")


def synthesize_many(items: Sequence[tuple[str, str | None, str]], backend: str = "auto") -> list[np.ndarray]:
    """Render (text, voice, lang) items in ONE engine call (SAPI start-up costs about a second). 16 kHz float32 each."""
    be = get_backend(backend)
    resolved = [(clean_text(t), resolve_voice(v, lg, be.name).name) for t, v, lg in items]
    if not resolved:
        return []
    with tempfile.TemporaryDirectory(prefix="he_tts_") as d:
        files = be.render(resolved, Path(d))
        return [load_audio(str(p), SR) for p in files]


def synthesize(text: str, voice: str | None = None, lang: str = "en", backend: str = "auto") -> np.ndarray:
    """Speak ``text`` into memory: mono float32 at 16 kHz. Raises TTSError / TTSUnavailable."""
    return synthesize_many([(text, voice, lang)], backend)[0]


def play(text: str, lang: str = "en", voice: str | None = None, backend: str = "auto") -> bool:
    """Speak ``text`` on the default output and return when it has finished. False on any failure (never raises):
    the caller (voice inquiry) must treat a failed prompt as a degraded condition, not as a response."""
    try:
        be = get_backend(backend)
        if not be.available():
            return False
        be.speak(clean_text(text), resolve_voice(voice, lang, be.name).name)
        return True
    except TTSError as e:
        log.warning("TTS playback failed: %s", e)
        return False
    except Exception:  # pragma: no cover - audio device / platform specific
        log.exception("TTS playback failed")
        return False
