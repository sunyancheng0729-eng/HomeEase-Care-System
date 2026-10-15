# Third-party components

| Component | Licence | Where |
|---|---|---|
| Silero VAD (`silero-vad` package) | MIT | installed by pip |
| YAMNet (Google), ONNX conversion by audiomagic | Apache-2.0 | `models/audio/yamnet/yamnet.onnx` |
| Whisper tiny, CTranslate2 conversion by Systran (`faster-whisper`) | MIT | `models/audio/faster-whisper-tiny/` |
| WeSpeaker ResNet34-LM (VoxCeleb) | CC BY 4.0 | `models/audio/wespeaker/voxceleb_resnet34_LM.onnx` |
| faster-whisper, onnxruntime, scikit-learn, FastAPI, ... | their own licences | installed by pip |

The sound classifier head (`models/audio_sound/sound_events/v2/`) was trained for this project on ESC-50 (CC BY-NC 3.0), VIVAE
(CC BY-NC 4.0), FSD50K clips (per clip CC0 / CC BY / CC BY-NC) and Mini LibriSpeech (CC BY 4.0); see `docs/DATA_CARD.md`.
The demonstration scenes in `samples/` mix clips of those corpora and inherit their licences (non-commercial research use).
The short replies in `samples/` were synthesised with the operating-system voices and contain no third-party recordings.
