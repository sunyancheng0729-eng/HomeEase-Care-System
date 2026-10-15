# 算法 3 交接文档:语音交互与声音监测

负责人:林锴(lk,GitHub ALTERLK)。对应提案 3.3 节 Function 3 Algorithm 和图 3-3。
代码来源:HomeEase 仓库 `main` 的提交 `a90b2862`(2026-10-13 合并的 PR #7),见 `VERSION.txt`。

## 0. 一页概览

算法 3 做三件事,全部在本机运行,不调用任何云服务,音频只在内存里处理:

| 能力 | 输入 | 输出 | 在系统里的作用 |
|---|---|---|---|
| 声音事件监测 | 一段连续音频 | HELP / IMPACT / ABNORMAL / DAILY / UNKNOWN 标签(时间、置信度、电平) | 只是提示,**不会报警、也不会取消报警** |
| 喊救命检测 | 连续音频里切出的说话片段 | "报警"或被忽略的原因 | 中文或英文的明确求救(救命、help me)直接报紧急 |
| 询问与回复识别 | 摔倒候选之后问"请问你怎么啦",录下回答 | 回复意图 HELP / REASSURANCE / OTHER / UNCERTAIN,以及是否有效取消 | 没有有效取消就把 PENDING 升级为 EMERGENCY;有效取消只停止升级 |

另有一条规则:固定 10 秒窗口内,加速度、视觉、音频三个通道里有 1 / 2 / 3 个给出报警标记,事件标签分别是 `single_channel` / `dual_channel` / `triple_channel`。

**怎么跑起来**:解压后双击 `start.bat`。第一次会在文件夹里建 `.venv` 并联网安装依赖(几分钟),之后几秒就能启动,浏览器会自动打开演示页。页面上有四张卡片,每张都带示例音频,也可以上传文件或用浏览器录音 8 秒。

## 1. 快速开始

### 1.1 双击运行(Windows)

1. 解压整个 zip(路径里最好不要有中文和空格以外的特殊字符)。
2. 双击 `start.bat`。需要 Python 3.12 和网络(只在第一次需要)。
3. 看到 "Algorithm 3 demo: http://127.0.0.1:87xx/" 后浏览器会自动打开。关闭黑色窗口即停止。

`start.bat` 的顺序:已有 `.venv\ready.txt` 就直接用;否则找 Python 3.12;如果这台电脑的 Python 已经装好了全部依赖就直接用它;否则建 `.venv` 并安装 `requirements.txt`。

### 1.2 命令行

```
set PYTHONPATH=src
python -m algorithm3                  # 打开演示页,默认端口 8730
python -m algorithm3 --vad energy     # 不装 torch 时强制用能量检测
python -m pytest -q                   # 运行包内测试(需要 requirements-dev.txt)
```

### 1.3 在你自己的代码里用

```python
import sys; sys.path.insert(0, "src")          # 或者设置 PYTHONPATH
from algorithm3 import Algorithm3, inquiry_outcome, alarm_tag
from homeease.ml.audio.features import load_audio   # 读 wav,转成 16 kHz 单声道

a3 = Algorithm3()                                    # 模型第一次用到时才加载
x = load_audio("samples/scene_fall_help_call.wav")

a3.label_sounds(x, 16000)        # -> {"events": [...], "speech": [...], ...}
a3.detect_spoken_help(x, 16000)  # -> {"help": True, "segments": [{"verdict": "help", ...}]}
rec = a3.classify_reply(audio=x[9*16000:14*16000], sr=16000)   # 或 text="I'm fine"
inquiry_outcome({**rec, "prompt_played": True}, decision_level="PENDING")   # -> 是否升级
alarm_tag([{"channel": "vision", "at_s": 0}, {"channel": "audio", "at_s": 3}])  # -> dual_channel
```

### 1.4 常见问题

| 现象 | 处理 |
|---|---|
| 提示找不到 Python 3.12 | 安装 Python 3.12(python.org),勾选 Add to PATH;已有 3.12 在别处(例如 conda 环境)时,在命令行里先 `set PYTHON312=C:\路径\python.exe` 再运行 `start.bat` |
| pip 很慢或失败 | 按窗口里的提示换清华镜像,然后重新双击 |
| 页面顶部显示"语音门限:energy(silero unavailable)" | 没装 torch,功能可用,只是语音门限换成能量检测(评估里 Silero 的精确率更高) |
| 浏览器录音没反应 | 浏览器要允许使用麦克风;也可以上传 wav 文件 |
| 8730 端口被占用 | 程序会自动换下一个端口,以窗口里打印的地址为准 |

## 2. 目录说明

```
algorithm3_handover/
  start.bat  requirements.txt  requirements-dev.txt  pytest.ini  pyproject.toml  VERSION.txt
  README.txt                      三行说明
  docs/HANDOVER.md                本文件
  docs/DATA_CARD.md               数据来源与划分
  THIRD_PARTY_NOTICES.md          预训练模型和数据的许可
  config/default.yaml             HomeEase 的完整默认配置,算法 3 用其中的 audio 和 privacy 段
  src/homeease/                   从 HomeEase 仓库原样裁剪(包路径不变)
    core/{paths,util,clock,config}.py
    ml/registry.py
    ml/audio/*.py                 算法 3 的全部算法代码(见第 3 节)
    backend/services/alarm_tags.py   10 秒标签规则(纯函数,和节点里是同一个文件)
  src/algorithm3/                 新增的外壳,仓库里没有
    api.py                        Algorithm3 类:对外的 4 个调用
    policy.py                     询问结果和标签的纯函数(镜像节点里的 events.py)
    decode.py                     任意音频文件 -> 16 kHz 单声道
    webdemo.py + static/index.html  本地演示页
  models/                         权重(第 7 节)
  samples/                        6 段示例场景和 10 段示例回复(index.json 有说明)
  tests/                          包内测试
```

`src/homeease` 里的文件和 HomeEase 仓库 `main` 完全一致,没有改过一行。需要和仓库保持同步时直接用仓库版本替换即可。

## 3. 算法组成与数据流

```
麦克风 16 kHz 单声道 (内存环形缓冲 15 s)
  -> 前端 ActivityGate (ml/audio/pipeline.py)
       语音门限 SpeechGate (Silero VAD)         -> 说话片段 SpeechSegment  -> 语音分支
       起始门限 SoundTrigger + SoundGate        -> 2 s 窗口                -> 声音分支

声音分支: 窗口 -> RMS 归一化 -> YAMNet 嵌入 (2048 维) -> 逻辑回归 -> 每类阈值 -> 标签或 UNKNOWN
          -> SoundMonitor (连续 2 个窗口同标签才成事件, 20 s 内合并)        (monitor.py, sound_package.py, embed.py)

语音分支: 说话片段 -> Whisper tiny 识别 -> 回复规则 v2 -> HELP / REASSURANCE / OTHER / UNCERTAIN   (asr.py, intent_v2.py)
   a) 喊救命监听 HelpListener:0.3 到 6 s、置信度 >= -1.0、是一句明确的求救话、30 s 冷却      (help_listener.py)
   b) 询问 AudioInquiryService:问"请问你怎么啦",10 s 窗口,收回答                        (inquiry.py,需要节点运行时)
   c) 声纹 CancelGate:登记了声纹才有 speaker_verified,只是证据                           (cancel_gate.py, speaker.py)

决策: 询问结果 -> inquiry_outcome (policy.py) -> 音频标记 + 是否升级
      传感器标记 + 音频标记 -> alarm_tag (alarm_tags.py) -> 10 s 窗口内的通道数 -> 标签
```

### 3.1 文件与作用

| 文件 | 作用 |
|---|---|
| `ml/audio/pipeline.py` | `Function3Pipeline`:环形缓冲、两个门限、分支调度;`SpeechSegment` |
| `ml/audio/vad.py` | `EnergyVAD`、`WebRTCVAD`、`SileroVAD`;`create(name)` |
| `ml/audio/monitor.py` | `SoundMonitor`:窗口打分、确认、合并、事件;`SoundEvent`(只有标量,没有音频) |
| `ml/audio/sound_package.py` | 声音模型包的加载和校验(`load_sound_classifier`) |
| `ml/audio/embed.py` | YAMNet ONNX 嵌入 |
| `ml/audio/asr.py` | `WhisperASR`(进程内)和 `ASRProcess`(独立进程,带超时) |
| `ml/audio/intent_v2.py` | `RuleIntentV2` 回复规则:繁简转换、否定范围、求救线索表 |
| `ml/audio/help_listener.py` | `HelpListener`:喊救命监听的全部判断 |
| `ml/audio/cancel_gate.py`, `speaker.py` | 声纹登记、`CancelGate.evaluate`、WeSpeaker 嵌入 |
| `ml/audio/inquiry.py` | `AudioInquiryService`:提示语、回复窗口、写回复记录(依赖节点的 runtime 对象,演示页不用它) |
| `ml/audio/live.py` | `LiveSoundMonitor`:实时麦克风线程和健康状态(依赖数据库层,演示页不用它) |
| `ml/audio/tts.py` | 语音提示(Windows SAPI / macOS say) |
| `backend/services/alarm_tags.py` | 10 秒窗口和通道标签 |

## 4. 对外接口(`algorithm3`)

### 4.1 `Algorithm3(vad="auto", overrides=None)`

`vad`:`auto`(装了 torch 用 Silero,否则能量检测)、`silero`、`energy`。`overrides` 是配置的嵌套字典,例如 `{"audio": {"help_listener": {"cooldown_s": 60}}}`。一次只处理一个调用(内部有锁)。

| 方法 | 参数 | 返回 |
|---|---|---|
| `label_sounds(samples, sr=16000)` | 一维 float 或整型数组 | `events`: 类别、置信度、`start_s`、`end_s`、电平、窗口数;`speech`: 说话片段的时间;`vad`、`status`、`raw_audio_stored: False` |
| `classify_reply(audio=None, sr=16000, text=None, enrollment=None)` | 回答窗口的录音,或直接给文字 | 回复记录:`status`(RESPONDED / NO_RESPONSE / audio_unavailable)、`intent`、`intent_reason`、`cues`、`transcript`、`asr_avg_logprob`、`cancel_eligible`、`cancel_reasons`、`trusted_reassurance` |
| `detect_spoken_help(samples, sr=16000)` | 连续音频 | `segments`: 每个说话片段的 `verdict`(`help` 或被忽略的原因);`help`: 是否有报警 |
| `inquiry_outcome(rec, ...)` | 回复记录加 `prompt_played` | 见 4.2 |
| `alarm_tag(markers)` | 标记列表 | `tag`、`channels`、带 `counted` 的 `markers` |
| `warm_up()`, `info()` | | 加载全部模型;当前使用的 VAD、模型、关键参数 |

`detect_spoken_help` 的 `verdict` 取值:`help`(报警);被忽略的原因有 `too_short`、`too_long`、`cooldown`、`low_confidence`、`no_text`、`not_help`、`not_a_call`、`too_many_words`;状态类的有 `disabled`、`no_consent`、`inquiry_listening`(这段话属于正在进行的询问,由询问流程处理)、`asr_unavailable`、`asr_error`、`stopped`、`callback_error`。

### 4.2 `inquiry_outcome(rec, *, decision_level="PENDING", resolved=False, button_ok=False, escalate_on_no_cancel=True, min_asr_logprob=-1.0)`

纯函数,没有数据库和时钟,镜像节点里 `EventService._voice_outcome` 和 `_voice_cancel_verdict`。返回:

| 字段 | 含义 |
|---|---|
| `state` | `cancelled` / `no_cancel` / `not_run` / `audio_unavailable` |
| `cancelled`, `by` | 有效取消,以及是 `voice`(可信的"我没事")还是 `button`(老人页按钮) |
| `marker` | 要记的音频标记:`inquiry_no_cancel` 或 `spoken_help`,没有则 `None` |
| `escalate` | 是否把 PENDING 升级为 EMERGENCY |
| `notify` | 事件已经是 EMERGENCY 时又收到求救,是否再通知照护者 |
| `call` | 事件已被关闭时的求救:单独成为一个报警 |

## 5. 规则(这是算法 3 的核心约定)

**一条总则:回复永远不会关闭或降级一个事件。**

| 询问的结果 | 动作 |
|---|---|
| 可信的"我没事"(意图 REASSURANCE,识别置信度够,登记了声纹时还要匹配),或老人页的"我没事"按钮 | 只停止升级;事件保持 PENDING,由照护者关闭;不记音频标记 |
| 沉默、不清楚或无关的话、说了求救的话、不可信的"我没事" | 记音频标记,PENDING 升级为 EMERGENCY(音频是第二个通道) |
| "我需要帮助"按钮 | 立刻 EMERGENCY,和求助按钮一样 |
| 音频关闭、没有同意、麦克风或识别器故障、问题没有传达 | 不记标记、不升级,界面显示为降级,**绝不当作安全** |
| 说了求救的话,但事件已经是 EMERGENCY | 不再升级,但通知照护者 |

**10 秒通道标签**(`alarm_tags.py`,窗口固定为 10 秒,不可配置):

- 通道 accel、vision、audio 计数;`manual`(求助按钮)只记录不计数。
- 窗口从最早的计数标记开始;窗口内不同通道的个数 1 / 2 / 3 对应 single / dual / triple。
- 窗口之外的标记保留(`counted: false`),不改标签;所以事件级别可以是 EMERGENCY 而标签仍是 `single_channel`。
- 音频标记的时间是"问题被问出的时刻",和传感器用同一个时间基准。
- 声音事件标签(尖叫、撞击等)**不会**变成标记。

## 6. 参数与阈值

配置在 `config/default.yaml` 的 `audio:` 段(模型定义在 `homeease/core/config.py`)。演示页以 `Algorithm3(overrides=...)` 的方式覆盖。

| 项 | 默认 | 说明 |
|---|---|---|
| `audio.vad` | silero | 语音门限;可选 energy、webrtc |
| `audio.buffer_seconds` | 15 | 内存环形缓冲 |
| `audio.gate.vad_window_s / vad_hop_s` | 1.5 / 0.25 | 语音门限重算窗口 |
| `audio.gate.speech_min_s / speech_end_silence_s / speech_max_s` | 0.25 / 0.6 / 12 | 切说话片段 |
| `audio.gate.onset_db / min_level_dbfs / hold_s / periodic_s` | 12 / -60 / 0.3 / 60 | 声音分支的起始门限 |
| `audio.sound_monitor.window_s / hop_s / confirm_windows / cooldown_s` | 2.0 / 0.5 / 2 / 20 | 声音事件 |
| 每类阈值(在模型包 `meta.json` 里) | HELP 0.42, IMPACT 0.99, ABNORMAL 0.91, DAILY 0.52 | 低于阈值记 UNKNOWN |
| `audio.help_listener.min_utterance_s / max_utterance_s` | 0.3 / 6.0 | 喊救命监听的时长门限 |
| `audio.help_listener.cooldown_s / min_asr_logprob` | 30 / -1.0 | 冷却;识别置信度 |
| `audio.inquiry_response_window_s` | 10 | 回答窗口 |
| `audio.escalate_on_no_cancel` | true | 没有有效取消就升级 |
| `audio.voiceprint.threshold` | null | **默认没有阈值,所以没有任何回复会被判为 cancel_eligible**;要用必须先用真实声音标定 |
| `audio.voiceprint.min_speech_s / min_asr_logprob` | 1.0 / -1.0 | 声纹门限 |
| `audio.asr_model`, `audio.intent_method` | tiny, rule_v2 | |
| `audio.auto_cancel_emergency` | false | **配置校验器禁止设为 true**;`privacy.persist_raw_audio` 同样不能打开 |

这些值都没有用家庭录音调过;连续音频上的误报率没有测过。

## 7. 模型与权重(`models/`)

| 文件 | 大小 | 来源 | 许可 | 用途 |
|---|---|---|---|---|
| `audio/yamnet/yamnet.onnx` | 16.1 MB | audiomagic/yamnet-onnx @ f25b741c(Google YAMNet) | Apache-2.0 | 声音嵌入 |
| `audio_sound/sound_events/v2/model.joblib` + `meta.json` | 83 KB | 本项目训练(折 1 到 4) | n/a | 嵌入上的逻辑回归和每类阈值 |
| `audio/faster-whisper-tiny/` | 75.5 MB | Systran/faster-whisper-tiny @ d90ca5fe | MIT | 语音识别 |
| `audio/wespeaker/voxceleb_resnet34_LM.onnx` | 26.5 MB | Wespeaker/wespeaker-voxceleb-resnet34-LM @ f0c48c29 | CC BY 4.0 | 声纹嵌入 |
| Silero VAD | 在 pip 包 `silero-vad` 里 | snakers4/silero-vad | MIT | 语音门限 |

校验值见 `models/MODELS.json`。声音模型包由 `sound_package.load_sound_classifier` 校验 sha256,并检查 YAMNet 文件和训练时是同一个。

## 8. 评估结果摘要

**所有数字都来自干净的、朗读的、合成的或表演的音频,多数是英文。没有老年人、没有真实电视、没有真实住宅的数据。** 完整的实验和表格在最终报告第 8.4 节。

| 实验 | 结果 |
|---|---|
| A1 语音门限(45 个构造混合,113,863 帧) | Silero 精确率 0.995 / 召回 0.813 / F1 0.895;能量检测 0.871 / 0.946 / 0.907;WebRTC 0.869 / 0.931 / 0.899。有撞击声落在说话间隙时,能量和 WebRTC 的精确率降到 0.81,Silero 保持 0.988 |
| A2 声音事件(四类,2,382 个窗口,五折分组) | YAMNet 嵌入加逻辑回归 macro-F1 0.677(区间 0.633 到 0.740);HELP 召回 0.688、精确率 0.494 |
| A2b 随包模型(折 5,466 个窗口) | macro-F1 0.658;HELP 召回 0.610、精确率 0.575(HELP 占 17.6% 时;占 1% 时精确率约 6%);ABNORMAL 召回 0.408;77% 的窗口得到类别 |
| A3b 回复识别(测试集 196 条,38 条本人求救) | 规则 v1 macro-F1 0.480、漏 23 条;规则 v2 0.894 [0.82, 0.94]、漏 5 条。**这是上界**:词表是在看过两个划分的情况下写的,另外写的 20 条没见过的求救话只认出 6 条 |
| A4 说话人验证(LibriSpeech) | VoxCeleb 模型 EER 0.007(干净)、0.022(5 dB 噪声)、0.022(回放模拟) |
| A6 误取消(150+150 个合成说话人,2,370 条评估) | 严格阈值下,别人说同样的话被接受 0.05%,访客 0.09%,电视 0.00%;但真实用户只有 41.7% 被接受(5 dB 噪声下 21.3%);回放录音被接受 5.3% |
| 自动测试 | 包内 2,200 多个测试通过(含参数化);节点侧另有 132 个新增测试 |

A6 的含义:**只看文字,别人或电视说"我没事"有 78% 到 89% 会被当成取消;声纹能压到 0.1% 以下,但会拒掉一半以上的真实用户,而且挡不住回放。** 所以声纹在系统里只是证据,出厂没有登记流程,也没有阈值。

## 9. 隐私和一致性约定

- 音频只存在于:15 s 环形缓冲、传给下一个组件的说话片段、一次询问的回复缓冲;每条路径结束都会清零。不写文件,不写日志。
- 只存标签、时间和统计量。识别出来的文字不存(`privacy.persist_voice_transcript` 默认 false);事件记录里只有意图、线索类别、识别置信度和时长。
- 声纹是生物特征,不落盘;登记对象只存在内存里。
- 麦克风使用以同意为前提;撤回同意要立即停止监听并清空缓冲。
- 不可用不等于安全:模型缺失、麦克风故障、识别器故障、问题没有传达,界面必须显示为降级。
- 配置校验器禁止打开 `audio.auto_cancel_emergency` 和 `privacy.persist_raw_audio`。

## 10. 集成到新系统:你需要自己做的事

算法包只负责"判断"。下面这些在 HomeEase 节点里由别的模块完成,新系统需要自己提供:

| 事项 | HomeEase 节点里怎么做 | 新系统需要 |
|---|---|---|
| 采集 | `ml/audio/live.py`:sounddevice 输入流,16 kHz 单声道 0.1 s 块,健康检查和重连 | 自己的采集线程把块送进 `Function3Pipeline.process_chunk(x, t_mono_ms)`(`algorithm3/api.py` 里是离线整段的用法)。`LiveSoundMonitor` 构造时需要节点的 `db` 对象,换了存储就不要直接用它,把它当作采集、重连和健康检查的参考 |
| 语音提示 | `tts.py` 加浏览器 speechSynthesis;老人页弹出带倒计时的对话框 | 问"请问你怎么啦 / What's wrong? Are you OK?",并记录提示是否真的送达(`prompt_played`) |
| 同意 | 设置页的音频同意开关,每次读取 | 提供 `consent()` 回调,默认应当是"没有同意" |
| 事件库 | `EventService`:PENDING / EMERGENCY、标记列表、通知照护者 | 保存事件级别和 `alarm_markers`;把询问结果交给 `inquiry_outcome`,按返回值升级和通知 |
| 传感器标记 | `_mark_candidates`:加速度或视觉判为候选时写标记 | 在候选出现时调用 `alarm_tag` 的输入 `{"channel": "accel" 或 "vision", ...}` |
| 降级显示 | 音频健康单独成为一个通道,不影响跌倒覆盖状态 | 显示 `degraded:*` 状态 |

如果新系统也是 FastAPI + SQLite,可以直接把 `webdemo.py` 的四个接口当作参考:`/api/sounds`、`/api/reply`、`/api/help`、`/api/alarm`。

## 11. 已知问题与没有验证的事

- 没有在真实麦克风、扬声器和电视环境里测过;识别延迟(本机 Whisper tiny 中位数约 0.6 秒,p95 约 1.2 秒)和电视的误报率没有测。
- HELP 声音类很弱,只能当提示;喊救命靠的是识别出来的文字。
- 回复规则的准确率是上界;新措辞下的真实数字未知,很可能更低。普通话的漏报比英文多(19 条里漏 4 条,英文漏 1 条)。
- 语音提示的普通话访客和电视声音是用重采样模拟的,不是真实的第二个说话人。
- 声纹登记没有产品流程;阈值为空,所以默认没有任何回复是 cancel_eligible。
- 一个事件的 10 秒窗口固定,晚于首个标记 10 秒以上才开始的询问会被记录但不计数,可能出现事件级别 EMERGENCY 而标签 `single_channel`。
- Windows 上用 Git 检出仓库时换行符转换会让模型校验失败;本包是 zip,没有这个问题。
- `homeease.ml.audio.tts` 在 Windows 用 SAPI(PowerShell),macOS 用 say;只在 Windows 上实测过。

## 12. 版本

| 项 | 值 |
|---|---|
| 来源 | HomeEase `main` @ a90b2862(PR #7 合并) |
| 日期 | 2026-10-14 |
| Python / 平台 | 3.12 / Windows 11 |
| 包内测试 | 见 `pytest -q` 输出 |
