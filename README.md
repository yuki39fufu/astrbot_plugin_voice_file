# astrbot_plugin_voice_file

让她用 AI 语音（Edge-TTS）说话，并能读文本文件。

## 语音识别（她听语音）

本地 SenseVoice，用 `sherpa-onnx` 运行时，不需要 torch，也不需要 API key。

- 模型目录：留空时使用 `{插件数据目录}/models/sherpa-onnx-sense-voice-…`，
  也可以填绝对路径（例如 `D:\AI\Models\...` 或 `/opt/models/...`）
  （`model.int8.onnx` + `tokens.txt`，约 239MB）
- 收到语音消息时，插件在 `on_llm_request` 里把 WAV 转写成文字，以
  `[用户发来一条语音，转写内容：…]` 注入上下文
- 首次识别会加载模型（约 1~2 秒、常驻内存 ~300MB），之后每条 1 秒内
- 转写结果按文件缓存，避免同一轮工具调用里重复识别

> AstrBot 自带的 STT 管线保持关闭即可，两边同时开会重复识别。

## 前置（都是可选的）

| 依赖 | 用在哪里 | 不装会怎样 |
|---|---|---|
| `edge-tts` | 语音合成 | 插件照常加载，`/语音` 与 `voice_say` 不可用（建议关闭 `enable_edge_tts`） |
| `sherpa-onnx` + 模型 | 语音识别 | 插件照常加载，收到语音时跳过转写（可关闭 `stt_enable`） |
| `ffmpeg` | 音频格式转换 | 语音合成/识别不可用，文件读取照常 |

系统里 `ffmpeg` 不在 PATH 时，在配置 `ffmpeg_path` 填绝对路径（如 `/usr/bin/ffmpeg`）。

## 跨平台

插件本身不依赖 Windows：

- **Edge-TTS 可关闭**：配置 `enable_edge_tts=false`，插件仍会加载，语音识别和文件读取不受影响
- **语音识别可关闭**：`stt_enable=false`，不装 `sherpa-onnx` 也能用
- **RVC 默认关闭**：`rvc_enable=false`，它是 Windows + 独立 RVC 运行时的方案，Linux/Arch 不需要它
- 模型目录、RVC 目录都支持绝对路径；留空时会落到插件数据目录或直接报清晰错误，不会导致插件加载失败


## 语音（她说话）

| 方式 | 用法 |
|---|---|
| 指令 | `/语音 <文本>` |
| LLM 工具 | `voice_say(text)`，模型自己决定要不要用声音 |

Edge-TTS 生成基础语音，然后（可选）用 RVC 模型把音色换成你的声音库。
音色在配置 `voice_name` 里改（如 `zh-CN-YunxiNeural`）。

### RVC 变声（Windows，默认关闭）

`D:\vtuber project` 里的是 RVC 变声模型（`.pth` + `.index`），不是 TTS 模型，
所以流程是「Edge-TTS 出声 → RVC 换音色」。直接复用 RVC 目录自带的 Python 3.9
运行时（含 torch + CUDA），不需要往 AstrBot 里装 torch。

- 常驻 worker：首次启动加载模型约 6 秒，之后每句 0.4~3 秒（RTX 3050）
- 配置：`rvc_enable`（默认 false）、`rvc_dir`、`rvc_weight_root`、`rvc_model`、
  `rvc_index`、`rvc_pitch`、`rvc_f0_method`、`rvc_index_rate`、`rvc_protect`
- 路径留空或平台不支持时会返回明确错误，不影响插件加载
- 想听不同模型的区别：`D:\AI\voice_samples` 里有每个模型的 3 秒样音

## 文件（她看文件）

| 方式 | 说明 |
|---|---|
| 聊天附件 | 你在 QQ 里直接发 `.txt/.md/.csv/.json` 等文件给她，内容会自动放进本轮上下文 |
| `/读文件 <路径>` | 读本机文本文件，管理员限定 |
| LLM 工具 | `read_file(path)`，管理员限定 |

限制：只读文本类后缀（可配置），单文件默认 256KB，注入上下文默认最多 4000 字。

## 配置

| 项 | 默认 | 说明 |
|---|---|---|
| `enable_voice` | `true` | 语音发送总开关 |
| `enable_edge_tts` | `true` | Edge-TTS 合成开关；装不了 edge-tts 的平台请关掉 |
| `voice_name` | `zh-CN-XiaoxiaoNeural` | Edge-TTS 音色 |
| `voice_rate` | 空 | 语速，如 `+10%` |
| `voice_max_chars` | `300` | 单条语音最大字数 |
| `stt_enable` | `true` | 本地 SenseVoice 语音识别开关 |
| `stt_model_dir` | 空 | 留空用插件数据目录下的模型 |
| `rvc_enable` | `false` | RVC 变声开关（Windows） |
| `enable_read_file_tool` | `true` | 是否允许读本机文件 |
| `attachment_auto_read` | `true` | 是否自动读聊天附件 |
| `file_extensions` | 常见文本后缀 | 允许的后缀 |
| `file_max_kb` | `256` | 单文件大小上限 |
| `attachment_max_chars` | `4000` | 注入上下文字符上限 |

## 语音识别说明

本插件自己实现语音识别（`on_llm_request` 里转写并注入上下文），AstrBot 自带的
STT 管线保持关闭即可，两边同时开会重复识别。
