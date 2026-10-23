"""语音发送 + 文本文件读取插件。

- 让她用 AI 语音（Edge-TTS）说话：/语音 <文本>、LLM 工具 voice_say
- 读文本文件：聊天里收到的 txt/md/csv/json 等附件自动注入上下文；
  管理员也可以用 /读文件 <路径> 或 LLM 工具 read_file 读本机文本文件

语音识别（听语音）不在这里实现：AstrBot 自带 STT 管线（preprocess_stage），
开启 provider_stt_settings 并配置一个语音识别 Provider 即可。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import File, Record, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.message.message_event_result import MessageChain

DEFAULT_EXTS = [
    ".txt", ".md", ".markdown", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".log", ".ini", ".conf",
    ".py", ".js", ".ts", ".html", ".css", ".xml",
]

VOICE_COMMANDS = ("语音", "发语音")
READ_COMMANDS = ("读文件", "查看文件")
HELP_COMMANDS = ("语音帮助", "文件帮助")

STT_MODEL_DIRNAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


class VoiceFileError(Exception):
    """可直接展示给用户的错误。"""


class VoiceFilePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_voice_file")
        self.tts_dir = Path(tempfile.gettempdir()) / "astrbot_voice_file"
        self._recognizer = None
        self._stt_lock = asyncio.Lock()
        self._transcript_cache: dict[tuple[str, float], str] = {}
        self._rvc_proc = None
        self._rvc_lock = asyncio.Lock()

    async def initialize(self) -> None:
        edge_ok = self._edge_tts_available()
        if not edge_ok and self._conf_bool("enable_edge_tts", True):
            logger.warning(
                "[voice_file] 未检测到 edge-tts，语音合成不可用；"
                "如果这台机器装不了它，请在插件配置里关闭 enable_edge_tts。",
            )
        if (
            self._conf_bool("enabled", True)
            and self._conf_bool("enable_voice", True)
            and self._conf_bool("enable_edge_tts", True)
            and edge_ok
        ):
            self.context.activate_llm_tool("voice_say")
        else:
            self.context.deactivate_llm_tool("voice_say")
        if self._conf_bool("enable_read_file_tool", True) and self._conf_bool("enabled", True):
            self.context.activate_llm_tool("read_file")
        else:
            self.context.deactivate_llm_tool("read_file")

    @staticmethod
    def _edge_tts_available() -> bool:
        try:
            import edge_tts  # noqa: F401
        except Exception:
            return False
        return True

    async def terminate(self) -> None:
        proc = self._rvc_proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass
        self._rvc_proc = None

    # ------------------------------------------------------------------ 配置

    def _conf(self, key: str, default):
        value = self.conf.get(key, default)
        return default if value is None else value

    def _conf_bool(self, key: str, default: bool) -> bool:
        value = self._conf(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _conf_int(self, key: str, default: int) -> int:
        try:
            return int(self._conf(key, default))
        except (TypeError, ValueError):
            return default

    def _authorized(self, event: AstrMessageEvent) -> bool:
        if event.is_admin():
            return True
        sender = str(event.get_sender_id() or "")
        whitelist = self._conf("extra_whitelist", []) or []
        try:
            return sender in {str(item) for item in whitelist}
        except TypeError:
            return False

    def _strip_command(self, event: AstrMessageEvent, names: tuple[str, ...]) -> str:
        raw = (event.get_message_str() or "").strip()
        for name in names:
            if raw == name:
                return ""
            if raw.startswith(name):
                return raw[len(name):].strip()
        return raw

    # ------------------------------------------------------------------ 语音

    async def _synthesize(self, text: str) -> Path:
        if not self._conf_bool("enable_edge_tts", True):
            raise VoiceFileError("Edge-TTS 已在插件配置中关闭（enable_edge_tts=false）。")
        try:
            import edge_tts
        except Exception as exc:
            raise VoiceFileError(
                "未安装 edge-tts，无法合成语音。可以 pip install edge-tts，"
                "或在插件配置里关闭 enable_edge_tts。",
            ) from exc

        text = (text or "").strip()
        if not text:
            raise VoiceFileError("没有要说的内容。")

        max_chars = self._conf_int("voice_max_chars", 300)
        if max_chars > 0 and len(text) > max_chars:
            raise VoiceFileError(f"文本太长了（{len(text)} 字，上限 {max_chars} 字）。")

        voice = str(self._conf("voice_name", "zh-CN-XiaoxiaoNeural") or "zh-CN-XiaoxiaoNeural")
        rate = str(self._conf("voice_rate", "") or "").strip()

        self.tts_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_files()

        stem = uuid.uuid4().hex
        mp3_path = self.tts_dir / f"{stem}.mp3"

        kwargs = {"text": text, "voice": voice}
        if rate:
            kwargs["rate"] = rate

        try:
            communicate = edge_tts.Communicate(**kwargs)
            await asyncio.wait_for(communicate.save(str(mp3_path)), timeout=45)
        except asyncio.TimeoutError as exc:
            raise VoiceFileError("语音合成超时，检查一下网络。") from exc
        except Exception as exc:
            raise VoiceFileError(f"语音合成失败：{exc}") from exc

        ffmpeg = self._ffmpeg_path()

        src_wav = self.tts_dir / f"{stem}_edge.wav"
        await self._ffmpeg_convert(mp3_path, src_wav, ["-ar", "16000", "-ac", "1"])

        if self._conf_bool("rvc_enable", False):
            rvc_wav = self.tts_dir / f"{stem}_rvc.wav"
            await self._rvc_convert(src_wav, rvc_wav)
            out_wav = self.tts_dir / f"{stem}.wav"
            await self._ffmpeg_convert(rvc_wav, out_wav, ["-ar", "24000", "-ac", "1"])
            for leftover in (mp3_path, src_wav, rvc_wav):
                leftover.unlink(missing_ok=True)
            return out_wav

        out_wav = self.tts_dir / f"{stem}.wav"
        await self._ffmpeg_convert(src_wav, out_wav, ["-ar", "24000", "-ac", "1"])
        for leftover in (mp3_path, src_wav):
            leftover.unlink(missing_ok=True)
        return out_wav

    def _ffmpeg_path(self) -> str:
        ffmpeg = str(self._conf("ffmpeg_path", "ffmpeg") or "ffmpeg")
        if not shutil.which(ffmpeg):
            raise VoiceFileError("找不到 ffmpeg，无法转换语音格式。")
        return ffmpeg

    async def _ffmpeg_convert(self, src: Path, dst: Path, extra: list[str]) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._ffmpeg_path(), "-y", "-i", str(src), *extra,
                "-acodec", "pcm_s16le", str(dst),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError as exc:
            raise VoiceFileError("语音转码超时。") from exc

        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            detail = (stderr or b"").decode("utf-8", "ignore")[-300:]
            raise VoiceFileError(f"语音转码失败：{detail}")

    # ------------------------------------------------------------------ RVC 变声

    def _rvc_settings(self) -> dict:
        raw_dir = str(self._conf("rvc_dir", "") or "").strip()
        if not raw_dir:
            raise VoiceFileError(
                "未配置 rvc_dir，无法使用 RVC 变声（可在配置里关闭 rvc_enable）。",
            )
        rvc_dir = Path(raw_dir)
        model = str(self._conf("rvc_model", "") or "").strip()
        if not model:
            raise VoiceFileError("未配置 rvc_model，无法使用 RVC 变声。")
        python = str(self._conf("rvc_python", "") or "").strip()
        if not python:
            if sys.platform.startswith("win"):
                python = str(rvc_dir / "runtime" / "python.exe")
            else:
                python = str(rvc_dir / "runtime" / "bin" / "python")
        weight_root = str(self._conf("rvc_weight_root", "") or "").strip()
        if not weight_root:
            weight_root = str(rvc_dir / "assets" / "weights")
        return {
            "rvc_dir": rvc_dir,
            "python": python,
            "weight_root": weight_root,
            "model": model,
            "index": str(self._conf("rvc_index", "") or ""),
            "pitch": self._conf_int("rvc_pitch", 0),
            "f0_method": str(self._conf("rvc_f0_method", "rmvpe")),
            "index_rate": float(self._conf("rvc_index_rate", 0.75)),
            "protect": float(self._conf("rvc_protect", 0.33)),
            "timeout": self._conf_int("rvc_timeout_seconds", 180),
        }

    async def _ensure_rvc_worker(self):
        proc = self._rvc_proc
        if proc is not None and proc.returncode is None:
            return proc

        settings = self._rvc_settings()
        python = Path(settings["python"])
        worker = Path(__file__).parent / "rvc_worker.py"
        if not python.exists():
            raise VoiceFileError(f"找不到 RVC 的 Python：{python}")
        if not worker.exists():
            raise VoiceFileError(f"找不到 rvc_worker.py：{worker}")

        args = [
            str(python), str(worker),
            "--rvc-dir", str(settings["rvc_dir"]),
            "--weight-root", settings["weight_root"],
            "--model", settings["model"],
            "--index", settings["index"],
            "--pitch", str(settings["pitch"]),
            "--f0-method", settings["f0_method"],
            "--index-rate", str(settings["index_rate"]),
            "--protect", str(settings["protect"]),
        ]
        logger.info(f"[voice_file] 启动 RVC worker：{settings['model']}")
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(settings["rvc_dir"]),
        )
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=180)
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise VoiceFileError("RVC 模型加载超时。") from exc
        if line.strip() != b"READY":
            proc.kill()
            raise VoiceFileError("RVC worker 启动失败。")

        self._rvc_proc = proc
        return proc

    async def _rvc_convert(self, src: Path, dst: Path) -> None:
        async with self._rvc_lock:
            proc = await self._ensure_rvc_worker()
            settings = self._rvc_settings()
            payload = json.dumps({"input": str(src), "output": str(dst)}) + "\n"
            try:
                proc.stdin.write(payload.encode("utf-8"))
                await proc.stdin.drain()
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=settings["timeout"],
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                self._rvc_proc = None
                raise VoiceFileError("RVC 变声失败（进程异常）。")

            if not line:
                self._rvc_proc = None
                raise VoiceFileError("RVC 变声失败（进程已退出）。")
            try:
                reply = json.loads(line.decode("utf-8", "ignore"))
            except json.JSONDecodeError as exc:
                raise VoiceFileError("RVC 返回格式异常。") from exc
            if not reply.get("ok"):
                raise VoiceFileError(f"RVC 变声失败：{reply.get('error')}")

    def _cleanup_old_files(self) -> None:
        try:
            now = time.time()
            for item in self.tts_dir.glob("*"):
                if now - item.stat().st_mtime > 3600:
                    item.unlink(missing_ok=True)
        except Exception:
            pass

    async def _send_voice(self, event: AstrMessageEvent, wav_path: Path) -> None:
        await event.send(MessageChain(chain=[Record(file=str(wav_path))]))

    async def synthesize_voice(self, text: str) -> Path:
        """供其它插件调用：文字 → 语音文件路径。"""
        return await self._synthesize(text)

    async def send_voice_to(self, umo: str, text: str) -> bool:
        """供其它插件调用：把文字合成语音并发送到指定会话。"""
        wav_path = await self._synthesize(text)
        return bool(
            await self.context.send_message(
                umo,
                MessageChain(chain=[Record(file=str(wav_path))]),
            ),
        )

    # ------------------------------------------------------------------ 文件

    def _allowed_exts(self) -> set[str]:
        raw = self._conf("file_extensions", DEFAULT_EXTS) or []
        try:
            return {str(item).lower() for item in raw}
        except TypeError:
            return {item.lower() for item in DEFAULT_EXTS}

    def _read_text_file(self, path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise VoiceFileError(f"文件不存在：{file_path}")

        exts = self._allowed_exts()
        if exts and file_path.suffix.lower() not in exts:
            raise VoiceFileError(f"不支持读取 {file_path.suffix or '无后缀'} 文件。")

        max_kb = self._conf_int("file_max_kb", 256)
        size_kb = file_path.stat().st_size / 1024
        if max_kb > 0 and size_kb > max_kb:
            raise VoiceFileError(f"文件太大了（{size_kb:.0f} KB，上限 {max_kb} KB）。")

        for encoding in ("utf-8-sig", "utf-8", "gbk", "utf-16"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise VoiceFileError("看起来不是文本文件（编码无法识别）。")

    async def _attachment_text(self, component: File) -> str | None:
        name = component.name or ""
        exts = self._allowed_exts()
        if exts and Path(name).suffix.lower() not in exts:
            return None
        try:
            path = await asyncio.wait_for(component.get_file(), timeout=30)
        except Exception as exc:
            logger.warning(f"[voice_file] 下载附件失败：{exc}")
            return None
        if not path:
            return None
        try:
            return self._read_text_file(path)
        except VoiceFileError as exc:
            logger.warning(f"[voice_file] 读取附件失败：{exc}")
            return None

    @filter.on_llm_request()
    async def inject_attachment_text(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
        *args,
        **kwargs,
    ):
        """把聊天里收到的文本附件内容注入本轮上下文。"""
        if not self._conf_bool("enabled", True):
            return
        if not self._conf_bool("attachment_auto_read", True):
            return

        components: list[File] = []
        for comp in event.get_messages():
            if isinstance(comp, File):
                components.append(comp)
            elif isinstance(comp, Reply) and comp.chain:
                for sub in comp.chain:
                    if isinstance(sub, File):
                        components.append(sub)
        if not components:
            return

        prompt = getattr(request, "prompt", "") or ""
        if "[用户发送的文件" in prompt:
            return

        chunks: list[str] = []
        budget = self._conf_int("attachment_max_chars", 4000)
        used = 0
        for comp in components:
            text = await self._attachment_text(comp)
            if not text:
                continue
            remain = max(0, budget - used)
            if remain <= 0:
                break
            clipped = text[:remain]
            used += len(clipped)
            note = "" if len(clipped) >= len(text) else "\n……（内容已截断）"
            chunks.append(f"[用户发送的文件：{comp.name or '未命名'}]\n{clipped}{note}")

        if not chunks:
            return
        request.prompt = prompt + "\n\n" + "\n\n".join(chunks)

    # ------------------------------------------------------------------ 语音识别

    def _stt_model_dir(self) -> Path:
        raw = str(self._conf("stt_model_dir", "") or "").strip()
        if raw:
            return Path(raw)
        return self.data_dir / "models" / STT_MODEL_DIRNAME

    async def _get_recognizer(self):
        if self._recognizer is not None:
            return self._recognizer
        async with self._stt_lock:
            if self._recognizer is not None:
                return self._recognizer
            model_dir = self._stt_model_dir()
            model_file = model_dir / "model.int8.onnx"
            tokens_file = model_dir / "tokens.txt"
            if not model_file.exists() or not tokens_file.exists():
                raise VoiceFileError(
                    f"找不到 SenseVoice 模型：{model_dir}（需要 model.int8.onnx 和 tokens.txt）",
                )

            def _load():
                try:
                    import sherpa_onnx
                except Exception as exc:
                    raise VoiceFileError(
                        "未安装 sherpa-onnx，无法进行语音识别；"
                        "可在插件配置里关闭 stt_enable。",
                    ) from exc

                return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model_file),
                    tokens=str(tokens_file),
                    num_threads=2,
                    use_itn=True,
                    language=str(self._conf("stt_language", "auto") or "auto"),
                    debug=False,
                )

            logger.info("[voice_file] 正在加载 SenseVoice 模型 ...")
            self._recognizer = await asyncio.to_thread(_load)
            logger.info("[voice_file] SenseVoice 模型加载完成")
            return self._recognizer

    async def _to_16k_wav(self, wav_path: str) -> str:
        """把任意采样率的音频转成 SenseVoice 需要的 16kHz 单声道 WAV。"""
        ffmpeg = str(self._conf("ffmpeg_path", "ffmpeg") or "ffmpeg")
        if not shutil.which(ffmpeg):
            return wav_path

        self.tts_dir.mkdir(parents=True, exist_ok=True)
        try:
            tag = uuid.uuid5(uuid.NAMESPACE_URL, f"{wav_path}:{os.path.getmtime(wav_path)}").hex
        except OSError:
            return wav_path

        out = self.tts_dir / f"stt_{tag}.wav"
        if out.exists() and out.stat().st_size > 0:
            return str(out)

        try:
            proc = await asyncio.create_subprocess_exec(
                ffmpeg, "-y", "-i", wav_path,
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                str(out),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=60)
        except Exception as exc:
            logger.warning(f"[voice_file] 音频重采样失败，改用原文件：{exc}")
            return wav_path
        if proc.returncode != 0 or not out.exists():
            return wav_path
        return str(out)

    async def _transcribe(self, wav_path: str) -> str:
        try:
            mtime = os.path.getmtime(wav_path)
        except OSError:
            return ""
        key = (wav_path, mtime)
        if key in self._transcript_cache:
            return self._transcript_cache[key]

        recognizer = await self._get_recognizer()
        normalized = await self._to_16k_wav(wav_path)

        def _decode() -> str:
            import wave

            import numpy as np

            with wave.open(normalized, "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                frames = handle.readframes(handle.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)

            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            return (stream.result.text or "").strip()

        text = await asyncio.to_thread(_decode)
        if len(self._transcript_cache) > 128:
            self._transcript_cache.clear()
        self._transcript_cache[key] = text
        return text

    def _voice_paths(self, event: AstrMessageEvent, request=None) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()

        def add(path: str | None) -> None:
            if path and path not in seen and os.path.isfile(path):
                seen.add(path)
                paths.append(path)

        for comp in event.get_messages():
            if isinstance(comp, Record):
                add(comp.path or comp.file)
            elif isinstance(comp, Reply) and comp.chain:
                for sub in comp.chain:
                    if isinstance(sub, Record):
                        add(sub.path or sub.file)

        if request is not None:
            for url in getattr(request, "audio_urls", None) or []:
                if isinstance(url, str):
                    add(url)
        return paths

    async def _transcribe_all(self, paths: list[str]) -> list[str]:
        texts: list[str] = []
        for path in paths:
            try:
                text = await self._transcribe(path)
            except VoiceFileError as exc:
                logger.warning(f"[voice_file] 语音识别不可用：{exc}")
                break
            except Exception as exc:
                logger.exception(f"[voice_file] 语音识别失败：{exc}")
                continue
            if text:
                texts.append(text)
                logger.info(f"[voice_file] 语音转写：{text}")
        return texts

    @filter.on_llm_request(priority=15)
    async def transcribe_voice_early(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
        *args,
        **kwargs,
    ):
        """先转写，并把结果写回 event.message_str。

        优先级 15 让它排在 reply_merge（0）之前：否则 reply_merge 会把语音
        当成“空消息”，先干等十几秒再放行。写进 message_str 后，它的合并
        窗口与 virtual_life 的延迟批次也能看到这段文字。
        """
        if not self._conf_bool("enabled", True) or not self._conf_bool("stt_enable", True):
            return

        paths = self._voice_paths(event)
        if not paths:
            return

        texts = await self._transcribe_all(paths)
        if not texts:
            return

        event.set_extra("voice_file_transcripts", texts)
        joined = " ".join(texts)
        current = event.get_message_str() or ""
        if joined not in current:
            event.message_str = (current + " " + joined).strip()
            try:
                event.message_obj.message_str = event.message_str
            except Exception:
                pass

    @filter.on_llm_request(priority=-5)
    async def inject_voice_transcript(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
        *args,
        **kwargs,
    ):
        """注入转写文本，并剥掉原始音频片段。

        优先级 -5：排在 virtual_life（0）之后，避免它把 audio_urls 重新加回来
        导致 deepseek 直接 422。
        """
        if not self._conf_bool("enabled", True) or not self._conf_bool("stt_enable", True):
            return

        paths = self._voice_paths(event, request)
        texts = list(event.get_extra("voice_file_transcripts") or [])
        if not texts and paths:
            texts = await self._transcribe_all(paths)

        # 不论有没有识别出文字，音频片段都必须清掉
        self._strip_audio_parts(request)
        if hasattr(request, "audio_urls"):
            request.audio_urls = []

        if not texts:
            return

        prompt = getattr(request, "prompt", "") or ""
        if "[用户发来一条语音" in prompt:
            return
        chunks = [f"[用户发来一条语音，转写内容：{text}]" for text in texts]
        request.prompt = prompt + "\n" + "\n".join(chunks)

    @staticmethod
    def _strip_audio_parts(request: ProviderRequest) -> None:
        """移除请求里的原始音频片段。

        部分模型（如 DeepSeek）不接受 input_audio，AstrBot 的 OpenAI 适配器
        会把语音原样转成该片段，导致整轮请求 422。转写文字已经放进 prompt，
        原始音频在这里丢弃。
        """
        contexts = getattr(request, "contexts", None)
        if not isinstance(contexts, list):
            return
        for message in contexts:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            kept = [
                part
                for part in content
                if not (
                    isinstance(part, dict)
                    and part.get("type") in ("input_audio", "audio_url")
                )
            ]
            if len(kept) == len(content):
                continue
            message["content"] = kept or "[用户发来一条语音，内容已转写为文字]"

    # ------------------------------------------------------------------ 指令

    @filter.command("语音", alias={"发语音"})
    async def voice_command(self, event: AstrMessageEvent):
        if not self._authorized(event):
            return
        if not self._conf_bool("enabled", True) or not self._conf_bool("enable_voice", True):
            return event.plain_result("语音功能已关闭。")

        text = self._strip_command(event, VOICE_COMMANDS)
        if not text:
            return event.plain_result("用法：/语音 <要说的话>")

        try:
            wav_path = await self._synthesize(text)
        except VoiceFileError as exc:
            return event.plain_result(f"语音失败：{exc}")

        try:
            await self._send_voice(event, wav_path)
        except Exception as exc:
            logger.exception("[voice_file] 发送语音失败")
            return event.plain_result(f"语音发送失败：{exc}")
        return None

    @filter.command("读文件", alias={"查看文件"})
    async def read_command(self, event: AstrMessageEvent):
        if not self._authorized(event):
            return
        if not self._conf_bool("enabled", True):
            return event.plain_result("插件已关闭。")

        path = self._strip_command(event, READ_COMMANDS)
        if not path:
            return event.plain_result("用法：/读文件 <本机文件路径>")

        try:
            text = self._read_text_file(path.strip("\"'"))
        except VoiceFileError as exc:
            return event.plain_result(f"读取失败：{exc}")

        limit = self._conf_int("attachment_max_chars", 4000)
        clipped = text[:limit]
        note = "" if len(clipped) >= len(text) else f"\n……（已截断，共 {len(text)} 字）"
        return event.plain_result(f"{clipped}{note}")

    @filter.command("语音帮助", alias={"文件帮助"})
    async def help_command(self, event: AstrMessageEvent):
        if not self._authorized(event):
            return
        return event.plain_result(
            "语音与文件\n"
            "/语音 <文本>        用 AI 语音说一段话\n"
            "/读文件 <路径>       读本机文本文件（txt/md/csv/json 等）\n"
            "直接发 txt 等文本文件给她，她会自动读内容。\n"
            "/语音帮助            查看本帮助",
        )

    # ------------------------------------------------------------------ LLM 工具

    @filter.llm_tool(name="voice_say")
    async def voice_say(self, event: AstrMessageEvent, text: str) -> str:
        """用语音把一段话发给用户。适合用户明确想听声音、或者你想用语气表达情绪时使用。

        Args:
            text(string): 要说的内容，口语化、简短
        """
        if not self._conf_bool("enabled", True) or not self._conf_bool("enable_voice", True):
            return "语音功能已关闭，未发送。"
        if not self._authorized(event):
            return "当前用户没有使用语音的权限，未发送。"

        try:
            wav_path = await self._synthesize(text)
            await self._send_voice(event, wav_path)
        except VoiceFileError as exc:
            return f"语音发送失败：{exc}"
        except Exception as exc:
            logger.exception("[voice_file] LLM 发送语音失败")
            return f"语音发送失败：{exc}"
        return "已用语音发送。"

    @filter.llm_tool(name="read_file")
    async def read_file(self, event: AstrMessageEvent, path: str) -> str:
        """读取本机上的一个文本文件并返回内容（管理员限定，仅支持文本类后缀）。

        Args:
            path(string): 文件的绝对路径
        """
        if not self._conf_bool("enabled", True) or not self._conf_bool("enable_read_file_tool", True):
            return "文件读取功能已关闭。"
        if not self._authorized(event):
            return "当前用户没有读取文件的权限。"
        try:
            text = self._read_text_file(path)
        except VoiceFileError as exc:
            return f"读取失败：{exc}"
        limit = self._conf_int("attachment_max_chars", 4000)
        if len(text) > limit:
            return text[:limit] + f"\n……（已截断，共 {len(text)} 字）"
        return text
