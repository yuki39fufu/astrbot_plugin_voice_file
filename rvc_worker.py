"""RVC 推理 worker（用 RVC 自带的 Python 3.9 运行时执行）。

由 astrbot_plugin_voice_file 以子进程方式启动，常驻内存。协议：
- 启动后先输出一行 READY（模型加载完成）
- 之后每行一个 JSON 请求：{"input": "x.wav", "output": "y.wav"}
- 每行一个 JSON 回复：{"ok": true, "elapsed": 1.2} 或 {"ok": false, "error": "..."}
"""

import argparse
import json
import os
import sys
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-dir", required=True)
    parser.add_argument("--weight-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--filter-radius", type=int, default=3)
    args = parser.parse_args()

    # RVC 的 configs.config 在 import 时会解析 sys.argv，必须先清掉我们自己的参数
    sys.argv = [sys.argv[0]]

    os.chdir(args.rvc_dir)
    sys.path.insert(0, args.rvc_dir)
    os.environ["weight_root"] = args.weight_root
    os.environ["index_root"] = os.path.join(args.rvc_dir, "assets", "indices")
    os.environ["rmvpe_root"] = os.path.join(args.rvc_dir, "assets", "rmvpe")

    import numpy as np
    import soundfile as sf
    from configs.config import Config
    from infer.modules.vc.modules import VC

    vc = VC(Config())
    vc.get_vc(args.model)
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            started = time.time()
            info, result = vc.vc_single(
                0,
                request["input"],
                args.pitch,
                None,
                args.f0_method,
                args.index or "",
                None,
                args.index_rate,
                args.filter_radius,
                0,
                0.25,
                args.protect,
            )
            sample_rate, audio = result
            if audio is None:
                raise RuntimeError(str(info))
            audio = np.asarray(audio)
            # RVC 返回 int16（±32767），直接按 float 写会被播放器削波成“炸麦”
            if audio.dtype == np.int16:
                sf.write(request["output"], audio, sample_rate, subtype="PCM_16")
            else:
                sf.write(
                    request["output"],
                    audio.astype(np.float32),
                    sample_rate,
                    subtype="PCM_16",
                )
            print(
                json.dumps({"ok": True, "elapsed": round(time.time() - started, 2)}),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
