#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import os, time, re, threading
from indextts.infer_v2 import IndexTTS2
import random,string

# ====== 固定配置（按需改你本地路径即可） ======
MODEL_DIR = r"./checkpoints"             # 模型目录
CFG_PATH  = None                         # None 则自动用 <MODEL_DIR>/config.yaml
USE_FP16 = False
USE_DEEPSPEED = False
USE_CUDA_KERNEL = False

# 生成输出目录
OUT_DIR = r"H:/index-tts2-0911-win/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 写死的生成参数（与 webui 对齐）
GEN_KW = dict(
    do_sample=True,
    top_p=0.8,
    top_k=30,
    temperature=0.8,
    length_penalty=0.0,
    num_beams=3,
    repetition_penalty=10.0,
    max_mel_tokens=1500,
    max_text_tokens_per_segment=120,
    verbose=False,
)

# 情感控制：默认“与音色一致”（不启用情感）
EMO_ARGS = dict(
    emo_audio_prompt=None,
    emo_alpha=1.0,
    emo_vector=None,
    use_emo_text=False,
    emo_text=None,
    use_random=False,
)

# ====== 初始化（只做一次，避免每次请求都加载模型） ======
_cfg_path = CFG_PATH or os.path.join(MODEL_DIR, "config.yaml")
_tts_lock = threading.Lock()  # 若底层不线程安全，可串行化
tts = IndexTTS2(
    model_dir=MODEL_DIR,
    cfg_path=_cfg_path,
    use_fp16=USE_FP16,
    use_deepspeed=USE_DEEPSPEED,
    use_cuda_kernel=USE_CUDA_KERNEL,
)

app = Flask(__name__)


def __generate_random_filename(length=10):  # 生成一个由字母和数字组成的随机字符串
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
def _safe_stem(s: str, maxlen: int = 40) -> str:
    """基于文本生成安全文件名片段"""
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff-]", "", s)
    return (s[:maxlen] or "utt")

@app.route("/index_tts", methods=["POST"])
def index_tts():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "")
        prompt_audio = data.get("prompt_audio", "")
        if not text or not isinstance(text, str):
            return jsonify({"error": "text is required"}), 400
        if not prompt_audio or not isinstance(prompt_audio, str):
            return jsonify({"error": "prompt_audio is required"}), 400
        if not os.path.exists(prompt_audio):
            return jsonify({"error": f"prompt_audio not found: {prompt_audio}"}), 400
        out_path = os.path.join(OUT_DIR,__generate_random_filename()+".wav")
        # 串行推理（更稳妥）；如果你确认线程安全，可去掉锁
        with _tts_lock:
            result_path = tts.infer(
                spk_audio_prompt=prompt_audio,
                text=text,
                output_path=out_path,
                **EMO_ARGS,
                **GEN_KW,
            )
        return jsonify({"file_path": result_path})

    except Exception as e:
        # 打印到控制台，方便排查
        print("ERROR /index_tts:", repr(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 监听 7860：对应你要的 http://localhost:7860/index_tts
    app.run(host="0.0.0.0", port=7860, threaded=True)