# -*- coding: utf-8 -*-
from threading import Thread
import openai
import os, re, time, base64, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_session = None
text = "用于直接运行进行测试生图接口的文本"
os.environ["NO_PROXY"] = "api.openai.com,localhost,127.0.0.1" #使用加速器代理的话需要加上此句。可以自己调一下
openai.api_base = "https://api.openai.com/v1/"
print("api_base =", openai.api_base)
openai.api_key = "sk-xxxx" #本地用python 3.7的调用openai接口方式
NEG_DEFAULT = (
    "text, calligraphy, chinese characters, poster, logo, watermark, "
        "distant shot, tiny subject, soft focus, motion blur, "
        "armor, weapon, sci-fi suit"
)

_tok = None
_mdl = None
_device = "cuda"
MODEL_DIR = "H:/index-tts2-0911-win/opus-mt-zh-en" #写了两种生成prompt的方式。此为jieba本地翻译库地址

class QueueImage:
    def __init__(self):
        self.result = {}
        self.current_img=""
        self.is_running = False
    def push_text_to_queue(self,index:int,text:str):
        if not self.is_running:
            Thread(target=self.post_image, args=(index, text), daemon=True).start()

    def post_image(self,index,text):
        self.is_running = True
        try:
            self.result[index]=get_stable_image(text)
            self.current_img=self.result[index]
            self.is_running = False
        finally:
            self.is_running = False
    def get_image(self,index):
        if self.result.get(index):
            return self.result[index]
        else: return self.current_img

def generator_prompt(text):
    def _get_session():
        global _session
        if _session is None:
            s = requests.Session()
            retry = Retry(
                total=4, connect=4, read=4,
                backoff_factor=0.6,  # 退避: 0.6, 1.2, 2.4, 4.8s
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(["GET", "POST"])
            )
            adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
            s.mount("http://", adapter);
            s.mount("https://", adapter)
            s.headers.update({"Connection": "keep-alive"})
            _session = s
        return _session
    from jieba import analyse
    if len(text) <10:
        return "",""
    # elif len(text) <100:
    #     analyse_word = analyse.tfidf(text, topK=8, allowPOS=('n','a'))
    # else:
    #     analyse_word = analyse.textrank(text, topK=8, allowPOS=('n','a'))
    try:
        openai.requestssession = _get_session()
        response = openai.ChatCompletion.create(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "system", "content": "你是一个阅读器中根据文字绘画的Agent。请根据文本内容，生成一个契合内容的，用于生成图片的英文提示词。适用于stable diffution绘画。动漫风格。"
                                              "\n提示词由几个特征单词和短句构成，中间用逗号隔开。尽可能简短，不超过8项内容"
                                              "\n下游直接对接程序，在任何时候直接生成提示词就好。不要输出多余的解释或选项"},
                {"role": "user", "content": text}
            ]
        )
        content=response.choices[0].message.content
        print(content)
    except Exception as e:
        # 失败兜底：直接不给 prompt，让上层跳过本次出图
        print(f"[prompt-api] fail: {e}")
        return "", ""

    # result_word="masterpiece, best quality, nahida_genshin, cross-shaped pupils, <lora:Nahida3:1>," + translate(",".join(analyse_word))
    result_word = "masterpiece, best quality, " + content
    return result_word,NEG_DEFAULT

def translate(word):
    from transformers import MarianMTModel, MarianTokenizer
    from transformers import GenerationConfig
    def _lazy_load():
        global _tok, _mdl, _device
        if _tok is not None and _mdl is not None:
            return
        _tok = MarianTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        _mdl = MarianMTModel.from_pretrained(MODEL_DIR, local_files_only=True)
        _mdl.eval()
    _lazy_load()
    inputs = _tok(word, return_tensors="pt", truncation=True, max_length=64)
    gen_cfg = GenerationConfig(
        max_new_tokens=16,
        num_beams=3,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
    )
    import torch
    with torch.inference_mode():
        out = _mdl.generate(**inputs, generation_config=gen_cfg)
    en = _tok.batch_decode(out, skip_special_tokens=True)[0]
    return re.sub(r'[^a-zA-Z],', '', en.lower())

def sd_txt2img(
    prompt: str,
    negative: str = NEG_DEFAULT,
    width: int = 540,
    height: int = 315,
    steps: int = 30,
    cfg: float = 6.4,
    sampler: str = "DPM++ SDE Karras",
    seed: int = -1,
    host: str = "http://127.0.0.1:7861",
    timeout: int = 600,
    out_dir: str = "H:\index-tts2-0911-win\.cache_imgs",
    return_bytes: bool = False,
):
    """
    调本地 stable diffution 文生图接口：
      - prompt: 正向提示词（英文更稳）
      - negative: 反向提示词（可空）
      - return_bytes=True 则直接返回字节；否则保存到 out_dir 并返回文件路径
    """
    url = host.rstrip("/") + "/sdapi/v1/txt2img"
    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg,
        "sampler_name": sampler,
        "seed": seed,
        # 需要可再加：clip_skip、hrfix、loras 等
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    if not data.get("images"):
        raise RuntimeError(f"No image returned: {data}")

    b64 = data["images"][0]
    img_bytes = base64.b64decode(b64.split(",", 1)[-1])

    if return_bytes or out_dir is None:
        return img_bytes

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(
        out_dir,
        f"sd_{int(time.time()*1000)}.jpg"
    )
    with open(fname, "wb") as f:
        f.write(img_bytes)
    return fname

def get_stable_image(text):
    pos, neg = generator_prompt(text)
    if not pos:
        return ""  # 外部挂了就跳过，不阻塞
    return sd_txt2img(pos, negative=neg)
# get_stable_image(text)
# generator_prompt(text)
# sd_txt2img(text)