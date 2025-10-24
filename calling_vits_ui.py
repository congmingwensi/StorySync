#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch TTS Reader — 极简滚动小窗 + 预取无缝播放（默认 TK）
=====================================================
这版专门按你的习惯调整：
- 顶部提供一个 **TEXT** 变量，直接在代码里写小说文本，运行就读；无需命令行参数。
- 修复高亮：改为**基于正则 span 的精确区间**，并用 "\n" 计算行列索引。
- 预取：播放当前句时，后台生成后续 2 句，句间几乎无间隔。
- Windows + `.wav` 用 winsound 异步播放；其他情况按时长静默等待。
运行： python calling_vits_ui.py
"""
from __future__ import annotations
import argparse
import os
import re
import wave
import time
import threading
import winsound
import tkinter as tk
from get_image import QueueImage,get_stable_image
from tkinter import ttk
from PIL import Image, ImageTk
from contextlib import closing
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests
TEXT_PANEL_RATIO = 0.34
TEXT = (
"""
默认阅读的文本，也可以打开阅读器后修改再播放。
"""
)
PROMPT_AUDIO = r"H:\index-tts2-0911-win\test_audio\index_tts.mp3" #支持bert-vits2中文特化版和index-tts，此为index-tts参考音频
queue_img=QueueImage()
def split_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """返回 (sentence, start, end)。start/end 为**原文本**中的字符区间。
    我们会 strip 两端空白并把区间校正到去掉空白后的范围，便于精确高亮。
    """
    items: List[Tuple[str, int, int]] = []
    target=30
    punct="。.！？!?~♡"
    items: List[Tuple[str, int, int]] = []
    if not text:
        return items
    seg_start = 0
    seg_len = 0
    for i, ch in enumerate(text):
        seg_len += 1
        # 仅当遇到标点，且当前段长度至少达到 target 时切分
        if ch in punct and seg_len >= target:
            end = i + 1  # 包含这个标点
            items.append((text[seg_start:end], seg_start, end))
            seg_start = end
            seg_len = 0
    # 收尾：把最后一段补上（可能不足 target 或没有标点）
    if seg_start < len(text):
        items.append((text[seg_start:], seg_start, len(text)))
    return items

def _wav_duration(path: str) -> Optional[float]:
    try:
        with closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        return None
    return None
def get_audio_duration(path: Optional[str]) -> Optional[float]:
    if not path or not isinstance(path, str):
        return None
    if not os.path.exists(path):
        return None
    if path.lower().endswith(".wav"):
        return _wav_duration(path)
    try:
        from mutagen import File as MutagenFile  # type: ignore
        mf = MutagenFile(path)
        if mf is not None and hasattr(mf, "info") and getattr(mf.info, "length", None):
            return float(mf.info.length)
    except Exception:
        return None
    return None
def post_tts(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 600) -> Dict[str, Any]:
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()
EventHook = Callable[[str, Dict[str, Any]], None]
def compute_duration(sentence: str, file_path: Optional[str], resp_duration: Optional[float]) -> float:
    if isinstance(resp_duration, (int, float)):
        return float(resp_duration)
    d = get_audio_duration(file_path)
    if isinstance(d, (int, float)):
        return float(d)
    # 兜底估算：≈12 字/秒，最小 1.2s，保证不卡顿
    return max(1.2, len(sentence) / 12.0)


# =====================================================
# 6) Tk 小窗（默认），带预取
# =====================================================
class ReaderApp:
    """极简阅读器：滚动文本 + 当前句高亮 + 预取下一句。"""
    def __init__(self, endpoint: str, field: str = "text", pad: float = 0.05, prompt_audio: str = PROMPT_AUDIO):
        self.endpoint = endpoint
        self.field = field
        self.pad = float(pad)
        self.prefetch_k = 3  # 预取窗口大小
        self.prompt_audio = prompt_audio

        self.cache: Dict[str, Tuple[str, float]] = {}  # sentence -> (file_path, duration)
        self.inflight_texts: set[str] = set()
        self.inflight: Dict[int, threading.Thread] = {}
        self.lock = threading.Lock()

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("📖 TTS Reader")
        self.root.geometry("1080x860")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 控制区
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        self.btn_start = ttk.Button(top, text="开始 ▶", command=self.start)
        self.btn_toggle = ttk.Button(top, text="暂停/继续 ⏯", command=self.toggle_pause_resume, state="disabled")
        self.btn_prev = ttk.Button(top, text="上一句 ◀", command=self.prev, state="disabled")
        self.btn_next = ttk.Button(top, text="下一句 ▶", command=self.next, state="disabled")

        for w in (self.btn_start, self.btn_toggle, self.btn_prev, self.btn_next):
            w.pack(side="left", padx=4)

        # 文本区（改为“背景铺满 + 文本在上”叠层布局）
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=6)

        # 背景图：直接铺满 body，作为底层
        self.bg_label = self.tk.Label(body, bd=0)
        self.bg_label.place(relx=0, rely=0, relwidth=1, relheight=1 - TEXT_PANEL_RATIO, anchor="nw")
        self._bg_imgtk = None
        self._bg_path: Optional[str] = None
        self.bg_cache: Dict[str, str] = {}
        self._last_img_t = 0.0

        # —— 背景生成：初始化条件变量/队首队尾/线程 ——
        self._bg_cv = threading.Condition()
        self._bg_head: Optional[str] = None  # 需要立刻生成并显示的句子（队首）
        self._bg_tail: Optional[str] = None  # 最新请求（生成完队首后再生成它；期间会被覆盖）
        self._bg_running = True
        self._bg_worker = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_worker.start()

        def _on_resize_bg(event=None):
            if self._bg_path and os.path.exists(self._bg_path):
                self._set_bg_image(self._bg_path)
        self.bg_label.bind("<Configure>", _on_resize_bg)

        # 顶层放文本区域（留一点内边距，让背景能“露边”）
        text_wrap = ttk.Frame(body)
        text_wrap.place(relx=0, rely=1.0, relwidth=1, relheight=TEXT_PANEL_RATIO, anchor="sw")
        self.txt = tk.Text(
            text_wrap,
            wrap="word",
            font=("Segoe UI", 14, "bold"),
            undo=False,
            bg="#FFFFFF",
            fg="#222222",
        )
        self.txt.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        scroll = ttk.Scrollbar(text_wrap, command=self.txt.yview)
        scroll.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=scroll.set)
        # 热键：Alt+← / Alt+→
        self.root.bind_all("<Alt-Left>", self._hotkey_prev)
        self.root.bind_all("<Alt-Right>", self._hotkey_next)
        self.root.bind_all("<Control-Left>", self._hotkey_prev)
        self.root.bind_all("<Control-Right>", self._hotkey_next)
        self.root.bind_all("<space>", self._hotkey_toggle_space)
        # 基础编辑快捷键：在 Text 聚焦时可复制/粘贴/全选/剪切
        self.txt.bind("<Control-a>", lambda e: (self.txt.tag_add("sel", "1.0", "end"), "break"))
        self.txt.bind("<Control-A>", lambda e: (self.txt.tag_add("sel", "1.0", "end"), "break"))
        self.txt.bind("<Control-c>", lambda e: (self.txt.event_generate("<<Copy>>"), "break"))
        self.txt.bind("<Control-C>", lambda e: (self.txt.event_generate("<<Copy>>"), "break"))
        self.txt.bind("<Control-x>", lambda e: (self.txt.event_generate("<<Cut>>"), "break"))
        self.txt.bind("<Control-X>", lambda e: (self.txt.event_generate("<<Cut>>"), "break"))
        # 把你在代码里写的 TEXT 放进来
        self.txt.insert("1.0", re.sub(r'\n{2,}', '\n', TEXT))
        self.txt.tag_configure("current", background="#1f2937", foreground="#e5e7eb")
        # 状态
        self.items: List[Tuple[str, int, int]] = []  # (sentence, start, end)
        self.index = 0
        self.is_running = False
        self.is_paused = False
        self.after_id = None
    # --- 索引与高亮 ---
    def _tk_index_from_abs(self, text_all: str, pos: int) -> str:
        pre = text_all[:pos]
        line = pre.count("\n") + 1
        col = pos - (pre.rfind("\n") + 1 if "\n" in pre else 0)
        return f"{line}.{col}"

    def _highlight_current(self):
        self.txt.tag_remove("current", "1.0", "end")
        if not (0 <= self.index < len(self.items)):
            return
        text_all = self.txt.get("1.0", "end-1c")
        _, s0, e0 = self.items[self.index]
        s = self._tk_index_from_abs(text_all, s0)
        e = self._tk_index_from_abs(text_all, e0)
        self.txt.tag_add("current", s, e)
        self.txt.see(s)
        self.txt.update_idletasks()

        # 优先用像素信息精确居中
        info = self.txt.dlineinfo(s)
        if info:
            line_y = info[1]  # 当前行相对文本框顶部的 y 像素
            line_h = info[3] or 1  # 行高像素
            vis_h = self.txt.winfo_height()
            delta_pix = line_y - (vis_h // 2 - line_h // 2)
            n_lines = int(delta_pix / line_h)
            if n_lines:
                self.txt.yview_scroll(n_lines, "units")
        else:
            total_lines = max(1, int(self.txt.count("1.0", "end-1c", "displaylines")[0]))
            line_idx = int(self.txt.count("1.0", s, "displaylines")[0])
            # 估算可见行数（行高来自 1.0 行）
            h0 = (self.txt.dlineinfo("1.0") or (0, 0, 0, 1))[3] or 1
            vis_lines = max(1, int(round(self.txt.winfo_height() / h0)))
            target_frac = max(0.0, min(1.0, (line_idx - vis_lines / 2) / total_lines))
            self.txt.yview_moveto(target_frac)
        # 可再次设置样式以防被刷新覆盖
        self.txt.tag_configure("current", background="#1f2937", foreground="#e5e7eb")

    # —— 新增：背景图设置与重绘 ——
    def _set_bg_image(self, path: Optional[str]) -> None:
        """把右侧预览替换为 path 指向的图片（自适应缩放）。"""
        self._bg_path = path
        w = max(2, self.bg_label.winfo_width())
        h = max(2, self.bg_label.winfo_height())
        im = Image.open(self._bg_path).convert("RGB")
        iw, ih = im.size
        # 等比放大以覆盖（避免拉伸变形）
        scale = max(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
        # 居中裁剪到目标尺寸
        left = max(0, (nw - w) // 2)
        top = max(0, (nh - h) // 2)
        im = im.crop((left, top, left + w, top + h))
        self._bg_imgtk = ImageTk.PhotoImage(im)
        self.bg_label.config(image=self._bg_imgtk)
        print(f"[IMG] set -> {self._bg_path}  size=({w}x{h}))")

    def _bg_request(self, current: Optional[str] = None, latest: Optional[str] = None) -> None:
        with self._bg_cv:
            if current:
                self._bg_head = current
            if latest:
                self._bg_tail = latest  # 只保留“最新”的一个，自动覆盖旧 tail
            self._bg_cv.notify()

    def _bg_loop(self) -> None:
        while self._bg_running:
            with self._bg_cv:
                while self._bg_running and not (self._bg_head or self._bg_tail):
                    self._bg_cv.wait()
                if not self._bg_running:
                    break
                target = None
                if self._bg_head:
                    target = self._bg_head
                    self._bg_head = None
                elif self._bg_tail:
                    target = self._bg_tail
                    self._bg_tail = None
            if not target:
                continue
            # 先试缓存
            path = self.bg_cache.get(target)
            if not path:
                path = get_stable_image(target)
                self.bg_cache[target] = path
                # 缓存上限控制
                old_key = next(iter(self.bg_cache.keys()))
                del self.bg_cache[old_key]
            # 无论缓存或新图，都切回主线程更新 UI
            self.root.after(0, lambda p=path: self._set_bg_image(p))

    def _on_close(self):
        with self._bg_cv:
            self._bg_running = False
            self._bg_cv.notify_all()
        self.root.destroy()

    def _schedule(self, seconds: float, fn: Callable[[], None]):
        import math
        ms = int(math.ceil(max(0.0, seconds) * 1000))
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(ms, fn)
        # --- 辅助：停止计时/声音 & 刷新按钮 ---

    def _stop_pending(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if os.name == "nt":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)

    def _update_nav_buttons(self):
        prev_state = "normal" if self.index > 0 else "disabled"
        next_state = "normal" if self.index < len(self.items) - 1 else "disabled"
        self.btn_prev.config(state=prev_state)
        self.btn_next.config(state=next_state)

    def _update_toggle(self):
        if self.is_running:
            self.btn_toggle.config(state="normal", text=("继续 ▶" if self.is_paused else "暂停 ⏸"))
        else:
            self.btn_toggle.config(state="disabled", text="暂停/继续 ⏯")

    def toggle_pause_resume(self):
        """空格或按钮：在“暂停/继续”之间切换；未开始则直接开始。"""
        if not self.is_running:
            self.start()
            return
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    # --- 预取 ---
    def _prefetch_one(self, idx: int, sentence: str):
        try:
            if sentence in self.inflight_texts:return
            self.inflight_texts.add(sentence)
            payload = {self.field: sentence, "is_play": False,"prompt_audio": self.prompt_audio}
            print(f"[PREFETCH] idx={idx} text={sentence[:24]}...")
            resp = post_tts(self.endpoint, payload)
            queue_img.push_text_to_queue(idx,sentence)
            fp = resp.get("file_path")
            dur = compute_duration(sentence, fp, resp.get("duration"))
            with self.lock:
                self.cache[sentence] = (fp, float(dur))
        finally:
            with self.lock:
                self.inflight.pop(idx, None)

    def _ensure_prefetch_window(self):
        for off in range(1, self.prefetch_k + 1):
            idx = self.index + off
            if idx >= len(self.items):
                break
            sent = self.items[idx][0]
            with self.lock:
                if sent in self.cache or idx in self.inflight:
                    continue
                t = threading.Thread(target=self._prefetch_one, args=(idx, sent), daemon=True)
                self.inflight[idx] = t
                t.start()

    # --- 主循环 ---
    def _step(self):
        if not self.is_running or self.is_paused:
            return
        if self.index >= len(self.items):
            self.is_running = False
            self._update_nav_buttons()
            self._update_toggle()
            return
        sentence, _, _ = self.items[self.index]
        self._highlight_current()
        img_path=queue_img.get_image(self.index)
        self.root.after(0, lambda p=img_path: self._set_bg_image(p))
        with self.lock:
            cached = self.cache.get(sentence)
        if cached:
            fp, dur = cached
        else:
            # 防止并发重复：若预取已在路上则等待其结果
            issued_sync = False
            with self.lock:
                already_issuing = sentence in self.inflight_texts
                if not already_issuing:
                    self.inflight_texts.add(sentence)
                    issued_sync = True
            if issued_sync:
                payload = {"text": sentence, "prompt_audio": self.prompt_audio}
                print(f"[POST] idx={self.index} text={sentence[:24]}...")
                resp = post_tts(self.endpoint, payload)
                fp = resp.get("file_path")
                dur = compute_duration(sentence, fp, resp.get("duration"))
                with self.lock:
                    self.cache[sentence] = (fp, float(dur))
            else:
                # 有预取在进行：轮询等待缓存
                import time
                for _ in range(1200):  # 最多 120s
                    with self.lock:
                        cached = self.cache.get(sentence)
                    if cached:
                        break
                    time.sleep(0.1)
                if not cached:  # 兜底：再同步发一次
                    payload = {"text": sentence, "prompt_audio": self.prompt_audio}
                    print(f"[POST-FALLBACK] idx={self.index} text={sentence[:24]}...")
                    resp = post_tts(self.endpoint, payload)
                    fp = resp.get("file_path")
                    dur = compute_duration(sentence, fp, resp.get("duration"))
                    with self.lock:
                        self.cache[sentence] = (fp, float(dur))
            fp, dur = self.cache[sentence]
        winsound.PlaySound(fp, winsound.SND_FILENAME | winsound.SND_ASYNC)
        self._ensure_prefetch_window()
        self.index += 1
        self._update_nav_buttons()
        self._schedule(float(dur) + float(self.pad), self._step)

    # --- 控制 ---
    def start(self):
        if self.is_running:
            return
        text_all = self.txt.get("1.0", "end-1c")
        self.items = split_sentences_with_spans(text_all)
        self.index = 0
        self.is_running = True
        self.is_paused = False
        self._update_nav_buttons()
        self._update_toggle()
        self._ensure_prefetch_window()
        self._step()

    def pause(self):
        if not self.is_running or self.is_paused:
            return
        # 1) 立即停止计时与声音
        self._stop_pending()
        if os.name == "nt":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        # 2) 将 index 回退到“当前正在播的句子”，以便继续时从**同一句**开始
        if self.index > 0:
            self.index -= 1
        self.is_paused = True
        self._update_toggle()
        # 3) 立刻刷新高亮到这一句（视觉不跳）
        self._highlight_current()

    def resume(self):
        if not self.is_running or not self.is_paused:
            return
        self.is_paused = False
        self._update_toggle()
        self._step()

    def prev(self):
        if self.index > 0:
            self.is_paused = False
            self._stop_pending()  # 取消已排程的跳转，避免被顶回去
            self.index -= 2
            self._update_nav_buttons()
            self._update_toggle()
            self._step()

    def next(self):
        """跳到“下一句”。不要自增 index；_step 内部会处理。"""
        if not self.items:
            return
        if self.index >= len(self.items):
            self._stop_pending()
            self._update_nav_buttons()
            self._update_toggle()
            return
        self.is_paused = False
        self._stop_pending()
        self._update_nav_buttons()
        self._update_toggle()
        self._step()

    def _hotkey_prev(self, event=None):
        # 使用 bind_all；确保任何控件聚焦时也能触发
        self.prev()
        return "break"

    def _hotkey_next(self, event=None):
        self.next()
        return "break"

    def _hotkey_toggle_space(self, event=None):
        self.toggle_pause_resume()
        return "break"

    def run(self):
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Batch TTS Reader")
    parser.add_argument("--ui", choices=["tk"], default="tk")
    parser.add_argument("--endpoint", default="http://localhost:5002/generate_audio") #用flask实现的本地文字转语音接口
    parser.add_argument("--field", default="text")
    parser.add_argument("--pad", type=float, default=0.05, help="额外等待秒数（无缝建议极小值）")
    parser.add_argument("--prompt_audio", default=PROMPT_AUDIO)  # IndexTTS 的参考音频路径
    args, _ = parser.parse_known_args()
    ReaderApp(endpoint=args.endpoint, field=args.field, pad=args.pad, prompt_audio=args.prompt_audio).run()
if __name__ == "__main__":
    main()
