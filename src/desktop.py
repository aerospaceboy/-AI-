"""Local Windows desktop front end for the video2ai CLI."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .config import load_yaml_options
from .utils import default_output_path


PROJECT_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent
)
CONFIG_PATH = PROJECT_DIR / "video2ai.yaml"
VIDEO_TYPES = [
    ("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv *.m4v"),
    ("所有文件", "*.*"),
]
STAGE_PATTERN = re.compile(r"^\[(\d)/(\d)\] (.+)$")
SCAN_PROGRESS_PATTERN = re.compile(r"Progress:\s*(\d+)%")


def build_command(
    video: Path,
    output: Path | None,
    model: str,
    device: str,
    ocr: bool,
    llm: bool,
) -> list[str]:
    """Produce a shell-free command, retaining all other YAML defaults."""
    command = (
        [sys.executable, "--worker", str(video)]
        if getattr(sys, "frozen", False)
        else [sys.executable, "-u", str(PROJECT_DIR / "video2ai.py"), str(video)]
    )
    if CONFIG_PATH.is_file():
        command.extend(["--config", str(CONFIG_PATH)])
    if output is not None:
        command.extend(["--output", str(output)])
    command.extend(["--whisper-model", model, "--device", device])
    command.append("--ocr" if ocr else "--no-ocr")
    command.append("--llm" if llm else "--no-llm")
    return command


class Video2AIApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"Video → AI 资料包  {__version__}")
        self.root.geometry("900x680")
        self.root.minsize(700, 520)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.running = False
        self.stopping = False
        self.output_path: Path | None = None
        self.portable = bool(getattr(sys, "frozen", False))

        defaults = load_yaml_options(CONFIG_PATH) if CONFIG_PATH.is_file() else {}
        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.model_var = tk.StringVar(value=str(defaults.get("whisper_model", "base" if self.portable else "small")))
        self.device_var = tk.StringVar(value=str(defaults.get("device", "auto")))
        self.ocr_var = tk.BooleanVar(value=not bool(defaults.get("no_ocr", False)))
        self.llm_var = tk.BooleanVar(value=not bool(defaults.get("no_llm", False)))
        if self.portable:
            self.device_var.set("cpu")
            self.ocr_var.set(False)
            self.llm_var.set(False)
        self.status_var = tk.StringVar(value="请选择一个视频")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="Video → AI 资料包", font=("Microsoft YaHei UI", 19, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 14)
        )
        ttk.Label(frame, text="视频").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.video_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="选择视频…", command=self._choose_video).grid(row=1, column=2)

        ttk.Label(frame, text="输出目录").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="选择目录…", command=self._choose_output).grid(row=2, column=2)

        settings = ttk.Frame(frame)
        settings.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 6))
        ttk.Label(settings, text="语音模型").pack(side="left")
        ttk.Combobox(
            settings, textvariable=self.model_var, state="readonly", width=12,
            values=("tiny", "base", "small") if self.portable else ("tiny", "base", "small", "medium", "large-v3"),
        ).pack(side="left", padx=(7, 20))
        ttk.Label(settings, text="计算设备").pack(side="left")
        ttk.Combobox(
            settings, textvariable=self.device_var,
            state="disabled" if self.portable else "readonly", width=8,
            values=("cpu",) if self.portable else ("auto", "cuda", "cpu"),
        ).pack(side="left", padx=(7, 20))
        ttk.Checkbutton(
            settings, text="画面文字 OCR", variable=self.ocr_var,
            state="disabled" if self.portable else "normal",
        ).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(
            settings, text="AI 总结", variable=self.llm_var,
            state="disabled" if self.portable else "normal",
        ).pack(side="left")

        actions = ttk.Frame(frame)
        actions.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 10))
        self.start_button = ttk.Button(actions, text="开始生成", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(actions, text="打开资料包", command=self._open_output, state="disabled")
        self.open_button.pack(side="left", padx=(12, 8))
        self.report_button = ttk.Button(actions, text="查看网页报告", command=self._open_report, state="disabled")
        self.report_button.pack(side="left", padx=(0, 8))
        self.bundle_button = ttk.Button(actions, text="打开上传包目录", command=self._open_bundle, state="disabled")
        self.bundle_button.pack(side="left")

        log_frame = ttk.LabelFrame(frame, text="处理日志", padding=8)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, state="disabled", wrap="word", font=("Consolas", 10))
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        bottom = ttk.Frame(frame)
        bottom.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(bottom, mode="indeterminate", length=210)
        self.progress.pack(side="right")

    def _choose_video(self) -> None:
        selected = filedialog.askopenfilename(title="选择视频", filetypes=VIDEO_TYPES)
        if selected:
            previous = self.video_var.get().strip()
            old_default = str(default_output_path(Path(previous))) if previous else ""
            update_output = not self.output_var.get().strip() or self.output_var.get().strip() == old_default
            self.video_var.set(selected)
            if update_output:
                self.output_var.set(str(default_output_path(Path(selected))))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="选择输出目录")
        if selected:
            self.output_var.set(selected)

    def _append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        if self.running:
            return
        video = Path(self.video_var.get().strip().strip('"')).expanduser()
        if not video.is_file():
            messagebox.showerror("无法开始", "请选择存在的视频文件。")
            return
        output_value = self.output_var.get().strip().strip('"')
        output = Path(output_value).expanduser() if output_value else default_output_path(video)
        if output.resolve() == video.resolve():
            messagebox.showerror("无法开始", "输出目录不能与视频文件相同。")
            return
        self.output_path = output.resolve()
        self.running = True
        self.stopping = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.report_button.configure(state="disabled")
        self.bundle_button.configure(state="disabled")
        self.status_var.set("正在启动…")
        self.progress.start(12)
        self._append_log(f"开始处理：{video}")
        command = build_command(
            video.resolve(), self.output_path, self.model_var.get(), self.device_var.get(),
            self.ocr_var.get(), self.llm_var.get(),
        )
        threading.Thread(target=self._run_process, args=(command,), daemon=True).start()

    def _run_process(self, command: list[str]) -> None:
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(
                command, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=creationflags,
            )
            self.process = process
            if self.stopping:
                self._terminate_process(process)
            assert process.stdout is not None
            last_scan_progress = -1
            for line in process.stdout:
                line = line.rstrip("\r\n")
                scan = SCAN_PROGRESS_PATTERN.search(line)
                if scan:
                    percent = int(scan.group(1))
                    if percent // 5 > last_scan_progress // 5 or percent == 100:
                        self.events.put(("progress", percent))
                        last_scan_progress = percent
                    continue
                if line:
                    self.events.put(("log", line))
            return_code = process.wait()
            self.events.put(("done", return_code))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.process = None

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    line = str(payload)
                    self._append_log(line)
                    stage = STAGE_PATTERN.match(line)
                    if stage:
                        self.status_var.set(f"步骤 {stage.group(1)}/{stage.group(2)}：{stage.group(3)}")
                elif kind == "done":
                    self._finish(int(payload))
                elif kind == "progress":
                    self.status_var.set(f"正在检测场景：{payload}%")
                elif kind == "error":
                    self._append_log(f"[ERROR] {payload}")
                    self._finish(1)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _finish(self, return_code: int) -> None:
        self.running = False
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.stopping:
            self.status_var.set("已停止。再次开始会按缓存继续处理。")
        elif return_code == 0:
            self.status_var.set("完成。资料包可以打开或上传。")
        else:
            self.status_var.set(f"处理失败，退出码 {return_code}。请查看日志。")
        if self.output_path and self.output_path.is_dir():
            self.open_button.configure(state="normal")
            if return_code == 0 and (self.output_path / "report.html").is_file():
                self.report_button.configure(state="normal")
            if return_code == 0 and (self.output_path / "chatgpt" / "upload_bundle.zip").is_file():
                self.bundle_button.configure(state="normal")

    def _stop(self) -> None:
        process = self.process
        if not self.running:
            return
        self.stopping = True
        self.status_var.set("正在停止…")
        self.stop_button.configure(state="disabled")
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            # Terminate this job's exact process tree, including ffmpeg when active.
            subprocess.Popen(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()

    def _open_output(self) -> None:
        if self.output_path and self.output_path.is_dir():
            os.startfile(self.output_path)

    def _open_report(self) -> None:
        if self.output_path:
            report = self.output_path / "report.html"
            if report.is_file():
                os.startfile(report)

    def _open_bundle(self) -> None:
        if self.output_path:
            bundle = self.output_path / "chatgpt" / "upload_bundle.zip"
            if bundle.is_file():
                os.startfile(bundle.parent)

    def _on_close(self) -> None:
        if self.running:
            messagebox.showinfo("正在处理", "请先点击“停止”，等待任务结束后再关闭窗口。")
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Video2AIApp(root)
    root.mainloop()
