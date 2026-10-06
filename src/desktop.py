"""Local Windows desktop front end for the video2ai CLI."""

from __future__ import annotations

import json
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
from .config import load_dotenv, load_yaml_options
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
SUMMARY_PROGRESS_PATTERN = re.compile(r"^Summary: chunk (\d+)/(\d+)")
OCR_PROGRESS_PATTERN = re.compile(r"^OCR progress: (\d+) unique frames")
VISION_PROGRESS_PATTERN = re.compile(r"^Vision: (\d+)/(\d+) frames described")
BATCH_PATTERN = re.compile(r"^=== \[(\d+)/(\d+)\] (.+?) ===$")
KEY_FILE_NAME = "OPENAI_API_KEY.txt"
FONT = "Microsoft YaHei UI"

# Shared palette with report.html so the app and the generated report feel alike.
COLORS = {
    "bg": "#f4f7fa",
    "panel": "#ffffff",
    "header": "#10243a",
    "header_sub": "#9cc8e3",
    "text": "#172333",
    "muted": "#5c7183",
    "border": "#dfe7ef",
    "accent": "#2b7cd3",
    "accent_active": "#1f66b4",
    "log_bg": "#101828",
    "log_fg": "#d7e2ee",
}


def load_api_key_from_project(project_dir: Path) -> bool:
    """GUI-side API key detection: .env first, then the beginner-friendly key file.

    Windows Notepad cannot easily create a ".env" file, so the portable build
    also accepts OPENAI_API_KEY.txt next to the exe; the worker subprocess
    inherits the exported variable.
    """
    if os.getenv("OPENAI_API_KEY"):
        return True
    key_file = project_dir / KEY_FILE_NAME
    if key_file.is_file():
        try:
            value = key_file.read_text(encoding="utf-8-sig", errors="ignore").strip()
        except OSError:
            return False
        if value:
            os.environ["OPENAI_API_KEY"] = value
            return True
    return False


def _settings_path() -> Path:
    base = os.getenv("APPDATA")
    directory = Path(base) / "Video2AI" if base else PROJECT_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "gui_settings.json"
    except OSError:
        return PROJECT_DIR / "gui_settings.json"


def load_gui_settings() -> dict:
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_gui_settings(settings: dict) -> None:
    try:
        _settings_path().write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def write_env_key(env_path: Path, key: str) -> None:
    """Merge OPENAI_API_KEY into .env without touching other lines."""
    lines: list[str] = []
    if env_path.is_file():
        try:
            lines = [
                line for line in env_path.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("OPENAI_API_KEY=")
            ]
        except OSError:
            lines = []
    if key:
        lines.append(f"OPENAI_API_KEY={key}")
    try:
        env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except OSError:
        pass


def build_command(
    video: Path,
    output: Path | None,
    model: str,
    device: str,
    ocr: bool,
    llm: bool,
    vlm: bool | None = None,
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
    if vlm is not None:
        command.append("--vlm" if vlm else "--no-vlm")
    return command


class Video2AIApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"Video → AI 资料包  {__version__}")
        self.root.minsize(780, 620)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.running = False
        self.stopping = False
        self.output_path: Path | None = None
        self.portable = bool(getattr(sys, "frozen", False))

        load_dotenv(PROJECT_DIR / ".env")
        self.config_error: str | None = None
        try:
            defaults = load_yaml_options(CONFIG_PATH) if CONFIG_PATH.is_file() else {}
        except (FileNotFoundError, ValueError) as exc:
            # A broken config must not stop the app from opening at all.
            defaults = {}
            self.config_error = str(exc)
        self.settings = load_gui_settings()
        geometry = str(self.settings.get("geometry") or "960x740")
        try:
            self.root.geometry(geometry)
        except tk.TclError:
            self.root.geometry("960x740")

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.info_var = tk.StringVar(value="支持单个视频，或选择整个文件夹进行批量处理")
        self.model_var = tk.StringVar(value=str(defaults.get("whisper_model", "base" if self.portable else "small")))
        self.device_var = tk.StringVar(value=str(defaults.get("device", "auto")))
        self.ocr_var = tk.BooleanVar(value=not bool(defaults.get("no_ocr", False)))
        self.llm_var = tk.BooleanVar(value=not bool(defaults.get("no_llm", False)))
        self.vlm_var = tk.BooleanVar(value=not bool(defaults.get("no_vlm", True)))
        if self.portable:
            self.device_var.set("cpu")
            self.ocr_var.set(False)
            # Beginners get AI summary out of the box once the key file exists.
            self.llm_var.set(load_api_key_from_project(PROJECT_DIR))
            self.vlm_var.set(load_api_key_from_project(PROJECT_DIR))
        self.key_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择视频或文件夹")
        self.stage_total = 8
        self._setup_style()
        self._build_ui()
        self._refresh_key_status()
        if self.config_error:
            self._append_log(f"[WARN] 配置文件读取失败，已使用默认设置：{self.config_error}")
            self.status_var.set("配置文件无效，已使用默认设置")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_events)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.root.configure(bg=COLORS["bg"])
        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"], font=(FONT, 10))
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", foreground=COLORS["muted"], font=(FONT, 9))
        style.configure("FieldLabel.TLabel", font=(FONT, 9, "bold"), foreground=COLORS["muted"])
        style.configure("TButton", padding=(12, 6), borderwidth=1, relief="solid", bordercolor=COLORS["border"], background="white")
        style.map("TButton", background=[("active", "#e3ebf2"), ("pressed", "#d5e2ee")])
        style.configure(
            "Accent.TButton", background=COLORS["accent"], foreground="white",
            font=(FONT, 10, "bold"), padding=(16, 6),
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#a8c4e4"), ("active", COLORS["accent_active"])],
            foreground=[("disabled", "#eef4fb")],
        )
        style.configure("TEntry", fieldbackground="white", bordercolor=COLORS["border"], lightcolor=COLORS["border"], padding=5)
        style.map("TEntry", bordercolor=[("focus", COLORS["accent"])])
        style.configure("TCombobox", fieldbackground="white", background="white", bordercolor=COLORS["border"], arrowcolor=COLORS["text"], padding=3)
        style.map("TCombobox", fieldbackground=[("readonly", "white")], background=[("readonly", "white")])
        style.configure("TLabelframe", background=COLORS["bg"], bordercolor=COLORS["border"], relief="solid")
        style.configure("TLabelframe.Label", background=COLORS["bg"], foreground=COLORS["muted"], font=(FONT, 9, "bold"))
        style.configure(
            "Horizontal.TProgressbar", background=COLORS["accent"],
            troughcolor="#e3ebf2", bordercolor=COLORS["bg"], lightcolor=COLORS["accent"], darkcolor=COLORS["accent"],
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=COLORS["header"], height=76)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_box = tk.Frame(header, bg=COLORS["header"])
        title_box.pack(side="left", padx=22, pady=12)
        tk.Label(
            title_box, text="Video → AI 资料包", font=(FONT, 17, "bold"),
            bg=COLORS["header"], fg="white",
        ).pack(anchor="w")
        tk.Label(
            title_box, text=f"v{__version__} · 本地视频转 AI 证据资料包", font=(FONT, 9),
            bg=COLORS["header"], fg=COLORS["header_sub"],
        ).pack(anchor="w")
        if self.portable:
            tk.Label(
                header, text=" 新手版 ", font=(FONT, 9, "bold"),
                bg="#28445f", fg=COLORS["header_sub"], padx=10, pady=4,
            ).pack(side="right", padx=22)

        body = ttk.Frame(self.root, padding=(20, 14, 20, 8))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(6, weight=1)

        ttk.Label(body, text="视频或文件夹", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w")
        video_row = ttk.Frame(body)
        video_row.grid(row=1, column=0, sticky="ew")
        video_row.columnconfigure(0, weight=1)
        ttk.Entry(video_row, textvariable=self.video_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(video_row, text="选择视频…", command=self._choose_video).grid(row=0, column=1)
        ttk.Button(video_row, text="选择文件夹（批量）…", command=self._choose_folder).grid(row=0, column=2, padx=(8, 0))
        ttk.Label(body, textvariable=self.info_var, style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(4, 8))

        output_row = ttk.Frame(body)
        output_row.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        output_row.columnconfigure(1, weight=1)
        ttk.Label(output_row, text="输出目录", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(output_row, textvariable=self.output_var).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(output_row, text="选择目录…", command=self._choose_output).grid(row=0, column=2)

        settings = ttk.Frame(body)
        settings.grid(row=4, column=0, sticky="ew", pady=(2, 10))
        ttk.Label(settings, text="语音模型", style="FieldLabel.TLabel").pack(side="left")
        ttk.Combobox(
            settings, textvariable=self.model_var, state="readonly", width=11,
            values=("tiny", "base", "small") if self.portable else ("tiny", "base", "small", "medium", "large-v3"),
        ).pack(side="left", padx=(6, 18))
        ttk.Label(settings, text="计算设备", style="FieldLabel.TLabel").pack(side="left")
        ttk.Combobox(
            settings, textvariable=self.device_var,
            state="disabled" if self.portable else "readonly", width=7,
            values=("cpu",) if self.portable else ("auto", "cuda", "cpu"),
        ).pack(side="left", padx=(6, 4 if self.portable else 18))
        if self.portable:
            ttk.Label(
                settings, text="新手版仅支持 CPU，完整版可用 GPU", style="Muted.TLabel",
            ).pack(side="left", padx=(0, 12))
        # Classic tk checkbuttons draw a real checkmark; the clam ttk theme
        # draws an "x", which reads as "disabled" to Chinese users.
        check_style = dict(
            bg=COLORS["bg"], activebackground=COLORS["bg"], font=(FONT, 10),
            selectcolor="white", highlightthickness=0, bd=0, padx=0, pady=0,
        )
        tk.Checkbutton(
            settings, text="画面文字 OCR", variable=self.ocr_var,
            state="disabled" if self.portable else "normal", **check_style,
        ).pack(side="left", padx=(0, 14))
        tk.Checkbutton(settings, text="AI 总结", variable=self.llm_var, **check_style).pack(side="left", padx=(0, 14))
        tk.Checkbutton(settings, text="AI 看图", variable=self.vlm_var, **check_style).pack(side="left", padx=(0, 14))
        ttk.Button(settings, text="密钥设置…", command=self._open_key_dialog).pack(side="left")
        ttk.Label(settings, textvariable=self.key_var, style="Muted.TLabel").pack(side="left", padx=(10, 0))

        actions = ttk.Frame(body)
        actions.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        self.start_button = ttk.Button(actions, text="▶ 开始生成", style="Accent.TButton", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="■ 停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(actions, text="打开资料包", command=self._open_output, state="disabled")
        self.open_button.pack(side="left", padx=(14, 8))
        self.report_button = ttk.Button(actions, text="查看网页报告", command=self._open_report, state="disabled")
        self.report_button.pack(side="left", padx=(0, 8))
        self.bundle_button = ttk.Button(actions, text="打开上传包目录", command=self._open_bundle, state="disabled")
        self.bundle_button.pack(side="left")

        log_frame = ttk.Labelframe(body, text=" 处理日志 ", padding=10)
        log_frame.grid(row=6, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame, state="disabled", wrap="word", font=("Consolas", 10),
            bg=COLORS["log_bg"], fg=COLORS["log_fg"], insertbackground="white",
            selectbackground="#33415c", relief="flat", borderwidth=0, padx=10, pady=8, height=8,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        self.log.tag_configure("error", foreground="#ff9b9b")
        self.log.tag_configure("warn", foreground="#f5c26b")
        self.log.tag_configure("skip", foreground="#7d8da1")
        self.log.tag_configure("stage", foreground="#7cb8f5", font=("Consolas", 10, "bold"))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x")
        status_bar = ttk.Frame(self.root, padding=(20, 8, 20, 12))
        status_bar.pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(status_bar, mode="determinate", maximum=100, length=240)
        self.progress.pack(side="right")

    def _choose_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择视频", filetypes=VIDEO_TYPES,
            initialdir=str(self.settings.get("last_video_dir") or ""),
        )
        if selected:
            path = Path(selected)
            self._apply_video_selection(path, remember_dir=str(path.parent))

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="选择视频文件夹（批量处理）",
            initialdir=str(self.settings.get("last_video_dir") or ""),
        )
        if selected:
            path = Path(selected)
            self._apply_video_selection(path, remember_dir=str(path))

    def _apply_video_selection(self, path: Path, remember_dir: str) -> None:
        previous = self.video_var.get().strip()
        old_default = str(default_output_path(Path(previous))) if previous else ""
        update_output = not self.output_var.get().strip() or self.output_var.get().strip() == old_default
        self.video_var.set(str(path))
        if update_output:
            self.output_var.set(str(default_output_path(path)) if path.is_file() else "")
        self.settings["last_video_dir"] = remember_dir
        self._show_video_info(path)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(
            title="选择输出目录",
            initialdir=str(self.settings.get("last_output_dir") or ""),
        )
        if selected:
            self.output_var.set(selected)
            self.settings["last_output_dir"] = selected

    def _show_video_info(self, path: Path) -> None:
        try:
            if path.is_dir():
                from .cli import expand_inputs

                count = len(expand_inputs(path, None))
                self.info_var.set(f"批量模式：{count} 个视频，将逐个生成到各自的 _ai 文件夹")
                return
            import cv2

            capture = cv2.VideoCapture(str(path))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                capture.release()
            duration = frames / fps if fps > 0 else 0.0
            minutes, seconds = divmod(int(duration), 60)
            size_mb = path.stat().st_size / 1024 / 1024
            self.info_var.set(f"时长 {minutes:02d}:{seconds:02d} · {width}x{height} · {size_mb:.1f} MB")
        except Exception:
            self.info_var.set("")

    def _append_log(self, line: str) -> None:
        tag = None
        if line.startswith("[ERROR]"):
            tag = "error"
        elif line.startswith("[WARN]"):
            tag = "warn"
        elif line.startswith("[SKIP]"):
            tag = "skip"
        elif STAGE_PATTERN.match(line) or line.startswith("==="):
            tag = "stage"
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_progress(self, value: float) -> None:
        self.progress.configure(value=min(100.0, max(0.0, value)))

    def _stage_progress(self, stage: int, total: int | None = None) -> None:
        total = total or self.stage_total
        self._set_progress((stage - 1) / total * 100)

    def _sub_progress(self, stage: int, fraction: float, total: int | None = None) -> None:
        total = total or self.stage_total
        self._set_progress((stage - 1 + min(max(fraction, 0.0), 1.0)) / total * 100)

    def _refresh_key_status(self) -> None:
        self.key_var.set("密钥已配置" if load_api_key_from_project(PROJECT_DIR) else "未配置密钥")

    def _open_key_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("API 密钥设置")
        dialog.configure(bg=COLORS["bg"])
        dialog.transient(self.root)
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="OPENAI_API_KEY", style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(frame, show="*", width=54)
        entry.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        entry.insert(0, os.getenv("OPENAI_API_KEY", ""))
        ttk.Label(
            frame,
            text="密钥保存在本程序目录的 .env 文件中，不会进入 Git。\n用于 AI 总结与 AI 看图；"
                 "支持阿里云百炼等 OpenAI-compatible 服务。",
            style="Muted.TLabel", justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(0, 14))
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, sticky="e")

        def save() -> None:
            value = entry.get().strip()
            write_env_key(PROJECT_DIR / ".env", value)
            if value:
                os.environ["OPENAI_API_KEY"] = value
            self._refresh_key_status()
            dialog.destroy()

        ttk.Button(buttons, text="保存", style="Accent.TButton", command=save).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="left")
        entry.focus_set()
        dialog.grab_set()

    def _start(self) -> None:
        if self.running:
            return
        video = Path(self.video_var.get().strip().strip('"')).expanduser()
        if video.is_dir():
            # Batch mode: the CLI picks per-video default outputs itself.
            output = None
            self.output_path = video
        elif video.is_file():
            output_value = self.output_var.get().strip().strip('"')
            output = Path(output_value).expanduser() if output_value else default_output_path(video)
            if output.resolve() == video.resolve():
                messagebox.showerror("无法开始", "输出目录不能与视频文件相同。")
                return
            self.output_path = output.resolve()
        else:
            messagebox.showerror("无法开始", "请选择存在的视频文件或文件夹。")
            return
        if self.llm_var.get() or self.vlm_var.get():
            if not load_api_key_from_project(PROJECT_DIR):
                if not messagebox.askyesno(
                    "AI 功能",
                    "未检测到 API Key，本次将生成不含 AI 总结和 AI 看图的基础资料包。\n\n"
                    "启用方法：点击“密钥设置…”粘贴 API Key，或在本程序文件夹内新建文本文件 "
                    "OPENAI_API_KEY.txt。\n\n是否继续生成？",
                ):
                    return
        self.running = True
        self.stopping = False
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.report_button.configure(state="disabled")
        self.bundle_button.configure(state="disabled")
        self.status_var.set("批量模式，正在启动…" if video.is_dir() else "正在启动…")
        self._set_progress(0)
        self._append_log(f"开始处理：{video}")
        command = build_command(
            video, output, self.model_var.get(), self.device_var.get(),
            self.ocr_var.get(), self.llm_var.get(), vlm=self.vlm_var.get(),
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
                        total = int(stage.group(2))
                        self.stage_total = total
                        self.status_var.set(f"步骤 {stage.group(1)}/{total}：{stage.group(3)}")
                        self._stage_progress(int(stage.group(1)), total)
                        continue
                    summary_progress = SUMMARY_PROGRESS_PATTERN.match(line)
                    if summary_progress:
                        done, total = int(summary_progress.group(1)), int(summary_progress.group(2))
                        self.status_var.set(f"正在生成 AI 总结：第 {done}/{total} 块")
                        self._sub_progress(self.stage_total, done / total if total else 0)
                        continue
                    vision_progress = VISION_PROGRESS_PATTERN.match(line)
                    if vision_progress:
                        done, total = int(vision_progress.group(1)), int(vision_progress.group(2))
                        self.status_var.set(f"正在理解画面：{done}/{total} 帧")
                        self._sub_progress(6, done / total if total else 0, 8)
                        continue
                    ocr_progress = OCR_PROGRESS_PATTERN.match(line)
                    if ocr_progress:
                        self.status_var.set(f"正在识别画面文字：已处理 {ocr_progress.group(1)} 帧")
                        self._sub_progress(5, 0.5, 8)
                        continue
                    batch = BATCH_PATTERN.match(line)
                    if batch:
                        self.status_var.set(f"批量处理 [{batch.group(1)}/{batch.group(2)}]：{batch.group(3)}")
                        self._set_progress(0)
                elif kind == "done":
                    self._finish(int(payload))
                elif kind == "progress":
                    percent = int(payload)
                    self.status_var.set(f"正在检测场景：{percent}%")
                    self._sub_progress(4, percent / 100)
                elif kind == "error":
                    self._append_log(f"[ERROR] {payload}")
                    self._finish(1)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _finish(self, return_code: int) -> None:
        self.running = False
        if return_code == 0:
            self._set_progress(100)
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
        self.settings["geometry"] = self.root.winfo_geometry()
        save_gui_settings(self.settings)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Video2AIApp(root)
    root.mainloop()
