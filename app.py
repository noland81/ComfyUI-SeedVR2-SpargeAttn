#!/usr/bin/env python3
"""
SeedVR2 GPU-Direct Upscaler — Standalone PySide6 Desktop App

Bypasses ComfyUI entirely, calling the SeedVR2 pipeline directly on GPU.
Models are pre-loaded into VRAM and cached between runs for maximum speed.

Usage:
    python app.py

Requirements:
    pip install PySide6
"""

import sys
import os
import io
import json
import time
import subprocess

# Fix Windows console encoding (cp1252 can't handle emoji in SeedVR2 logs)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Setup path before any project imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# CUDA config before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("LOCAL_RANK", "0")

import psutil
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QProgressBar,
    QFileDialog, QGroupBox, QSplitter, QStatusBar, QSizePolicy, QScrollArea,
    QPlainTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QMimeData, QTimer
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QFont, QPalette, QColor

# Pipeline module — shared between GUI app and backend
from pipeline import (
    RESOLUTION_PRESETS, SeedVR2Pipeline, run_upscale,
    _resize_to_megapixels, _resize_keep_proportion,
    _make_dimensions_even, _apply_gaussian_blur,
    _tile_and_upscale,
)

# SeedVR2 imports (still needed for model registry in GUI)
from src.utils.model_registry import get_available_dit_models, get_available_vae_models, DEFAULT_VAE


# ─── QThread Worker ───────────────────────────────────────────────────

class ModelLoadWorker(QThread):
    """Loads models in background thread."""
    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, pipeline, blocks_to_swap=9, vae_tile_size=768,
                 vae_tile_overlap=32):
        super().__init__()
        self.pipeline = pipeline
        self.blocks_to_swap = blocks_to_swap
        self.vae_tile_size = vae_tile_size
        self.vae_tile_overlap = vae_tile_overlap

    def run(self):
        try:
            self.pipeline.load_models(
                blocks_to_swap=self.blocks_to_swap,
                vae_tile_size=self.vae_tile_size,
                vae_tile_overlap=self.vae_tile_overlap,
                progress_fn=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class UpscaleWorker(QThread):
    """Runs upscaling in background thread."""
    progress = Signal(int, str)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, pipeline, image_np, preset_name, seed, color_correction,
                 input_noise_scale, latent_noise_scale):
        super().__init__()
        self.pipeline = pipeline
        self.image_np = image_np
        self.preset_name = preset_name
        self.seed = seed
        self.color_correction = color_correction
        self.input_noise_scale = input_noise_scale
        self.latent_noise_scale = latent_noise_scale

    def run(self):
        try:
            result = run_upscale(
                self.pipeline, self.image_np, self.preset_name,
                seed=self.seed,
                color_correction=self.color_correction,
                input_noise_scale=self.input_noise_scale,
                latent_noise_scale=self.latent_noise_scale,
                progress_fn=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ─── Image Display Widget ────────────────────────────────────────────

class ImageLabel(QLabel):
    """QLabel with drag & drop and click-to-browse support."""
    clicked = Signal()

    def __init__(self, placeholder_text="Drop image here\nor click to browse"):
        super().__init__()
        self.placeholder_text = placeholder_text
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAcceptDrops(True)
        self.setText(placeholder_text)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                border-radius: 8px;
                background-color: #1a1a2e;
                color: #888;
                font-size: 14px;
                padding: 10px;
            }
        """)
        self._pixmap = None

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._update_display()

    def _update_display(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    event.acceptProposedAction()
                    self.setStyleSheet(self.styleSheet().replace("#555", "#4fc3f7"))
                    return

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.styleSheet().replace("#4fc3f7", "#555"))

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self.styleSheet().replace("#4fc3f7", "#555"))
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                self.clicked.emit()  # Will be handled by parent
                # Parent will handle via the path
                self.setProperty("dropped_path", path)
                self.clicked.emit()
                return


# ─── System Monitor Widget ────────────────────────────────────────────

def _query_nvidia_smi():
    """Query GPU utilization, VRAM, and temperature via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            vram_used = int(parts[1].strip())
            vram_total = int(parts[2].strip())
            return {
                'gpu_util': int(parts[0].strip()),
                'vram_used_mb': vram_used,
                'vram_total_mb': vram_total,
                'vram_pct': int(vram_used / vram_total * 100) if vram_total > 0 else 0,
                'temp': int(parts[3].strip()),
            }
    except Exception:
        pass
    return None


class SystemMonitor(QWidget):
    """Crystools-style system monitor: CPU, RAM, GPU, VRAM, Temp."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        # Create monitor bars
        self._bars = {}
        for name, color in [("CPU", "#4caf50"), ("RAM", "#2196f3"),
                             ("GPU", "#ff9800"), ("VRAM", "#9c27b0")]:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; "
                              "background: transparent; padding: 0; margin: 0;")
            lbl.setFixedWidth(32)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedWidth(60)
            bar.setFixedHeight(14)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #333; border-radius: 3px;
                    background-color: #161b22; text-align: center;
                    font-size: 10px; color: #ccc;
                }}
                QProgressBar::chunk {{
                    background-color: {color}; border-radius: 2px;
                }}
            """)

            layout.addWidget(lbl)
            layout.addWidget(bar)
            self._bars[name] = bar

        # Temperature label
        self._temp_label = QLabel("--°C")
        self._temp_label.setStyleSheet(
            "color: #ef5350; font-size: 11px; font-weight: bold; "
            "background: transparent; padding: 0 4px; margin: 0;")
        layout.addWidget(self._temp_label)

        # Timer — update every 2 seconds
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(2000)
        self._update()

    def _update(self):
        # CPU & RAM (psutil)
        try:
            self._bars["CPU"].setValue(int(psutil.cpu_percent(interval=0)))
            mem = psutil.virtual_memory()
            self._bars["RAM"].setValue(int(mem.percent))
        except Exception:
            pass

        # GPU, VRAM, Temp (nvidia-smi)
        stats = _query_nvidia_smi()
        if stats:
            self._bars["GPU"].setValue(stats['gpu_util'])
            self._bars["VRAM"].setValue(stats['vram_pct'])
            temp = stats['temp']
            self._temp_label.setText(f"{temp}°C")
            # Color temperature: green < 60, yellow 60-80, red > 80
            if temp < 60:
                tcolor = "#4caf50"
            elif temp < 80:
                tcolor = "#ff9800"
            else:
                tcolor = "#ef5350"
            self._temp_label.setStyleSheet(
                f"color: {tcolor}; font-size: 11px; font-weight: bold; "
                "background: transparent; padding: 0 4px; margin: 0;")


# ─── Log Stream Redirector ────────────────────────────────────────────

class LogStream(io.TextIOBase):
    """Redirect stdout/stderr to a Qt signal while keeping original stream."""

    def __init__(self, original_stream, signal_fn):
        super().__init__()
        self._original = original_stream
        self._signal_fn = signal_fn

    def write(self, text):
        if text:
            # Write to original stream (terminal)
            if self._original:
                try:
                    self._original.write(text)
                    self._original.flush()
                except Exception:
                    pass
            # Emit to GUI (strip trailing newline for cleaner display)
            self._signal_fn(text)
        return len(text) if text else 0

    def flush(self):
        if self._original:
            try:
                self._original.flush()
            except Exception:
                pass

    def fileno(self):
        if self._original:
            return self._original.fileno()
        raise io.UnsupportedOperation("fileno")

    @property
    def encoding(self):
        return getattr(self._original, 'encoding', 'utf-8')

    def isatty(self):
        return False


class LogBridge(QWidget):
    """Bridge to emit log text as Qt signal (thread-safe for cross-thread writes)."""
    log_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)


# ─── Main Window ──────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SeedVR2 GPU-Direct Upscaler")
        self.setMinimumSize(900, 650)
        self.resize(1100, 750)

        # State
        self.input_image_path = None
        self.input_image_np = None
        self.output_image_np = None
        self.pipeline = None
        self.worker = None
        self._pending_upscale = False
        self._last_open_dir = ""   # remembered between sessions via config.json
        self._last_save_dir = ""

        # Dark theme
        self._apply_dark_theme()

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Left panel — main content (images, controls, progress, buttons)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # ── Image panels ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Input image
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        self.input_label = ImageLabel("Drop image here\nor click to browse")
        self.input_label.clicked.connect(self._on_input_click)
        input_layout.addWidget(self.input_label)
        self.input_info = QLabel("")
        self.input_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_info.setStyleSheet("color: #aaa; font-size: 11px;")
        input_layout.addWidget(self.input_info)
        splitter.addWidget(input_group)

        # Output image
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        self.output_label = ImageLabel("Output will appear here")
        self.output_label.setAcceptDrops(False)
        output_layout.addWidget(self.output_label)
        self.output_info = QLabel("")
        self.output_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_info.setStyleSheet("color: #aaa; font-size: 11px;")
        output_layout.addWidget(self.output_info)
        splitter.addWidget(output_group)

        splitter.setSizes([500, 500])
        left_layout.addWidget(splitter, stretch=1)

        # ── Controls ──
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(8)

        # Row 1: Resolution presets
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.res_buttons = {}
        for name in RESOLUTION_PRESETS:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumWidth(50)
            btn.setStyleSheet("""
                QPushButton { padding: 6px 12px; border-radius: 4px; border: 1px solid #444; }
                QPushButton:checked { background-color: #1976d2; border-color: #42a5f5; color: white; }
                QPushButton:hover { border-color: #666; }
            """)
            btn.clicked.connect(lambda checked, n=name: self._on_preset_click(n))
            self.res_buttons[name] = btn
            res_row.addWidget(btn)
        res_row.addStretch()
        controls_layout.addLayout(res_row)

        # Row 2: Model selection (always visible)
        model_row = QHBoxLayout()

        model_row.addWidget(QLabel("DiT Model:"))
        self.model_combo = QComboBox()
        self._populate_models()
        self.model_combo.setMinimumWidth(320)
        self.model_combo.setToolTip("DiT model for upscaling.\n"
                                     "Q8_0 = best quality GGUF (8.8 GB)\n"
                                     "Q6_K = good balance (6.9 GB)\n"
                                     "Q5_K_M / Q4_K_M = smaller, faster\n"
                                     "fp16 = full precision (16.5 GB)\n"
                                     "sharp = enhanced detail variant")
        model_row.addWidget(self.model_combo)

        model_row.addSpacing(16)

        model_row.addWidget(QLabel("VAE:"))
        self.vae_combo = QComboBox()
        self._populate_vae_models()
        self.vae_combo.setMinimumWidth(200)
        model_row.addWidget(self.vae_combo)

        model_row.addStretch()
        controls_layout.addLayout(model_row)

        # Row 3: Basic params
        params_row = QHBoxLayout()

        params_row.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        self.seed_spin.setMinimumWidth(100)
        params_row.addWidget(self.seed_spin)

        params_row.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
        self.color_combo.setMinimumWidth(100)
        params_row.addWidget(self.color_combo)

        params_row.addStretch()
        controls_layout.addLayout(params_row)

        # Row 4: Advanced settings (collapsible)
        self.advanced_group = QGroupBox("Advanced Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        adv_layout = QHBoxLayout(self.advanced_group)

        # Attention mode
        adv_layout.addWidget(QLabel("Attention:"))
        self.attn_combo = QComboBox()
        self.attn_combo.addItems(["spargeattn", "sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"])
        adv_layout.addWidget(self.attn_combo)

        # Blocks to swap (DiT offload to CPU for VRAM saving)
        adv_layout.addWidget(QLabel("Blocks swap:"))
        self.blocks_spin = QSpinBox()
        self.blocks_spin.setRange(0, 40)
        self.blocks_spin.setValue(9)
        self.blocks_spin.setToolTip("DiT transformer blocks offloaded to CPU.\n"
                                     "Higher = less VRAM but slower.\n"
                                     "2K-4K: 9, 6K: 16, 9K-12K: 32")
        adv_layout.addWidget(self.blocks_spin)

        # VAE tile size
        adv_layout.addWidget(QLabel("VAE tile:"))
        self.vae_tile_spin = QSpinBox()
        self.vae_tile_spin.setRange(256, 2048)
        self.vae_tile_spin.setSingleStep(128)
        self.vae_tile_spin.setValue(768)
        self.vae_tile_spin.setToolTip("VAE encode/decode tile size.\n"
                                      "2K-4K: 1024, 6K+: 768\n"
                                      "Smaller = less VRAM")
        adv_layout.addWidget(self.vae_tile_spin)

        # VAE tile overlap
        adv_layout.addWidget(QLabel("Overlap:"))
        self.vae_overlap_spin = QSpinBox()
        self.vae_overlap_spin.setRange(8, 128)
        self.vae_overlap_spin.setSingleStep(8)
        self.vae_overlap_spin.setValue(32)
        self.vae_overlap_spin.setToolTip("VAE tile overlap.\n2K-4K: 64, 6K+: 32")
        adv_layout.addWidget(self.vae_overlap_spin)

        adv_layout.addStretch()
        controls_layout.addWidget(self.advanced_group)

        left_layout.addLayout(controls_layout)

        # ── Progress bar ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% — %v")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 4px;
                text-align: center;
                height: 24px;
                background-color: #1a1a2e;
                color: #ccc;
            }
            QProgressBar::chunk {
                background-color: #1976d2;
                border-radius: 3px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        # ── Action buttons ──
        btn_row = QHBoxLayout()

        self.load_btn = QPushButton("Load Models")
        self.load_btn.setMinimumHeight(38)
        self.load_btn.setStyleSheet("""
            QPushButton { padding: 8px 20px; background-color: #2e7d32; border-radius: 6px;
                          font-weight: bold; font-size: 13px; color: white; }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.load_btn.clicked.connect(self._on_load_models)
        btn_row.addWidget(self.load_btn)

        self.upscale_btn = QPushButton("Upscale")
        self.upscale_btn.setMinimumHeight(38)
        self.upscale_btn.setEnabled(False)
        self.upscale_btn.setStyleSheet("""
            QPushButton { padding: 8px 20px; background-color: #1565c0; border-radius: 6px;
                          font-weight: bold; font-size: 13px; color: white; }
            QPushButton:hover { background-color: #1976d2; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.upscale_btn.clicked.connect(self._on_upscale)
        btn_row.addWidget(self.upscale_btn)

        self.save_btn = QPushButton("Save Output")
        self.save_btn.setMinimumHeight(38)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton { padding: 8px 20px; background-color: #6a1b9a; border-radius: 6px;
                          font-weight: bold; font-size: 13px; color: white; }
            QPushButton:hover { background-color: #7b1fa2; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)

        left_layout.addLayout(btn_row)

        # Add left panel to main layout
        main_layout.addWidget(left_widget, stretch=1)

        # ── Terminal log panel (right side, toggleable) ──
        self._log_visible = True
        self.terminal_widget = QWidget()
        terminal_layout = QVBoxLayout(self.terminal_widget)
        terminal_layout.setContentsMargins(0, 0, 0, 0)
        terminal_layout.setSpacing(4)

        terminal_header = QHBoxLayout()
        self.terminal_toggle_btn = QPushButton("Hide \u25b6")
        self.terminal_toggle_btn.setFixedHeight(24)
        self.terminal_toggle_btn.setStyleSheet("""
            QPushButton { padding: 2px 12px; background-color: #21262d; border: 1px solid #333;
                          border-radius: 4px; font-size: 11px; color: #8b949e; }
            QPushButton:hover { background-color: #30363d; border-color: #555; color: #c9d1d9; }
        """)
        self.terminal_toggle_btn.clicked.connect(self._toggle_terminal)
        terminal_header.addStretch()
        terminal_header.addWidget(self.terminal_toggle_btn)
        terminal_layout.addLayout(terminal_header)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(5000)  # limit buffer
        self.log_console.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 4px;
                font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 4px;
                selection-background-color: #264f78;
            }
        """)
        terminal_layout.addWidget(self.log_console)

        self.terminal_widget.setFixedWidth(340)
        main_layout.addWidget(self.terminal_widget)

        # Setup log redirection (stdout + stderr → log_console)
        self._log_bridge = LogBridge(self)
        self._log_bridge.log_signal.connect(self._append_log)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = LogStream(self._original_stdout, self._log_bridge.log_signal.emit)
        sys.stderr = LogStream(self._original_stderr, self._log_bridge.log_signal.emit)

        # ── Elapsed timer (for upscale time tracking) ──
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_msg = ""  # last progress message

        # ── Status bar with system monitor ──
        self.sys_monitor = SystemMonitor()
        self.statusBar().addPermanentWidget(self.sys_monitor)
        self.statusBar().showMessage("Ready — Load models to begin")

        # ── Restore saved config or use defaults ──
        config = self._load_config()
        if config:
            # Restore combo boxes (only if value exists in dropdown)
            idx = self.model_combo.findText(config.get("dit_model", ""))
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            idx = self.vae_combo.findText(config.get("vae_model", ""))
            if idx >= 0:
                self.vae_combo.setCurrentIndex(idx)
            idx = self.attn_combo.findText(config.get("attention_mode", ""))
            if idx >= 0:
                self.attn_combo.setCurrentIndex(idx)
            idx = self.color_combo.findText(config.get("color_correction", ""))
            if idx >= 0:
                self.color_combo.setCurrentIndex(idx)

            # Restore preset (also sets blocks/vae_tile/overlap)
            saved_preset = config.get("preset", "4K")
            if saved_preset in RESOLUTION_PRESETS:
                self._on_preset_click(saved_preset)
            else:
                self._on_preset_click("4K")

            # Restore seed
            self.seed_spin.setValue(config.get("seed", 42))

            # Restore UI state
            if not config.get("terminal_visible", True):
                self._toggle_terminal()
            self.advanced_group.setChecked(config.get("advanced_expanded", False))

            # Restore last directories
            self._last_open_dir = config.get("last_open_dir", "")
            self._last_save_dir = config.get("last_save_dir", "")

            print(f"[CONFIG] Restored settings from config.json "
                  f"(preset={saved_preset}, model={config.get('dit_model', '?')})")
        else:
            # First run — use defaults
            self._on_preset_click("4K")

        # Auto-load models on startup (uses whatever model is currently selected)
        QTimer.singleShot(500, self._on_load_models)

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QWidget { background-color: #0d1117; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 8px;
                        padding-top: 14px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QComboBox, QSpinBox { background-color: #161b22; border: 1px solid #333;
                                  border-radius: 4px; padding: 4px 8px; min-height: 24px; }
            QComboBox:hover, QSpinBox:hover { border-color: #555; }
            QComboBox::drop-down { border: none; }
            QLabel { background: transparent; }
            QCheckBox { spacing: 5px; }
            QStatusBar { background-color: #161b22; border-top: 1px solid #333; }
            QSplitter::handle { background-color: #333; width: 2px; }
        """)

    def _populate_models(self):
        """Populate DiT model dropdown — 7B only, no Q3 (user preference)."""
        try:
            all_models = get_available_dit_models()
            # Filter: 7B only, exclude Q3 variants
            models = [m for m in all_models
                      if "3b" not in m.lower() and "q3" not in m.lower()]
            # Ensure Q8_0 is always in list (used in all workflows)
            q8_name = "seedvr2_ema_7b-Q8_0.gguf"
            if q8_name not in models:
                models.insert(0, q8_name)
            # Sort: GGUF quants by size (Q8>Q6>Q5>Q4), then safetensors
            def sort_key(name):
                n = name.lower()
                if "sharp" in n:
                    group = 1  # sharp after standard
                else:
                    group = 0
                # Priority within group: Q8_0, Q6_K, Q5_K_M, Q4_K_M, fp8, fp16
                for rank, pat in enumerate(["q8_0", "q6_k", "q5_k_m", "q4_k_m",
                                            "fp8_e4m3fn_mixed", "fp8_e4m3fn", "fp16"]):
                    if pat in n:
                        return (group, rank)
                return (group, 99)
            models.sort(key=sort_key)
            self.model_combo.addItems(models)
            # Default: Q8_0
            for i, m in enumerate(models):
                if "7b-q8_0" in m.lower() and "sharp" not in m.lower():
                    self.model_combo.setCurrentIndex(i)
                    break
        except Exception:
            self.model_combo.addItem("seedvr2_ema_7b-Q8_0.gguf")

    def _populate_vae_models(self):
        """Populate VAE model dropdown."""
        try:
            models = get_available_vae_models()
            self.vae_combo.addItems(models)
        except Exception:
            self.vae_combo.addItem(DEFAULT_VAE)

    def _on_preset_click(self, name):
        for n, btn in self.res_buttons.items():
            btn.setChecked(n == name)
        # Update all settings from preset
        preset = RESOLUTION_PRESETS[name]
        self.blocks_spin.setValue(preset["blocks_to_swap"])
        self.vae_tile_spin.setValue(preset["vae_tile_size"])
        self.vae_overlap_spin.setValue(preset["vae_tile_overlap"])

        # Show reload hint if settings differ from loaded model
        if self.pipeline and self.pipeline._loaded:
            if self.pipeline.needs_reload(
                preset["blocks_to_swap"], preset["vae_tile_size"],
                preset["vae_tile_overlap"]
            ):
                self.statusBar().showMessage(
                    f"{name} preset: blocks_to_swap={preset['blocks_to_swap']}, "
                    f"vae_tile={preset['vae_tile_size']} "
                    f"— will auto-reload models on Upscale"
                )

    def _get_selected_preset(self):
        for name, btn in self.res_buttons.items():
            if btn.isChecked():
                return name
        return "4K"

    def _on_input_click(self):
        # Check for dropped path first
        dropped = self.input_label.property("dropped_path")
        if dropped:
            self.input_label.setProperty("dropped_path", None)
            self._load_image(dropped)
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", self._last_open_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
        )
        if path:
            self._last_open_dir = os.path.dirname(path)
            self._load_image(path)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            self.input_image_np = np.array(img)
            self.input_image_path = path

            # Display
            h, w = self.input_image_np.shape[:2]
            qimg = QImage(self.input_image_np.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            self.input_label.set_image(QPixmap.fromImage(qimg))
            self.input_info.setText(f"{os.path.basename(path)} — {w}x{h}")

            # Enable upscale if models loaded
            if self.pipeline and self.pipeline._loaded:
                self.upscale_btn.setEnabled(True)

            self.statusBar().showMessage(f"Loaded: {os.path.basename(path)} ({w}x{h})")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading image: {e}")

    def _on_load_models(self):
        dit_model = self.model_combo.currentText()
        vae_model = self.vae_combo.currentText()
        attention = self.attn_combo.currentText()
        blocks = self.blocks_spin.value()
        vae_tile = self.vae_tile_spin.value()
        vae_overlap = self.vae_overlap_spin.value()

        self.pipeline = SeedVR2Pipeline(
            dit_model=dit_model,
            vae_model=vae_model,
            device="cuda:0",
            attention_mode=attention,
        )

        self.load_btn.setEnabled(False)
        self.upscale_btn.setEnabled(False)
        self.statusBar().showMessage(
            f"Loading models (swap={blocks}, tile={vae_tile}, overlap={vae_overlap})..."
        )

        self._load_worker = ModelLoadWorker(
            self.pipeline, blocks_to_swap=blocks,
            vae_tile_size=vae_tile, vae_tile_overlap=vae_overlap
        )
        self._load_worker.progress.connect(self._on_progress)
        self._load_worker.finished.connect(self._on_models_loaded)
        self._load_worker.error.connect(self._on_error)
        self._load_worker.start()

    def _on_models_loaded(self):
        self.load_btn.setText("Reload Models")
        self.load_btn.setEnabled(True)
        if self.input_image_np is not None:
            self.upscale_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Show VRAM usage and loaded settings
        info_parts = []
        if self.pipeline and self.pipeline._loaded:
            info_parts.append(f"swap={self.pipeline._loaded_blocks_to_swap}")
            info_parts.append(f"tile={self.pipeline._loaded_vae_tile_size}")
            info_parts.append(f"VAE tiled=ON")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info_parts.append(f"VRAM: {allocated:.1f}GB / {total:.1f}GB")
        self.statusBar().showMessage(f"Models loaded — {', '.join(info_parts)}")

        # If upscale was pending (auto-load from Upscale button), start it
        if hasattr(self, '_pending_upscale') and self._pending_upscale:
            self._pending_upscale = False
            preset_name = self._get_selected_preset()
            self._start_upscale(preset_name)

    def _on_upscale(self):
        if self.input_image_np is None:
            return
        if not self.pipeline or not self.pipeline._loaded:
            # Auto-load models first
            self._pending_upscale = True
            self._on_load_models()
            return

        preset_name = self._get_selected_preset()
        preset = RESOLUTION_PRESETS[preset_name]

        # Auto-reload if preset requires different model settings
        if self.pipeline.needs_reload(
            preset["blocks_to_swap"], preset["vae_tile_size"],
            preset["vae_tile_overlap"]
        ):
            self.statusBar().showMessage(
                f"Reloading models for {preset_name} "
                f"(swap: {self.pipeline._loaded_blocks_to_swap}→{preset['blocks_to_swap']}, "
                f"tile: {self.pipeline._loaded_vae_tile_size}→{preset['vae_tile_size']})..."
            )
            # Update UI spins to match preset
            self.blocks_spin.setValue(preset["blocks_to_swap"])
            self.vae_tile_spin.setValue(preset["vae_tile_size"])
            self.vae_overlap_spin.setValue(preset["vae_tile_overlap"])
            # Reload then upscale
            self._pending_upscale = True
            self._do_reload_for_preset()
            return

        self._start_upscale(preset_name)

    def _do_reload_for_preset(self):
        """Reload models with current UI settings, then start upscale."""
        blocks = self.blocks_spin.value()
        vae_tile = self.vae_tile_spin.value()
        vae_overlap = self.vae_overlap_spin.value()

        self.upscale_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.load_btn.setEnabled(False)

        self._load_worker = ModelLoadWorker(
            self.pipeline, blocks_to_swap=blocks,
            vae_tile_size=vae_tile, vae_tile_overlap=vae_overlap
        )
        self._load_worker.progress.connect(self._on_progress)
        self._load_worker.finished.connect(self._on_reload_then_upscale)
        self._load_worker.error.connect(self._on_error)
        self._load_worker.start()

    def _on_reload_then_upscale(self):
        """Called after auto-reload, starts the upscale."""
        self._on_models_loaded()
        if self._pending_upscale:
            self._pending_upscale = False
            preset_name = self._get_selected_preset()
            self._start_upscale(preset_name)

    def _start_upscale(self, preset_name):
        """Start the actual upscale worker."""
        self.upscale_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.load_btn.setEnabled(False)

        preset = RESOLUTION_PRESETS[preset_name]
        info = f"swap={preset['blocks_to_swap']}, vae_tile={preset['vae_tile_size']}"
        if preset.get("pre_resize"):
            info += f", pre-resize={preset['pre_resize']}"
        if preset.get("tile_mode"):
            info += f", tile={preset['tile_mode']}"
        self.statusBar().showMessage(f"Upscaling to {preset_name} ({info})...")

        self.worker = UpscaleWorker(
            pipeline=self.pipeline,
            image_np=self.input_image_np,
            preset_name=preset_name,
            seed=self.seed_spin.value(),
            color_correction=self.color_combo.currentText(),
            input_noise_scale=0.0,
            latent_noise_scale=preset.get("latent_noise_scale", 0.0),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_upscale_done)
        self.worker.error.connect(self._on_error)
        self._start_time = time.time()
        self._elapsed_msg = ""
        self._elapsed_timer.start(1000)  # update every 1 second
        self.worker.start()

    def _on_upscale_done(self, result_np):
        self._elapsed_timer.stop()
        self.output_image_np = result_np
        elapsed = time.time() - self._start_time

        # Display output
        h, w = result_np.shape[:2]
        qimg = QImage(result_np.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.output_label.set_image(QPixmap.fromImage(qimg))
        self.output_info.setText(f"{w}x{h}")

        self.upscale_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{elapsed:.1f}s"
        self.progress_bar.setFormat(f"100% — Done! {w}x{h} in {time_str}")
        self.statusBar().showMessage(f"Done! {w}x{h} in {time_str}")

    def _on_save(self):
        if self.output_image_np is None:
            return

        # Default filename based on input
        default_name = "upscaled.jpg"
        if self.input_image_path:
            stem = Path(self.input_image_path).stem
            preset = self._get_selected_preset()
            default_name = f"{stem}_{preset}.jpg"

        save_dir = self._last_save_dir or self._last_open_dir or ""
        default_path = os.path.join(save_dir, default_name) if save_dir else default_name

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Upscaled Image", default_path,
            "JPEG sRGB (*.jpg);;PNG (*.png);;All Files (*)"
        )
        if path:
            self._last_save_dir = os.path.dirname(path)
            img = Image.fromarray(self.output_image_np)
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                # sRGB JPEG 95% quality
                from PIL import ImageCms
                srgb_profile = ImageCms.createProfile("sRGB")
                icc_bytes = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
                img.save(path, "JPEG", quality=95, icc_profile=icc_bytes)
            else:
                img.save(path)
            self.statusBar().showMessage(f"Saved: {path}")

    def _on_progress(self, percent, message):
        self._elapsed_msg = message
        elapsed = int(time.time() - self._start_time) if hasattr(self, '_start_time') else 0
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"%p% — {message}  [{elapsed}s]")

    def _update_elapsed(self):
        """Update elapsed time display every second during upscale."""
        if hasattr(self, '_start_time'):
            elapsed = int(time.time() - self._start_time)
            pct = self.progress_bar.value()
            self.progress_bar.setFormat(f"{pct}% — {self._elapsed_msg}  [{elapsed}s]")

    def _save_config(self):
        """Save current settings to config.json on app close."""
        config = {
            "dit_model": self.model_combo.currentText(),
            "vae_model": self.vae_combo.currentText(),
            "preset": self._get_selected_preset(),
            "attention_mode": self.attn_combo.currentText(),
            "seed": self.seed_spin.value(),
            "color_correction": self.color_combo.currentText(),
            "terminal_visible": self._log_visible,
            "advanced_expanded": self.advanced_group.isChecked(),
            "last_open_dir": self._last_open_dir,
            "last_save_dir": self._last_save_dir,
        }
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save config: {e}")

    def _load_config(self):
        """Load saved settings from config.json. Returns dict or None."""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}")
        return None

    def closeEvent(self, event):
        """Save config and restore stdout/stderr on close."""
        self._save_config()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        super().closeEvent(event)

    def _toggle_terminal(self):
        """Toggle terminal log panel visibility (right-side panel)."""
        self._log_visible = not self._log_visible
        self.log_console.setVisible(self._log_visible)
        if self._log_visible:
            self.terminal_widget.setFixedWidth(340)
            self.terminal_toggle_btn.setText("Hide \u25b6")
        else:
            self.terminal_widget.setFixedWidth(40)
            self.terminal_toggle_btn.setText("\u25c0")

    def _append_log(self, text):
        """Append text to terminal log panel (thread-safe via signal)."""
        # Strip trailing newline for cleaner display
        if text.endswith('\n'):
            text = text[:-1]
        if text:
            self.log_console.appendPlainText(text)
            # Auto-scroll to bottom
            scrollbar = self.log_console.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _on_error(self, error_msg):
        self._elapsed_timer.stop()
        self.upscale_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"Error: {error_msg}")
        print(f"[ERROR] {error_msg}")


# ─── Entry Point ──────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SeedVR2 Upscaler")
    app.setStyle("Fusion")

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


def _hide_console_window():
    """Hide the console window on Windows. All output goes to the in-app terminal."""
    try:
        import ctypes
        ctypes.windll.user32.ShowWindow(
            ctypes.windll.kernel32.GetConsoleWindow(), 0  # SW_HIDE
        )
    except Exception:
        pass  # Non-Windows or no console — ignore


if __name__ == "__main__":
    _hide_console_window()
    main()
