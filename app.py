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
import time

# Setup path before any project imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# CUDA config before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QProgressBar,
    QFileDialog, QGroupBox, QSplitter, QStatusBar, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QMimeData
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QFont, QPalette, QColor

# SeedVR2 imports
from src.core.generation_utils import (
    setup_generation_context, prepare_runner, load_text_embeddings,
    compute_generation_info, log_generation_start
)
from src.core.generation_phases import (
    encode_all_batches, upscale_all_batches,
    decode_all_batches, postprocess_all_batches
)
from src.utils.debug import Debug
from src.utils.constants import get_script_directory, SEEDVR2_FOLDER_NAME, get_base_cache_dir
from src.utils.model_registry import get_available_dit_models, DEFAULT_VAE
from src.optimization.memory_manager import clear_memory, get_gpu_backend


# ─── Resolution Presets ───────────────────────────────────────────────

RESOLUTION_PRESETS = {
    "2K": {"resolution": 2000, "max_resolution": 2000, "tile_grid": None, "blur_radius": 0,
           "blocks_to_swap": 0, "vae_tile_size": 768},
    "3K": {"resolution": 3000, "max_resolution": 3000, "tile_grid": None, "blur_radius": 0,
           "blocks_to_swap": 0, "vae_tile_size": 768},
    "4K": {"resolution": 4000, "max_resolution": 4000, "tile_grid": None, "blur_radius": 1,
           "blocks_to_swap": 8, "vae_tile_size": 768},
    "6K": {"resolution": 6000, "max_resolution": 6000, "tile_grid": None, "blur_radius": 1,
           "blocks_to_swap": 16, "vae_tile_size": 768},
    "9K": {"resolution": 9000, "max_resolution": 9000, "tile_grid": (2, 2), "blur_radius": 3,
           "blocks_to_swap": 8, "vae_tile_size": 768},
    "12K": {"resolution": 12000, "max_resolution": 12000, "tile_grid": (3, 3), "blur_radius": 3,
            "blocks_to_swap": 8, "vae_tile_size": 768},
}


# ─── Pipeline Backend ─────────────────────────────────────────────────

class SeedVR2Pipeline:
    """Direct GPU pipeline — models stay in VRAM between runs."""

    def __init__(self, dit_model, device="cuda:0", attention_mode="spargeattn",
                 use_torch_compile=False):
        self.device = device
        self.dit_model = dit_model
        self.vae_model = DEFAULT_VAE
        self.attention_mode = attention_mode
        self.use_torch_compile = use_torch_compile

        self.runner = None
        self.ctx = None
        self.debug = Debug(enabled=True)
        self.script_directory = get_script_directory()
        self.model_dir = get_base_cache_dir()
        self._loaded = False

    def load_models(self, blocks_to_swap=0, vae_tile_size=768, progress_fn=None):
        """Load models into VRAM (one-time cost)."""
        if progress_fn:
            progress_fn(5, "Setting up CUDA optimizations...")

        # GPU optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if progress_fn:
            progress_fn(10, "Initializing generation context...")

        self.ctx = setup_generation_context(
            dit_device=self.device,
            vae_device=self.device,
            debug=self.debug
        )

        if progress_fn:
            progress_fn(20, f"Loading DiT model: {self.dit_model}...")

        torch_compile_args = None
        if self.use_torch_compile:
            torch_compile_args = {
                "backend": "inductor",
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
                "dynamo_cache_size_limit": 64,
                "dynamo_recompile_limit": 128,
            }

        self.runner, cache_context = prepare_runner(
            dit_model=self.dit_model,
            vae_model=self.vae_model,
            model_dir=self.model_dir,
            debug=self.debug,
            ctx=self.ctx,
            block_swap_config={
                'blocks_to_swap': blocks_to_swap,
                'swap_io_components': False,
                'offload_device': torch.device("cpu") if blocks_to_swap > 0 else None,
            },
            encode_tiled=True,
            encode_tile_size=(vae_tile_size, vae_tile_size),
            encode_tile_overlap=(64, 64),
            decode_tiled=True,
            decode_tile_size=(vae_tile_size, vae_tile_size),
            decode_tile_overlap=(64, 64),
            attention_mode=self.attention_mode,
            torch_compile_args_dit=torch_compile_args,
            torch_compile_args_vae=torch_compile_args,
        )
        self.ctx['cache_context'] = cache_context

        if progress_fn:
            progress_fn(80, "Loading text embeddings...")

        self.ctx['text_embeds'] = load_text_embeddings(
            self.script_directory, self.ctx['dit_device'],
            self.ctx['compute_dtype'], self.debug
        )

        if progress_fn:
            progress_fn(100, "Models loaded!")

        self._loaded = True

    def _reset_ctx_for_run(self):
        """Reset context state for a new run, keeping device config and models."""
        keys_to_keep = {
            'dit_device', 'vae_device', 'dit_offload_device',
            'vae_offload_device', 'tensor_offload_device', 'compute_dtype',
            'cache_context', 'text_embeds', 'interrupt_fn', 'comfyui_available',
        }
        for key in list(self.ctx.keys()):
            if key not in keys_to_keep:
                self.ctx[key] = None if key in ('video_transform', 'final_video') else []

    @torch.inference_mode()
    def upscale_image(self, image_np, resolution, max_resolution, seed=42,
                      color_correction="lab", blocks_to_swap=0,
                      input_noise_scale=0.0, latent_noise_scale=0.0,
                      progress_fn=None):
        """
        Upscale a single image through the 4-phase pipeline.

        Args:
            image_np: Input image as numpy array (H, W, 3), uint8 or float32
            resolution: Target resolution (shortest edge)
            max_resolution: Maximum resolution (any edge)
            seed: Random seed
            color_correction: Color correction mode
            blocks_to_swap: BlockSwap count
            progress_fn: Callback (percent, message)

        Returns:
            Upscaled image as numpy array (H', W', 3), uint8
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        self._reset_ctx_for_run()

        # Convert numpy to tensor [1, H, W, 3] float32 range [0,1]
        if image_np.dtype == np.uint8:
            image_tensor = torch.from_numpy(image_np).float() / 255.0
        else:
            image_tensor = torch.from_numpy(image_np).float()
        image_tensor = image_tensor.unsqueeze(0)  # [1, H, W, C]

        if progress_fn:
            progress_fn(5, "Preparing image...")

        # Compute generation info
        image_tensor, gen_info = compute_generation_info(
            ctx=self.ctx, images=image_tensor,
            resolution=resolution, max_resolution=max_resolution,
            batch_size=1, seed=seed, debug=self.debug
        )
        log_generation_start(gen_info, self.debug)

        # Phase 1: Encode
        if progress_fn:
            progress_fn(10, "Phase 1/4: VAE Encoding...")
        self.ctx = encode_all_batches(
            self.runner, ctx=self.ctx, images=image_tensor,
            debug=self.debug, batch_size=1, seed=seed,
            resolution=resolution, max_resolution=max_resolution,
            input_noise_scale=input_noise_scale,
            color_correction=color_correction
        )

        # Phase 2: Upscale (DiT inference — the heavy part)
        if progress_fn:
            progress_fn(25, "Phase 2/4: DiT Upscaling...")
        self.ctx = upscale_all_batches(
            self.runner, ctx=self.ctx, debug=self.debug,
            seed=seed, latent_noise_scale=latent_noise_scale,
            cache_model=True
        )

        # Phase 3: Decode
        if progress_fn:
            progress_fn(75, "Phase 3/4: VAE Decoding...")
        self.ctx = decode_all_batches(
            self.runner, ctx=self.ctx, debug=self.debug,
            cache_model=True
        )

        # Phase 4: Post-process
        if progress_fn:
            progress_fn(90, "Phase 4/4: Color correction...")
        self.ctx = postprocess_all_batches(
            ctx=self.ctx, debug=self.debug,
            color_correction=color_correction,
            batch_size=1
        )

        # Get result
        result = self.ctx['final_video']
        if result.is_cuda:
            result = result.cpu()
        if result.dtype != torch.float32:
            result = result.to(torch.float32)

        # Convert to numpy uint8 [H, W, 3]
        result_np = (result.squeeze(0).clamp(0, 1).numpy() * 255).astype(np.uint8)

        if progress_fn:
            progress_fn(100, "Done!")

        return result_np


def _apply_gaussian_blur(image_np, radius):
    """Apply Gaussian blur to numpy image using PIL."""
    from PIL import ImageFilter
    img = Image.fromarray(image_np)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img)


def _tile_and_upscale(pipeline, image_np, preset, seed, color_correction,
                      input_noise_scale, latent_noise_scale, progress_fn):
    """
    Split image into overlapping tiles, upscale each, and reassemble.
    Used for 9K+ resolutions that exceed single-pass VRAM capacity.
    """
    rows, cols = preset["tile_grid"]
    h, w, c = image_np.shape
    overlap_rate = 0.05

    # Compute tile sizes with overlap
    tile_h = h // rows
    tile_w = w // cols
    overlap_h = max(int(tile_h * overlap_rate), 16)
    overlap_w = max(int(tile_w * overlap_rate), 16)

    tiles = []
    positions = []
    for r in range(rows):
        for col_idx in range(cols):
            y0 = max(0, r * tile_h - overlap_h)
            y1 = min(h, (r + 1) * tile_h + overlap_h)
            x0 = max(0, col_idx * tile_w - overlap_w)
            x1 = min(w, (col_idx + 1) * tile_w + overlap_w)
            tiles.append(image_np[y0:y1, x0:x1])
            positions.append((r, col_idx, y0, y1, x0, x1))

    total_tiles = len(tiles)
    upscaled_tiles = []

    # Per-tile resolution: scale proportionally
    per_tile_resolution = preset["resolution"] // max(rows, cols)
    per_tile_max = preset["max_resolution"] // max(rows, cols)

    for i, tile in enumerate(tiles):
        def tile_progress(pct, msg):
            overall = int((i / total_tiles + pct / 100 / total_tiles) * 100)
            if progress_fn:
                progress_fn(overall, f"Tile {i+1}/{total_tiles}: {msg}")

        result = pipeline.upscale_image(
            tile, resolution=per_tile_resolution, max_resolution=per_tile_max,
            seed=seed, color_correction=color_correction,
            blocks_to_swap=preset["blocks_to_swap"],
            input_noise_scale=input_noise_scale,
            latent_noise_scale=latent_noise_scale,
            progress_fn=tile_progress
        )
        upscaled_tiles.append(result)

    # Compute scale factor from first tile
    scale_h = upscaled_tiles[0].shape[0] / (positions[0][3] - positions[0][2])
    scale_w = upscaled_tiles[0].shape[1] / (positions[0][5] - positions[0][4])

    out_h = int(h * scale_h)
    out_w = int(w * scale_w)
    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

    for idx, (r, col_idx, y0, y1, x0, x1) in enumerate(positions):
        tile_up = upscaled_tiles[idx].astype(np.float32)
        th, tw = tile_up.shape[:2]

        out_y0 = int(y0 * scale_h)
        out_x0 = int(x0 * scale_w)
        out_y1 = out_y0 + th
        out_x1 = out_x0 + tw

        # Clip to output bounds
        out_y1 = min(out_y1, out_h)
        out_x1 = min(out_x1, out_w)
        th = out_y1 - out_y0
        tw = out_x1 - out_x0
        tile_up = tile_up[:th, :tw]

        # Create weight mask with linear falloff at overlap edges
        w_mask = np.ones((th, tw, 1), dtype=np.float32)
        scaled_overlap_h = int(overlap_h * scale_h)
        scaled_overlap_w = int(overlap_w * scale_w)

        if r > 0 and scaled_overlap_h > 0:
            ramp = np.linspace(0, 1, min(scaled_overlap_h, th)).reshape(-1, 1, 1)
            w_mask[:min(scaled_overlap_h, th)] *= ramp
        if r < rows - 1 and scaled_overlap_h > 0:
            ramp = np.linspace(1, 0, min(scaled_overlap_h, th)).reshape(-1, 1, 1)
            w_mask[-min(scaled_overlap_h, th):] *= ramp
        if col_idx > 0 and scaled_overlap_w > 0:
            ramp = np.linspace(0, 1, min(scaled_overlap_w, tw)).reshape(1, -1, 1)
            w_mask[:, :min(scaled_overlap_w, tw)] *= ramp
        if col_idx < cols - 1 and scaled_overlap_w > 0:
            ramp = np.linspace(1, 0, min(scaled_overlap_w, tw)).reshape(1, -1, 1)
            w_mask[:, -min(scaled_overlap_w, tw):] *= ramp

        output[out_y0:out_y1, out_x0:out_x1] += tile_up * w_mask
        weights[out_y0:out_y1, out_x0:out_x1] += w_mask

    # Normalize by weights
    weights = np.maximum(weights, 1e-6)
    output = (output / weights).clip(0, 255).astype(np.uint8)
    return output


# ─── QThread Worker ───────────────────────────────────────────────────

class ModelLoadWorker(QThread):
    """Loads models in background thread."""
    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, pipeline, blocks_to_swap=0, vae_tile_size=768):
        super().__init__()
        self.pipeline = pipeline
        self.blocks_to_swap = blocks_to_swap
        self.vae_tile_size = vae_tile_size

    def run(self):
        try:
            self.pipeline.load_models(
                blocks_to_swap=self.blocks_to_swap,
                vae_tile_size=self.vae_tile_size,
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
            preset = RESOLUTION_PRESETS[self.preset_name]
            image = self.image_np.copy()

            # Apply blur if configured
            if preset["blur_radius"] > 0:
                self.progress.emit(2, f"Applying blur (radius={preset['blur_radius']})...")
                image = _apply_gaussian_blur(image, preset["blur_radius"])

            # Tiled or direct processing
            if preset["tile_grid"] is not None:
                result = _tile_and_upscale(
                    self.pipeline, image, preset,
                    self.seed, self.color_correction,
                    self.input_noise_scale, self.latent_noise_scale,
                    lambda p, m: self.progress.emit(p, m)
                )
            else:
                result = self.pipeline.upscale_image(
                    image,
                    resolution=preset["resolution"],
                    max_resolution=preset["max_resolution"],
                    seed=self.seed,
                    color_correction=self.color_correction,
                    blocks_to_swap=preset["blocks_to_swap"],
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

        # Dark theme
        self._apply_dark_theme()

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

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
        main_layout.addWidget(splitter, stretch=1)

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

        # Row 2: Basic params
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

        # Row 3: Advanced settings (collapsible)
        self.advanced_group = QGroupBox("Advanced Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        adv_layout = QHBoxLayout(self.advanced_group)

        # Model selection
        adv_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self._populate_models()
        self.model_combo.setMinimumWidth(200)
        adv_layout.addWidget(self.model_combo)

        # Attention mode
        adv_layout.addWidget(QLabel("Attention:"))
        self.attn_combo = QComboBox()
        self.attn_combo.addItems(["spargeattn", "sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"])
        adv_layout.addWidget(self.attn_combo)

        # Blocks to swap
        adv_layout.addWidget(QLabel("Blocks swap:"))
        self.blocks_spin = QSpinBox()
        self.blocks_spin.setRange(0, 36)
        self.blocks_spin.setValue(0)
        adv_layout.addWidget(self.blocks_spin)

        # torch.compile
        self.compile_check = QCheckBox("torch.compile")
        self.compile_check.setToolTip("Enable torch.compile (inductor backend). First run slower, subsequent faster.")
        adv_layout.addWidget(self.compile_check)

        adv_layout.addStretch()
        controls_layout.addWidget(self.advanced_group)

        main_layout.addLayout(controls_layout)

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
        main_layout.addWidget(self.progress_bar)

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

        main_layout.addLayout(btn_row)

        # ── Status bar ──
        self.statusBar().showMessage("Ready — Load models to begin")

        # Default preset
        self._on_preset_click("4K")

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
        try:
            models = get_available_dit_models()
            self.model_combo.addItems(models)
            # Select 7B GGUF if available
            for i, m in enumerate(models):
                if "7b" in m.lower() and "q8" in m.lower():
                    self.model_combo.setCurrentIndex(i)
                    break
        except Exception:
            self.model_combo.addItem("seedvr2_ema_7b-Q8_0.gguf")

    def _on_preset_click(self, name):
        for n, btn in self.res_buttons.items():
            btn.setChecked(n == name)
        # Update blocks_to_swap from preset
        preset = RESOLUTION_PRESETS[name]
        self.blocks_spin.setValue(preset["blocks_to_swap"])

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
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
        )
        if path:
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
        attention = self.attn_combo.currentText()
        use_compile = self.compile_check.isChecked()
        blocks = self.blocks_spin.value()

        self.pipeline = SeedVR2Pipeline(
            dit_model=dit_model,
            device="cuda:0",
            attention_mode=attention,
            use_torch_compile=use_compile,
        )

        self.load_btn.setEnabled(False)
        self.upscale_btn.setEnabled(False)
        self.statusBar().showMessage("Loading models into VRAM...")

        self._load_worker = ModelLoadWorker(self.pipeline, blocks_to_swap=blocks)
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

        # Show VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            self.statusBar().showMessage(
                f"Models loaded — VRAM: {allocated:.1f}GB / {total:.1f}GB"
            )
        else:
            self.statusBar().showMessage("Models loaded — Ready")

    def _on_upscale(self):
        if self.input_image_np is None or not self.pipeline or not self.pipeline._loaded:
            return

        preset_name = self._get_selected_preset()
        self.upscale_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.statusBar().showMessage(f"Upscaling to {preset_name}...")

        self.worker = UpscaleWorker(
            pipeline=self.pipeline,
            image_np=self.input_image_np,
            preset_name=preset_name,
            seed=self.seed_spin.value(),
            color_correction=self.color_combo.currentText(),
            input_noise_scale=0.0,
            latent_noise_scale=0.0,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_upscale_done)
        self.worker.error.connect(self._on_error)
        self._start_time = time.time()
        self.worker.start()

    def _on_upscale_done(self, result_np):
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
        self.statusBar().showMessage(f"Done! {w}x{h} in {elapsed:.1f}s")

    def _on_save(self):
        if self.output_image_np is None:
            return

        # Default filename based on input
        default_name = "upscaled.png"
        if self.input_image_path:
            stem = Path(self.input_image_path).stem
            preset = self._get_selected_preset()
            default_name = f"{stem}_{preset}.png"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Upscaled Image", default_name,
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if path:
            img = Image.fromarray(self.output_image_np)
            img.save(path)
            self.statusBar().showMessage(f"Saved: {path}")

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"%p% — {message}")

    def _on_error(self, error_msg):
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
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
