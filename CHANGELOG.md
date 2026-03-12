# Changelog

## v1.1.0 (2026-03-12) — Standalone GPU-Direct Upscaler App

### Tính năng mới
- PySide6 desktop app (`app.py`) chạy SeedVR2 pipeline trực tiếp trên GPU, bỏ qua ComfyUI hoàn toàn
- Model pre-loading: giữ DiT + VAE trong VRAM giữa các lần chạy (không cần reload)
- Resolution presets: 2K, 3K, 4K, 6K, 9K, 12K với auto-config từ workflow
- Auto-tiling cho 9K/12K: tự động chia ảnh thành tiles, xử lý từng tile, ghép lại
- GPU optimizations: TF32, cuDNN benchmark, SpargeAttn default, torch.compile optional
- Dark theme UI với drag & drop image, progress bar, save output
- Background threading (QThread): UI không bị freeze khi GPU đang xử lý

### Tại sao nhanh hơn ComfyUI
- Không có node graph execution overhead
- Không có HTTP/WebSocket API layer
- Model cached trong VRAM (không reload mỗi lần chạy)
- torch.compile tùy chọn (20-40% speedup cho DiT)
- Direct CUDA calls, không qua memory manager của ComfyUI

### Files thay đổi
- `app.py` — Mới: standalone PySide6 desktop app (~550 lines)
- `VERSION.txt` — 1.0.0 → 1.1.0
- `CHANGELOG.md` — Thêm entry v1.1.0
- `FEATURES.md` — Thêm F4 (Standalone App) + F5 (Auto-Tiling)

## v1.0.0 (2026-03-12) — Initial release: SeedVR2 + SpargeAttn integrated

### Tính năng mới
- Tích hợp SpargeAttn (Sparse + Sage Attention) trực tiếp vào SeedVR2 VideoUpscaler
- SpargeAttn là attention backend mặc định trong DiT Model Loader
- Auto-install script (`install.py`) tự động cài SpargeAttn + SageAttention 2 + ninja
- Fallback chain: SpargeAttn → SageAttention 2 → PyTorch SDPA
- Hỗ trợ cả 3B và 7B model variants

### Base
- Fork từ [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) by numz
- SpargeAttn từ [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn) (Tsinghua University)

### Files thay đổi (so với SeedVR2 gốc)
- `src/optimization/compatibility.py` — Thêm SpargeAttn detection, config, wrapper function `call_sparge_attn_varlen()`
- `src/models/dit_7b/attention.py` — Thêm import `call_sparge_attn_varlen`, thêm case 'spargeattn' trong `FlashAttentionVarlen.forward()`
- `src/models/dit_3b/attention.py` — Tương tự dit_7b
- `src/interfaces/dit_model_loader.py` — Thêm 'spargeattn' vào dropdown (mặc định)
- `src/core/model_configuration.py` — Thêm description mapping cho 'spargeattn'
- `install.py` — Mới: auto-install SpargeAttn + dependencies
- `VERSION.txt` — Mới: version tracking
- `CHANGELOG.md` — Mới: changelog
- `FEATURES.md` — Mới: feature spec
