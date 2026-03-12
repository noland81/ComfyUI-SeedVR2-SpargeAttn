# Changelog

## v1.2.1 (2026-03-12) — Auto-download models + Windows encoding fix

### Tính năng mới
- Auto-download models: `download_weight()` tự tải DiT + VAE từ HuggingFace nếu chưa có
- Khi click "Load Models", app sẽ tự kiểm tra + tải model cần thiết

### Bug fixes
- Fix Windows console encoding crash: emoji characters (📊⚡) trong SeedVR2 logs gây UnicodeEncodeError trên cp1252
- Thêm `sys.stdout.reconfigure(encoding='utf-8')` trước tất cả SeedVR2 imports

### Files thay đổi
- `app.py` — Thêm download_weight() import + call, thêm UTF-8 encoding fix
- `VERSION.txt` — 1.2.0 → 1.2.1
- `CHANGELOG.md` — Thêm entry v1.2.1

## v1.2.0 (2026-03-12) — Deep audit fix: match exact ComfyUI workflow processing chain

### Bug fixes (CRITICAL)
- **9K/12K preprocessing chain was completely missing**: Workflows do LoadImage → Resize(keep proportion) → Blur → Make-even → Tile → SeedVR2. App was: Load → Blur → Tile raw input → SeedVR2. Output would be ~1/4 expected size and wrong quality.
- **9K tile grid wrong**: Was fixed 2x2 (4 tiles). Workflow uses adaptive: landscape 2x1, portrait 1x2 (2 tiles). Too many tiles = unnecessary seams + slower.
- **12K tile grid wrong**: Was 3x3 (9 tiles). Workflow uses 2x2 (4 tiles). Same issue.
- **Per-tile resolution wrong**: Was `preset_resolution / grid_size`. Should be `min(tile_w, tile_h)` matching ImpactMinMax(mode=false). This caused SeedVR2 to produce wrong output scale for each tile.
- **4K/6K missing pre-resize**: Workflows pre-resize to 4096/6000 (keep proportion, lanczos) before blur + SeedVR2. App skipped this, sending raw input directly.
- **4K blocks_to_swap wrong**: Was 9, workflow uses 8.
- **4K max_resolution wrong**: Was 4000, workflow uses 6000.
- **4K/6K resolution was fixed**: Should be auto-computed as min(resized_w, resized_h) from pre-resized dimensions, matching ImpactMinMax.

### Tính năng mới
- Pre-resize step (keep proportion, lanczos) for 4K+ presets
- Adaptive tile grid for 9K: landscape → 2x1, portrait → 1x2
- Make-dimensions-even stretch resize for tiled presets (matches workflow math)
- Auto-computed resolution from pre-resized image dimensions

### Removed
- torch.compile option: Not officially supported on Windows GPU (requires triton-windows + MSVC build tools + vcvars64.bat). Too fragile for production use.

### Preset values (matching ComfyUI workflows exactly)
| Preset | pre_resize | blur | tile_mode | resolution | max_res | blocks | vae_tile | overlap |
|--------|-----------|------|-----------|------------|---------|--------|----------|---------|
| 2K     | —         | 0    | —         | 2000       | 2000    | 9      | 1024     | 64      |
| 3K     | —         | 0    | —         | 3000       | 3000    | 9      | 1024     | 64      |
| 4K     | 4096      | 1    | —         | auto       | 6000    | 8      | 768      | 32      |
| 6K     | 6000      | 2    | —         | auto       | 6000    | 16     | 768      | 32      |
| 9K     | 9000      | 2    | adaptive  | auto       | 9000    | 32     | 768      | 32      |
| 12K    | 12000     | 3    | 2x2       | auto       | 12000   | 32     | 768      | 32      |

### Processing flow by preset
- **2K/3K**: Load → SeedVR2 (direct upscale, no preprocessing)
- **4K/6K**: Load → Resize(keep prop) → Blur → SeedVR2
- **9K**: Load → Resize(9000, keep prop) → Blur → Make-even → Tile(adaptive) → SeedVR2/tile → Reassemble
- **12K**: Load → Resize(12000, keep prop) → Blur → Make-even → Tile(2x2) → SeedVR2/tile → Reassemble

### Files thay đổi
- `app.py` — Complete preprocessing rewrite, fix all presets, remove torch.compile
- `VERSION.txt` — 1.1.1 → 1.2.0
- `CHANGELOG.md` — Thêm entry v1.2.0
- `FEATURES.md` — Cập nhật F4 + F5

## v1.1.1 (2026-03-12) — Fix VRAM safety: match exact workflow parameters

### Bug fixes
- blocks_to_swap quá thấp (0 thay vì 9 cho 2K-3K, 8 thay vì 32 cho 9K-12K) → OOM trên 24GB
- VAE tile size sai (768 cho tất cả, đúng phải 1024 cho 2K-3K)
- VAE tile overlap sai (64 cho tất cả, đúng phải 32 cho 4K+)
- latent_noise_scale sai (0.0 cho 4K+, đúng phải 0.001)
- blur_radius 6K sai (1 thay vì 2)

### Tính năng mới
- Auto-reload models: khi đổi preset cần blocks_to_swap/VAE tile khác → tự động reload
- UI thêm controls: VAE tile size + VAE tile overlap trong Advanced Settings
- Auto-load: click Upscale khi chưa load models → tự động load rồi upscale
- Status bar hiển thị chi tiết: swap, tile, VAE tiled=ON, VRAM usage

### Preset values (khớp ComfyUI workflow thực tế cho RTX 4090 24GB)
| Preset | blocks_to_swap | vae_tile | overlap | latent_noise | blur |
|--------|---------------|----------|---------|-------------|------|
| 2K     | 9             | 1024     | 64      | 0.0         | 0    |
| 3K     | 9             | 1024     | 64      | 0.0         | 0    |
| 4K     | 9             | 768      | 32      | 0.001       | 1    |
| 6K     | 16            | 768      | 32      | 0.001       | 2    |
| 9K     | 32            | 768      | 32      | 0.001       | 2    |
| 12K    | 32            | 768      | 32      | 0.001       | 3    |

### Files thay đổi
- `app.py` — Fix presets, thêm auto-reload, thêm VAE controls, thêm _pending_upscale flow
- `VERSION.txt` — 1.1.0 → 1.1.1
- `CHANGELOG.md` — Thêm entry v1.1.1

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
