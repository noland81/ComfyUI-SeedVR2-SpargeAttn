# Changelog

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
