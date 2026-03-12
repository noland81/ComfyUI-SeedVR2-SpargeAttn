# Features Specification

## F1. SpargeAttn Attention Backend

**Yêu cầu**: Tích hợp SpargeAttn (block-sparse quantized attention) làm attention backend cho SeedVR2 DiT model.

**Cách hoạt động**:
1. User chọn `spargeattn` trong DiT Model Loader dropdown (mặc định)
2. `model_configuration.py` validate attention_mode, set `module.attention_mode = 'spargeattn'` cho tất cả FlashAttentionVarlen instances
3. Khi inference, `FlashAttentionVarlen.forward()` dispatch tới `call_sparge_attn_varlen()`
4. `call_sparge_attn_varlen()` trong compatibility.py:
   - Nhận varlen format `(total_seq, heads, head_dim)` + `cu_seqlens`
   - Kiểm tra sequence lengths đều nhau (uniform) → nếu không → fallback
   - Kiểm tra seq_len >= 128 và head_dim in {64, 128} → nếu không → fallback
   - Reshape từ varlen `(total_seq, H, D)` → batched `(B, H, N, D)`
   - Gọi `spas_sage2_attn_meansim_topk_cuda(q, k, v, topk, smooth_k)`
   - Reshape output về varlen format `(total_seq, H, D)`
5. Fallback chain: SpargeAttn → SageAttention 2 → PyTorch SDPA

**QUAN TRỌNG**:
- SpargeAttn CHỈ hỗ trợ batched format `[B, H, N, D]`, KHÔNG hỗ trợ varlen → PHẢI reshape
- Sequence lengths PHẢI đều nhau để reshape được (SeedVR2 thường đều vì xử lý 1 video/image)
- Minimum seq_len = 128, head_dim phải là 64 hoặc 128
- FlashAttentionVarlen tồn tại ở 2 module riêng biệt: `dit_3b/attention.py` VÀ `dit_7b/attention.py`
- SM89 (RTX 4090): dùng INT8 quantization cho Q*K, FP8 cho V

**Files**:
- `src/optimization/compatibility.py` — `call_sparge_attn_varlen()`, `SPARGE_ATTN_CONFIG`, detection logic
- `src/models/dit_7b/attention.py` — `FlashAttentionVarlen.forward()` dispatch
- `src/models/dit_3b/attention.py` — `FlashAttentionVarlen.forward()` dispatch
- `src/interfaces/dit_model_loader.py` — dropdown options
- `src/core/model_configuration.py` — description mapping + validation

## F2. Auto-Install Dependencies

**Yêu cầu**: Tự động cài đặt SpargeAttn và dependencies khi ComfyUI load node lần đầu.

**Cách hoạt động**:
1. ComfyUI gọi `install.py` khi phát hiện custom node mới
2. Script kiểm tra và cài: ninja → sageattention → spas_sage_attn (từ GitHub source)
3. Nếu cài thất bại → in warning, node vẫn hoạt động với SDPA fallback

**QUAN TRỌNG**:
- SpargeAttn cần build từ source (CUDA extensions) → cần CUDA toolkit
- SpargeAttn phụ thuộc SageAttention 2 → phải cài SageAttention 2 trước
- Dùng `--no-build-isolation` khi pip install SpargeAttn

**Files**:
- `install.py` — auto-install script

## F3. Attention Backend Selection (inherited from SeedVR2)

**Yêu cầu**: User có thể chọn attention backend từ DiT Model Loader UI.

**Backends**: spargeattn (default), sdpa, flash_attn_2, flash_attn_3, sageattn_2, sageattn_3

**Cách hoạt động**:
1. User chọn backend trong dropdown
2. Config dict truyền qua pipeline
3. `model_configuration.py` validate và apply cho tất cả FlashAttentionVarlen modules

**Files**:
- `src/interfaces/dit_model_loader.py` — UI dropdown
- `src/core/model_configuration.py` — validation + apply

## F4. Standalone GPU-Direct Desktop App

**Yêu cầu**: PySide6 desktop app chạy SeedVR2 pipeline trực tiếp trên GPU, không cần ComfyUI.

**Cách hoạt động**:
1. User chạy `python app.py` → cửa sổ PySide6 mở
2. Click "Load Models" → DiT + VAE load vào VRAM trên QThread (UI không freeze)
3. Drag & drop hoặc browse ảnh input
4. Chọn resolution preset (2K-12K) → auto-config blocks_to_swap, tiling, blur
5. Click "Upscale" → pipeline chạy trên QThread, progress bar cập nhật theo 4 phases
6. Output hiển thị → click "Save Output" để lưu

**Pipeline flow (trong UpscaleWorker QThread)**:
1. Blur preprocessing (nếu preset yêu cầu)
2. Nếu 9K+: chia tiles → xử lý từng tile → ghép lại
3. Nếu <9K: xử lý trực tiếp qua 4-phase pipeline
4. Phase 1: encode_all_batches (VAE encode)
5. Phase 2: upscale_all_batches (DiT inference)
6. Phase 3: decode_all_batches (VAE decode)
7. Phase 4: postprocess_all_batches (color correction)

**GPU Optimizations**:
- TF32 enabled cho matmul + cudnn
- cudnn.benchmark = True
- SpargeAttn default attention
- torch.compile optional (inductor backend)
- Model cached trong VRAM giữa các lần chạy
- torch.inference_mode() context

**QUAN TRỌNG**:
- GPU work PHẢI chạy trên QThread, KHÔNG ĐƯỢC chạy trên main thread (UI freeze)
- QImage cần contiguous memory → result_np phải là C-contiguous
- Pipeline imports cần script_dir trong sys.path
- PYTORCH_CUDA_ALLOC_CONF phải set TRƯỚC import torch

**Files**:
- `app.py` — `SeedVR2Pipeline`, `UpscaleWorker(QThread)`, `ModelLoadWorker(QThread)`, `MainWindow(QMainWindow)`

## F5. Auto-Tiling for High Resolution (9K/12K)

**Yêu cầu**: Tự động chia ảnh thành tiles cho resolution 9K+ (DiT không xử lý nổi full image).

**Cách hoạt động**:
1. Chia ảnh thành grid tiles với overlap (5% overlap_rate)
2. Xử lý từng tile qua SeedVR2 pipeline (resolution = preset / grid_size)
3. Ghép lại với weighted blending tại vùng overlap (linear ramp)

**Tiling grid**:
- 9K: 2x2 tiles (4 tiles total)
- 12K: 3x3 tiles (9 tiles total)

**QUAN TRỌNG**:
- Scale factor tính từ tile đầu tiên, áp dụng cho tất cả tiles
- Weight mask dùng linear ramp tại overlap edges → tránh seam artifacts
- Overlap phải đủ lớn (>= 16px) để blending smooth

**Files**:
- `app.py` — `_tile_and_upscale()` function
