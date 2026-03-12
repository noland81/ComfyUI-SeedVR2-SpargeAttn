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
4. Chọn resolution preset (2K-12K) → auto-config blocks_to_swap, tiling, blur, pre-resize
5. Click "Upscale" → preprocessing + pipeline chạy trên QThread, progress bar cập nhật
6. Output hiển thị → click "Save Output" để lưu

**Pipeline flow (trong UpscaleWorker QThread)**:
1. Pre-resize (keep proportion, lanczos) — chỉ 4K+ (2K/3K bỏ qua)
2. Compute resolution = min(resized_w, resized_h) hoặc fixed (2K/3K)
3. Blur preprocessing (nếu preset yêu cầu)
4. Nếu 9K/12K: make-even (stretch) → tile → xử lý từng tile → ghép lại
5. Nếu <9K: xử lý trực tiếp qua 4-phase pipeline
6. Phase 1: encode_all_batches (VAE encode)
7. Phase 2: upscale_all_batches (DiT inference)
8. Phase 3: decode_all_batches (VAE decode)
9. Phase 4: postprocess_all_batches (color correction)

**Processing flow by preset (khớp chính xác ComfyUI workflow)**:
- 2K/3K: Load → SeedVR2 (direct, resolution=2000/3000, max=2000/3000)
- 4K: Load → Resize(4096, keep prop) → Blur(1) → SeedVR2 (res=auto, max=6000)
- 6K: Load → Resize(6000, keep prop) → Blur(2) → SeedVR2 (res=auto, max=6000)
- 9K: Load → Resize(9000) → Blur(2) → Make-even → Tile(adaptive) → SeedVR2/tile → Assemble
- 12K: Load → Resize(12000) → Blur(3) → Make-even → Tile(2x2) → SeedVR2/tile → Assemble

**GPU Optimizations**:
- TF32 enabled cho matmul + cudnn
- cudnn.benchmark = True
- SpargeAttn default attention
- Model cached trong VRAM giữa các lần chạy
- torch.inference_mode() context

**QUAN TRỌNG**:
- GPU work PHẢI chạy trên QThread, KHÔNG ĐƯỢC chạy trên main thread (UI freeze)
- QImage cần contiguous memory → result_np phải là C-contiguous
- Pipeline imports cần script_dir trong sys.path
- PYTORCH_CUDA_ALLOC_CONF phải set TRƯỚC import torch
- torch.compile ĐÃ BỊ GỠ — không hỗ trợ chính thức trên Windows GPU
- Pre-resize PHẢI dùng lanczos và keep proportion (khớp ImageResize+ node)
- Resolution cho 4K+ PHẢI là min(w, h) của ảnh sau pre-resize (khớp ImpactMinMax)

**Files**:
- `app.py` — `SeedVR2Pipeline`, `UpscaleWorker(QThread)`, `ModelLoadWorker(QThread)`, `MainWindow(QMainWindow)`

## F5. Preprocessing & Tiling for High Resolution

**Yêu cầu**: Preprocessing chain và auto-tiling cho resolution 4K+ khớp chính xác ComfyUI workflows.

**Preprocessing chain (4K+)**:
1. Resize ảnh input giữ tỷ lệ, longest edge = target (lanczos) — `_resize_keep_proportion()`
2. Compute resolution = min(resized_w, resized_h) — khớp ImpactMinMax(mode=false)
3. Blur (nếu preset yêu cầu) — `_apply_gaussian_blur()`
4. Nếu tiled preset (9K/12K): make dimensions even — `_make_dimensions_even()`

**Tiling (9K/12K) — khớp chính xác ComfyUI TTP nodes**:
1. Xác định grid dựa trên tile_mode:
   - 9K "adaptive": landscape (w>=h) → 2 cols x 1 row, portrait → 1 col x 2 rows
   - 12K (2,2): fixed 2x2 grid (4 tiles)
2. Cut tiles với 5% overlap (min 16px)
3. Resolution per tile = min(tile_w, tile_h) — KHÔNG PHẢI preset/grid_size
4. max_resolution per tile = preset max_resolution (KHÔNG chia)
5. Ghép lại với weighted blending (linear ramp tại overlap edges)

**QUAN TRỌNG**:
- 9K grid là ADAPTIVE (2 tiles, không phải 4) — landscape split horizontal, portrait split vertical
- 12K grid là 2x2 (4 tiles, KHÔNG PHẢI 3x3 = 9 tiles)
- Resolution per tile = min(tile_w, tile_h), KHÔNG PHẢI preset_resolution / grid_size
- max_resolution KHÔNG được chia cho grid size
- Make-even stretch resize: round(dim/2)*2 — khớp workflow math nodes
- Scale factor tính từ tile đầu tiên, áp dụng cho tất cả tiles
- Weight mask dùng linear ramp tại overlap edges → tránh seam artifacts
- Overlap phải đủ lớn (>= 16px) để blending smooth

**Files**:
- `app.py` — `_resize_keep_proportion()`, `_make_dimensions_even()`, `_apply_gaussian_blur()`, `_tile_and_upscale()`
