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
- SeedVR2's 720p windowing (window.py) LUÔN tạo variable-length windows (4 sizes khác nhau: interior + 3 loại edge). Mọi resolution (2K-6K) đều non-uniform → SpargeAttn KHÔNG BAO GIỜ chạy được trực tiếp
- Group-by-size workaround ĐÃ THỬ VÀ THẤT BẠI: Python gather/scatter loop (420+ windows × 36 layers × 3 tensors = ~15,000 tensor slices/step) tạo overhead 2-8x → PHẢI dùng SageAttention 2 fallback
- Fallback có 1-time warning log: `[ATTN] SpargeAttn -> SageAttn2 fallback: N different window sizes detected`
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
2. Chọn DiT model (dropdown luôn hiển thị, 7B only, default Q8_0) + VAE model
3. Click "Load Models" → DiT + VAE load vào VRAM trên QThread (UI không freeze)
4. Drag & drop hoặc browse ảnh input
5. Chọn resolution preset (2K-12K) → auto-config blocks_to_swap, tiling, blur, pre-resize
6. Click "Upscale" → preprocessing + pipeline chạy trên QThread, progress bar cập nhật
7. Output hiển thị → click "Save Output" để lưu

**Model selection**:
- DiT dropdown: 7B only (lọc bỏ 3B + Q3), sắp xếp Q8>Q6>Q5>Q4>fp8>fp16
- VAE dropdown: từ get_available_vae_models()
- Cả 2 hiển thị trực tiếp (không ẩn trong Advanced Settings)
- Default DiT: seedvr2_ema_7b-Q8_0.gguf (khớp tất cả workflow)
- Auto-download qua download_weight() nếu model chưa có

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
- 2K/3K/4K: Load → SeedVR2 (direct, resolution=2000/3000/4000, max=2000/3000/4000)
- 6K: Load → Resize(6000, keep prop) → Blur(2) → SeedVR2 (res=auto, max=6000)
- 9K: Load → Resize(9000) → Blur(2) → Make-even → Tile(adaptive) → SeedVR2/tile → Assemble
- 12K: Load → Resize(12000) → Blur(3) → Make-even → Tile(2x2) → SeedVR2/tile → Assemble

**GPU Optimizations**:
- TF32 enabled cho matmul + cudnn
- cudnn.benchmark = False (True gay 22GB VRAM spike voi VAE tiles)
- SpargeAttn default attention (fallback to SageAttn2 do variable-length windows)
- Model cached trong VRAM giua cac lan chay
- torch.inference_mode() context
- **BlockSwap async prefetch + lazy pin**: CUDA prefetch stream load block N+1 song song voi compute block N. Lazy pin (`_ensure_pinned`) truoc prefetch thay vi sync offload + re-pin. **Adaptive offload**: async (non_blocking) cho ≤12 blocks (4K), sync cho >12 blocks (6K+) — non_blocking voi nhieu blocks gay VRAM accumulation (17-36s stalls). Flag `_use_async_offload` set luc configure, ko per-block overhead
- **Skip-offload warm run**: Giu DiT blocks tai vi tri BlockSwap giua cac warm run. Skip 3s offload/restore round-trip. cleanup_dit() detect BlockSwap active + cache → giu blocks, chi deactivate
- Between-phase empty_cache() thay vi per-block clear_memory (giam overhead)

**QUAN TRỌNG**:
- GPU work PHẢI chạy trên QThread, KHÔNG ĐƯỢC chạy trên main thread (UI freeze)
- QImage cần contiguous memory → result_np phải là C-contiguous
- Pipeline imports cần script_dir trong sys.path
- PYTORCH_CUDA_ALLOC_CONF phải set TRƯỚC import torch
- torch.compile ĐÃ BỊ GỠ — không hỗ trợ chính thức trên Windows GPU
- Pre-resize PHẢI dùng lanczos và keep proportion (khớp ImageResize+ node)
- Resolution cho 4K+ PHẢI là min(w, h) của ảnh sau pre-resize (khớp ImpactMinMax)
- setup_generation_context PHẢI truyền CẢ 3 offload devices: dit_offload_device, vae_offload_device, tensor_offload_device = "cpu". Thiếu vae/tensor offload → VRAM tích tụ 22GB → VAE encode/decode chậm 6-12x
- 4K preset khớp workflow-refix-4k.json: NO pre-resize, resolution=4000, max_resolution=4000 → output ~12 Mpx. KHÔNG dùng comfyui_workflow_4k.json (pre-resize 4096, output 13.4 Mpx)
- cudnn.benchmark PHẢI = False. Với True + 1024x1024 VAE tiles, cuDNN profiling allocate workspace 22GB và thêm 35s+ warmup mỗi shape mới (tiles 1-5 + edge tiles 16-20). Heuristic algorithm chỉ chậm ~5%/tile nhưng không có VRAM spike
- torch.cuda.empty_cache() PHẢI gọi giữa các phases VÀ cuối load_models(). CUDA allocator giữ reserved VRAM từ operations trước → BlockSwap config chậm 3.5x (7.3s vs 2.1s), mỗi block swap chậm hơn (638ms vs 488ms). empty_cache() giải phóng reserved memory về CUDA driver
- BlockSwap PHẢI dùng pinned CPU memory + CUDA prefetch stream cho async loading. non_blocking=True cho CPU→GPU CHỈ hoạt động với pinned memory — nếu không pin, PyTorch fallback về synchronous cudaMemcpy
- BlockSwap dùng **lazy pin** (`_ensure_pinned`): non_blocking GPU→CPU offload tạo unpinned tensors, lazy pin trước prefetch (~20ms) thay vì sync offload + re-pin (~170ms/block). Quick-check first param is_pinned → no-op trên cold start
- BlockSwap **adaptive offload**: `_use_async_offload` flag set luc configure_block_swap. ≤12 blocks → async (non_blocking, saves ~150ms/block). >12 blocks → sync (prevents VRAM accumulation). NON_BLOCKING GPU→CPU voi nhieu blocks (6K, 16 blocks) gay deferred VRAM release → CUDA allocator stall 17-36s/block khi headroom ko du
- Warm run: DiT blocks PHẢI giữ tại vị trí BlockSwap (skip offload). cleanup_dit detect BlockSwap active + cache_model → skip manage_model_device offload. _handle_blockswap_model_movement detect blocks already in place → skip restore, chỉ reactivate. Tiết kiệm ~3s/warm run
- KHÔNG gọi clear_memory per-block trong BlockSwap — overhead ~5ms × N blocks × 36 layers, giá trị thực tế gần 0 (force=False). Between-phase empty_cache() đã đủ

**Files**:
- `app.py` — `SeedVR2Pipeline`, `UpscaleWorker(QThread)`, `ModelLoadWorker(QThread)`, `MainWindow(QMainWindow)`

## F5. Preprocessing & Tiling for High Resolution

**Yêu cầu**: Preprocessing chain và auto-tiling cho resolution 6K+ khớp ComfyUI workflows. Output size dùng megapixel target (24/54/96 Mpx).

**Preprocessing chain (6K+)**:
1. Resize ảnh input theo megapixel target — `_resize_to_megapixels(target_mpx)`:
   - scale = sqrt(target_pixels / current_pixels), giữ tỷ lệ
   - Dimensions aligned to even (// 2 * 2)
   - 6K=24 Mpx, 9K=54 Mpx, 12K=96 Mpx
2. Compute resolution = min(resized_w, resized_h) — khớp ImpactMinMax(mode=false)
3. Blur (nếu preset yêu cầu) — `_apply_gaussian_blur()`
4. Nếu tiled preset (9K/12K): make dimensions even — `_make_dimensions_even()`

**Tiling (9K/12K) — khớp chính xác ComfyUI TTP nodes**:
1. Xác định grid (tile_mode):
   - 9K "adaptive": landscape (w>=h) → 2×1, portrait → 1×2
   - 12K (2,2): fixed 2×2 grid (4 tiles)
2. Tính tile size — `_ttp_tile_size()`:
   - `tile = int(img_size / (1 + (factor-1) * (1-overlap_rate)))`
   - Round DOWN to multiple of 8
3. Tính step + cutting — `_ttp_tile_step()`:
   - `num_tiles = ceil(img_size / tile_size)`
   - `overlap = (num_tiles * tile_size - img_size) // (num_tiles - 1)`
   - `step = tile_size - overlap`
   - Last tile clamped to exact tile_size
4. Resolution per tile = min(tile_w, tile_h) — KHÔNG PHẢI preset_resolution / grid_size
5. max_resolution = fixed từ preset (6000/9000/12000) — KHÔNG chia cho grid
6. Ghép lại — `_ttp_blend_tiles()` (matching TTP_Image_Assy padding=64):
   - Linear gradient mask 255→0 chỉ 64px ở GIỮA overlap
   - Hard-cut phần còn lại (offset_left, offset_right)
   - PIL Image.composite() cho blending
   - Blend rows horizontally trước, rồi blend vertically

**QUAN TRỌNG**:
- Output size LUÔN theo megapixel target, KHÔNG phải longest edge
- Tile size PHẢI align to 8 (diffusion model requirement)
- Blending chỉ 64px (BLEND_PADDING=64), KHÔNG blend toàn bộ overlap
- 9K grid là ADAPTIVE (2 tiles) — landscape split horizontal, portrait split vertical
- 12K grid là 2×2 (4 tiles, KHÔNG PHẢI 3×3 = 9 tiles)
- blocks_to_swap: 6K=20 (spargeattn needs more VRAM than flash_attn), 9K/12K=32
- empty_cache() PHẢI gọi giữa mọi phase (encode/dit/decode) — skip = VRAM thrashing

**Files**:
- `app.py` — `_resize_to_megapixels()`, `_resize_keep_proportion()`, `_make_dimensions_even()`, `_apply_gaussian_blur()`, `_ttp_tile_size()`, `_ttp_tile_step()`, `_ttp_create_gradient_mask()`, `_ttp_blend_tiles()`, `_tile_and_upscale()`

## F6. Embedded Terminal Log Panel (Right Side)

**Yeu cau**: Hien thi realtime pipeline output (stdout/stderr) trong panel ben phai app, co the bat/tat.

**Cach hoat dong**:
1. `LogStream(io.TextIOBase)` wrap sys.stdout va sys.stderr
2. Moi dong write() vua ghi ra terminal goc (giu terminal output) vua emit signal toi GUI
3. `LogBridge(QWidget)` lam thread-safe bridge — QThread write stdout → signal → main thread append
4. `QPlainTextEdit` (read-only, monospace font) hien thi log, auto-scroll xuong cuoi
5. Terminal nam ben PHAI trong `terminal_widget` (fixedWidth=340px), main content ben trai (stretch=1)
6. Toggle button: "Hide ▶" thu nho panel xuong 40px, "◀" mo rong lai 340px
7. `maxBlockCount = 5000` gioi han buffer tranh memory leak
8. App mo len luon fullscreen (showMaximized)

**Layout**:
```
main_layout (QHBoxLayout)
├─ left_widget (stretch=1) ← images, controls, progress, buttons
└─ terminal_widget (fixedWidth=340) ← toggle button + log_console
```

**QUAN TRONG**:
- stdout/stderr redirect PHAI thread-safe — dung Qt Signal (LogBridge) de cross-thread communication
- Giu original stream de terminal van hoat dong (debug, tqdm progress bars)
- closeEvent() PHAI restore sys.stdout/sys.stderr ve original de tranh crash khi thoat
- LogStream ke thua io.TextIOBase de tuong thich voi tqdm, print(), logging
- Terminal panel PHAI o ben PHAI (KHONG phai ben duoi) — user preference
- main_layout PHAI la QHBoxLayout (KHONG phai QVBoxLayout)

**Files**:
- `app.py` — `LogStream`, `LogBridge`, `_toggle_terminal()`, `_append_log()`, `closeEvent()`, `terminal_widget`

## F7. System Monitor (Crystools-style)

**Yeu cau**: Hien thi realtime CPU%, RAM%, GPU%, VRAM%, GPU Temperature tren status bar.

**Cach hoat dong**:
1. `SystemMonitor(QWidget)` tao 4 mini QProgressBar (CPU, RAM, GPU, VRAM) + temperature label
2. QTimer update moi 2 giay
3. CPU/RAM: psutil.cpu_percent(), psutil.virtual_memory()
4. GPU/VRAM/Temp: `_query_nvidia_smi()` — subprocess goi nvidia-smi voi --query-gpu
5. Mau temperature: xanh < 60°C, vang 60-80°C, do > 80°C

**Files**:
- `app.py` — `SystemMonitor`, `_query_nvidia_smi()`

## F8. Elapsed Timer

**Yeu cau**: Bo dem giay realtime trong progress bar khi upscale.

**Cach hoat dong**:
1. Khi bat dau upscale: ghi `_start_time = time.time()`, start QTimer 1s
2. Moi giay: `_update_elapsed()` cap nhat progress bar format voi `[Xs]`
3. Khi xong: stop timer, hien thi tong thoi gian (vd: "Done! 3072x4096 in 2m 35s")

**Files**:
- `app.py` — `_elapsed_timer`, `_update_elapsed()`, `_start_upscale()`, `_on_upscale_done()`

## F9. Session Persistence + Auto-Load Models

**Yeu cau**: Luu settings giua cac phien. Khi mo app → tu dong restore settings + load models → user chi can chon anh va bam Upscale.

**Cach hoat dong**:
1. Khi dong app (`closeEvent`): `_save_config()` luu tat ca settings vao `config.json` (cung thu muc voi app.py)
2. Khi mo app (`__init__` cuoi): `_load_config()` doc config.json
3. Neu co config → restore tat ca combo boxes, preset, seed, UI state
4. Sau khi restore → `QTimer.singleShot(500, self._on_load_models)` tu dong load models
5. Neu khong co config (lan dau) → dung defaults + van auto-load models

**Settings duoc persist**:
| Setting | Widget | Default |
|---------|--------|---------|
| DiT model | `model_combo` | seedvr2_ema_7b-Q8_0.gguf |
| VAE model | `vae_combo` | DEFAULT_VAE |
| Resolution preset | `res_buttons` | 4K |
| Attention mode | `attn_combo` | spargeattn |
| Seed | `seed_spin` | 42 |
| Color correction | `color_combo` | lab |
| Terminal visible | `_log_visible` | True |
| Advanced expanded | `advanced_group` | False |

**KHONG persist**: blocks_to_swap, vae_tile, overlap — vi chung auto-set theo preset.

**QUAN TRONG**:
- Config luu khi DONG app (closeEvent), KHONG luu moi lan doi setting — don gian va du
- Restore combo box chi khi value ton tai trong dropdown (`findText >= 0`) — tranh crash neu model bi xoa
- Preset restore qua `_on_preset_click()` de dong bo blocks/vae_tile/overlap
- Auto-load dung QTimer.singleShot(500ms) de cho UI hien thi xong truoc khi bat dau load
- Neu config.json bi corrupt (invalid JSON) → `_load_config()` return None → dung defaults

**Files**:
- `app.py` — `_save_config()`, `_load_config()`, `closeEvent()`, cuoi `__init__`
- `config.json` — Auto-generated, cung thu muc voi app.py

## F10. Backend API Integration

**Yeu cau**: Backend Flask server goi SeedVR2Pipeline truc tiep thay vi qua ComfyUI (HTTP+WebSocket).

**Cach hoat dong**:
1. `pipeline.py` module tach rieng pipeline code (RESOLUTION_PRESETS, SeedVR2Pipeline, helpers, run_upscale) — import duoc boi ca GUI app va backend
2. `backend/seedvr2_api.py` cung cap `SeedVR2Client` class:
   - `load_models()`: Load DiT+VAE 1 lan, giu trong VRAM
   - `upscale(input_path, output_path, resolution, progress_fn)`: Upscale 1 anh
   - `is_ready()`: Check models loaded
   - `reset()`: Clear caches giua jobs
   - `full_reload()`: Reload sau OOM
3. Backend `_process_job()` goi `seedvr2.upscale()` truc tiep — khong can upload/queue/poll
4. Worker loop lazy load: chi load models khi co job dau tien
5. GPU error recovery: OOM → `full_reload()` → re-queue job

**Architecture**:
```
GUI app (app.py) → import pipeline.py → SeedVR2Pipeline
Backend (backend/app.py) → import seedvr2_api.py → SeedVR2Client → pipeline.py → SeedVR2Pipeline
```

**QUAN TRONG**:
- Models loaded 1 lan, giu giua cac job — KHONG reload moi job
- `threading.Lock()` trong SeedVR2Client chong concurrent GPU access — model reload PHAI nam trong lock
- Auto-reload khi doi preset co blocks_to_swap khac (vd: 4K=9 → 9K=32)
- **QUAN TRONG**: Model reload (load_models, full_reload) PHAI: (1) sync `_prefetch_stream`, (2) `model.to("cpu")` + `vae.to("cpu")` — EXPLICIT move GPU tensors ve CPU, (3) `self.runner = None`, (4) `torch.cuda.empty_cache()`. PHAI dung `model.to("cpu")` vi Python refcount KHONG free old model do circular refs (BlockSwap wrappers) → 1+GB GPU mem bi giu → new model load → tong vuot 24GB → CUDA swap qua RAM → 13s/block
- **QUAN TRONG — gc.collect() INCOMPATIBLE voi cudaMallocAsync**: KHONG DUOC goi gc.collect() O BAT KY DAU trong khi GPU tensors ton tai. gc.collect() trigger Python GC finalizers → GPU tensors freed OUT-OF-BAND (ngoai CUDA stream order) → CUDA pool corrupt → block swap 20-45s. Chi duoc goi trong debug endpoint (memory_profile) khi KHONG co job chay
- Points deducted SAU khi output thanh cong — KHONG TRUOC
- Output format: JPEG 95% + sRGB ICC profile (match GUI app)
- `seedvr2.reset()` goi trong `_worker_loop` finally block — dam bao cleanup moi job (success/fail/cancel)
- OOM retry gioi han 2 lan — tranh infinite loop (reload → OOM → reload → OOM...)
- Status API tra ve `system_healthy: seedvr2.is_ready()` thay vi `comfyui.is_healthy`
- Health endpoint `/api/comfyui/health` tra ve SeedVR2 info (GPU tu torch.cuda, queue tu job_queue, pipeline_mode)
- Admin clear-history endpoint clear VRAM cache (seedvr2.reset + empty_cache) thay vi ComfyUI history
- **QUAN TRONG**: Clear VRAM (ca admin va user endpoint) PHAI check `job_queue.processing` truoc khi goi `seedvr2.reset()` — neu co job dang chay, chi `empty_cache()`, KHONG reset pipeline (se corrupt ctx mid-inference). KHONG goi gc.collect()
- **QUAN TRONG**: VRAM monitoring (admin dashboard, pipeline status, clear-vram) dung `torch.cuda.memory_allocated(0)` de hien actual tensor usage. `mem_get_info()` bao cao CUDA pool la "used" voi cudaMallocAsync backend → hien 100% sai. `mem_get_info()` van dung cho total VRAM capacity
- Frontend admin.html hien thi "SeedVR2 Pipeline" + mode OPTIMAL/UNIVERSAL + VRAM info
- Cancel vs Stop: `cancelled` flag → status 'cancelled'; `stopped` flag → status 'pending' (re-runnable)
- Refix chi ho tro 2k/3k/4k (KHONG co 5k) — phai match RESOLUTION_PRESETS
- Stuck processing jobs tu crash: `restore_pending_jobs()` reset ve 'queued' trong DB
- Cancellation chi co effect giua cac pipeline phases (khong interrupt mid-phase)
- Pipeline progress monotonically increasing: pre-resize 1% → blur 2% → phases 5-100%
- **QUAN TRONG**: Tat ca admin endpoints (`/api/admin/*`) PHAI co `@admin_auth` decorator — bao gom logs, error-log, error-stats, queue-stats, force-cleanup
- **QUAN TRONG**: `generate_token()` nhan 1 dict `{'id': ..., 'username': ...}` — KHONG truyen 2 args rieng
- **QUAN TRONG**: Password min length = 6 chars cho CA register va admin reset
- **QUAN TRONG**: Refix enhance call PHAI gui `original_name` — neu khong, history se hien thi mangled filename
- **QUAN TRONG**: Admin gen-history view anh phai dung `/api/download/<fn>` — KHONG dung `/api/output/<fn>` (khong ton tai)
- **QUAN TRONG**: Upload PHAI dung `save_uploaded_image()` — auto convert webp/heic → JPEG. iPhone webp co encoding dac biet, PIL khong doc duoc truc tiep. KHONG dung `file.save()` truc tiep cho non-JPEG/PNG
- **QUAN TRONG**: `get_base_cache_dir()` khi khong co ComfyUI PHAI dung absolute path (`get_script_directory()`) — KHONG dung relative `./models/` (se resolve tu CWD cua caller)
- **QUAN TRONG**: `get_job_status()` khi job failed PHAI tra ve CA HAI `error` va `error_message` — web frontend doc `data.error`, desktop app doc `status.error_message`. Thieu 1 trong 2 se lam 1 client mat error message
- **QUAN TRONG**: `redeem_code()` response PHAI co field `points` (= `points_added`) — desktop app doc `result.points`, thieu se hien "undefined" trong toast

**Files**:
- `pipeline.py` — Pipeline module (shared giua GUI + backend)
- `backend/seedvr2_api.py` — SeedVR2Client wrapper, AdaptiveReloadStrategy, ResolutionLogger
- `backend/app.py` — Import SeedVR2Client, _process_job, _worker_loop

## F11. Adaptive Reload Strategy

**Yeu cau**: Tranh model reload ping-pong khi queue co cac resolution xen ke (4K→6K→4K→9K). Moi lan reload ton 30-60s. Thay vi luon reload, tu dong phat hien pattern xen ke va chuyen sang UNIVERSAL mode.

**Cach hoat dong**:
1. RESOLUTION_PRESETS chia 3 tier theo model settings:
   - LOW (2K/3K/4K): blocks=9, vae=1024, overlap=64
   - MID (6K): blocks=20, vae=768, overlap=32
   - HIGH (9K/12K): blocks=32, vae=768, overlap=32
2. `AdaptiveReloadStrategy` class trong `seedvr2_api.py` co 2 modes:
   - **OPTIMAL**: Dung settings toi uu cho tung tier (nhanh nhat, nhung can reload khi doi tier)
   - **UNIVERSAL**: Dung HIGH tier settings (blocks=32, vae=768, overlap=32) cho tat ca resolution.
     Cham hon ~6s cho LOW tier, nhung tranh 30-60s reload
3. Chuyen mode:
   - OPTIMAL → UNIVERSAL: khi >=3 tier transitions trong 6 job gan nhat (sliding window)
   - UNIVERSAL → OPTIMAL: khi 4 job lien tiep cung tier (streak)
4. Strategy integrate vao `SeedVR2Client.upscale()` — goi `strategy.get_settings()` truoc khi check `needs_reload()`
5. Sau moi job, goi `strategy.record_job()` de cap nhat mode

**QUAN TRONG**:
- UNIVERSAL mode dung settings HIGH tier — AN TOAN cho moi resolution vi blocks=32 > blocks can thiet
- Khi chuyen UNIVERSAL→OPTIMAL tai HIGH tier, KHONG can reload vi dang dung cung settings
- Strategy KHONG bi reset khi full_reload() (OOM recovery) — chi pipeline reload, history giu nguyen
- Strategy reset ve OPTIMAL khi server restart (khong co data → chap nhan)
- Thresholds (WINDOW_SIZE=6, TRANSITION_THRESHOLD=3, STREAK_TO_OPTIMAL=4) co the dieu chinh sau khi phan tich log

**Files**:
- `backend/seedvr2_api.py` — AdaptiveReloadStrategy class
- `backend/seedvr2_api.py` — SeedVR2Client.upscale() integrate strategy

## F12. Resolution Queue Log

**Yeu cau**: Log rieng ghi lai thu tu cac job (resolution, tier, mode, reload, timing) de phan tich pattern va toi uu hoa backend sau nay.

**Cach hoat dong**:
1. `ResolutionLogger` class trong `seedvr2_api.py`
2. File log: `backend/logs/resolution_queue.log` — human-readable, Claude-readable
3. Moi job ghi 1 dong: timestamp, preset, tier, strategy mode, reload status, timing
4. Moi 10 job ghi summary block: queue sequence, tier transitions, reload rate
5. Khi shutdown ghi final summary + resolution distribution
6. Session boundaries marked voi `=== Session started/ended ===`

**Format mau**:
```
[14:30:05] #1  4K (LOW)  | OPTIMAL | blocks=9  | reload=YES 30.2s | process=45.3s | job=abc123 user=u1
[14:31:20] #2  6K (MID)  | OPTIMAL | blocks=20 | reload=YES 32.1s | process=51.2s | job=def456 user=u2
```

**QUAN TRONG**:
- Log append-only, khong overwrite — tich luy qua nhieu session
- Claude doc file nay khi user noi "check log queue" → phan tich pattern → goi y toi uu
- Log KHONG anh huong performance (write 1 dong text sau moi job)
- Log errors khong lam fail job (try/except trong log_job)

**Files**:
- `backend/seedvr2_api.py` — ResolutionLogger class
- `backend/logs/resolution_queue.log` — Output log file

## F10. Trang Deblur — Blur Analysis + Multi-pass SeedVR2 Upscale

**Yeu cau**: Trang web moi cho workflow deblur: phan tich blur de tim effective resolution, resize ve muc do, roi chay SeedVR2 nhieu lan de khoi phuc chat luong.

**Cach hoat dong**:
1. User drop/upload anh vao queue tren trang deblur.html
2. Khi bam Run, backend chay blur_score.py (TOPIQ-NR) phan tich blur score (0-100) va tim effective resolution (do phan giai thuc su cua noi dung)
3. Anh duoc resize ve effective resolution (cv2.INTER_AREA)
4. Branching logic dua tren effective longest edge:
   - **Branch A** (> 1000px va < 4000px): chay 1 pass 4K workflow
   - **Branch B** (>= 4000px): skip — anh da net, tra ket qua "anh da net"
   - **Branch C** (<= 1000px): multi-pass
     - Pass 1: SeedVR2 upscale (eff * 4, max 2000px, LOW tier blocks=9)
     - Kiem tra output: neu <= 1000px → Pass 2 (current * 4, max 2000px) → Pass 3 (4K workflow)
     - Neu output Pass 1 > 1000px → Pass 2 (4K workflow)
5. Pre-check: neu input longest edge > 8000px → resize ve 6000px truoc

**VRAM management cho blur_score.py**:
- BlurAnalyzer singleton, lazy-load TOPIQ-NR chi khi can
- Sau analyze() xong, model.to("cpu") + empty_cache() de giai phong GPU cho SeedVR2
- Thread-safe (lock)

**QUAN TRONG**:
- Blur analyzer PHAI giai phong GPU truoc khi SeedVR2 chay — goi model.to("cpu")
- Giua cac pass SeedVR2, goi seedvr2.reset() de don dep
- Progress reporting: blur analysis 0-5%, resize 5-8%, cac pass chia deu phan con
- Pre-check 8000px TRUOC khi analyze (tranh analyze anh qua lon)
- Branch B KHONG tru diem — anh da net, khong can xu ly
- Pre-check 8000px resize KHONG DUOC ghi de file goc — phai luu vao temp file (can file goc cho before/after viewer)
- syncWithBackend phai combine `pending + queued + processing + completed + failed` arrays va filter `is_deblur` only
- `get_user_jobs()` phai SELECT `is_deblur` column trong tat ca queries
- pollJobStatus CAN co timeout guard (30 min) de tranh chay vo han
- Status endpoint completed phai tra `blur_result` va `passes` tu in-memory completed data

**Files**:
- `backend/blur_score.py` — BlurAnalyzer module (TOPIQ-NR, lazy-load, singleton)
- `backend/app.py` — `_process_deblur_job()`, `/api/deblur/analyze`, `/api/deblur/enhance`
- `frontend/deblur.html` — Trang deblur (3-column layout)
- `frontend/deblur.js` — IIFE JavaScript: queue, blur analysis, polling, viewer
- `frontend/style.css` — Blur badges, queue item styles, deblur badges
