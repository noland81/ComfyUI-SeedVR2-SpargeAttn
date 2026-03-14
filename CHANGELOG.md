# Changelog

## v2.4.2 (2026-03-14) — Deblur logging + optimize blur analysis

### Cai tien
- **Detailed logging cho toan bo deblur pipeline** — Log tu API endpoint, blur analysis, den tung pass SeedVR2 voi timing chi tiet
  - API endpoints: log request received, file info, analyze duration, errors
  - `_process_deblur_job`: log input size, blur score, branch decision, tung pass start/done voi duration
  - `blur_score.py`: log tung phase probe voi score + timing, GPU model move
  - Final summary: score, effective resolution, passes, total duration
- **Toi uu blur analysis: binary search** — Phase 1 chuyen tu linear scan (18 probes) sang binary search (~6 probes), giam thoi gian phan tich ~3-4x cho anh lon
- **Early exit trong Phase 1** — Ngung scan khi da tim duoc threshold, khong can scan toan bo range
- **Retry logic cho analyze** — Frontend tu dong retry 2 lan khi analyze that bai (fix "Failed to fetch" do timeout)

### Files thay doi
- `backend/app.py` — Detailed logging cho deblur API endpoints + `_process_deblur_job` pipeline
- `backend/blur_score.py` — Binary search Phase 1 + timing logs cho moi probe + GPU move logs
- `frontend/deblur.js` — Retry logic cho analyzeItem (max 2 retries)

## v2.4.1 (2026-03-14) — Fix deblur bugs: pre-resize overwrite, sync filter, polling timeout

### Bug fixes
- **Pre-resize ghi de file goc** — Khi input > 8000px, `cv2.imwrite(input_path, ...)` ghi de file upload goc. Fix: luu pre-resize vao temp file, giu nguyen file goc cho before/after viewer
- **syncWithBackend sai data format** — `data.jobs` khong ton tai trong API response (API tra `{pending, queued, processing, completed, failed}`). Fix: combine all arrays va filter `is_deblur` only
- **get_user_jobs thieu is_deblur** — 3 SELECT queries trong `get_user_jobs()` khong include `is_deblur` column → syncWithBackend khong the filter deblur jobs. Fix: them `is_deblur` vao tat ca queries
- **runQueue stale indices** — Thu thap indices upfront nhung queue co the thay doi giua cac run. Fix: dung item references va re-lookup index moi lan
- **Polling khong co timeout** — `setInterval` chay vo han neu backend khong tra completed/failed. Fix: them 30-minute timeout guard
- **Passes info luon hien '--'** — Status endpoint khong tra `blur_result` va `passes` cho completed jobs. Fix: them in-memory completed data vao status response
- **Temp file leak Branch B** — Khi image pre-resized va "already sharp", pre_resize_temp khong duoc cleanup vi early return truoc try/finally. Fix: them cleanup truoc return
- **Client-side "already sharp" bug** — Frontend set `output_filename = input_filename` khi skip, nhung file khong ton tai trong output folder. Fix: xoa client-side check, de backend handle Branch B (copy file + no points)
- **get_user_jobs thieu duration** — Query no-status-filter thieu `duration` trong SELECT. Fix: them `duration`

### Files thay doi
- `backend/app.py` — Fix `_process_deblur_job()` pre-resize temp, `get_user_jobs()` is_deblur + duration, `get_job_status()` blur_result/passes
- `frontend/deblur.js` — Fix syncWithBackend, runQueue, pollJobStatus timeout + blur_result capture, remove client-side sharp check
- `backend/VERSION.txt` — 3.32.0 → 3.32.1

## v2.4.0 (2026-03-14) — Trang Deblur: blur analysis + multi-pass SeedVR2 upscale

### Tinh nang moi
- **Trang Deblur** (`deblur.html`) — Trang web moi cho workflow deblur: phan tich blur → resize ve effective resolution → multi-pass SeedVR2 upscale de khoi phuc do net
- **Blur Analyzer** (`blur_score.py`) — Module TOPIQ-NR phan tich blur score (0-100), tim effective resolution qua 2-phase search (coarse 5% + fine 1%)
- **Multi-pass processing logic** trong `_process_deblur_job()`:
  - Pre-check: input > 8000px → resize ve 6000px
  - Branch A: effective > 1000px va < 4000px → single 4K pass
  - Branch B: effective >= 4000px → skip (anh da net)
  - Branch C: effective <= 1000px → Pass 1 (eff*4) → Pass 2 neu can (max 2000px) → Final 4K pass
- **API endpoints moi**: `POST /api/deblur/analyze` (preview blur score), `POST /api/deblur/enhance` (submit deblur job)
- **DB migration**: them cot `is_deblur` vao jobs + history tables, `count_deblur` vao users table
- **Points system**: them `price_deblur` (default 15 pts), admin co the chinh qua settings
- **Navigation**: them link Deblur vao index.html, advanced.html, refix.html
- **Frontend**: deblur.js voi queue management, blur analysis preview (score + badge + effective resolution), multi-pass progress tracking, before/after viewer

### Files thay doi
- `backend/blur_score.py` — **MOI** — BlurAnalyzer module (TOPIQ-NR, lazy-load, VRAM management)
- `backend/app.py` — DB migration, `_process_deblur_job()`, deblur API endpoints, points integration
- `frontend/deblur.html` — **MOI** — Trang deblur 3-column layout
- `frontend/deblur.js` — **MOI** — IIFE JavaScript: upload, queue, blur analysis, polling, viewer
- `frontend/style.css` — Them styles: blur badges, queue item inner, deblur recent badge
- `frontend/index.html` — Them Deblur link vao titlebar-nav + mobile-links
- `frontend/advanced.html` — Them Deblur link vao mode-switcher
- `frontend/refix.html` — Them mode-switcher voi link Deblur
- `backend/VERSION.txt` — 3.32.0
- `backend/start_server.bat` — v2.4.0

## v2.3.11 (2026-03-14) — Xoa toan bo gc.collect() + them diagnostic logging

### Bug fixes
- **REGRESSION v2.3.10: clear_memory(deep=True) lam CHAM hon** — gc.collect() trong clear_memory(deep=True) khi reload model CUNG corrupt cudaMallocAsync pool. Job dau tien sau restart cham 20-45s/block thay vi 500ms. v2.3.9 chi cham tu job thu 2, v2.3.10 lam cham ca job dau → WORSE
- **gc.collect() O NHIEU CHO khac trong app.py va seedvr2_api.py** — Phat hien 7+ cho goi gc.collect() chay SAU MOI JOB:
  1. `seedvr2_api.py reset()` — chay sau MOI job qua worker finally block
  2. `app.py add_completed()` — chay sau moi job thanh cong
  3. `app.py add_failed()` — chay sau moi job that bai
  4. `app.py _cleanup_old_jobs()` — chay khi cleanup old jobs
  5. `app.py admin_clear_vram()` — admin endpoint
  6. `app.py pipeline_clear_vram()` — user endpoint
  7. `app.py` resize endpoint, memory cleanup endpoint

### Fix
- **pipeline.py reload**: EXPLICIT GPU cleanup — `model.to("cpu")` + `vae.to("cpu")` TRUOC khi deref runner. Bo `clear_memory(deep=True)`. Root cause: `self.runner = None` KHONG free old model do circular refs (BlockSwap wrappers, weakrefs) → 1.07GB GPU mem con bi giu → new model load len → tong VRAM vuot 24GB → CUDA swap qua system RAM → 13s/block
- **seedvr2_api.py reset()**: Bo gc.collect()
- **seedvr2_api.py full_reload()**: Bo gc.collect(), them model.to("cpu") + vae.to("cpu")
- **app.py**: Bo TAT CA gc.collect() (7 cho): add_completed, add_failed, _cleanup_old_jobs, admin_clear_vram, pipeline_clear_vram, resize endpoint, memory cleanup
- **Giu gc.collect() DUY NHAT**: trong `memory_profile()` debug endpoint (can cho object counting, co warning)

### Diagnostic logging (ghi ra file `backend/logs/diag.log` — Claude doc truc tiep)
- `diag_logger.py` — Module ghi DIAG log ra ca console + file, thread-safe
- `pipeline.py load_models()` reload: [DIAG-RELOAD] VRAM at each step
- `pipeline.py upscale_image()`: [DIAG-RUN] VRAM before encode + before DiT
- `blockswap.py wrapped_forward()`: [DIAG-BLOCK] Per-block timing (load/prefetch/compute/offload)
- `seedvr2_api.py reset()`: [DIAG-RESET] VRAM before/after cleanup

### Key insight — Circular refs giu old model tren GPU
- `self.runner = None` KHONG du de free GPU memory — BlockSwap wrappers (bound methods, weakrefs, `_original_forward` refs) tao circular references giu model alive
- VRAM log chung minh: alloc van 1.07GB SAU khi deref runner, SAU empty_cache
- Khi new model (blocks_to_swap=20, nhieu blocks on GPU) load len → tong > 24GB → cudaMallocAsync dung system RAM → compute 13s/block
- Fix: `model.to("cpu")` explicitly move GPU tensors ve CPU TRUOC khi deref → GPU memory freed ngay qua normal CUDA path, khong can gc.collect()

### Files thay doi
- `pipeline.py` — Explicit model.to("cpu") + vae.to("cpu") trong reload, them DIAG logging
- `diag_logger.py` — NEW: diagnostic logger ghi ra file + console
- `src/optimization/blockswap.py` — Them DIAG timing per-block
- `backend/seedvr2_api.py` — Bo gc.collect(), them model.to("cpu") trong full_reload(), DIAG logging
- `backend/app.py` — Bo 7 cho gc.collect(), BACKEND_VERSION 1.11.0 → 1.11.1
- `VERSION.txt` — 2.3.10 → 2.3.11
- `backend/VERSION.txt` — 3.31.84 → 3.31.85
- `backend/start_server.bat` — Version 2.3.10 → 2.3.11

## v2.3.10 (2026-03-14) — Fix cross-job block swap degradation khi switch preset (12K→6K)

### Bug fixes
- **Block swap 12-29s/block khi chay job thu 2 voi preset khac (vd: 12K→6K) — ROOT CAUSE: load_models() reload khong cleanup old model dung cach** — Khi `blocks_to_swap` thay doi giua jobs (12K=32 → 6K=20), pipeline reload model moi. Nhung code cu chi deref `self.runner = None` roi goi `clear_memory()` (deep=False, chi `empty_cache()`). Van de:
  1. Old model's `_prefetch_stream` (async DMA stream) KHONG duoc sync truoc khi free → async GPU ops co the reference freed memory
  2. `clear_memory(deep=False)` KHONG goi `gc.collect()` → Python GC chua release old model tensors → GPU memory van bi giu
  3. New model load len tren old memory → CUDA pool fragmented → block swaps cham 12-29s thay vi 500ms
- **full_reload() trong seedvr2_api.py cung bug** — Deref `pipeline.runner` ma khong sync prefetch stream. Them: set `_loaded=False` truoc `load_models()` nen skip block `if self._loaded:` trong load_models() → bypass fix

### Fix
- `pipeline.py load_models()`: Sync `_prefetch_stream` TRUOC khi deref old model, doi `clear_memory()` → `clear_memory(deep=True)` de gc.collect() chi chay 1 lan luc reload (KHONG phai moi job/tile)
- `seedvr2_api.py full_reload()`: Them sync `_prefetch_stream` truoc deref, doi thu tu `gc.collect()` truoc `empty_cache()`

### Key insight
- `gc.collect()` chi CO HAI khi chay SAU MOI JOB/TILE (v2.3.6 bug) — fragment CUDA pool moi job
- `gc.collect()` la CAN THIET khi RELOAD MODEL — force Python release old model tensors truoc khi load new model
- Fix nay chi dung gc.collect() 1 cho duy nhat: trong `clear_memory(deep=True)` luc reload — KHONG anh huong performance cua jobs binh thuong

### Files thay doi
- `ComfyUI-SeedVR2-SpargeAttn/pipeline.py` — Fix load_models() reload: sync prefetch stream + clear_memory(deep=True)
- `backend/seedvr2_api.py` — Fix full_reload(): sync prefetch stream truoc deref
- `VERSION.txt` — 2.3.9 → 2.3.10
- `backend/start_server.bat` — Version 2.3.9 → 2.3.10

## v2.3.9 (2026-03-14) — Fix block swap 14-15s regression tu gc.collect() + synchronize()

### Bug fixes
- **Block swap 14-15s/block tu job thu 2 tro di — ROOT CAUSE: gc.collect() + synchronize()** — gc.collect() tuong tac xau voi `cudaMallocAsync` CUDA allocator backend. Khi gc.collect() chay giua jobs/tiles, no trigger Python GC scan TOAN BO heap → finalizers chay → GPU tensors duoc free OUT-OF-BAND (ngoai CUDA stream order) → CUDA pool bi fragment/corrupt → allocations cham. Ket hop voi `synchronize()` trong reset() (v2.3.7) force-drain toan bo pool → job tiep theo phai allocate tu CUDA driver → 14-15s/block thay vi 500ms
- **Dashboard VRAM hien 100% sai** — `torch.cuda.mem_get_info()` bao cao CUDA memory pool la "used" voi cudaMallocAsync backend. Fix: dung `torch.cuda.memory_allocated()` de hien actual tensor usage

### Changes reverted (tu v2.3.6 + v2.3.7)
- Bo `gc.collect()` sau `upscale()` finally trong seedvr2_api.py (v2.3.6)
- Bo `gc.collect()` sau `reset()` trong app.py worker finally (v2.3.6)
- Bo `gc.collect()` sau del tiles/upscaled_tiles/row_images trong pipeline.py (v2.3.6)
- Bo `synchronize()` truoc `empty_cache()` trong reset() va full_reload() (v2.3.7)
- GIU `del` statements (chi drop references, khong goi GC)
- GIU prefetch stream sync + _prefetched_idx reset (dung, xu ly async DMA state)

### Root cause analysis
- `cudaMallocAsync` (PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync) su dung CUDA native memory pools
- `gc.collect()` trigger Python GC finalizers → GPU tensors freed ngoai CUDA stream → pool management bi anh huong
- `synchronize() + empty_cache()` = `cudaMemPoolTrimTo(0)` → xoa toan bo pool → allocations tu CUDA driver cham
- Job dau sau restart: clean pool → nhanh (500ms/block)
- Job thu 2+: pool bi drain/fragment → cham (14-15s/block)
- Giai phap: de CUDA allocator tu quan ly pool, chi goi empty_cache() KHONG synchronize()

### Files thay doi
- `backend/seedvr2_api.py` — Bo synchronize() trong reset()/full_reload(), bo gc.collect() trong upscale() finally
- `backend/app.py` — Bo gc.collect() sau reset() trong worker finally, fix VRAM monitoring dung memory_allocated()
- `ComfyUI-SeedVR2-SpargeAttn/pipeline.py` — Bo gc.collect() trong _tile_and_upscale() (3 cho)
- `VERSION.txt` — 2.3.8 → 2.3.9
- `backend/start_server.bat` — Version 2.3.8 → 2.3.9

## v2.3.8 (2026-03-14) — Revert synchronize() regression: VRAM 100% tren moi resolution

### Bug fixes
- **VRAM 100% tren tat ca resolution (6K, 9K, 12K) — REGRESSION tu v2.3.7** — `torch.cuda.synchronize()` truoc `empty_cache()` giua cac phases (Phase1→2, Phase2→3, Phase3→4) pha huy CUDA allocator cache pool. Allocator pool la CO CHE QUAN TRONG giup PyTorch tai su dung memory blocks giua phases ma khong can xin lai tu CUDA driver. Khi `synchronize() + empty_cache()` chay giua phases, toan bo pool bi tra lai driver → phase tiep theo phai allocate lai tu scratch → fragmentation nang hon → VRAM 100%. Trieu chung: 6K block swap 8-31s/block, 9K va 12K cung VRAM 100%. Fix: bo `synchronize()` giua phases, CHI giu `synchronize()` trong `reset()` va `full_reload()` (giua JOBS, khong phai giua phases)
- **End-of-function cleanup qua aggressive trong upscale_image()** — Block `_reset_ctx_for_run() + synchronize() + empty_cache()` cuoi `upscale_image()` (them o v2.3.7) chay SAU MOI TILE trong tiled processing. Pha huy allocator pool giua tiles → tile 2+ phai re-allocate tat ca. Original code khong co block nay vi `_reset_ctx_for_run()` dau tile tiep theo da du de free references, allocator tu tai su dung blocks. Fix: bo toan bo end-of-function cleanup block

### Root cause analysis
- `torch.cuda.synchronize()` doi TAT CA async CUDA ops hoan thanh, sau do `empty_cache()` tra TOAN BO free blocks ve CUDA driver
- Giua phases trong 1 job, day la KHONG CAN THIET va CO HAI: allocator pool giup phases sau tai su dung memory tu phases truoc
- Chi can `synchronize()` giua JOBS (trong reset/full_reload) de dam bao cross-job cleanup
- `empty_cache()` KHONG co `synchronize()` van hoat dong: no chi tra unused cached blocks, khong anh huong pending ops

### Files thay doi
- `ComfyUI-SeedVR2-SpargeAttn/pipeline.py` — Bo `synchronize()` truoc 4 `empty_cache()` calls (load_models, Phase1→2, Phase2→3, Phase3→4). Bo toan bo end-of-function cleanup block trong upscale_image()
- `VERSION.txt` — 2.3.7 → 2.3.8
- `backend/start_server.bat` — Version 2.3.7 → 2.3.8
- `CHANGELOG.md` — Them entry v2.3.8

## v2.3.7 (2026-03-14) — Fix VRAM thrashing khi gen nhieu anh lien tiep

### Bug fixes
- **Block swap 11-13s/block thay vi ~250ms sau 4+ jobs** — `torch.cuda.empty_cache()` duoc goi KHONG co `synchronize()` truoc → pending async CUDA operations (non_blocking .to(), prefetch stream DMA) chua hoan thanh → CUDA allocator khong the giai phong VRAM → fragmentation tich luy qua nhieu jobs → block swap phai doi allocator search contiguous memory. Fix: them `torch.cuda.synchronize()` truoc MOI `empty_cache()` call (4 cho trong pipeline.py + reset() + full_reload())
- **Prefetch stream khong sync giua jobs** — `_prefetch_stream` (CUDA stream rieng cho async block loading) khong duoc synchronize khi `cleanup_blockswap(keep_state_for_cache=True)`. Pending DMA ops tu job cu linger tren stream → job moi `wait_stream()` phai doi stale operations. Fix: them `model._prefetch_stream.synchronize()` + reset `_prefetched_idx = -1` trong cleanup_blockswap voi keep_state
- **_prefetched_idx khong reset giua jobs** — `seedvr2.reset()` khong reset `_prefetched_idx`, co the dang = block index cu → next job tin rang block da duoc prefetch nhung thuc te khong → load sync fallback bat ngo. Fix: reset `_prefetched_idx = -1` trong reset()
- **12K VRAM 100%: khong free VRAM giua tiles** — `upscale_image()` return numpy result nhung KHONG free CUDA tensors (encoded latents, attention cache, ctx data) truoc khi return. Khi tiled processing (12K = 4 tiles), VRAM tu tile 1 (~2-4GB) van tren GPU khi tile 2 bat dau → VRAM tich luy → 100%. Fix: them `_reset_ctx_for_run()` + `synchronize()` + `empty_cache()` tai cuoi `upscale_image()` truoc khi return

### Root cause analysis
- User chay 4x 12K + 1x 9K → job thu 5 block swap 11-13 GIAY/block
- Ca 12K va 9K deu dung blocks_to_swap=32 voi "sync" offload strategy (blocking)
- Nhung VRAM fragmentation tu cac job truoc KHONG duoc giai phong vi thieu synchronize()
- CUDA allocator giu ~10GB reserved memory tu pending async ops → tim contiguous block cho moi swap mat 11-13s

### Files thay doi
- `backend/seedvr2_api.py` — Them `synchronize()` truoc `empty_cache()` trong `reset()` va `full_reload()`, reset `_prefetched_idx` va sync prefetch stream trong `reset()`
- `ComfyUI-SeedVR2-SpargeAttn/pipeline.py` — Them `synchronize()` truoc 4 `empty_cache()` calls (load_models, Phase1→2, Phase2→3, Phase3→4)
- `ComfyUI-SeedVR2-SpargeAttn/src/optimization/blockswap.py` — Sync prefetch stream + reset prefetched_idx trong `cleanup_blockswap(keep_state_for_cache=True)`
- `backend/app.py` — BACKEND_VERSION 1.10.1 → 1.10.2
- `backend/start_server.bat` — Version 2.3.6 → 2.3.7
- `VERSION.txt` — 2.3.6 → 2.3.7
- `CHANGELOG.md` — Them entry v2.3.7

## v2.3.6 (2026-03-14) — Fix RAM memory leak khi gen nhieu lan

### Bug fixes
- **progress dict khong clear khi job fail/cancel** — `self.progress[job_id]` chi duoc xoa khi job thanh cong (line 2115), khong xoa khi fail/cancel/OOM → dict tich tu entries theo thoi gian. Fix: them `self.clear_progress(job_id)` tai tat ca error paths (OOM retry, OOM fail, cancel, generic error)
- **oom_retries khong clear sau retry thanh cong** — Khi OOM retry thanh cong va job duoc re-queue, `oom_retries[job_id]` khong duoc pop → dict tich tu. Fix: them `oom_retries.pop(job_id, None)` sau successful re-queue
- **finally block thieu gc.collect + silent reset failure** — `seedvr2.reset()` trong finally block voi `except: pass` → (1) khong biet khi reset that bai, (2) `gc.collect()` khong duoc goi doc lap. Fix: log reset failure, them `gc.collect()` rieng biet luon chay bat ke reset thanh cong hay khong
- **seedvr2 upscale() khong cleanup khi exception** — `img`, `image_np`, `result_np`, `result_img` (co the hang tram MB moi doi tuong) chi duoc `del` tren success path (line 509). Neu exception xay ra giua line 458-508, cac object nay leak cho den khi GC tu dong chay. Fix: wrap trong try/finally, `del` + `gc.collect()` luon chay
- **_tile_and_upscale() khong del arrays trung gian** — `tiles[]`, `upscaled_tiles[]`, `row_images[]` (moi array co the hang GB cho 9K/12K) khong duoc del sau khi reassemble xong → nam trong RAM cho den khi function scope ket thuc va GC chay. Fix: `del tiles` ngay sau upscale xong, `del upscaled_tiles` ngay sau blend rows, `del row_images` ngay sau blend columns, kem `gc.collect()` moi buoc

### Files thay doi
- `backend/app.py` — Fix progress cleanup (5 cho), oom_retries cleanup, finally block gc.collect + log, BACKEND_VERSION 1.10.0 → 1.10.1
- `backend/seedvr2_api.py` — Wrap upscale() image processing trong try/finally
- `ComfyUI-SeedVR2-SpargeAttn/pipeline.py` — Del intermediate arrays trong _tile_and_upscale() sau moi phase
- `backend/start_server.bat` — Version 2.3.5 → 2.3.6
- `VERSION.txt` — 2.3.5 → 2.3.6
- `CHANGELOG.md` — Them entry v2.3.6

## v2.3.5 (2026-03-13) — Fix API contract mismatches with desktop app

### Bug fixes
- **`/api/status` missing `error_message` field (CRITICAL)** — Backend tra ve `error` nhung desktop app (TypeScript) doc `error_message` → error message luon `undefined`, UI hien "Processing failed" thay vi loi thuc te. Fix: tra ve **ca hai** `error` va `error_message` de tuong thich ca web frontend (`data.error`) va desktop app (`status.error_message`)
- **Redeem code response missing `points` field** — Backend tra ve `points_added` nhung desktop app doc `result.points` → toast hien "Redeemed! +undefined points". Fix: them field `points` = `points_added` vao response

### Cross-check findings (desktop app vs backend)
- 23 API endpoints checked — 21 OK, 2 bugs fixed
- `syncJobs()` trong desktop app typed sai (`JobsResponse` thay vi actual `/api/jobs/sync` shape) nhung chua duoc su dung — khong anh huong runtime
- `imagePrep.ts` convert tat ca uploads sang JPEG truoc khi gui → iPhone HEIC khong anh huong desktop app (chi anh huong web browser)
- Auth flow (login, register, refresh) — tuong thich tot, `refreshToken()` bug da fix o v2.3.2

### Files thay doi
- `backend/app.py` — Fix `get_job_status()` tra ve `error_message`, fix `redeem_code()` tra ve `points`, BACKEND_VERSION 1.9.4 → 1.9.5
- `backend/start_server.bat` — Version 2.3.3 → 2.3.5
- `VERSION.txt` — 2.3.4 → 2.3.5
- `CHANGELOG.md` — Them entry v2.3.5

## v2.3.4 (2026-03-13) — Fix webp upload from iPhone + model path

### Bug fixes
- **iPhone webp upload crash** — iPhone screenshots/photos upload dang `.webp` nhung PIL khong doc duoc (encoding dac biet). Fix: them `save_uploaded_image()` helper tu dong convert webp → JPEG (quality 95, giu ICC profile) khi upload. Ap dung cho ca 3 endpoints: `/api/upload`, `/api/queue/add`, `/api/upload-resize`
- **Model path sai khi chay khong co ComfyUI** — `get_base_cache_dir()` fallback sang `./models/SEEDVR2` (relative path) → resolve tu CWD cua backend (`backend/models/SEEDVR2`) thay vi project root → khong tim thay models → download lai. Fix: dung `os.path.join(get_script_directory(), "models", SEEDVR2_FOLDER_NAME)` (absolute path)

### Files thay doi
- `backend/app.py` — Them `save_uploaded_image()` helper, ap dung cho 3 upload endpoints
- `ComfyUI-SeedVR2-SpargeAttn/src/utils/constants.py` — Fix `get_base_cache_dir()` dung absolute path
- `VERSION.txt` — 2.3.3 → 2.3.4
- `CHANGELOG.md` — Them entry v2.3.4

## v2.3.3 (2026-03-13) — Remove server chat

### Thay doi
- **Go bo toan bo chat system** — Chat server (WebSocket port 5001) va chat widget khong con su dung
  - Xoa `backend/chat_server.py` — WebSocket chat server (Flask-SocketIO, SQLite)
  - Xoa `frontend/chat-widget.js` — Chat bubble widget (embedded trong 3 trang user)
  - Xoa `frontend/admin-chat.html` — Admin chat panel
  - Xoa chat widget includes + CHAT_SERVER_URL config tu `index.html`, `advanced.html`, `refix.html`
  - Xoa link "Support Chat" tu `admin.html`
  - Don gian hoa `run_all.py` — chi chay main server (xoa chat server thread + flask-socketio/gevent deps)
- Backend version: 1.9.2 → 1.9.3

### Files thay doi
- `backend/chat_server.py` — **XOA**
- `frontend/chat-widget.js` — **XOA**
- `frontend/admin-chat.html` — **XOA**
- `frontend/index.html` — Xoa chat widget script + CHAT_SERVER_URL config
- `frontend/advanced.html` — Xoa chat widget script + CHAT_SERVER_URL config
- `frontend/refix.html` — Xoa chat widget script + CHAT_SERVER_URL config
- `frontend/admin.html` — Xoa "Support Chat" link
- `backend/run_all.py` — Don gian hoa, chi chay main server
- `VERSION.txt` — 2.3.2 → 2.3.3
- `CHANGELOG.md` — Them entry v2.3.3

## v2.3.2 (2026-03-13) — Security fixes + frontend-backend connection review

### Security fixes
- **`/api/auth/refresh` crash** — `generate_token()` nhan dict nhung bi goi voi 2 args rieng → TypeError at runtime. Fix: truyen dict `{'id': user['id'], 'username': user['username']}`
- **Admin log endpoints khong co auth** — 4 endpoints (`/api/admin/logs`, `/api/admin/logs/<fn>`, `/api/admin/error-log`, `/api/admin/error-stats`) khong co `@admin_auth` → bat ky ai cung doc duoc server logs. Fix: them `@admin_auth` cho ca 4
- **`/api/admin/queue-stats` khong co admin auth** — Dung `@token_optional` thay vi `@admin_auth` → exposed queue stats. Fix: doi sang `@admin_auth`
- **`/api/force-cleanup` khong co auth** — Bat ky ai cung trigger force cleanup xoa het completed/failed jobs. Fix: them `@admin_auth`
- **Password min length inconsistency** — Register yeu cau 6 chars, admin reset chi yeu cau 4 chars. Fix: thong nhat 6 chars cho ca 2

### Bug fixes
- **Admin gen-history 404** — Admin panel gen-history dung `/api/output/<fn>` (khong ton tai) → 404 khi click xem anh. Fix: doi sang `/api/download/<fn>`
- **Duplicate `/api/preview` route** — 2 route definitions cho cung URL → Flask overwrite, first one (1500px, q90) la dead code. Fix: xoa first definition, giu second (1200px, q85, check ca output + input folder)
- **Refix missing original_name** — refix.js khong gui `original_name` trong `/api/enhance` call → history hien thi mangled server filename thay vi ten goc. Fix: them `original_name: state.localFileName || state.originalFilename`
- **Admin log calls missing headers** — Frontend admin.html goi 3 log endpoints khong gui `getAdminHeaders()`. Fix: them headers cho ca 3 calls

### Dead code cleanup
- Xoa `WORKFLOWS` va `REFIX_WORKFLOWS` dicts + loading loops (~30 dong) — ComfyUI workflow files khong con su dung
- Backend version: 1.9.1 → 1.9.2

### Files thay doi
- `backend/app.py` — Security fixes (5), bug fixes (3), dead code cleanup
- `frontend/admin.html` — Fix gen-history URL, add auth headers to log calls
- `frontend/refix.js` — Add original_name to enhance call
- `VERSION.txt` — 2.3.1 → 2.3.2
- `CHANGELOG.md` — Them entry v2.3.2

## v2.3.1 (2026-03-13) — Bug fixes + dead ComfyUI code cleanup

### Bug fixes
- **VRAM free calculation inaccurate** — Health endpoint va clear-VRAM endpoint dung `total_mem - memory_allocated()` khong tinh CUDA context, cuDNN workspace, fragmentation. Fix: dung `torch.cuda.mem_get_info(0)` tra ve actual CUDA driver-level (free, total)
- **Clear VRAM crash running job** — Ca admin endpoint (`/api/admin/comfyui/clear-history`) va user endpoint (`/api/comfyui/clear-queue`) goi `seedvr2.reset()` khong check job dang chay. `reset()` set ctx keys = None → corrupt pipeline mid-inference. Fix: check `job_queue.processing` truoc, neu co job → skip reset, chi `empty_cache() + gc.collect()`

### Dead code cleanup
- Xoa `_start_comfyui_if_not_running()` + `_auto_recover_comfyui()` (~145 dong dead code — khong bao gio duoc goi)
- Xoa env vars: `COMFYUI_BAT_PATH`, `COMFYUI_RESTART_WAIT`, `COMFYUI_INPUT`
- Xoa `COMFYUI_URL` env var + `from comfyui_api import ComfyUIClient` + `comfyui = ComfyUIClient(...)` instance
- Xoa `subprocess` import (khong con dung)
- Fix comment: "let ComfyUI handle it" → "let pipeline handle it"
- Backend version: 1.9.0 → 1.9.1

### Files thay doi
- `backend/app.py` — Bug fixes (2), dead code cleanup (env vars, imports, ~145 dong methods, ComfyUI client)
- `VERSION.txt` — 2.3.0 → 2.3.1
- `CHANGELOG.md` — Them entry v2.3.1

## v2.3.0 (2026-03-13) — Frontend + backend: remove ComfyUI references

### Thay doi
- **Backend health endpoint** — `/api/comfyui/health` gio tra ve SeedVR2 Pipeline status:
  - GPU info tu `torch.cuda` (thay vi ComfyUI system_stats)
  - Queue info tu `job_queue` (thay vi ComfyUI queue)
  - Them `pipeline_mode` field (OPTIMAL/UNIVERSAL tu AdaptiveReloadStrategy)
  - Xoa endpoint `/api/comfyui/restart` va `/api/comfyui/clear-queue` (khong can thiet)
- **Backend clear-history endpoint** — `/api/admin/comfyui/clear-history` gio clear VRAM cache:
  - Goi `seedvr2.reset()` + `torch.cuda.empty_cache()` + `gc.collect()`
  - Tra ve VRAM free sau cleanup
- **Admin dashboard** — Cap nhat UI:
  - "ComfyUI Server" → "SeedVR2 Pipeline"
  - "Server:" → "Mode:" (hien thi OPTIMAL/UNIVERSAL)
  - "Clear History" → "Clear VRAM"
  - Offline text: "Models not loaded" thay vi "Offline"
  - Auto-refresh status sau khi clear VRAM
- **Advanced.js** — Cap nhat comments (ComfyUI → SeedVR2 pipeline)
- Backend version: 1.8.0 → 1.9.0

### Files thay doi
- `backend/app.py` — Rewrite `/api/comfyui/health`, `/api/admin/comfyui/clear-history`, xoa restart/clear-queue endpoints
- `frontend/admin.html` — UI labels, JS functions (loadComfyUIStatus, clearVRAMCache)
- `frontend/advanced.js` — Comments update
- `VERSION.txt` — 2.2.0 → 2.3.0
- `CHANGELOG.md` — Them entry v2.3.0

## v2.2.0 (2026-03-13) — Adaptive reload strategy + resolution queue log

### Tinh nang moi
- **Adaptive Reload Strategy** — Tu dong phat hien pattern xen ke resolution (4K→6K→4K→9K) va chuyen sang UNIVERSAL mode de tranh reload ping-pong (30-60s moi lan reload):
  - `AdaptiveReloadStrategy` class trong `seedvr2_api.py`
  - 3 tier: LOW (2K/3K/4K), MID (6K), HIGH (9K/12K)
  - 2 modes: OPTIMAL (per-tier, nhanh nhat) va UNIVERSAL (blocks=32, safe cho tat ca, ~6s cham hon cho LOW)
  - OPTIMAL→UNIVERSAL: >=3 tier transitions trong 6 job gan nhat
  - UNIVERSAL→OPTIMAL: 4 job lien tiep cung tier
- **Resolution Queue Log** — Log rieng ghi lai thu tu cac job de phan tich pattern:
  - `ResolutionLogger` class trong `seedvr2_api.py`
  - File: `backend/logs/resolution_queue.log` (human/Claude readable)
  - Moi job: timestamp, preset, tier, mode, reload, timing
  - Summary block moi 10 job + final summary khi shutdown

### Files thay doi
- `backend/seedvr2_api.py` — Them `AdaptiveReloadStrategy`, `ResolutionLogger`, sua `upscale()` (them `job_id`, `user_id` params, strategy integration, logging)
- `backend/app.py` — Truyen `job_id`/`user_id` vao `seedvr2.upscale()`, them `atexit.register(seedvr2.shutdown)`, startup messages cho adaptive reload + log path
- `VERSION.txt` — 2.1.0 → 2.2.0
- `CHANGELOG.md` — Them entry v2.2.0
- `FEATURES.md` — Them F11 Adaptive Reload Strategy, F12 Resolution Queue Log

## v2.1.0 (2026-03-13) — Backend integration + pipeline extraction + bug fixes

### Tinh nang moi
- **Pipeline module extraction** — Tach pipeline code tu `app.py` ra `pipeline.py`:
  - `RESOLUTION_PRESETS`, `SeedVR2Pipeline`, tat ca helper functions
  - `run_upscale()` — headless upscale entry point (khong can PySide6)
  - Import duoc boi: GUI app, backend server, bat ky Python script
- **Backend SeedVR2 integration** — Thay the ComfyUI (HTTP+WebSocket) bang direct pipeline:
  - `backend/seedvr2_api.py` — `SeedVR2Client` wrapper class
  - Models load 1 lan, giu trong VRAM giua cac job
  - Thread-safe voi `threading.Lock()`
  - Auto-reload models khi doi preset (blocks_to_swap khac nhau)
  - Progress callback truc tiep tu pipeline → backend update_progress
  - Output: JPEG 95% + sRGB ICC profile
- **Simplified worker loop** — Xoa ComfyUI health checks, WebSocket polling, auto-start
  - Lazy model loading: chi load khi co job dau tien
  - GPU error recovery: full_reload() khi OOM, re-queue job tu dong

### Bug fixes (logic review — 2 rounds)
- **Fix VRAM leak on error/cancel** — `seedvr2.reset()` gio duoc goi trong `finally` block cua `_worker_loop`, dam bao VRAM duoc giai phong sau moi job (thanh cong, that bai, hoac cancel). Truoc do chi goi tren success path → VRAM leak khi job fail/cancel
- **Fix OOM infinite loop** — Them `oom_retries` dict theo doi so lan retry moi job. Gioi han 2 lan OOM retry. Truoc do, OOM → reload → OOM → reload lien tuc khong bao gio dung
- **Fix thread safety** — Move model reload (needs_reload + load_models) vao trong `with self._lock` trong `seedvr2_api.py`. Truoc do reload xay ra ngoai lock → race condition tiem an
- **Fix status API** — Thay `comfyui.is_healthy` bang `seedvr2.is_ready()`. Them `system_healthy` field, giu `comfyui_healthy` cho legacy compat. Truoc do luon tra ve False vi ComfyUI khong chay
- **Fix cancel/stop mismatch** — Worker gio check `stopped` flag: neu stop → chuyen ve 'pending', neu cancel → chuyen ve 'cancelled'. Truoc do ca 2 deu set 'cancelled' du frontend expect 'pending' cho stop
- **Fix 5k refix crash** — Xoa '5k' khoi danh sach refix validation (khong co preset '5K' trong RESOLUTION_PRESETS). Truoc do, refix voi resolution='5k' luon crash voi ValueError
- **Fix stuck processing jobs** — `restore_pending_jobs()` gio reset stuck 'processing' jobs ve 'queued' trong DB. Truoc do, processing jobs tu crash cu khong bao gio duoc pick up lai vi worker chi doc status='queued'
- **Fix interrupt endpoint** — Xoa `comfyui.interrupt()` call (ComfyUI khong chay). Cancel mechanism qua `cancelled` flag van hoat dong
- **Clean up dead code** — Xoa `prompt_id` references, update startup messages, fix stale comments

### Refactoring
- `app.py` — Import tu `pipeline.py`, `UpscaleWorker.run()` goi `run_upscale()`
- `backend/app.py` — `_process_job()` goi `seedvr2.upscale()` truc tiep
- `backend/app.py` — `_worker_loop()` simplified, OOM retry tracking, VRAM cleanup in finally
- `backend/app.py` — Backend version 1.6.0 → 1.7.0

### Files thay doi
- `pipeline.py` — **NEW** — Pipeline module (extracted tu app.py)
- `app.py` — Import tu pipeline.py, UpscaleWorker simplified
- `backend/seedvr2_api.py` — **NEW** — SeedVR2Client wrapper (model reload inside lock)
- `backend/app.py` — SeedVR2 integration, bug fixes (VRAM leak, OOM loop, status API, dead code)
- `VERSION.txt` — 1.10.0 → 2.1.0
- `CHANGELOG.md` — Them entry v2.1.0
- `FEATURES.md` — Them F10 Backend API Integration

## v1.10.0 (2026-03-13) — Match ComfyUI tiling + megapixel output + VRAM fixes

### Tinh nang moi
- **Megapixel-based output sizing** — 6K/9K/12K presets dung target megapixel (24/54/96 Mpx) thay vi fixed longest edge. Dam bao output dung so megapixel bat ke aspect ratio cua anh input. Function `_resize_to_megapixels()` tinh scale = sqrt(target_pixels / current_pixels)
- **TTP-matching tiling** — Rewrite toan bo `_tile_and_upscale()` de match chinh xac ComfyUI TTP nodes:
  - `_ttp_tile_size()`: Match TTP_Tile_image_size — tile_size = int(img / (1 + (factor-1) * (1-overlap))), align down to 8
  - `_ttp_tile_step()`: Match TTP_Image_Tile_Batch — step-based cutting voi overlap deu
  - `_ttp_blend_tiles()`: Match TTP_Image_Assy — linear gradient 64px o giua overlap (khong blend toan bo overlap nhu truoc)
  - `_ttp_create_gradient_mask()`: PIL gradient mask 255→0 cho Image.composite()

### Bug fixes
- **Fix 6K OOM** — Tang blocks_to_swap tu 16 len 20 cho 6K. Spargeattn/SageAttention can nhieu VRAM hon flash_attn (ComfyUI dung 16 voi flash_attn)
- **Fix 9K tile 2 VRAM thrashing** — Xoa `_warm_tile` optimization — skip empty_cache() giua phases khien CUDA allocator cache ~3.5GB khong duoc giai phong → block swap mat 22s/block. Gio luon empty_cache() giua moi phase
- **Fix max_resolution** — Doi tu "auto" sang fixed values (6000/9000/12000) matching ComfyUI workflow
- **Revert streamed offload** — "streamed" strategy (dedicated _offload_stream) gay OOM vi VRAM tich tu giua sync points. Revert ve simple async/sync 2-way strategy

### Files thay doi
- `app.py` — RESOLUTION_PRESETS (target_mpx, max_resolution fixed), `_resize_to_megapixels()`, rewrite `_tile_and_upscale()` + TTP helper functions, xoa `_warm_tile` parameter
- `src/optimization/blockswap.py` — Revert streamed offload, simplify to async/sync 2-way
- `VERSION.txt` — 1.9.1 → 1.10.0
- `CHANGELOG.md` — Them entry v1.10.0
- `FEATURES.md` — Update F5 tiling section

## v1.9.1 (2026-03-13) — Fix 6K VRAM thrashing (adaptive offload)

### Bug fixes
- **Fix 6K block swap thrashing** — Non-blocking GPU→CPU offload (v1.9.0) defers VRAM release. Voi blocks_to_swap≤12 (4K), du headroom → fast. Voi blocks_to_swap>12 (6K, 16 blocks), VRAM tich tu → CUDA allocator phai doi pending DMA hoan thanh truoc khi free memory → moi block swap mat 17-36 GIAY (thay vi ~250ms). Fix: adaptive offload — async cho ≤12 blocks, sync cho >12 blocks. Flag `_use_async_offload` set 1 lan luc configure, khong co per-block overhead
- **IO forward wrapper** — Ap dung cung adaptive offload cho I/O components (embeddings, norms)

### Phan tich ky thuat
**6K truoc (v1.9.0) — cold start:**
```
Block 0: 17,496ms  Block 1: 5,425ms  Block 2: 11,638ms
Block 3: 16,115ms  Block 4: 36,482ms  Block 5: 23,642ms
→ VRAM accumulation tu non_blocking offload (10.90GB base + pending DMA)
```

**6K sau (v1.9.1) — cold start (du kien):**
```
Block N: ~250-500ms (sync offload, GPU memory freed immediately)
→ Khong con VRAM thrashing
```

**4K khong anh huong:** Van dung async offload (blocks_to_swap=9 ≤ 12)

### Files thay doi
- `src/optimization/blockswap.py` — Them `_use_async_offload` flag tren model luc `configure_block_swap()`. `_wrap_block_forward()` STEP 4: check flag, async neu ≤12, sync neu >12. `_wrap_io_forward()`: tuong tu
- `VERSION.txt` — 1.9.0 → 1.9.1
- `CHANGELOG.md` — Them entry v1.9.1

## v1.9.0 (2026-03-13) — Lazy pin + skip-offload warm run optimization

### Tinh nang moi
- **Lazy pin** — Thay vi sync offload + re-pin sau moi block compute (~170ms/block), chuyen sang non_blocking offload (~0ms) + lazy pin truoc khi prefetch (~20ms). Tiet kiem ~150ms × 9 blocks = ~1.35s/inference. Them `_ensure_pinned()` helper: quick-check first param, chi pin khi can
- **Skip-offload warm run** — Giu DiT blocks tai vi tri BlockSwap giua cac warm run (swapped blocks tren CPU, non-swapped tren GPU) thay vi offload TOAN BO ve CPU roi restore lai. Tiet kiem ~3s/warm run (1.54s offload + 1.46s restore). VRAM an toan: 6.24GB DiT + 4.89GB VAE = ~11GB < 24GB
- **Block 0 lazy pin** — Block dau tien (khong co prefetch) duoc lazy pin truoc sync load, giup DMA nhanh hon qua page-locked memory path

### Phan tich ky thuat
**Truoc (v1.8.1) — warm run:**
```
Phase 2: 11.60s
  DiT inference: 8.56s (9 blocks × ~268ms avg, bao gom ~170ms sync offload + pin)
  Model restore: 1.46s (27 blocks CPU→GPU)
  Model offload: 1.54s (27 blocks GPU→CPU)
```

**Sau (v1.9.0) — warm run (du kien):**
```
Phase 2: ~7.3s
  DiT inference: ~7.0s (9 blocks × ~100ms avg, lazy pin ~20ms + non_blocking offload ~0ms)
  Model restore: 0s (blocks already in place!)
  Model offload: 0s (skip offload!)
```

### Files thay doi
- `src/optimization/blockswap.py` — Them `_ensure_pinned()` helper. `_wrap_block_forward()`: STEP 1 lazy pin truoc sync load, STEP 2 lazy pin truoc prefetch, STEP 4 non_blocking offload. `_wrap_io_forward()`: lazy pin + non_blocking offload
- `src/optimization/memory_manager.py` — `cleanup_dit()`: khi BlockSwap active + cache_model, skip full offload, giu blocks tai vi tri. `_handle_blockswap_model_movement()`: detect blocks already in place, skip restore, chi reactivate
- `VERSION.txt` — 1.8.1 → 1.9.0
- `CHANGELOG.md` — Them entry v1.9.0

## v1.8.1 (2026-03-13) — Fix warm run prefetch (re-pin after offload)

### Bug fixes
- **Fix warm run async prefetch** — `non_blocking=True` GPU→CPU offload tao NEW unpinned CPU tensors, khien warm run ke tiep prefetch bi fallback ve synchronous `cudaMemcpy`. Fix: doi sang synchronous offload + re-pin CPU memory sau moi block offload. Cost ~170ms/block (150ms sync + 20ms pin), nhung dam bao warm runs dat ~225ms/prefetched block thay vi ~350ms
- **I/O components re-pin** — Ap dung fix tuong tu cho I/O component offload (embeddings, norms)

### Files thay doi
- `src/optimization/blockswap.py` — `_wrap_block_forward()`: doi `non_blocking=True` sang sync offload + `_pin_module_memory()`. `_wrap_io_forward()`: tuong tu
- `VERSION.txt` — 1.8.0 → 1.8.1
- `CHANGELOG.md` — Them entry v1.8.1

## v1.8.0 (2026-03-13) — BlockSwap async prefetch pipeline optimization

### Tinh nang moi
- **BlockSwap async prefetch** — Khi GPU dang compute block N, block N+1 duoc load tu CPU→GPU tren CUDA prefetch stream rieng (song song). GPU→CPU offload dung non_blocking=True. Giam thoi gian BlockSwap tu ~3.15s xuong ~1.05s (tiet kiem ~2s) cho 9 blocks
- **Pinned CPU memory** — Block offloaded tren CPU dung pinned (page-locked) memory, cho phep non_blocking=True CPU→GPU thuc su async (bypass cudaMemcpy dong bo)
- **Non-blocking I/O offload** — I/O components (embeddings, norms) offload voi non_blocking=True
- **Bo per-block clear_memory** — Loai bo kiem tra memory pressure sau moi block swap (~5ms x 9 blocks overhead). Between-phase empty_cache() da du

### Toi uu nho (app.py)
- Bo `image.copy()` khong can thiet trong UpscaleWorker (pipeline tao tensor tu numpy, khong modify goc)
- Dung `.float()` thay vi `.to(torch.float32)` cho result conversion

### Phan tich ky thuat
**Truoc (v1.7.1):**
```
Block N: [BLOCK Load 150ms] [Compute 50ms] [BLOCK Offload 150ms] [Check 5ms]
x 9 blocks = ~3150ms overhead
```

**Sau (v1.8.0):**
```
Block 0: [Load sync 150ms] [Prefetch B1 async] [Compute 50ms] [Offload async]
Block 1: [Wait prefetch ~0ms] [Prefetch B2 async] [Compute 50ms] [Offload async]
...
= 150ms + 9 x ~100ms = ~1050ms overhead
```

### Files thay doi
- `src/optimization/blockswap.py` — Them `_pin_module_memory()` helper. `_configure_blocks()` pin CPU blocks. `apply_block_swap_to_dit()` tao CUDA prefetch stream. Rewrite `_wrap_block_forward()` voi async prefetch + non-blocking offload + bo per-block clear_memory. `_wrap_io_forward()` non-blocking offload. `cleanup_blockswap()` cleanup prefetch stream
- `app.py` — Bo `image.copy()` trong UpscaleWorker, dung `.float()` cho result
- `VERSION.txt` — 1.7.1 → 1.8.0
- `CHANGELOG.md` — Them entry v1.8.0

## v1.7.1 (2026-03-13) — Revert SpargeAttn group-by-size (performance regression fix)

### Bug fixes
- **Revert SpargeAttn group-by-size** — Fix group-by-size trong v1.7.0 lam DiT inference cham 2-8x (cold: 71.8s vs 12.6s, warm: 16-22s vs ~9s). Nguyen nhan: Python-level gather/scatter loop (420+ windows x 36 layers x 3 tensors = ~15,000 tensor slices/step) tao overhead lon hon speedup cua SpargeAttn. Revert ve simple SageAttention 2 fallback cho variable-length sequences
- **Them fallback logging** — Khi SpargeAttn fall back sang SageAttn2 do variable-length windows, in 1 log message (chi 1 lan) de user biet: `[ATTN] SpargeAttn -> SageAttn2 fallback: N different window sizes detected`

### Ket luan ky thuat
- SpargeAttn **khong tuong thich** voi SeedVR2's 720p windowing — moi resolution (2K-6K) deu tao 4 window sizes khac nhau (interior + 3 loai edge). SpargeAttn yeu cau uniform sequence lengths de reshape (B,H,N,D)
- SageAttention 2 la lua chon tot nhat cho SeedVR2 vi ho tro native varlen format, dat ~9s DiT inference (warm)

### Files thay doi
- `src/optimization/compatibility.py` — Xoa group-by-size path (lines 699-765), thay bang simple SageAttn2 fallback. Them 1-time warning log khi fallback xay ra
- `VERSION.txt` — 1.7.0 → 1.7.1
- `CHANGELOG.md` — Them entry v1.7.1

## v1.7.0 (2026-03-13) — Terminal panel moved to right side + fullscreen on startup

### Tinh nang moi
- **Terminal panel ben phai** — Chuyen terminal log tu bottom len right side. Layout chuyen tu QVBoxLayout sang QHBoxLayout voi left_widget (main content, stretch=1) va terminal_widget (fixedWidth=340px, full height). Toggle button thu nho panel xuong 40px khi an, mo rong 340px khi hien
- **Fullscreen on startup** — App mo len luon maximize (showMaximized() thay vi show())

### Files thay doi
- `app.py` — Restructure layout: QVBoxLayout→QHBoxLayout, tao left_widget/left_layout cho main content, wrap terminal trong terminal_widget voi fixedWidth(340). Bo setFixedHeight(180) cua log_console. Cap nhat _toggle_terminal() resize panel 340↔40px. Doi window.show()→showMaximized()
- `VERSION.txt` — 1.6.1 → 1.7.0
- `CHANGELOG.md` — Them entry v1.7.0
- `FEATURES.md` — Cap nhat F6 (terminal position + layout)

## v1.6.1 (2026-03-13) — VRAM cache flush between phases (performance fix)

### Bug fixes
- **BlockSwap 3.5x cham hon ComfyUI** — CUDA allocator giu ~11GB reserved VRAM tu model loading phase. Khi Phase 2 (DiT) bat dau, BlockSwap phai lam viec trong VRAM bi fragmented → config 7.37s (vs ComfyUI 2.15s), moi block swap 638ms (vs 488ms). Fix: them `torch.cuda.empty_cache()` tai 4 diem:
  1. Cuoi `load_models()` — giai phong VRAM tu model preparation
  2. Giua Phase 1 (encode) va Phase 2 (DiT) — giai phong VRAM tu VAE tiles
  3. Giua Phase 2 (DiT) va Phase 3 (decode) — giai phong VRAM tu DiT inference
  4. Giua Phase 3 (decode) va Phase 4 (post-process) — giai phong VRAM cho CPU↔GPU transfers
- Du kien: DiT materialization 7.37s→~2s, DiT inference per-swap 638ms→~490ms, post-process 1.39s→~0.3s. Tong ~32.6s→~25s (khop ComfyUI)

### Phan tich chi tiet (App 32.6s vs ComfyUI 25.0s)
| Phase | App (truoc) | ComfyUI | Gap | Nguyen nhan |
|-------|-------------|---------|-----|-------------|
| DiT materialization | 7.37s | 2.15s | +5.22s | 11GB VRAM reserved → BlockSwap config 3.5x cham |
| DiT inference | 11.72s | 9.92s | +1.80s | VRAM fragmentation → moi swap cham hon |
| Post-process | 1.39s | 0.23s | +1.16s | Nhieu RAM hon → CPU↔GPU transfer cham |
| VAE decode | 6.67s | 5.64s | +1.03s | Model caching overhead |

### Files thay doi
- `app.py` — Them `torch.cuda.empty_cache()` tai 4 vi tri: cuoi load_models(), giua phase 1↔2, 2↔3, 3↔4
- `VERSION.txt` — 1.6.0 → 1.6.1
- `CHANGELOG.md` — Them entry v1.6.1
- `FEATURES.md` — Cap nhat F4 invariant ve empty_cache

## v1.6.0 (2026-03-13) — Session persistence + auto-load models on startup

### Tinh nang moi
- **Session persistence** — Luu tat ca settings (DiT model, VAE model, preset, attention mode, seed, color correction, terminal visibility, advanced expanded) vao `config.json` khi dong app. Khi mo lai → tu dong restore tat ca settings
- **Auto-load models on startup** — Sau khi restore settings, app tu dong goi `_on_load_models()` qua `QTimer.singleShot(500ms)` — user chi can chon anh va bam Upscale, khong can bam Load Models thu cong
- **Config fallback** — Neu `config.json` khong ton tai (lan dau chay) hoac bi loi → dung defaults binh thuong. Neu model trong config khong con trong dropdown → giu default

### Files thay doi
- `app.py` — Them `import json`. Them `_save_config()` va `_load_config()` methods. Sua `closeEvent()` goi `_save_config()` truoc khi dong. Sua cuoi `__init__` de restore config + auto-load models qua QTimer
- `VERSION.txt` — 1.5.2 → 1.6.0
- `CHANGELOG.md` — Them entry v1.6.0
- `FEATURES.md` — Them F9 (Session Persistence + Auto-Load)

## v1.5.2 (2026-03-13) — Disable cuDNN benchmark (critical VAE performance fix)

### Bug fixes
- **cuDNN benchmark gay 22GB VRAM spike + 35s warmup** — `cudnn.benchmark=True` voi 1024x1024 VAE tiles khien cuDNN profiling allocate workspace 22GB va mat 18s cho tiles 1-5 (warmup shape moi) + 16s cho tiles 16-20 (edge tiles shape khac). Tong 74s/90s la cuDNN overhead! Fix: `cudnn.benchmark=False` — heuristic algorithm selection chi cham ~5%/tile nhung loai bo hoan toan VRAM spike va warmup. Du kien: encode 37s→4s, decode 37s→7s

### Files thay doi
- `app.py` — Doi `cudnn.benchmark = True` thanh `False` voi comment giai thich
- `VERSION.txt` — 1.5.1 → 1.5.2
- `CHANGELOG.md` — Them entry v1.5.2

## v1.5.1 (2026-03-13) — Fix 4K preset + VRAM offload (critical performance fix)

### Bug fixes
- **4K preset sai workflow** — App dung `comfyui_workflow_4k.json` (pre-resize 4096, resolution=auto, max_resolution=6000) thay vi `workflow-refix-4k.json` (direct, resolution=4000, max_resolution=4000). Fix: 4K preset doi thanh no pre-resize, resolution=4000, max_resolution=4000, blocks_to_swap=9, vae_tile=1024, overlap=64, latent_noise=0. Output dung 12 Mpx thay vi 13.4 Mpx
- **Thieu VAE offload** — `setup_generation_context()` khong truyen `vae_offload_device="cpu"`. VAE khong offload sau encode → tensor tich tu tren GPU → 22GB VRAM peak (vs ComfyUI 3GB)
- **Thieu tensor offload** — `setup_generation_context()` khong truyen `tensor_offload_device="cpu"`. Encoded latents + upscaled latents nam tren GPU → CUDA fragmentation → VAE encode cham 12.7x, VAE decode cham 6.1x so voi ComfyUI

### Verified
- Da doc va so sanh tat ca 7 workflow files (workflow-refix-2k/3k/4k.json + comfyui_workflow_4k/6k/9k/12k.json)
- Tat ca 6 presets (2K/3K/4K/6K/9K/12K) da khop chinh xac voi workflow tuong ung
- 9K/12K tiling logic khop ComfyUI TTP_Image_Tile_Batch + TTP_Image_Assy
- Model device migration (manage_model_device) hoat dong dung voi cache_model=True
- Phase functions tu dong move model from CPU offload back to GPU khi can

### Files thay doi
- `app.py` — Fix 4K preset (pre_resize=None, resolution=4000, max_resolution=4000, blocks=9, vae_tile=1024, overlap=64, noise=0). Them vae_offload_device="cpu" va tensor_offload_device="cpu" vao setup_generation_context(). Cap nhat preset comments va tooltips cho khop 4K moi
- `FEATURES.md` — Cap nhat processing flow cho 4K (no pre-resize). Them invariants ve offload devices va 4K preset
- `VERSION.txt` — 1.5.0 → 1.5.1
- `CHANGELOG.md` — Them entry v1.5.1

## v1.5.0 (2026-03-12) — Embedded terminal log panel

### Tinh nang moi
- **Embedded terminal** — QPlainTextEdit hien thi realtime stdout/stderr cua SeedVR2 pipeline trong app. Xem duoc toan bo debug output (phase progress, tile encoding, memory stats, model loading...) ma khong can mo terminal rieng
- **Toggle terminal** — Nut "Show Terminal" / "Hide Terminal" de bat tat log panel. Mac dinh bat
- **LogStream** — Custom stream redirector giu stdout/stderr goc (terminal), dong thoi emit signal toi GUI (thread-safe qua Qt signal)
- **LogBridge** — QWidget bridge de thread-safe cross-thread log writes
- **Auto-scroll** — Terminal tu dong cuon xuong dong moi nhat
- **Buffer limit** — Gioi han 5000 dong de tranh memory leak

### Files thay doi
- `app.py` — Them LogStream, LogBridge classes. Them terminal QPlainTextEdit + toggle button. Them _toggle_terminal(), _append_log(), closeEvent(). Import io, QPlainTextEdit
- `VERSION.txt` — 1.4.0 → 1.5.0
- `CHANGELOG.md` — Them entry v1.5.0

## v1.4.0 (2026-03-13) — System monitor + elapsed timer

### Tinh nang moi
- **System Monitor (Crystools-style)** — Hien thi realtime CPU%, RAM%, GPU%, VRAM%, GPU Temp tren status bar. Cap nhat moi 2 giay. Mau sac temperature thay doi theo nhiet do (xanh < 60, vang 60-80, do > 80)
- **Elapsed timer** — Bo dem giay realtime trong progress bar khi upscale. Hien thi `[Xs]` lien tuc cap nhat moi giay. Khi xong hien thi tong thoi gian (vd: "Done! 3072x4096 in 2m 35s")

### Bug fixes
- Fix `total_mem` → `total_memory` (AttributeError khi hien thi VRAM trong status bar)

### Files thay doi
- `app.py` — Them SystemMonitor widget, _query_nvidia_smi(), elapsed timer (QTimer 1s), import subprocess/psutil/QTimer
- `VERSION.txt` — 1.3.0 → 1.4.0
- `CHANGELOG.md` — Them entry v1.4.0

## v1.3.0 (2026-03-12) — Model selection overhaul + registry expansion

### Tính năng mới
- **DiT Model dropdown luon hien thi** — Di chuyen tu Advanced Settings len khu vuc chinh, luon nhin thay
- **VAE Model dropdown** — Them dropdown chon VAE model (truoc do hardcode ema_vae_fp16.safetensors)
- **Danh sach model chi 7B** — Loc bo tat ca 3B va Q3 model khoi dropdown (theo yeu cau user)
- **Sap xep model thong minh** — Q8_0 > Q6_K > Q5_K_M > Q4_K_M > fp8 > fp16, standard truoc sharp
- **Default model: seedvr2_ema_7b-Q8_0.gguf** — Khop tat ca 7 workflow

### Bug fixes
- **BlockSwap crash** — `dit_offload_device` khong duoc truyen vao `setup_generation_context()`, gay loi "BlockSwap requires offload_device to be set and different from device". Fix: truyen `dit_offload_device="cpu"` khi `blocks_to_swap > 0`
- **Download progress khong hien thi tren GUI** — `download_with_resume()` chi show tqdm terminal. Fix: them `progress_fn` callback → progress bar GUI cap nhat real-time (file size + %)

### Model Registry mo rong
- Them 7B GGUF tu `cmeka/SeedVR2-GGUF`: Q8_0, Q6_K, Q5_K_M (standard + sharp = 6 model moi)
- Them 7B sharp Q8_0 tu `cmeka/SeedVR2-GGUF`
- Them fp8_e4m3fn safetensors tu `numz/SeedVR2_comfyUI` (standard + sharp)
- Sua repo Q4_K_M 7B tu AInVFX → cmeka (day du quant hon)
- Tong cong: 7B models tang tu 6 → 16

### Files thay doi
- `app.py` — Restructure UI (model row rieng), _populate_models() filter 7B-only, _populate_vae_models() moi, SeedVR2Pipeline nhan vae_model parameter
- `src/utils/model_registry.py` — Them 10 model 7B moi vao MODEL_REGISTRY (Q8_0, Q6_K, Q5_K_M, fp8 cho ca standard va sharp)
- `VERSION.txt` — 1.2.1 → 1.3.0
- `CHANGELOG.md` — Them entry v1.3.0
- `FEATURES.md` — Cap nhat F4 (model selection UI)

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
