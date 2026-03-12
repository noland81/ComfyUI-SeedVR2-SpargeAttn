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
