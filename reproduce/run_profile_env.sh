# --- Quantization: 8-bit weights ---
export LLAVA_USE_8BIT=1
export LOAD_IN_8BIT=1
export BNB_8BIT=1

# --- Generation length (limits ONLY output; input context remains large) ---
# Raise later if you want longer answers; lowering reduces KV-cache.
export MAX_NEW_TOKENS=64

# --- Batch & image size: keep memory predictable ---
export LLAVA_BATCH_SIZE=1
export LLAVA_IMAGE_SIZE=336   # try 256 if you still see OOM

# --- CUDA allocator: reduce fragmentation ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# --- Safety: single GPU, no surprise data parallel ---
export CUDA_VISIBLE_DEVICES=0

echo "[run_profile_env] 8-bit quant ON | MAX_NEW_TOKENS=$MAX_NEW_TOKENS | BATCH=1 | IMG=$LLAVA_IMAGE_SIZE"
