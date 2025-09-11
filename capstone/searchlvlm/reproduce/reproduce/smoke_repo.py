import os, json, time, platform

# Ensure cuBLAS determinism within this process
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Determinism (seed + strict algos)
from determinism import set_seed
set_seed(42, strict=True)

# Collect quick env/provenance
info = {
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "platform": platform.platform(),
}

# Core deps sanity checks based on environment.yml pins
mods = {}
def check(modname, attr=None):
    try:
        m = __import__(modname, fromlist=[] if attr is None else [attr])
        v = getattr(m, "__version__", "unknown")
        if attr:
            v = getattr(m, attr).__version__
        mods[modname if attr is None else f"{modname}.{attr}"] = v
    except Exception as e:
        mods[modname if attr is None else f"{modname}.{attr}"] = f"ERROR: {e.__class__.__name__}: {e}"

# Heavy hitters
for m in [
    "torch", "transformers", "tokenizers", "tiktoken",
    "sklearn", "PIL", "newspaper", "requests", "numpy"
]:
    check(m)

# GPU + torch detail
try:
    import torch
    info.update({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
    })
except Exception as e:
    info["torch"] = f"ERROR: {e.__class__.__name__}: {e}"

# Tiny tokenization sanity (common in SearchLVLM-style repos)
tok_sample = {}
try:
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_sample["transformers_tokenizer"] = t.convert_ids_to_tokens(t("Hello world!")["input_ids"])[:5]
except Exception as e:
    tok_sample["transformers_tokenizer"] = f"ERROR: {type(e).__name__}: {e}"

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tok_sample["tiktoken"] = enc.encode("Hello world!")[:5]
except Exception as e:
    tok_sample["tiktoken"] = f"ERROR: {type(e).__name__}: {e}"

# Write result
out = {
    "info": info,
    "modules": mods,
    "tokenization_samples": tok_sample,
    "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
}

os.makedirs("../runs/baseline", exist_ok=True)
with open("../runs/baseline/smoke.result.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
