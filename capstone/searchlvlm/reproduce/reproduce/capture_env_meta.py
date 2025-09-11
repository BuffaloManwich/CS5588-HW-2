import json, os, sys, platform, subprocess

def safe_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    except Exception as e:
        return f"UNAVAILABLE: {e.__class__.__name__}"

meta = {
    "python": sys.version,
    "platform": platform.platform(),
    "git_commit": safe_cmd(["git","rev-parse","HEAD"]),
    "git_status": safe_cmd(["git","status","--porcelain"]),
}

try:
    import torch
    meta.update({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    })
except Exception as e:
    meta["torch_info"] = f"UNAVAILABLE: {e.__class__.__name__}"

os.makedirs("../logs", exist_ok=True)
out = "../logs/env.meta.json"
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Wrote {out}")
print(json.dumps(meta, indent=2))
