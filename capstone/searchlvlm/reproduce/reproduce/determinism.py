import os, random
import numpy as np

# cuBLAS determinism
# ':4096:8' is safe on modern GPUs; if you hit "CUBLAS_WORKSPACE_CONFIG not set" warnings, ensure this runs before torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def set_seed(seed: int = 42, *, use_cuda=True, strict=True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # cuDNN settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Global deterministic algorithms (may raise on unsupported ops)
        if strict:
            # If you prefer warnings instead of errors, use warn_only=True in 2.2+:
            # torch.use_deterministic_algorithms(True, warn_only=True)
            torch.use_deterministic_algorithms(True)

    except Exception:
        # If torch is missing or something odd happens, we still return the seed.
        pass

    return seed
