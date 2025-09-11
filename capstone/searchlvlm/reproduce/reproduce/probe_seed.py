from determinism import set_seed
import random, numpy as np

set_seed(42)

print("py_random:", [random.random() for _ in range(3)])
print("np_random:", np.random.RandomState(42).rand(3).tolist())
try:
    import torch
    g = torch.Generator(device="cpu").manual_seed(42)
    print("torch_rand:", torch.rand(3, generator=g).tolist())
except Exception as e:
    print("torch_unavailable:", type(e).__name__)
