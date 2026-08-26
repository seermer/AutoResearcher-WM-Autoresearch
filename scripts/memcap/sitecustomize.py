"""Cap CUDA memory per process so a big-GPU box can honestly test a small-GPU recipe.

Put this directory on PYTHONPATH and set AR_CUDA_MEM_GB. The cap is applied right
after the rank binds its device, which is when a real 24 GB card's limit starts to
matter; exceeding it raises OOM exactly as it would on the smaller card.
"""
import os

_gb = os.environ.get("AR_CUDA_MEM_GB")
if _gb:
    import torch

    _orig_set_device = torch.cuda.set_device

    def _set_device(device, *a, **kw):
        _orig_set_device(device, *a, **kw)
        idx = device if isinstance(device, int) else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory / 2**30
        frac = min(1.0, float(_gb) / total)
        torch.cuda.set_per_process_memory_fraction(frac, idx)
        print(f"[memcap] rank device {idx}: capped at {float(_gb):.0f} GB "
              f"({frac:.3f} of {total:.0f} GB)", flush=True)

    torch.cuda.set_device = _set_device
