from __future__ import annotations
import torch

def flatten_modules_for_linear_weights(root: torch.nn.Module):
    module_map = {}
    for name, mod in root.named_modules():
        if hasattr(mod, "weight"):
            module_map[f"{name}.weight" if name else "weight"] = mod
    return module_map
