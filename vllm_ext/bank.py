from __future__ import annotations
from typing import Dict, Any
import torch

class WeightBank:
    """Two bank install and swap for FP tensors.
    module_map: Dict[str, torch.nn.Module] maps FQN to module instance.
    """
    def __init__(self, module_map: Dict[str, torch.nn.Module]):
        self.module_map = module_map
        self.staging: Dict[str, Dict[str, Any]] = {}

    def install_fp(self, fqn: str, payload: torch.Tensor, qmeta: dict) -> None:
        self.staging[fqn] = {"tensor": payload.detach()}

    def make_live(self) -> None:
        for fqn, pack in self.staging.items():
            mod = self.module_map[fqn]
            t = pack["tensor"]
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.nn.Parameter):
                with torch.no_grad():
                    # Align shapes if shard equals full tensor in TP=1
                    if mod.weight.shape == t.shape:
                        mod.weight.copy_(t)
                    else:
                        # Basic row slice install for sharded rows
                        s = 0
                        e = min(mod.weight.shape[0], t.shape[0])
                        mod.weight.data[s:e].copy_(t[: e - s])
            else:
                mod.register_buffer("weight", t, persistent=True)
        self.staging.clear()

    def clear(self) -> None:
        self.staging.clear()
