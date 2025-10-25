from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Protocol
import torch

@dataclass
class ShardMeta:
    global_shape: tuple[int, ...]
    shard_axis: int  # 0 for out_features in Linear
    start: int
    length: int
    layout_info: dict

@dataclass
class StreamRecord:
    fqn: str
    meta: ShardMeta
    payload: torch.Tensor  # device tensor, full precision here
    qmeta: dict  # unused in FP flow, reserved for parity with quant path
    version_id: int

class StreamingWeightProvider(Protocol):
    def iter_stream(self) -> Iterator[StreamRecord]: ...
    def manifest(self) -> Optional[dict]: ...
    def close(self) -> None: ...
