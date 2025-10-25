from __future__ import annotations
import threading
import queue
from typing import Iterator, Optional
import torch

from .provider_base import ShardMeta, StreamRecord, StreamingWeightProvider

try:
    import torchforge  # type: ignore
except ImportError as e:
    raise RuntimeError("TorchForge is required. pip install TorchForge or set PYTHONPATH") from e


class TorchForgeSender:
    """Sender class for sending tensor data to a TorchForge endpoint."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._client = torchforge.Client(endpoint)
    
    def send(self, data: dict):
        """Send tensor data to the TorchForge endpoint.
        
        Args:
            data: Dictionary containing 'fqn', 'meta', 'tensor', and 'version_id' keys
        """
        self._client.send(data)
    
    def close(self):
        """Close the client connection."""
        try:
            self._client.close()
        except Exception:
            pass

class TorchForgeProvider(StreamingWeightProvider):
    def __init__(self, endpoint: str, device: torch.device | None = None):
        self.endpoint = endpoint
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._q: "queue.Queue[StreamRecord]" = queue.Queue(maxsize=1024)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _run(self):
        def on_tensor(msg):
            # Expect a dict with fields: fqn, meta, tensor, version_id
            fqn = msg["fqn"]
            m = msg["meta"]
            meta = ShardMeta(
                global_shape=tuple(m["global_shape"]),
                shard_axis=int(m.get("shard_axis", 0)),
                start=int(m["start"]),
                length=int(m["length"]),
                layout_info=dict(m.get("layout_info", {})),
            )
            t: torch.Tensor = msg["tensor"].to(self.device, non_blocking=True)
            ver = int(msg["version_id"]) if "version_id" in msg else 0
            rec = StreamRecord(fqn=fqn, meta=meta, payload=t, qmeta={}, version_id=ver)
            self._q.put(rec)

        self._client = torchforge.Client(self.endpoint)
        self._client.subscribe(on_tensor)
        self._client.run_until(self._stop)

    def iter_stream(self) -> Iterator[StreamRecord]:
        while not self._stop.is_set():
            try:
                yield self._q.get(timeout=0.5)
            except queue.Empty:
                continue

    def manifest(self) -> Optional[dict]:
        return None

    def close(self) -> None:
        self._stop.set()
        try:
            self._client.close()
        except Exception:
            pass
