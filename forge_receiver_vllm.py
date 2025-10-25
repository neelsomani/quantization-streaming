from __future__ import annotations
import argparse
import threading
import time

import torch
from fastapi import FastAPI
import uvicorn

from vllm import LLM, SamplingParams

from torchforge_bridge.provider_torchforge import TorchForgeProvider
from vllm_ext.install_fp import flatten_modules_for_linear_weights
from vllm_ext.loader_stream import StreamLoader

app = FastAPI()

class Receiver:
    def __init__(self, model: str, tp: int, listen: str, max_model_len: int,
                 gpu_mem_util: float, dtype: str | None):
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            dtype=dtype or "auto",
        )
        # Build a module map, FQNs matching state dict names for weights
        root = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        module_map = flatten_modules_for_linear_weights(root)
        expected = sum(1 for k, v in root.state_dict().items() if k.endswith("weight") and v.ndim == 2)
        self.loader = StreamLoader(module_map, expected_tensors=expected, timeout_s=120.0)
        self.provider = TorchForgeProvider(listen)
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _run(self):
        print("receiver started, waiting for shards...")
        for completed_version in self.loader.consume(self.provider.iter_stream()):
            print(f"completed version {completed_version}, swapped live")

    def close(self):
        self.provider.close()

receiver: Receiver | None = None

@app.post("/v1/completions")
async def completions(payload: dict):
    prompt = payload.get("prompt") or payload.get("input")
    max_tokens = int(payload.get("max_tokens", 64))
    sp = SamplingParams(max_tokens=max_tokens)
    out = receiver.llm.generate(prompt, sp)
    text = out[0].outputs[0].text
    return {"id": f"cmpl-{int(time.time())}", "object": "text_completion", "model": receiver.llm.llm_engine.model_config.model, "choices": [{"text": text, "index": 0}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--listen", type=str, default="tcp://0.0.0.0:55001")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "auto", "float16", "bfloat16", "float32"])
    args = ap.parse_args()

    global receiver
    receiver = Receiver(
        model=args.model,
        tp=args.tp,
        listen=args.listen,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
        dtype=args.dtype,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
