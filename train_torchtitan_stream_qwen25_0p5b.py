from __future__ import annotations
import argparse
import time
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# TorchTitan imports, adjust if your install provides a different top level
try:
    import TorchTitan as titan  # type: ignore
except ImportError as e:
    # Some envs use lowercase torchtitan
    try:
        import torchtitan as titan  # type: ignore
    except Exception as e2:
        raise RuntimeError("TorchTitan not found. Install or adjust imports.") from e2

from torchforge_bridge import TorchForgeSender


def shard_rows(O: int, tp: int, r: int):
    per = (O + tp - 1) // tp
    s = min(r * per, O)
    e = min(s + per, O)
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-export", type=int, default=50)
    ap.add_argument("--send", type=str, required=True, help="TorchForge endpoint, e.g., tcp://127.0.0.1:55001")
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Simple toy data
    text = [
        "Hello world", "Goodbye world", "The quick brown fox", "Transformers are fun",
    ]
    data = [tok(t, return_tensors="pt").to(device) for t in text]

    # TorchForge client
    client = TorchForgeSender(args.send)

    global_step = 0

    for epoch in range(args.epochs):
        for batch in data:
            optimizer.zero_grad()
            out = model(**batch, labels=batch["input_ids"]) 
            out.loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % args.steps_per_export == 0:
                export_step(client, model, global_step, tp=args.tp)
                print(f"exported version {global_step}")
                time.sleep(0.2)

    # Final export
    export_step(client, model, global_step + 1, tp=args.tp)
    print(f"exported version {global_step + 1}")


def export_step(client, model, step_id: int, tp: int = 1):
    sd: Dict[str, torch.Tensor] = model.state_dict()
    for fqn, W in sd.items():
        if not fqn.endswith("weight"):
            continue
        if W.ndim != 2:
            continue
        O, _ = W.shape
        for r in range(tp):
            s, e = shard_rows(O, tp, r)
            payload = W[s:e].contiguous()
            meta = {
                "global_shape": list(W.shape),
                "shard_axis": 0,
                "start": int(s),
                "length": int(e - s),
                "layout_info": {},
            }
            client.send(
                {
                    "fqn": fqn,
                    "meta": meta,
                    "tensor": payload,
                    "version_id": int(step_id),
                }
            )


if __name__ == "__main__":
    main()
