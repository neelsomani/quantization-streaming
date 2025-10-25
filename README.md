End to end demo, TorchTitan for training, TorchForge for streaming, vLLM for inference, full precision weights only.

Flow
1. TorchTitan fine tunes Qwen2.5 0.5B for a few steps and emits versioned weight shards.
2. TorchForge transports those shards over RDMA or TCP, same API.
3. A small vLLM wrapper receives shards, installs into a staging bank, then atomically swaps when a version is complete.

Notes
- This demo runs with tensor parallel size 1 to keep it simple. The provider and bank code already carries shard metadata and versioning. Extending to TP>1 means sharding rows by out_features and starting one receiver per TP rank.
- No quantization here. We install full precision tensors directly.

Quick start

First, install Torchforge using the script in their repo: git@github.com:meta-pytorch/torchforge.git

Terminal A, start the vLLM receiver
```bash
pip install -r requirements.txt
python forge_receiver_vllm.py \
  --model Qwen/Qwen2.5-0.5B \
  --tp 1 \
  --listen tcp://0.0.0.0:55001 \
  --max-model-len 1024 \
  --gpu-mem-util 0.70 \
  --dtype float16
```

Terminal B, run the TorchTitan trainer that streams checkpoints every few steps
```bash
python train_torchtitan_stream_qwen25_0p5b.py \
  --model Qwen/Qwen2.5-0.5B \
  --epochs 1 \
  --steps-per-export 50 \
  --send tcp://127.0.0.1:55001
```

In a third shell, hit the vLLM API to see live swaps reflected in logits
```bash
curl -s localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B","prompt":"Hello, my name is","max_tokens":20}' | jq .
```

What you should see
- The receiver prints arrivals and version completions, then logs a swap.
- The API stays responsive during swaps since installs go to the staging bank.
