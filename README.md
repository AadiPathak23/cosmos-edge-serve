# cosmos-edge-serve

> ## 💵 COST WARNING — READ BEFORE RUNNING ANYTHING
>
> **Phase 1 (this repo as it stands) is free.** It runs entirely on your own machine.
> Nothing here calls AWS, and there are no paid APIs.
>
> **Phase 2 launches a GPU instance and will bill you.** A `g4dn.xlarge` spot instance is
> **~$0.32/hour**; on-demand is **$0.526/hour**. A forgotten instance costs about **$380/month**.
> An unattached Elastic IP bills hourly even with nothing running.
>
> If you have run Phase 2, **[`docs/TEARDOWN.md`](docs/TEARDOWN.md) is not optional.** Run it
> the same day and verify in the AWS console — not from memory.
>
> This project has a hard **$10 total budget**. Estimated real cost of a full Phase 2
> benchmark run: **~$1.05**.

An inference service for [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B),
a 2B-parameter reasoning vision-language model for physical AI. POST an image or a short
video plus a text prompt, get the model's chain-of-thought reasoning and answer back.

The headline deliverable is a benchmark: **throughput and p50/p95 latency under concurrency
on a T4 GPU.** Everything else here exists to produce that number honestly.

---

## Status

| Phase | What | State |
|---|---|---|
| 1 | Local FastAPI + Docker service | **in progress** |
| 2 | EC2 T4 deployment + k6 benchmark | not started (costs money) |
| 3 | Kubernetes (local k3s) | optional, not started |

Benchmark results will appear here once Phase 2 runs.

---

## Hardware reality check

Read this before you try to run it, because the obvious setup does not work.

**Your 6 GB laptop GPU cannot run this model at fp16.** The weights alone are ~4.9 GB
(2,438,696,960 params × 2 bytes). Add Windows' display reservation, the vision tower's
activations, and the KV cache and you are over 6 GB. The default config therefore uses
**4-bit NF4** (`COSMOS_QUANT=nf4`), which brings weights to roughly 2 GB.

**vLLM does not work for this model on a T4.** vLLM
[#29743](https://github.com/vllm-project/vllm/issues/29743) — "Turing support in Qwen3-VL
backends" — was closed as not planned. Every attention backend fails on SM75. This service
uses HuggingFace `transformers` with `attn_implementation="sdpa"`, which is also what
NVIDIA's own sample script does. Do not "upgrade" this to vLLM on a T4.

**T4 has no bfloat16 and no FP8.** It is Turing. Any FP8 Cosmos guidance you find online
targets Jetson Thor, Hopper, or Blackwell. Use `float16`.

| Target | VRAM | Setting |
|---|---|---|
| RTX 3060 Laptop | 6 GB | `COSMOS_QUANT=nf4` |
| Kaggle T4 / EC2 T4 | 16 GB | `COSMOS_QUANT=none` (fp16) |
| No GPU at all | — | `COSMOS_DEVICE=cpu` (works, 30–60 s/request) |

---

## Quickstart (local, free)

### Prerequisites

- **A HuggingFace token, and the model licence accepted.** `nvidia/Cosmos-Reason2-2B`
  is a **gated** repo — without this the download fails with a bare HTTP 401 that looks
  like a network fault. Two one-off steps:
  1. Signed in to HuggingFace, open
     [the model page](https://huggingface.co/nvidia/Cosmos-Reason2-2B) and accept the
     NVIDIA Open Model License. Access is automatic; there is no approval queue.
  2. Create a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
     and put it in `.env` as `HF_TOKEN=hf_...`.
- Docker Desktop with the **WSL2 backend**
- For GPU: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  inside WSL2, so `--gpus all` works. Verify with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
  ```
  If that prints your GPU, you're set. If it errors, fix this before going further —
  everything downstream depends on it.

### Run

```bash
cp .env.example .env          # defaults are already tuned for a 6 GB laptop
# then edit .env and set HF_TOKEN=hf_...   (required — the repo is gated)
docker compose up --build
```

First start downloads ~5 GB of weights into the `cosmos-models` Docker volume. This takes a
while and the container is intentionally **unhealthy** until the model is loaded and a warmup
pass has completed. Watch for the load report banner — it tells you exactly what got loaded.

```bash
curl http://localhost:8000/health
```

Returns `503` until warm, then `200` with the full load report as JSON.

### Send a request

```bash
python scripts/make_assets.py          # generates assets/sample.jpg and sample.mp4
python scripts/smoke_test.py           # image + video request against localhost:8000
```

Or by hand:

```bash
curl -X POST http://localhost:8000/infer \
  -F "file=@assets/sample.jpg" \
  -F "prompt=Describe what you see and whether it is safe to proceed."
```

### The startup banner

The service prints a load report at startup and refuses to start if anything is wrong.
There is **no mock fallback and no silent degradation** — a failed load is a non-zero exit:

```
================================================================================
COSMOS-EDGE-SERVE — model load report
--------------------------------------------------------------------------------
  model id           nvidia/Cosmos-Reason2-2B
  model class        Qwen3VLForConditionalGeneration
  device             cuda:0  (NVIDIA GeForce RTX 3060 Laptop GPU)
  compute capability sm_86   bf16 supported: True
  dtype              torch.float16
  quantization       bitsandbytes-nf4 (double quant, fp16 compute)
  attention          sdpa
  adapter            NONE  (COSMOS_ADAPTER_ENABLED=false)
  total params       2,438,696,960
  trainable params   0  (0.0000%)
  weights on device  2.02 GiB
  VRAM               2.61 GiB used / 6.00 GiB total
  vision tokens      256 .. 1024
  warmup             OK — 8 tokens in 3.41 s
================================================================================
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/infer` | POST | multipart: `file` (jpg/png/mp4) + `prompt`, optional `max_new_tokens`, `fps` |
| `/health` | GET | `503` until warm, then `200` + load report |
| `/metrics` | GET | Prometheus text exposition |
| `/docs` | GET | Interactive OpenAPI docs (FastAPI built-in) |

`/infer` response:

```json
{
  "answer": "It is not safe to turn right...",
  "reasoning": "The pedestrian is entering the crosswalk...",
  "truncated": false,
  "tokens": { "input": 1183, "text": 159, "visual": 1024, "output": 217 },
  "timing_ms": { "queue_wait": 2.1, "preprocess": 41.6, "generate": 7180.4, "total": 7228.9 },
  "tokens_per_second": 30.2,
  "request_id": "0f2c9a4e"
}
```

### Concurrency model

Requests land on a bounded `asyncio.Queue` in front of **one** GPU worker. HF `generate()`
monopolises the GPU, so running several at once would thrash VRAM without buying real
parallelism. A single worker makes the benchmark legible: latency decomposes cleanly into
queue wait plus compute. Past `COSMOS_MAX_QUEUE_DEPTH` you get `503` with `Retry-After`
rather than an unbounded backlog.

---

## Configuration

Every setting is an environment variable — see [`.env.example`](.env.example), which
documents all of them inline. The ones that matter most:

| Variable | Default | Why you'd change it |
|---|---|---|
| `COSMOS_QUANT` | `nf4` | `none` for fp16 on a T4 |
| `COSMOS_DEVICE` | `auto` | `cpu` if you have no GPU |
| `COSMOS_MAX_NEW_TOKENS` | `256` | `1024`+ for full reasoning traces |
| `COSMOS_MAX_VISION_TOKENS` | `1024` | `4096` on a T4 for finer visual detail |
| `COSMOS_ADAPTER_ENABLED` | `false` | `true` to serve a featherweight-ai LoRA adapter |

---

## Development

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -r requirements-dev.txt
pytest                                            # runs without a GPU
ruff check .
```

The test suite stubs the model, so it runs on any machine in seconds. It covers request
validation, the 503-before-warm contract, queue overflow, `<think>` parsing, and the
`/metrics` shape — but it deliberately does **not** verify model output. That is what
`scripts/smoke_test.py` is for, and it needs real weights.

---

## Project docs

- [`docs/CLAUDE.md`](docs/CLAUDE.md) — architecture, every key decision and the reasoning
  behind it, plus an append-only changelog
- [`docs/PLAN.md`](docs/PLAN.md) — phased task breakdown and current status
- `docs/TEARDOWN.md` — written in Phase 2; the checklist that stops AWS billing

## Related

- [`featherweight-ai`](https://github.com/) — benchmarks PEFT methods (LoRA/QLoRA/DoRA) on
  this same model and produces the adapter this service can load. The two repos share no
  code; they are linked only by that artifact.

## License

The service code is MIT. The model itself is covered by the
[NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Reason2-2B), which permits
commercial use and derivative models. You are responsible for complying with it.
