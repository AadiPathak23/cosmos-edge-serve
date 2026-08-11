# docs/CLAUDE.md — project memory

Two sections. `STABLE` is the durable truth and changes rarely. `CHANGELOG` is append-only —
add entries at the bottom, never rewrite what's above.

---

# STABLE

## Purpose

**Repo:** https://github.com/AadiPathak23/cosmos-edge-serve (public)

`cosmos-edge-serve` is an inference service for `nvidia/Cosmos-Reason2-2B`, NVIDIA's 2B-parameter
reasoning vision-language model for physical AI. It accepts an image or short video plus a text
prompt over HTTP and returns the model's response.

The repo exists to learn **production serving** properly — Docker, FastAPI, cloud GPU deployment,
S3, load testing — not to advance the model itself. The single headline deliverable is a
benchmark table: **throughput and p50/p95 latency under concurrency on a T4 GPU.** Everything else
in the repo is scaffolding for producing that number honestly.

A sibling repo, `featherweight-ai`, benchmarks PEFT methods (LoRA/QLoRA/DoRA) on this same model
and produces a fine-tuned adapter. **This** repo serves that adapter. The two repos share no code
and no deployment — they are linked *only* by the adapter artifact. Adapter loading here is a
config flag, off by default.

The author's ML background is solid; infrastructure is the thing being learned. Explain infra
choices as you make them; don't assume prior knowledge of Docker, EC2, S3, or load testing.

## Architecture

Single process, single GPU, model loaded once at startup.

```
POST /infer  (multipart: file + prompt)
      |
      v
  validate mime/size  --(bad)--> 4xx
      |
      v
  asyncio.Queue (bounded)  --(full)--> 503 + Retry-After
      |                                 [record queue_wait]
      v
  ONE worker task
      |
      v
  run_in_executor(1-thread pool)   <-- keeps the event loop free so
      |                                /health still answers under load
      v
  processor.apply_chat_template(media-before-text, fps=4)
      |
      v
  model.generate()                  [record generate_time]
      |
      v
  token accounting + <think> split
      |
      v
  JSON response

GET /health   -> 503 until warmup completes, then 200 + the load report as JSON
GET /metrics  -> Prometheus text exposition
```

Why one worker and not N: HuggingFace `generate()` monopolises the GPU for the duration of a
request. Running several concurrently would thrash VRAM and interleave badly without producing
real parallelism. A bounded queue in front of a single worker makes concurrency behaviour
*legible* — latency decomposes cleanly into queue wait plus compute, which is exactly the story
the benchmark needs to tell.

## Key decisions and the reasoning behind them

### Serving backend: HuggingFace transformers, **not** vLLM

NVIDIA's own docs recommend vLLM for deployment. **That recommendation does not apply to a T4.**

`vllm-project/vllm` issue **#29743** ("Turing support in Qwen3-VL backends") was **closed as not
planned**. On SM75 (T4, T4G) every attention backend fails for this architecture:

- FlashAttention requires SM80+ (Ampere or newer)
- XFormers doesn't work with the FlashAttention2 path as expected
- FlashInfer and Triton attention lack the required `context_layer` calculation logic
- Torch SDPA is no longer registered in vLLM's backend registry

The model explicitly raises `Qwen3-VL does not support {backend} backend now.` The only
workaround is `--enforce-eager`, reported at **under 10 tok/s (~20% throughput)** — useless for a
benchmark.

So: `transformers.Qwen3VLForConditionalGeneration` with `attn_implementation="sdpa"`. This is
also exactly what NVIDIA's own `scripts/inference_sample.py` does, so we are not off the beaten
path — we're on a different beaten path than the deployment docs assume.

**Do not "upgrade" this to vLLM on a T4.** If the GPU ever changes to Ada or newer (L4, A10G,
L40S), vLLM becomes viable and this decision should be revisited *then*, deliberately.

### Precision: fp16, never bf16, never FP8

T4 is Turing. Turing has **no BF16 hardware and no FP8 hardware**. The HF model card says the
model was "only tested doing inference with BF16 precision," but NVIDIA's own sample script uses
`torch.float16` — fp16 is the supported practical path on Turing.

All the FP8 Cosmos guidance you will find online (Jetson AI Lab, the `--precision fp4`
llmcompressor scripts, `gpu-memory-utilization` tuning) targets Jetson Thor, Hopper, or
Blackwell. It does not transfer to a T4.

### Local development: bitsandbytes NF4 on the laptop

The dev machine is an RTX 3060 **Laptop** GPU with **6 GB** VRAM. fp16 weights alone are
**~4.9 GB** (2,438,696,960 params x 2 bytes). Add Windows' ~0.6–1 GB display reservation, the
vision tower's activations, and the KV cache, and fp16 **does not fit**. This was verified by
arithmetic before any code was written, not discovered by OOM.

NF4 (4-bit, double-quantised, fp16 compute) brings weights to roughly 2 GB and fits comfortably.
`COSMOS_QUANT` switches between `nf4` and `none`, so the identical code path runs fp16 on a
Kaggle T4 or EC2 T4 by flipping one environment variable. `COSMOS_DEVICE=cpu` is the escape
hatch if VRAM still runs short.

### The "24 GB minimum" on the model card is not the real floor

Both the HF card and the GitHub README state 24 GB for the 2B model. That figure assumes the full
256K context window and unbounded vision tokens. In practice: weights are 4.9 GB at fp16,
NVIDIA's own sample caps vision tokens at 8192, vLLM guidance uses `--max-model-len 8192`, and
Jetson runs this model on an 8 GB Orin. With bounded vision tokens and a capped context, a 16 GB
T4 is comfortable.

### Benchmark shape: two profiles, both capped

The model emits a `<think>...</think>` chain of thought and the card recommends 4096+ output
tokens to avoid truncation. On a T4 that is roughly **2–3 minutes per request** — a concurrency-8
sweep would run over an hour and cost real money, and p95 would be pure queueing noise.

So the benchmark publishes two tables:

- **Profile A (headline)** — `max_new_tokens=256`, fixed prompt, fixed image. Comparable p50/p95,
  full 1/4/8 concurrency sweep in ~15 minutes.
- **Profile B (realistic)** — `max_new_tokens=1024`, full CoT. Shows what reasoning actually costs.

Both are committed. Reporting only Profile A would misrepresent the model; reporting only
Profile B would burn budget for a noisier number.

### Load generation runs on the EC2 instance, not the laptop

k6 runs in a container on the instance and hits `localhost:8000`. Driving load from a home
network would fold WAN latency and jitter into every p95 sample, which would make the headline
number unreproducible and indefensible. It also means port 8000 never needs to be public — the
security group only opens SSH from one IP.

## Model facts (verified against the HF card and NVIDIA's inference_sample.py)

| Fact | Value |
|---|---|
| Model id | `nvidia/Cosmos-Reason2-2B` |
| Base | Post-trained from `Qwen3-VL-2B-Instruct`, identical architecture |
| Model class | `transformers.Qwen3VLForConditionalGeneration` |
| Processor class | `transformers.Qwen3VLProcessor` |
| Total params | **2,438,696,960** — assert this at load |
| Min transformers | `>=4.57` (NVIDIA pins `4.57.3`) |
| Pinned peers | `torch==2.9.0`, `accelerate==1.12.0`, `av==16.1.0`, `pillow==12.0.0`, `torchcodec==0.9.1` |
| Attention | `sdpa` |
| Media formats | `mp4` video, `jpg` images |
| Video fps | **4** — matches the training setup, do not change casually |
| Message ordering | **Media before text** in the content list, to match training inputs |
| Vision token size | `PIXELS_PER_TOKEN = 32**2`; processor `size = {shortest_edge: min_tok*1024, longest_edge: max_tok*1024}` |
| Output format | `<think>reasoning</think>` then the answer |
| Recommended max tokens | 4096+ to avoid truncation (we cap deliberately — see benchmark shape) |
| Context | up to 256K input tokens |
| Frame timestamps | The model recognises timestamps burned into the bottom of each frame |
| License | NVIDIA Open Model License — commercial use and derivative models permitted |

## Conventions

- Python **3.11+**. **Every** dependency pinned to an exact version.
- All settings via environment variables with a `COSMOS_` prefix, parsed by `pydantic-settings`.
  `.env.example` is committed and documents every variable; `.env` is gitignored.
- **No secrets in the repo.** Ever. `HF_TOKEN` and AWS credentials come from the environment.
- Structured logging via `structlog`, JSON output, one line per request.
- Model weights are **never** baked into a Docker image. They live on a mounted volume.
- Each phase must be runnable and testable before the next one starts.
- Prefer boring, well-documented tools over clever ones.

### Git conventions

- **Commits are authored solely by the repo owner.** Never add a `Co-Authored-By:` trailer,
  a "Generated with ..." line, or any other assistant attribution to a commit message or PR
  body. This is a portfolio repo and the history is part of it.
- Repo-local identity is set to `AadiPathak23 <aadipathak2323@gmail.com>`. The *global*
  `user.name` has a stray leading space (`" AadiPathak23"`), which is why this repo sets it
  locally. Note the git email differs from the HuggingFace/Claude account email.
- Prefer several logical commits in the order the work actually happened over one large dump.
- `.gitattributes` forces `eol=lf`. The repo is developed on Windows but everything runs in
  Linux containers; autocrlf would otherwise write CRLF into the Docker build context.
- Generated artifacts are not committed: `assets/*.jpg` and `*.mp4` come from
  `scripts/make_assets.py`, and model weights never enter the repo or an image.

## Environment

| Where | Hardware | Precision | Notes |
|---|---|---|---|
| Dev laptop | Windows 11, RTX 3060 Laptop **6 GB** | NF4 only | Docker GPU passthrough needs WSL2 + NVIDIA Container Toolkit |
| Free GPU | Kaggle T4 (Turing, 16 GB) | fp16 | No bf16, no FP8. Primary debugging target for GPU work. |
| Paid target | EC2 `g4dn.xlarge` **spot**, T4 16 GB, 4 vCPU | fp16 | ~$0.32/hr spot, $0.526/hr on-demand (us-east-1) |

Cloud time is minimised on purpose: develop and debug free on Kaggle, use EC2 only for the single
final benchmark run.

## Anti-goals — things that must not happen

- **No mock or fallback model, under any circumstances.** If the real model fails to load, the
  process exits non-zero with a loud error. A pipeline that silently degrades to a stub is the
  specific failure mode this repo is built to avoid. Load-time guards are fatal, never warnings.
- No vLLM on a Turing GPU (see the decision above).
- No model weights inside a Docker image.
- No AWS service beyond EC2 and S3 without asking.
- No on-demand instances without a stated reason spot won't work.
- No overselling Phase 3. It is a single-node learning exercise and the README must say so.

---

# CHANGELOG

Append only. Newest at the bottom.

### 2026-08-11 — Repo bootstrap (docs only, $0.00 spent)

- Researched how Cosmos-Reason2-2B is actually loaded and served: HF model card,
  `nvidia-cosmos/cosmos-reason2` (incl. `scripts/inference_sample.py`), NVIDIA Cosmos docs,
  Jetson AI Lab. Findings recorded in STABLE above.
- **Found that vLLM cannot serve Qwen3-VL on Turing/T4** — vllm-project/vllm#29743, closed as
  not planned. Locked HF transformers + `sdpa` + fp16 as the serving backend. This overrides
  NVIDIA's own "we recommend vllm" deployment guidance, which assumes Hopper/Blackwell/Jetson.
- Confirmed by arithmetic that the 6 GB RTX 3060 Laptop **cannot** hold fp16 weights
  (4.9 GB weights + display reservation + activations + KV cache). Locked bitsandbytes NF4 as
  the local default, with `COSMOS_DEVICE=cpu` as an escape hatch.
- Established that the model card's "24 GB minimum" assumes 256K context and unbounded vision
  tokens; a 16 GB T4 is fine with bounded vision tokens.
- Chose a two-profile benchmark (256-token headline, 1024-token realistic) because uncapped
  4096-token CoT is ~2-3 min/request on a T4.
- Chose k6 running on the EC2 instance against localhost, to keep home-network jitter out of p95.
- Created `docs/PLAN.md`, `docs/CLAUDE.md`, and the root `CLAUDE.md` stub. **No code written.**
- Open questions carried into Phase 1 are listed at the bottom of `docs/PLAN.md`.

### 2026-08-11 — Phase 1 implemented ($0.00 spent, still local-only)

Built the whole local service: `app/{config,logging_conf,metrics,model,inference,queue,
schemas,main}.py`, multi-stage `Dockerfile`, `docker-compose.yml`, `scripts/{make_assets,
smoke_test}.py`, and a 33-case pytest suite. **32 passed, 1 skipped, ruff clean.** Not yet
run against real weights — that is the next step.

Findings that changed the design as it was built:

- **The `nvidia/Cosmos-Reason2-*` repos are GATED** (`gated: auto` on the HF API; the raw
  `config.json` returns 401 while `Qwen/Qwen3-VL-2B-Instruct` returns 200). `HF_TOKEN` is
  therefore **required**, not optional as originally documented. Access is granted
  automatically on accepting the licence, so there is no approval queue. A bare 401 reads
  like a network fault, so `_load_failure_message()` rewrites it into the two concrete
  steps. Secondary consequence for Phase 2: mirroring weights to S3 means the token never
  has to live on the EC2 instance.
- **bitsandbytes packs two 4-bit params per stored byte**, so a raw `numel()` reports
  ~1.2 B parameters for a perfectly healthy NF4 load. That would have tripped the
  `MIN_EXPECTED_PARAMS` guard and aborted startup on a *correct* model — the anti-mock
  guard eating a good load. `count_parameters()` doubles `Params4bit` counts.
- **Dependency resolution had two traps.** `huggingface-hub` is on 1.x, but
  `transformers==4.57.3` requires `<1.0`, so an unpinned install silently resolves
  transformers backwards past Qwen3-VL support; pinned to 0.36.2, the last 0.x.
  `torchvision==0.24.0` is the release that hard-requires `torch==2.9.0` (latest, 0.28.0,
  wants torch 2.13). Whole pin set verified to co-install in a clean venv.
- **`device_map={"": 0}` rather than `"auto"`.** `"auto"` spills layers to CPU RAM when
  VRAM is short, producing a service that starts fine and runs ~20x slower for invisible
  reasons. An explicit map turns that into an OOM at startup, which is the honest failure.
  A separate guard also rejects parameters split across devices.
- **`torch.cuda.synchronize()` after `generate()`.** CUDA is async; without it the timer
  measures kernel *launch* and every latency number comes out far too low. This one would
  have quietly invalidated the entire Phase 2 benchmark.
- **Prometheus default buckets top out at 10 s**, which would put nearly every real
  request into `+Inf`. Custom buckets span 0.1 s to 300 s.
- Builder and runtime Docker stages share the same CUDA base deliberately: a venv records
  an absolute interpreter path, so building on `python:3.12-slim` and copying across
  leaves it pointing at a binary that does not exist in the final image.
- CPU mode forces float32 — fp16 matmul kernels do not exist on CPU
  (`addmm_impl_cpu_ not implemented for 'Half'`). The override is surfaced as a banner
  note rather than applied silently.
- Fixed during testing: app state was not cleared on lifespan shutdown, so `/health`
  kept advertising a model whose worker had already stopped.

### 2026-08-11 — Published to GitHub ($0.00 spent)

- Initialised the repo and pushed to https://github.com/AadiPathak23/cosmos-edge-serve as
  five logical commits (`ab0aaf8` docs → `f13f8fb` scaffold → `10689c0` app → `1e098eb`
  Docker → `43c2674` tests), following the order the work actually happened.
- All commits authored **and** committed by `AadiPathak23 <aadipathak2323@gmail.com>`, with
  no assistant attribution anywhere. Recorded as a standing convention in STABLE above.
- Added `.gitattributes` (`* text=auto eol=lf`) — git was about to convert the working copy
  to CRLF, which would have put CRLF into the Linux Docker build context.
- Pre-publish checks: no `.env` present or tracked, no token/key-shaped strings anywhere in
  the tree, `models/` and generated `assets/*.jpg|mp4` correctly ignored. 31 files pushed.

**Nothing has run against real weights yet.** The service is code-complete and unit-tested
but unproven end to end — see the "start here" block at the top of `docs/PLAN.md`.
