# PLAN

**Status: Phase 1 is COMPLETE and verified end to end on the RTX 3060.**
The service loads real weights, and the smoke test passes on **both image and video**.
The image has been shrunk to 13.5 GB and re-verified.
**Budget spent to date: $0.00 / $10.00**

---

## ▶ START HERE (next session)

Phase 1 is done and the image shrink is done. These are the next actions, in order.

**1. ⚠️ File the AWS quota increase — BLOCKING, and only you can do it**
*(free, ~5 min to file, 1–3 days to land)*

This is the long pole. Nothing about the code can unblock it.

1. Console → **region selector → US East (N. Virginia) `us-east-1`** *first*. Quotas are
   per-region; filing in the wrong region wastes the 1–3 days.
2. **Service Quotas → AWS services → Amazon EC2 →** search
   **`All G and VT Spot Instance Requests`** (code **`L-3819A6DF`**).
3. **Read the applied value before filing.** It may already be ≥ 4, in which case Phase 2 is
   unblocked immediately.
4. If 0: **Request increase at account level → `4`**. The unit is **vCPUs**, not instances —
   `g4dn.xlarge` is exactly 4. Ask for 4, not 8; minimal asks auto-approve far more often.

Traps: **spot and on-demand are separate quotas** — `L-3819A6DF` is spot only. If it is
rejected, reopen as a support case with a one-line justification. If AWS refuses outright, the
fallback is **Lightning AI Studios** (free GPU hours on a real VM *with* a Docker daemon —
the only free option that can run this container at all).

Region note: connecting from outside the US changes **nothing** about the benchmark. k6 runs
on the instance against `localhost`, so the client is never in the measurement path; only SSH
round-trip time changes. us-east-1 also has the deepest g4dn spot capacity, which lowers the
interruption risk that would actually cost money in a redo.

**2. Kaggle T4 check** *(free — harness is written and ready)*
Follow **`docs/KAGGLE.md`** and run `scripts/kaggle_t4_check.py` on a Kaggle T4. Proves the
fp16 path and measures the real decode rate, resolving open question 5. On the laptop under
NF4 we saw only **0.6–2.9 tok/s**, which is not a T4 number and must not be used to size
anything. Paste `kaggle_t4_results.json` back to size profile B's runtime and cost.

**3. Decide on a billing guard before any AWS spend** *(free, needs a call)*
Signup credits are expired, so Phase 2 is real money on a real card. An AWS Budgets alert at
$5 is free and is the obvious guard — but **Budgets is a service beyond EC2 + S3**, which this
repo's constraints require asking about first. Unresolved; decide before Phase 2 starts.

**To reproduce the working local run:**
```bash
cp .env.example .env     # set HF_TOKEN — the repo is GATED
docker compose up --build
python scripts/smoke_test.py    # host-side, stdlib only; assets already exist
```

**Do not start Phase 2 without an explicit go.** It costs money.

---

Legend: `[x]` done and verified · `[~]` written but not yet exercised against real
weights/hardware · `[ ]` not started.

Do not start a phase until the previous one is confirmed working.

---

## Phase 0 — Research & docs (free) ✅

- [x] Research how Cosmos-Reason2-2B is actually loaded and served
- [x] Determine the right backend: **transformers, not vLLM** (vLLM #29743 — no Turing
      support for Qwen3-VL, closed as not planned)
- [x] Determine video input format: mp4, `fps=4`, media listed before text
- [x] Determine VRAM footprint: ~4.9 GB weights at fp16; no FP8 on Turing
- [x] Write `docs/PLAN.md`, `docs/CLAUDE.md`, root `CLAUDE.md` stub

---

## Phase 1 — Local service (FREE — local only, no cloud calls)

### 1.1 Scaffold ✅

- [x] `.gitignore`, `.dockerignore`
- [x] `requirements.txt` — every dep pinned, **and the full set verified to co-install**
      in a clean venv. Two traps found and avoided:
      - `huggingface-hub` must be `<1.0` (it is on 1.x now); `transformers==4.57.3`
        declares `huggingface-hub<1.0,>=0.34.0`, so an unpinned install silently
        resolves transformers *backwards* to a version with no Qwen3-VL.
      - `torchvision==0.24.0` is the version that hard-requires `torch==2.9.0`. The
        current latest, 0.28.0, would have dragged in torch 2.13.
      - `tokenizers` ceiling is really 0.22.2 — transformers caps at `<=0.23.0` and
        0.23.0 was never published.
- [x] `requirements-dev.txt`, `pyproject.toml` (pytest + ruff config)
- [x] `.env.example` with every `COSMOS_*` variable documented inline
- [x] `README.md` with the cost warning at the very top
- [x] `scripts/make_assets.py` — generates `assets/sample.jpg` and `sample.mp4`
      (3 s, 12 frames @ 4 fps, with burnt-in timestamps). Verified: decodes back to
      12 frames at rate 4. Real media is not committed.

### 1.2 Model loading — `app/config.py`, `app/model.py` ✅ / ~

- [x] `COSMOS_*` settings via pydantic-settings (`protected_namespaces=()` so `model_id`
      does not collide with pydantic's reserved `model_` prefix)
- [x] Loader: `Qwen3VLForConditionalGeneration` + `AutoProcessor`
- [x] Precision switch NF4 / fp16 via `BitsAndBytesConfig`
- [x] Device switch `auto`/`cuda`/`cpu`; CPU forces float32 with an explicit banner note
      (fp16 matmul kernels do not exist on CPU)
- [x] Vision token clamp on `image_processor` and `video_processor`
- [x] Optional PEFT adapter, off by default (unwrap path fixed and regression-tested)
- [x] Loud startup banner with the full load report
- [x] **Anti-mock guards, all fatal:**
  - [x] model class must be `Qwen3VLForConditionalGeneration` (unwraps PEFT first)
  - [x] param count must exceed 2 B — **with 4-bit unpacking**, because bitsandbytes
        packs 2 params per byte and a raw `numel()` would report ~1.2 B on a perfectly
        healthy NF4 load and abort startup
  - [x] parameters must be on the requested device (no silent CPU fallback)
  - [x] parameters must not be split across devices (catches accelerate offload, which
        would silently make every benchmark number meaningless)
  - [x] `device_map={"": 0}` rather than `"auto"`, so insufficient VRAM is an OOM at
        startup instead of a quiet 20x slowdown
- [x] Gated-repo detection: a bare HTTP 401 is rewritten into the licence + token steps
- [x] Warmup: synthetic 448×448 image + 8-token generate before healthy (18.5 s measured)

### 1.3 Inference path — `app/inference.py`, `app/schemas.py` ✅ / ~

- [x] Conversation built **media before text**; `fps` passed only for video
- [x] `processor.apply_chat_template(...)`
- [x] Token accounting — visual tokens read from `config.image_token_id` /
      `video_token_id` first (authoritative), with a `<|image_pad|>`/`<|video_pad|>`
      tokenizer lookup as fallback
- [x] `<think>` split, including the unterminated case (truncated mid-reasoning →
      reasoning returned, answer empty, `truncated: true`)
- [x] `torch.cuda.synchronize()` after generate — without it the timing measures kernel
      *launch*, not execution, and every latency number comes out far too low
- [x] Greedy decoding (`do_sample=False`) so benchmark runs are comparable
- [x] 415 / 413 / 422 for bad media type, oversize upload, empty prompt

### 1.4 Concurrency — `app/queue.py` ✅

- [x] Bounded `asyncio.Queue`, single worker, 1-thread executor
- [x] Queue full → 503 + `Retry-After`; timeout → 504
- [x] Queue wait and compute recorded separately
- [x] Worker survives a failed request; failures propagate to the caller, not into a
      hung future

### 1.5 API — `app/main.py` ✅

- [x] `POST /infer`, `GET /health`, `GET /metrics`, `GET /` , `/docs`
- [x] Model loads once in the lifespan handler; load failure → non-zero exit
- [x] App state cleared on shutdown so `/health` stops advertising a dead worker

### 1.6 Observability ✅

- [x] structlog JSON, one line per request with the full token and timing breakdown
- [x] Prometheus metrics with buckets that span 0.1 s → 300 s (the default buckets top
      out at 10 s and would lump nearly every real request into `+Inf`)

### 1.7 Docker ~

- [x] Multi-stage Dockerfile. Builder and runtime share the same CUDA base on purpose:
      a venv records an absolute interpreter path, so building against `python:3.12-slim`
      and copying across would leave the venv pointing at a binary that does not exist.
- [x] `nvidia/cuda:12.8.1-base-ubuntu24.04` (was `-runtime`; see the shrink below), ffmpeg
      **plus `libpython3.12t64`** and **`libnpp-12-8`** for torchcodec, curl for the
      healthcheck, non-root user (UID 1000 needs `userdel ubuntu` first on 24.04), and both
      torch/lib and cuda_nvrtc/lib appended to `LD_LIBRARY_PATH`
- [x] `HF_HOME=/models` + named volume — **weights never baked into the image**
- [x] `HF_TOKEN` threaded through compose (required: the repo is gated)
- [x] Build it and **record the measured image size** in the CHANGELOG — **18.1 GB**, vs 6–8 GB predicted
- [x] Verify GPU passthrough end to end on the laptop
- [x] **Shrink the image before Phase 2** — `-base` instead of `-runtime`:
      **13.5 GB reported / 8.71 GB layers / ~4.8 GB compressed pull**, down from
      18.1 / ~11.6 / ~6.5. The compressed pull is the figure that costs paid GPU minutes.
      The swap broke video decode until `libnpp-12-8` (+320 MB) was added back and
      `cuda_nvrtc/lib` was put on `LD_LIBRARY_PATH` — **torch bundles what torch needs and
      nothing more; torchcodec's CUDA dependencies are its own problem.** Re-verified: both
      smoke-test legs pass with byte-identical output and token counts to the pre-shrink run.

### 1.8 Verify Phase 1

- [x] `pytest` — **35 passed, 1 skipped** (the skip needs a bf16-less GPU). Runs with no
      GPU and no weights: `load_model` and `run_inference` are faked at the seam, so the
      queue, executor, metrics, and error mapping are all genuinely exercised.
- [x] `ruff check .` clean
- [x] Repo initialised and pushed to https://github.com/AadiPathak23/cosmos-edge-serve
      (5 commits, no assistant attribution, `.gitattributes` forcing LF, no secrets)
- [x] `docker compose up --build` on the RTX 3060 with NF4 → healthy, banner correct
- [x] `python scripts/smoke_test.py` → image + video both return sane text and non-zero
      visual token counts (visual=300 image, visual=480 video). Assets already exist on the
      host; do **not** run `make_assets.py` inside the container — compose mounts only
      `cosmos-models:/models`, so files written there are invisible to the host-side test.
- [x] Harness for the T4 check written: `scripts/kaggle_t4_check.py` + `docs/KAGGLE.md`.
      Runs natively (Kaggle has no Docker daemon) but through the service's own
      `load_model()` / `run_inference()`, so the timings come from the real code path.
- [ ] Run it: **Kaggle T4** with fp16 → prove the `COSMOS_QUANT=none` path ← NEXT (free)
- [ ] Measure real T4 decode rate on Kaggle (free) to size benchmark profile B
- [ ] ⚠️ **File the AWS G-family vCPU quota increase** — free, but takes 1–3 days.
      `us-east-1` → Service Quotas → EC2 → `All G and VT Spot Instance Requests`
      (`L-3819A6DF`) → request **4 vCPUs**. Default is 0 and Phase 2 cannot launch
      without it. Full steps in the START HERE block above.
- [ ] Decide whether to add a $5 AWS Budgets alert (free, but a service beyond EC2 + S3,
      so it needs an explicit call before Phase 2)

---

## Phase 2 — Cloud (💵 COSTS MONEY — do not start without explicit confirmation)

### Cost estimate — read before doing anything in this phase

| Item | Rate | Est. usage | Est. cost |
|---|---|---|---|
| `g4dn.xlarge` **spot** (T4, 16 GB) | $0.32/hr | 3 hr | $0.96 |
| EBS gp3 root, 100 GB | $0.08/GB-month | 3 hr | $0.03 |
| S3 storage, ~6 GB | $0.023/GB-month | 7 days | $0.03 |
| S3 requests | negligible | — | <$0.01 |
| EC2 → S3 transfer, same region | free | — | $0.00 |
| **Total** | | | **~$1.05** |

On-demand comparison: $0.526/hr → ~$1.58 for the same 3 hours. Spot is the default;
the benchmark is short and rerunnable, so interruption risk is acceptable.

**Free alternative to state up front:** weights can be pulled straight from HuggingFace
on the instance for $0. S3 is in the plan because it is an explicit learning goal, and
because the featherweight-ai adapter has no other home. A real secondary benefit found
during Phase 1: mirroring to S3 means `HF_TOKEN` never has to live on the EC2 box.

Worst case with one failed run and a redo: **~$3**. Ceiling is $10.

**Free-GPU alternatives were considered and consciously declined (decided 2026-08-13).**
Kaggle and Colab are free but run notebooks, not servers: no Docker daemon, so the container
this entire phase was spent building and debugging would go untested, and there is no way to
drive concurrent load at a running service. Lightning AI Studios *would* work — free monthly
GPU hours on a real VM with Docker — and is the fallback if the AWS quota is refused. It was
declined because "cloud GPU deployment" and S3 are stated learning goals of this repo, and
~$1 of a $10 budget is a reasonable price for them.

**Cost discipline that follows from this:** use Kaggle (free) for everything it *can* do —
the fp16 path and the decode-rate measurement — so the paid instance only ever runs the one
thing that needs it. Debugging on a rented GPU is what turns a $1 run into a $5 one.

### Tasks

- [ ] 🛑 **STOP — confirm the cost table above before any AWS action**
- [ ] Verify the G-family vCPU quota increase landed
- [ ] S3: create bucket, upload base weights + adapter
      *(undo: `aws s3 rb s3://<bucket> --force`)*
- [ ] EC2: launch `g4dn.xlarge` **spot**, Deep Learning Base OSS Nvidia Driver AMI
      (Ubuntu 22.04) — driver/docker/container-toolkit preinstalled, minimising **paid**
      GPU minutes spent on setup
      *(undo: `aws ec2 terminate-instances --instance-ids <id>`)*
- [ ] Security group: SSH from your IP only. **Port 8000 stays closed** — k6 runs on-box.
      *(undo: `aws ec2 delete-security-group --group-id <id>`)*
- [ ] Pull weights from S3 into the volume, then `docker compose up`
- [ ] Confirm the banner reads `dtype float16`, `quantization none`, Tesla T4 detected
- [ ] `loadtest/load.js` — k6, profile A (256 tok) and B (1024 tok), 1 / 4 / 8 VUs
- [ ] Run both sweeps on the instance against `localhost:8000`
- [ ] `loadtest/render_results.py` → markdown tables committed to README + docs
- [ ] Write `docs/TEARDOWN.md` (terminate, verify EBS deleted, release EIP, empty and
      delete the bucket, delete snapshots/AMIs, confirm $0 in Cost Explorer 24 h later)
- [ ] ✅ **Run teardown the same day. Verify in the console, not from memory.**

---

## Phase 3 — Kubernetes (OPTIONAL — only if explicitly requested)

- [ ] Local k3s or minikube only. Zero cloud spend.
- [ ] Deployment, Service, HPA manifests
- [ ] README must state plainly: single-node learning exercise, not production.

---

## Open questions

**Resolved during Phase 1**

1. ~~Visual placeholder token strings~~ → sidestepped. Visual tokens are counted from
   `config.image_token_id` / `video_token_id`. Confirmed non-zero on the first real run:
   visual=300 (image), visual=480 (video).
2. ~~`nvidia/Cosmos-Reason2-*` repos are gated~~ → confirmed against a real 401. `HF_TOKEN`
   is required. The loader rewrites the bare 401 into the licence + token steps; verified.
3. ~~Does NF4 quantize the ViT tower, or only the language model?~~ → **BOTH.** Weights land
   at 1.54 GB, under the ~2 GB estimate, using 1.59 GB of 6 GB VRAM. Only the tied token
   embedding and ~4 M of norms/biases stay in fp16. Image *and* video understanding are fine.
4. ~~torchcodec 0.9.1 / FFmpeg linkage~~ → **FFmpeg was never the problem**, despite the
   error text saying so. Two stacked causes: `libtorch.so` not on `LD_LIBRARY_PATH` (pip puts
   it in `site-packages/torch/lib`), then `libpython3.12.so.1.0` missing because Ubuntu's
   `python3` ships no shared libpython (package is `libpython3.12t64` on 24.04). Both fixed
   in the Dockerfile; video decode verified end to end.
7. ~~Measured Docker image size~~ → **18.1 GB**, vs 6–8 GB predicted. CUDA ships twice: the
   `-runtime` base's system CUDA libraries (3.11 GB) plus torch's bundled `+cu128` wheels.
   ~~Switching to the `-base` tag is a candidate fix, not yet attempted.~~ → **Done
   2026-08-13: 13.5 GB reported / 8.71 GB layers / ~4.8 GB compressed pull.** The swap
   required adding `libnpp-12-8` back (torchcodec links against `libnppicc.so.12` at load
   time, and torch does *not* bundle NPP) and putting the venv's `cuda_nvrtc/lib` on
   `LD_LIBRARY_PATH`. Also **corrected a wrong explanation**: the layer-sum vs `docker images`
   gap is the containerd image store counting compressed blobs *plus* the unpacked snapshot,
   not buildx attestation manifests (which are kilobytes).

**New, found during the first real run**

8. **The model card's param count cannot be asserted for equality.** A live model reports
   2,127,532,032, not 2,438,696,960 — `lm_head` is tied to `embed_tokens` and
   `model.parameters()` deduplicates. Guard with a floor only. Recheck this on the T4 at
   fp16: the tying is architectural, so it should report the same figure.

**Still open**

5. Real T4 decode rate is unmeasured. The laptop under NF4 managed only **0.6–2.9 tok/s**,
   which says nothing about a T4 — 4-bit dequant overhead dominates on a 6 GB laptop GPU.
   Benchmark profile B's runtime budget still depends on this. → Measure free on Kaggle.
6. Do NF4-on-3060 and fp16-on-T4 agree closely enough for the smoke test to assert on output
   *content*, or only on shape? Still asserts on shape plus non-zero visual tokens. The NF4
   outputs were accurate on both legs, so a content assertion now looks plausible. The
   2026-08-13 rebuild showed NF4 output is *byte-stable across rebuilds* under greedy
   decoding, which is a necessary but not sufficient condition — the open part is whether it
   survives a precision change. `scripts/kaggle_t4_check.py` records sample answers precisely
   so this can be compared.

**New, found during the image shrink (2026-08-13)**

9. **How much more of the 8.71 GB is removable?** The venv layer is 7.56 GB and dominates
   everything else now. It contains nvidia-* wheels torch pulls in unconditionally but which
   a single-GPU inference service never uses — `nccl` (multi-GPU collectives), `cusparselt`,
   `nvshmem`, `cufile`. Stripping them could plausibly halve the layer again. **Not attempted
   and not obviously safe**: torch imports some of these eagerly, and the failure mode would
   be a load-time `dlopen` error, i.e. exactly the class of bug this shrink just spent a
   rebuild cycle on. Only worth doing if pull time on the instance turns out to matter, and
   it must be gated on the same both-legs smoke test.
