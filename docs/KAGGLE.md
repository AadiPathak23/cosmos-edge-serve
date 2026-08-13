# Running the T4 check on Kaggle (free)

**Cost: $0.00.** Kaggle gives 30 GPU hours a week. This run takes 20–40 minutes, most of
which is downloading weights.

## Why this exists

Everything so far has run **NF4 on a 6 GB RTX 3060**. Phase 2 runs **fp16 on a 16 GB T4**.
Those are different precisions on a different architecture, and two things are unknown until
a real T4 has run the code:

1. **Does the fp16 path load and infer at all?** Finding out on a paid spot instance means
   debugging on a rented GPU, which is exactly what turns a $1 run into a $5 one.
2. **What is the real decode rate?** The laptop managed 0.6–2.9 tok/s under NF4. That number
   is dominated by 4-bit dequantisation overhead on a small laptop GPU and tells us nothing
   about a T4. Benchmark profile B (1024 tokens of full chain-of-thought) has no runtime
   budget — and therefore no dollar estimate — until this is measured.

## What Kaggle cannot do

**Kaggle has no Docker daemon.** The container this project spent Phase 1 building and
debugging cannot run here, and there is no way to drive concurrent load against a running
service from a notebook. That is precisely why Phase 2 still needs EC2. This check runs the
app **natively from a clone**, calling `load_model()` and `run_inference()` directly — the
same code path the benchmark uses, minus the HTTP and queue layers (which the pytest suite
already covers).

---

## Setup, once

1. Create a new Kaggle notebook.
2. **Notebook options** (right sidebar):
   - **Accelerator → GPU T4 x2.** The script pins `CUDA_VISIBLE_DEVICES=0` and uses only one,
     because `g4dn.xlarge` has one. Do not pick P100 — it is Pascal, a generation away from
     the target, and the script will refuse to record its numbers.
   - **Internet → On.** This requires phone verification on your Kaggle account. Without it
     the weight download fails.
3. **Add-ons → Secrets → Add secret**, name it exactly `HF_TOKEN`, value = a HuggingFace
   token with read access. Do **not** paste the token into a cell — notebooks are shareable
   and the repo's rule is no secrets, anywhere, ever.
4. The HF account behind that token must have **accepted the licence** at
   <https://huggingface.co/nvidia/Cosmos-Reason2-2B>. The `nvidia/Cosmos-Reason2-*` repos are
   gated; acceptance is automatic with no approval queue, but without it every download
   returns a bare 401. (`app/model.py` rewrites that 401 into the two concrete steps, so if
   you see it, the message tells you what to do.)

---

## The cells

### Cell 1 — environment

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# NOT /kaggle/working: that directory is capped at 20 GB and is packaged into the
# notebook's output on commit, so ~5 GB of fp16 safetensors would be uploaded with
# every save. /kaggle/temp is scratch and is discarded with the session.
os.environ["HF_HOME"] = "/kaggle/temp/hf"

# HF's default read timeout is 10 s, which any brief network stall trips during a
# ~5 GB pull — the download then wedges rather than retrying cleanly. This bit us
# on the laptop and there is no reason to relearn it here.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
```

### Cell 2 — clone

```python
!git clone --depth 1 https://github.com/AadiPathak23/cosmos-edge-serve.git /kaggle/working/cosmos
%cd /kaggle/working/cosmos
!git log --oneline -1
```

### Cell 3 — install the pinned stack (~5 minutes)

```python
!pip install -q -r requirements.txt
```

**Expect noise, and expect it to be fine.** Kaggle preinstalls its own `torch`,
`transformers`, and `huggingface-hub`; this replaces all three with the repo's pins
(`torch==2.9.0`, `transformers==4.57.3`, `huggingface-hub==0.36.2`). pip will print
dependency-conflict warnings about Kaggle's own packages (`datasets`, `kagglehub`, and
friends). Those are warnings, not errors — nothing in this check uses them.

The downgrades are deliberate, not accidental: `transformers==4.57.3` declares
`huggingface-hub<1.0`, and hub is on 1.x, so an unpinned install would silently resolve
transformers *backwards* to a version with no Qwen3-VL support at all.

Verify before continuing:

```python
!python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# must print: 2.9.0+cu128 4.57.3
```

If it prints anything else, **restart the session** (Run → Restart session) and re-run
cells 1–3. A `torch` already imported into the kernel cannot be replaced in place.

### Cell 4 — generate the test media

```python
!python scripts/make_assets.py
```

Writes `assets/sample.jpg` and `assets/sample.mp4` (3 s, 12 frames at 4 fps, with burnt-in
timestamps). Real media is never committed to the repo, so this step is required.

### Cell 5 — the check itself (~15–30 minutes)

```python
!python scripts/kaggle_t4_check.py
```

Run it as a **subprocess** (`!python`), not by importing the modules into the notebook. The
script sets `CUDA_VISIBLE_DEVICES=0` before torch is imported, which only works in a fresh
process.

First run downloads ~5 GB of fp16 weights. Add `--trials 5` for tighter medians, or
`--skip-video` if you only need the image leg.

### Cell 6 — hand the results back

```python
print(open("kaggle_t4_results.json").read())
```

Paste that JSON into the Claude Code session. It contains the GPU identification, the full
load report, and median tok/s for every leg — which is what profile B's runtime budget and
the Phase 2 cost estimate get computed from.

---

## What "passing" looks like

The script exits non-zero and lists problems if any of these are wrong:

| Check | Expected | Why it matters |
|---|---|---|
| GPU | `Tesla T4`, `sm_75` | A P100 or L4 number would not transfer to `g4dn.xlarge` |
| `dtype` | `float16` | Turing has no bf16 and no FP8 hardware |
| `quantization` | `none` | This run exists to prove the *un*quantised path |
| `params.total` | `2,127,532,032` | Must match the NF4 figure exactly — see below |
| decode rate | recorded | The number the whole check is for |

### On the parameter count

A live model reports **2,127,532,032**, not the model card's 2,438,696,960. The difference is
exactly one token embedding table (151936 × 2048 = 311,164,928): Qwen3-VL-2B ties `lm_head`
to `embed_tokens`, and `model.parameters()` deduplicates shared tensors. The tying is
architectural, so **fp16 on a T4 must report the identical figure**. If it does not, that is a
real finding worth chasing, not a rounding difference.

This is also why `MIN_EXPECTED_PARAMS` in `app/config.py` is a **floor** and not an equality
check. Do not "fix" it to assert the model card's number — it would abort every correct load.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `401` / gated repo message on download | Licence not accepted on the token's HF account, or the `HF_TOKEN` secret is missing/misnamed |
| `No CUDA device` | Accelerator not set to GPU, or the session needs restarting after changing it |
| Refuses to run, "Expected a Tesla T4" | Kaggle allocated a P100/L4. Change the accelerator and restart; or pass `--allow-any-gpu` for an off-target datapoint |
| `torch.__version__` is not `2.9.0+cu128` | Kernel had torch loaded before cell 3. Restart the session |
| CUDA OOM at load | Should not happen — fp16 weights are ~4.9 GB of 16 GB. If it does, another notebook process is holding the card; restart the session |
| Download stalls at some fixed byte count | Network stall past the HF read timeout. Cell 1 raises it to 60 s; if it still happens, restart and re-run — the cache resumes |
