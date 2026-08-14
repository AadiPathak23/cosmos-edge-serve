# Phase 2 runbook — the benchmark run on EC2

> 💵 **This is the only part of the project that costs money.** Estimated **~$1.05**, ceiling
> **$10**. Read the whole file before running the first command. Every step that creates
> something states the command that destroys it.

Teardown has its own file: **`docs/TEARDOWN.md`**. Run it **the same day**.

---

## 0. Before you spend anything

Four preconditions. If any fails, stop — none of them cost anything to fix, and all of them
cost money to discover late.

| Check | Command / where | Required |
|---|---|---|
| Spot quota landed | Service Quotas → EC2 → `L-3819A6DF`, **in `us-east-1`** | applied value ≥ 4 |
| $5 budget exists | Billing → Budgets | 40/80/100% actual + 100% forecasted |
| Kaggle decode rate known | `kaggle_t4_results.json` | needed to size §5 |
| Region | `aws configure get region` | `us-east-1` |

```bash
aws sts get-caller-identity          # confirm you are in the right account
aws service-quotas get-service-quota --service-code ec2 --quota-code L-3819A6DF \
    --region us-east-1 --query 'Quota.Value'
```

**The unit is vCPUs, not instances.** `g4dn.xlarge` is exactly 4, so a value of `4.0` permits
exactly one instance — which is all this needs.

### The decision you must make before §2

The instance has to get the weights from somewhere, and the two routes differ in what they
put on the box:

| Route | What it costs | What it puts on the instance | IAM needed |
|---|---|---|---|
| **A — S3 mirror** *(the plan's choice)* | one ~4.9 GB upload from home, then $0.03 of storage | nothing secret | yes, an instance profile |
| **B — pull from HuggingFace** | $0.00, no upload | **`HF_TOKEN` in a file on a rented box** | no |

Route A is why S3 is in this project at all — it is a stated learning goal, and the real
secondary benefit found in Phase 1 is that `HF_TOKEN` never has to live on EC2. It needs an
**IAM role**, which is free but is a *third* AWS service. `CLAUDE.md` says ask before adding
one, so **confirm route A explicitly before running §2**, or take route B and skip to §3.

Route B is not wrong — it is just a token on a machine you are renting for three hours and
then destroying. Decide deliberately rather than by whichever section you read first.

---

## 1. Cost, restated

| Item | Rate | Est. usage | Est. cost |
|---|---|---|---|
| `g4dn.xlarge` **spot** (T4, 16 GB) | ~$0.32/hr | 3 hr | $0.96 |
| EBS gp3 root, 100 GB | $0.08/GB-month | 3 hr | $0.03 |
| S3, ~6 GB | $0.023/GB-month | 7 days | $0.03 |
| EC2 → S3, same region | free | — | $0.00 |
| **Total** | | | **~$1.05** |

On-demand would be $0.526/hr. **Spot only** — the run is short and rerunnable, so interruption
is an inconvenience, not a loss. Worst case with one failed run and a redo: ~$3.

**There is no image registry in this plan.** ECR is a fourth AWS service and is not
authorised, so the image is **built on the instance** rather than pulled. That means the
"~4.8 GB compressed pull" figure from the Phase 1 shrink is *not* the number that costs paid
minutes here — nothing pulls that image. What costs minutes is the base-image pull plus pip,
roughly **8–12 minutes (~$0.06)**. The `-base` switch still pays off, just through a different
channel: a ~400 MB base pull instead of ~2 GB.

---

## 2. S3 — mirror the weights *(route A only)*

The weights are already on your laptop, inside the `cosmos-models` Docker volume, as the
original fp16 safetensors. NF4 is applied at load time, so the cache holds exactly what a T4
needs. Nothing has to be re-downloaded from HuggingFace.

```bash
BUCKET=cosmos-edge-serve-$(aws sts get-caller-identity --query Account --output text)
aws s3 mb "s3://$BUCKET" --region us-east-1
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
echo "$BUCKET" > .bucket-name    # gitignored; teardown reads it
```
*(undo: `aws s3 rb "s3://$BUCKET" --force`)*

Sync straight out of the Docker volume — no host copy, so this does not need 5 GB of free
space on `C:`:

```bash
docker run --rm \
  -v cosmos-models:/models:ro \
  -v "$HOME/.aws:/root/.aws:ro" \
  amazon/aws-cli s3 sync /models "s3://$BUCKET/models" --region us-east-1
```

**This is a slow upload on a home connection** — ~4.9 GB, so budget an hour or more. It costs
nothing and it happens *before* any instance exists, which is the point: it is unpaid time.

Verify:
```bash
aws s3 ls "s3://$BUCKET/models/" --recursive --summarize | tail -3
```

### Instance profile *(route A only)*

```bash
cat > /tmp/trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON

aws iam create-role --role-name cosmos-bench-s3-read \
  --assume-role-policy-document file:///tmp/trust.json

cat > /tmp/policy.json <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
 {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::$BUCKET"}]}
JSON

aws iam put-role-policy --role-name cosmos-bench-s3-read \
  --policy-name read-weights --policy-document file:///tmp/policy.json

aws iam create-instance-profile --instance-profile-name cosmos-bench
aws iam add-role-to-instance-profile --instance-profile-name cosmos-bench \
  --role-name cosmos-bench-s3-read
```

Read-only, and scoped to this one bucket. An instance you are about to hand a public IP has
no business holding write credentials.

*(undo, in this order — IAM refuses to delete a role that is still referenced:)*
```bash
aws iam remove-role-from-instance-profile --instance-profile-name cosmos-bench \
  --role-name cosmos-bench-s3-read
aws iam delete-instance-profile --instance-profile-name cosmos-bench
aws iam delete-role-policy --role-name cosmos-bench-s3-read --policy-name read-weights
aws iam delete-role --role-name cosmos-bench-s3-read
```

---

## 3. Network and key

```bash
MYIP=$(curl -s https://checkip.amazonaws.com)
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
      --query 'Vpcs[0].VpcId' --output text)

SG=$(aws ec2 create-security-group --group-name cosmos-bench \
     --description "cosmos-edge-serve benchmark, SSH only" --vpc-id "$VPC" \
     --query GroupId --output text)

aws ec2 authorize-security-group-ingress --group-id "$SG" \
  --protocol tcp --port 22 --cidr "$MYIP/32"
```
*(undo: `aws ec2 delete-security-group --group-id "$SG"` — only works once the instance is gone)*

**Port 8000 is never opened.** k6 runs on the instance against `localhost`, so the service does
not need to be reachable from the internet, and a public inference endpoint on a box holding
credentials is a liability with no upside. This is also what keeps home-network jitter out of
p95 — see `docs/CLAUDE.md`, "Load generation runs on the EC2 instance".

```bash
aws ec2 create-key-pair --key-name cosmos-bench \
  --query KeyMaterial --output text > ~/.ssh/cosmos-bench.pem
chmod 600 ~/.ssh/cosmos-bench.pem
```
*(undo: `aws ec2 delete-key-pair --key-name cosmos-bench && rm ~/.ssh/cosmos-bench.pem`)*

`*.pem` is already in `.gitignore`. Confirm it did not land in the repo directory.

---

## 4. Launch the spot instance

Find the AMI rather than hardcoding an ID — they are region-specific and are re-published
often:

```bash
aws ec2 describe-images --owners amazon --region us-east-1 \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name,CreationDate]' --output text
```

Read the name it prints before using it. This AMI ships the NVIDIA driver, Docker, and the
container toolkit preinstalled — installing those by hand is 15+ minutes of **paid** GPU time
for something the AMI gives away.

```bash
AMI=<the id printed above>

aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type g4dn.xlarge \
  --key-name cosmos-bench \
  --security-group-ids "$SG" \
  --instance-market-options 'MarketType=spot' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --iam-instance-profile Name=cosmos-bench \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cosmos-bench},{Key=Project,Value=cosmos-edge-serve}]' \
  --query 'Instances[0].InstanceId' --output text
```
*(undo: `aws ec2 terminate-instances --instance-ids "$IID"`)*

Notes on the flags, because each is a decision:

- **No `--spot-price`.** Omitting it means "pay up to on-demand", which is the *low*
  interruption setting. A low max price does not save money — spot bills the market rate
  regardless — it only makes you likelier to be evicted mid-benchmark and pay twice.
- **`DeleteOnTermination: true`** — the single most common way an AWS bill outlives a project
  is an orphaned EBS volume. Teardown verifies it anyway.
- **`--iam-instance-profile`** — drop this line entirely on route B.
- **100 GB gp3** — the unpacked image is ~8.7 GB, weights ~5 GB, plus the OS and the build
  cache. 100 GB is $0.03 for the run; sizing it tight to save a cent is not worth a
  disk-full at hour two.

```bash
IID=<the instance id>
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$IID $IP" | tee .instance-info      # gitignored; teardown reads it
ssh -i ~/.ssh/cosmos-bench.pem ubuntu@"$IP"
```

> ⏱️ **The meter is now running.** Everything from here is billed. If something goes wrong,
> the cheap move is almost always to terminate and start again, not to debug on the clock.

---

## 5. On the instance

```bash
nvidia-smi          # must show Tesla T4. If it does not, terminate — you are on the wrong box.
docker --version

git clone https://github.com/AadiPathak23/cosmos-edge-serve.git
cd cosmos-edge-serve
```

### The `.env` — and the one value that is not a safety limit

```bash
cat > .env <<'ENV'
COSMOS_QUANT=none
COSMOS_DTYPE=float16
COSMOS_DEVICE=cuda
COSMOS_MAX_VISION_TOKENS=4096
COSMOS_MAX_QUEUE_DEPTH=32

# NOT a safety limit — do not "tune" this back down.
#
# COSMOS_REQUEST_TIMEOUT_S covers queue wait PLUS compute (app/queue.py). With one
# serial worker, the last of N concurrent requests waits roughly (N-1) x per-request
# before it even starts. At profile B (1024 tokens) and 8 VUs, the default of 300
# returns 504 at every decode rate plausible for a T4 — 10, 15 and 25 tok/s all
# overrun — and the benchmark would publish timeouts instead of latencies.
#
# 900 covers 8 VUs at ~10 tok/s. Recompute from the Kaggle figure:
#     timeout >= VUs x (max_new_tokens / measured_tok_s) x 1.3
COSMOS_REQUEST_TIMEOUT_S=900
ENV
```

Route B only — otherwise the token does not belong here:
```bash
echo "HF_TOKEN=hf_..." >> .env
```

### Weights *(route A)*

```bash
docker volume create cosmos-models
BUCKET=<your bucket>
docker run --rm -v cosmos-models:/models amazon/aws-cli \
  s3 sync "s3://$BUCKET/models" /models --region us-east-1
```

The instance profile supplies credentials automatically — there is nothing to configure and
no key to leak. On route B, skip this: the container downloads from HuggingFace on first start.

### Build and start

```bash
docker compose up --build -d      # ~8-12 min: base pull + pip, then the model load
docker compose logs -f            # watch for the banner
```

### 🛑 Assert the banner before spending another minute

| Field | Required |
|---|---|
| model class | `Qwen3VLForConditionalGeneration` |
| device / GPU | `cuda:0` — `Tesla T4`, sm_75 |
| dtype | `float16` |
| quantization | `none` |
| params total | **2,127,532,032** |

The param count is **not** the model card's 2,438,696,960: `lm_head` is tied to
`embed_tokens` and `model.parameters()` deduplicates. The tying is architectural, so fp16 on a
T4 must report the identical figure the 3060 did. **A mismatch here is a finding, not a
rounding difference — stop and investigate rather than benchmarking it.**

A `quantization` of anything but `none` means the `.env` did not take effect, and every number
you are about to produce would be an NF4 number wearing a T4 label.

### Test media

```bash
docker run --rm -v "$PWD:/work" -w /work cosmos-edge-serve:0.1.0 \
  python scripts/make_assets.py
```

Bind-mounts the repo, because **compose mounts only `cosmos-models`** — files written by
`docker compose exec` land inside the container and are invisible to k6 on the host. This
exact trap already cost a debugging cycle on the smoke test.

### Smoke test before load

```bash
python3 scripts/smoke_test.py
```

Stdlib only, so the instance's system python3 runs it. Both legs must pass. Compare the answers
against the Phase 1 run — under greedy decoding the 3060/NF4 output was byte-stable across
rebuilds, so a *precision* change is the one variable left, and this is the moment it shows up.

---

## 6. The sweep

```bash
mkdir -p loadtest/results

run () {  # profile vus duration
  docker run --rm --network host \
    -v "$PWD:/work" -w /work \
    -e BASE_URL=http://localhost:8000 \
    -e PROFILE="$1" -e VUS="$2" -e DURATION="$3" \
    -e OUT="loadtest/results/$1-vu$2.json" \
    grafana/k6 run loadtest/load.js
}

run a 1 5m ; run a 4 5m ; run a 8 5m
run b 1 8m ; run b 4 8m ; run b 8 8m
```

`--network host` so `localhost:8000` reaches the service; the port stays closed to the world.

### Wall clock, and why it can overrun

The script uses `constant-vus` with a fixed **duration**, not a fixed iteration count, because
wall clock is what costs money and a fixed iteration count has unbounded cost if the model is
slower than assumed.

In-flight requests are allowed to finish (`GRACEFUL_STOP=900s`). Cutting them off would
discard exactly the requests sitting behind the deepest queue and would flatter p95 by
construction. The price is a bounded overrun:

> worst-case overrun per cell ≈ **VUs × per-request latency**

At an assumed 15 tok/s (**provisional — replace with the Kaggle figure**):

| Cell | Duration | Per-request | Worst overrun | Budget |
|---|---|---|---|---|
| A @ 1 / 4 / 8 | 5m each | ~17 s | ~2 min total | **~17 min** |
| B @ 1 / 4 / 8 | 8m each | ~68 s | ~14 min total | **~38 min** |

Plus build ~12 min, weight sync ~5 min, load + warmup ~3 min, smoke ~2 min.
**Total ≈ 1 h 20 m of the 3 h budgeted** — the slack is deliberate, and it is what a spot
interruption or one bad `.env` eats into.

> **Abort rule:** if any single cell has not finished within **2× its budgeted duration**, kill
> it and drop that cell rather than letting it run. A missing row is honest; an unbudgeted hour
> is a third of the project's ceiling.

### Watch it

```bash
curl -s localhost:8000/health | python3 -m json.tool | head -20
curl -s localhost:8000/metrics | grep -E 'cosmos_(requests_total|queue_depth)'
```

---

## 7. Render and get the results off the box

```bash
python3 loadtest/render_results.py loadtest/results --out docs/BENCHMARK.md
```

Read the **Caveats** block first. If it reports 504s or fewer than 20 samples in a cell, the
table below it is not publishable as-is — fix the timeout or lengthen the duration and rerun
that cell. That block exists because a run full of timeouts still produces a table of
plausible-looking numbers.

Pull the artifacts down **before** terminating — this is the only thing on the instance worth
keeping:

```bash
# from the laptop
scp -i ~/.ssh/cosmos-bench.pem -r \
  ubuntu@"$IP":cosmos-edge-serve/loadtest/results ./loadtest/results
scp -i ~/.ssh/cosmos-bench.pem \
  ubuntu@"$IP":cosmos-edge-serve/docs/BENCHMARK.md ./docs/BENCHMARK.md
```

Also save the load report and the container log — a benchmark without its hardware provenance
is just a number:

```bash
ssh -i ~/.ssh/cosmos-bench.pem ubuntu@"$IP" \
  'cd cosmos-edge-serve && curl -s localhost:8000/health' > loadtest/results/health.json
```

---

## 8. Tear down — the same day

**→ `docs/TEARDOWN.md`.** Do not defer it. Do not trust memory that you did it; the teardown
doc verifies each step in the console, because "I'm sure I terminated it" is how a $1 project
becomes a $40 one.
