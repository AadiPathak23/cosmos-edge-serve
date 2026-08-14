# Teardown

> **Run this the same day as the benchmark.** Not tomorrow, not "once I've looked at the
> numbers". A `g4dn.xlarge` left running costs **~$7.70/day** — most of this project's $10
> ceiling, spent on nothing.

Every step has a **verify** command. The rule for this file is: *confirm in the console or the
API, never from memory.* "I'm sure I terminated it" is the single most expensive sentence in
cloud computing, and the AWS bill arrives weeks later, which is far too late to notice.

The order matters — AWS refuses to delete things that are still referenced.

---

## 0. First, get your results off the box

If you have not already done §7 of `docs/EC2.md`, do it **now**. Everything below is
irreversible, and the instance holds the only copy of the run:

```bash
scp -i ~/.ssh/cosmos-bench.pem -r ubuntu@"$IP":cosmos-edge-serve/loadtest/results ./loadtest/results
scp -i ~/.ssh/cosmos-bench.pem ubuntu@"$IP":cosmos-edge-serve/docs/BENCHMARK.md ./docs/BENCHMARK.md
```

Confirm the files exist locally and are non-empty before continuing.

```bash
ls -la loadtest/results/ && head -30 docs/BENCHMARK.md
```

---

## 1. Terminate the instance — the one that actually costs money

```bash
IID=$(cut -d' ' -f1 .instance-info)
aws ec2 terminate-instances --instance-ids "$IID"
aws ec2 wait instance-terminated --instance-ids "$IID"
```

**Verify** — the state must read `terminated`, not `stopping`, not `stopped`:

```bash
aws ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```

> A **stopped** instance still bills for its EBS volume, and a stopped *spot* instance is not
> a thing you should be relying on at all. Only `terminated` stops the meter.

Then confirm nothing else is running in the region — including anything from an earlier
attempt you have forgotten about:

```bash
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=instance-state-name,Values=running,pending,stopping,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text
```

**Expected output: nothing.** Any line here is money being spent right now.

---

## 2. Verify the EBS volume actually went with it

`DeleteOnTermination=true` was set at launch, but *verify it rather than trust it* — an
orphaned 100 GB gp3 volume is ~$8/month, which would quietly exceed this project's entire
budget while the instance that created it no longer exists.

```bash
aws ec2 describe-volumes --region us-east-1 \
  --filters Name=status,Values=available \
  --query 'Volumes[].[VolumeId,Size,CreateTime]' --output text
```

**Expected: nothing.** `available` means "attached to no instance" — i.e. orphaned. Delete
anything listed that belongs to this project:

```bash
aws ec2 delete-volume --volume-id <vol-id>
```

---

## 3. Snapshots and AMIs

Nothing in `docs/EC2.md` creates either, so this is a check rather than a step. Do it anyway —
snapshots are the classic thing that survives a teardown, and they bill forever.

```bash
aws ec2 describe-snapshots --owner-ids self --region us-east-1 \
  --query 'Snapshots[].[SnapshotId,VolumeSize,StartTime,Description]' --output text
aws ec2 describe-images --owners self --region us-east-1 \
  --query 'Images[].[ImageId,Name]' --output text
```

*(if anything belongs to this project: `aws ec2 deregister-image --image-id <ami>` first, then
`aws ec2 delete-snapshot --snapshot-id <snap>` — a snapshot backing a registered AMI will not
delete.)*

---

## 4. Elastic IPs

Also not created by the runbook, but an unassociated EIP bills ~$3.60/month precisely *because*
it is idle, which makes it the one charge that starts **after** you finish.

```bash
aws ec2 describe-addresses --region us-east-1 \
  --query 'Addresses[].[PublicIp,AllocationId,AssociationId]' --output text
```

*(if any: `aws ec2 release-address --allocation-id <id>`)*

---

## 5. Security group and key pair

These are free, but they only delete cleanly once the instance is gone — which is why they
come after §1.

```bash
aws ec2 delete-security-group --group-id "$SG"
aws ec2 delete-key-pair --key-name cosmos-bench
rm -f ~/.ssh/cosmos-bench.pem
```

**Verify:**
```bash
aws ec2 describe-security-groups --group-names cosmos-bench 2>&1 | tail -1   # expect: not found
```

If the delete fails with a dependency error, the instance is not fully terminated yet. Wait and
retry — do not force anything.

---

## 6. S3 bucket *(route A only)*

The weights are ~5 GB at $0.023/GB-month — about **$0.12/month** if left. Small, but it is a
charge with no end date attached to a project that is finished.

```bash
BUCKET=$(cat .bucket-name)
aws s3 rb "s3://$BUCKET" --force        # --force empties it first
```

**Verify:**
```bash
aws s3 ls | grep cosmos                 # expect: nothing
```

The weights are always re-downloadable from HuggingFace, so nothing irreplaceable is being
destroyed here. Keep the bucket only if a rerun is genuinely planned within days — and if you
do, write down that it is still there.

---

## 7. IAM role and instance profile *(route A only)*

Free, so there is no cost argument. Delete them anyway: a role that can read a bucket which no
longer exists is confusing leftover state, and the order below is the one IAM will accept.

```bash
aws iam remove-role-from-instance-profile --instance-profile-name cosmos-bench \
  --role-name cosmos-bench-s3-read
aws iam delete-instance-profile --instance-profile-name cosmos-bench
aws iam delete-role-policy --role-name cosmos-bench-s3-read --policy-name read-weights
aws iam delete-role --role-name cosmos-bench-s3-read
```

**Verify:**
```bash
aws iam get-role --role-name cosmos-bench-s3-read 2>&1 | tail -1   # expect: NoSuchEntity
```

---

## 8. Keep the $5 budget

**Do not delete it.** It is the thing that catches a teardown which quietly missed something —
and the whole reason this file exists is that teardowns *do* miss things. Removing it on
teardown day would drop the guard at the exact moment it is most likely to be needed.

Delete it only when the project is finished for good, via Billing → Budgets → select → Delete.

---

## 9. Confirm $0 — but not today

Cost Explorer refreshes roughly three times a day and lags real usage by **8–24 hours**, so
checking it five minutes after teardown proves nothing at all. The same lag is why the budget
alert is a backstop and never a kill switch.

**Set a reminder for tomorrow**, then:

```bash
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '3 days ago' +%Y-%m-%d),End=$(date -d tomorrow +%Y-%m-%d) \
  --granularity DAILY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

Or Console → Billing and Cost Management → Cost Explorer, grouped by service.

**What you are looking for:** EC2 and S3 charges that stop on the benchmark day, and a
**$0.00 figure for the day after**. A non-zero number the day after teardown means something
survived — go back through §2, §3 and §4, which is where survivors hide.

---

## Final checklist

- [ ] Results and `BENCHMARK.md` copied to the laptop and verified non-empty
- [ ] Instance state is `terminated`
- [ ] No running/stopped instances anywhere in `us-east-1`
- [ ] No `available` EBS volumes
- [ ] No unexpected snapshots or AMIs
- [ ] No unassociated Elastic IPs
- [ ] Security group and key pair deleted, `.pem` removed from the laptop
- [ ] Bucket emptied and deleted *(route A)*
- [ ] IAM role and instance profile deleted *(route A)*
- [ ] **$5 budget still in place**
- [ ] Reminder set to check Cost Explorer tomorrow
- [ ] Actual total spend recorded in the `docs/CLAUDE.md` CHANGELOG

Record the **real** number in the changelog, whatever it is. A spot interruption or a rerun
makes the actual figure diverge from the ~$1.05 estimate, and the estimate is only worth
anything to the next project if it gets checked against reality.
