#!/usr/bin/env python3
"""Turn k6 summary JSON into the committed benchmark tables.

    python loadtest/render_results.py loadtest/results
    python loadtest/render_results.py loadtest/results --out docs/BENCHMARK.md

Standard library only, for the same reason as scripts/smoke_test.py: it has to run
inside the runtime container and on a bare EC2 box, neither of which has the dev
dependencies installed.

Input is whatever `handleSummary()` in loadtest/load.js wrote — one file per cell
of the grid. Each file carries its own config and its own /health load report, so
the tables cannot be assembled out of mismatched runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Order the profiles are presented in. A is the headline (comparable p50/p95),
# B is the realistic one (full chain-of-thought). Both get published — reporting
# only A would misrepresent the model, only B would be a noisier number for more
# money. See docs/CLAUDE.md, "Benchmark shape".
PROFILE_ORDER = ["a", "b"]

# Deliberately no token count in these titles. The count is read from the run's
# own config instead: a dry run at 32 tokens under a heading that says 256 is
# exactly the kind of mislabelled result this repo exists to avoid publishing.
PROFILE_TITLES = {
    "a": "Profile A — headline",
    "b": "Profile B — realistic, full chain-of-thought",
}


def load_summaries(directory: Path) -> list[dict[str, Any]]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json summaries in {directory}. Did the k6 sweep run?")

    summaries = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  skipping {path.name}: not valid JSON ({exc})", file=sys.stderr)
            continue
        if "config" not in data:
            print(
                f"  skipping {path.name}: no config block — not a load.js summary",
                file=sys.stderr,
            )
            continue
        data["_source"] = path.name
        summaries.append(data)

    if not summaries:
        raise SystemExit(f"No usable summaries in {directory}.")
    return summaries


def _ms(values: dict[str, Any] | None, key: str) -> float | None:
    if not values or values.get(key) is None:
        return None
    return float(values[key])


def _secs(values: dict[str, Any] | None, key: str) -> str:
    value = _ms(values, key)
    return "—" if value is None else f"{value / 1000:.1f}"


def _num(values: dict[str, Any] | None, key: str, fmt: str = "{:.1f}") -> str:
    value = _ms(values, key)
    return "—" if value is None else fmt.format(value)


def throughput_per_min(summary: dict[str, Any]) -> str:
    """Completed requests per minute of wall clock.

    Deliberately computed from wall clock rather than from the mean latency:
    with one serial worker those diverge, and wall clock is the number that
    corresponds to what the service actually delivered.
    """
    completed = summary["counts"]["completed"]
    wall_ms = summary.get("throughput", {}).get("wall_clock_ms")
    if not wall_ms:
        return "—"
    return f"{completed / (wall_ms / 60000):.2f}"


def render_table(summaries: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| VUs | Completed | p50 (s) | p95 (s) | req/min | tok/s (med) | "
        "queue p50 (s) | queue p95 (s) | generate p50 (s) | 503 | 504 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(summaries, key=lambda s: s["config"]["vus"]):
        latency = summary.get("latency_ms", {})
        counts = summary["counts"]
        rows.append(
            "| {vus} | {done} | {p50} | {p95} | {rpm} | {tps} | {q50} | {q95} | {g50} | "
            "{r503} | {r504} |".format(
                vus=summary["config"]["vus"],
                done=counts["completed"],
                p50=_secs(latency.get("e2e"), "med"),
                p95=_secs(latency.get("e2e"), "p(95)"),
                rpm=throughput_per_min(summary),
                tps=_num(summary.get("throughput", {}).get("tokens_per_second"), "med"),
                q50=_secs(latency.get("queue_wait"), "med"),
                q95=_secs(latency.get("queue_wait"), "p(95)"),
                g50=_secs(latency.get("generate"), "med"),
                r503=counts["queue_full_503"],
                r504=counts["timeout_504"],
            )
        )
    return rows


def render_hardware(summaries: list[dict[str, Any]]) -> list[str]:
    """The load report, so a table can never be separated from its hardware.

    A p95 without the GPU, dtype and quantization beside it is not a benchmark
    result, it is a number. This is also the block that would catch a run that
    silently fell back to NF4 or to a GPU other than the one being claimed.
    """
    reports = [s.get("model") for s in summaries if s.get("model")]
    if not reports:
        return ["> ⚠️ No load report captured — these numbers have no recorded hardware.", ""]

    report = reports[0]
    # Stringified before going into the set so sorted() cannot trip over a None
    # sitting next to a str (gpu_name is None on CPU).
    distinct = {
        (str(r.get("gpu_name")), str(r.get("dtype")), str(r.get("quantization")))
        for r in reports
    }
    lines = [
        "| | |",
        "|---|---|",
        f"| GPU | `{report.get('gpu_name') or 'CPU'}` |",
        f"| Model class | `{report.get('model_class')}` |",
        f"| Device | `{report.get('device')}` |",
        f"| dtype | `{report.get('dtype')}` |",
        f"| Quantization | `{report.get('quantization')}` |",
        f"| Attention | `{report.get('attention')}` |",
        f"| Params | {report.get('params', {}).get('total', 0):,} |",
        "",
    ]
    if len(distinct) > 1:
        mismatch = [
            "> ⚠️ **These summaries did not all come from the same configuration.**",
            "> Do not read the tables below as a single sweep:",
            ">",
        ]
        mismatch += [
            f"> - `{gpu}` / `{dtype}` / `{quant}`" for gpu, dtype, quant in sorted(distinct)
        ]
        mismatch.append("")
        lines = mismatch + lines
    return lines


def render_warnings(summaries: list[dict[str, Any]]) -> list[str]:
    """Anything that makes the numbers above less than they appear.

    Printed above the tables, not in a footnote. A run where most requests timed
    out still produces a table full of plausible-looking figures, and the failure
    mode this guards against is publishing it as if it were latency data.
    """
    lines: list[str] = []
    for summary in summaries:
        config = summary["config"]
        counts = summary["counts"]
        label = f"profile {config['profile'].upper()} @ {config['vus']} VU"
        completed = counts["completed"]

        if counts["hard_errors"]:
            lines.append(
                f"- ❌ **{label}: {counts['hard_errors']} hard errors** — the service faulted."
            )
        if counts["timeout_504"]:
            total = completed + counts["timeout_504"]
            share = counts["timeout_504"] / total * 100 if total else 0
            lines.append(
                f"- ⚠️ **{label}: {counts['timeout_504']} requests ({share:.0f}%) hit the "
                "server timeout.** Raise `COSMOS_REQUEST_TIMEOUT_S`; the p95 below is a "
                "floor, not the real value."
            )
        if counts["queue_full_503"]:
            lines.append(
                f"- {label}: {counts['queue_full_503']} requests rejected with 503 "
                "(queue at capacity). Expected under overload, but it caps offered load."
            )
        if counts["truncated"]:
            lines.append(
                f"- {label}: {counts['truncated']} responses hit `max_new_tokens` mid-reasoning."
            )
        if completed < 20:
            lines.append(
                f"- ⚠️ **{label}: only {completed} samples.** p95 from this few requests is "
                "not a stable estimate — lengthen `DURATION` before publishing."
            )

    return (["### Caveats", ""] + lines + [""]) if lines else []


def render(summaries: list[dict[str, Any]]) -> str:
    out = [
        "# Benchmark — cosmos-edge-serve",
        "",
        "Generated by `loadtest/render_results.py`. Do not hand-edit; re-run the renderer.",
        "",
        "Load was generated by k6 **on the instance, against `localhost`**, so no WAN",
        "latency is folded into these samples. One GPU worker serves requests serially,",
        "so latency above 1 VU is mostly queue wait — that decomposition is broken out",
        "in its own columns rather than hidden inside the end-to-end figure.",
        "",
        "## Hardware",
        "",
    ]
    out += render_hardware(summaries)
    out += render_warnings(summaries)

    by_profile: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_profile.setdefault(summary["config"]["profile"], []).append(summary)

    known = [p for p in PROFILE_ORDER if p in by_profile]
    extra = sorted(p for p in by_profile if p not in PROFILE_ORDER)

    for profile in known + extra:
        group = by_profile[profile]
        tokens = group[0]["config"]["max_new_tokens"]
        title = PROFILE_TITLES.get(profile, f"Profile {profile.upper()}")
        out += ["", f"## {title}", ""]

        # Rows in one table must be the same experiment at different concurrency.
        # Mixing token budgets or durations would put incomparable numbers in
        # adjacent rows under a single heading, which reads as a clean result and
        # is not one.
        mixed = {
            key: sorted({str(s["config"][key]) for s in group})
            for key in ("max_new_tokens", "media", "duration")
        }
        for key, values in mixed.items():
            if len(values) > 1:
                out += [
                    f"> ⚠️ **Rows below differ in `{key}` ({', '.join(values)}).** "
                    "They are not a single sweep and must not be compared as one.",
                    "",
                ]

        out += [
            f"`max_new_tokens={tokens}`, media `{group[0]['config']['media']}`, "
            f"duration `{group[0]['config']['duration']}` per level.",
            "",
        ]
        out += render_table(group)
        out.append("")

    out += [
        "",
        "---",
        "",
        "Source summaries: " + ", ".join(f"`{s['_source']}`" for s in summaries),
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="loadtest/results",
        help="Directory of k6 summary JSON files (default: loadtest/results)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Also write the markdown here, e.g. docs/BENCHMARK.md",
    )
    args = parser.parse_args()

    # The markdown carries ⚠️/❌ markers, and a Windows console defaults to
    # cp1252, which cannot encode them — printing would die with a
    # UnicodeEncodeError before writing a single table. The file itself is always
    # written as UTF-8; only the console needs persuading.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    summaries = load_summaries(Path(args.results_dir))
    markdown = render(summaries)

    print(markdown)

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8", newline="\n")
        print(f"\nwrote {destination}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
