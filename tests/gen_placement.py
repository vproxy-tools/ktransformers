#!/usr/bin/env python3
"""Convert a KT expert hit-probability CSV into a --kt-expert-placement-map spec.

Input CSV columns: layer,expert,hits,hit_prob,share_of_layer_pairs
(produced from the /tmp/kt-distribute.txt SIGUSR2 dump; one row per
(layer, expert), layer = MoE layer ordinal).

Selection: layers in --full-layers get ALL experts on GPU ('F'); the
remaining budget (--max-experts, counting only non-full layers) is filled by
global greedy on share_of_layer_pairs. Every MoE layer routes identical
total mass, so shares are comparable across layers and greedy-by-share is
the max-coverage assignment; measured vs per-layer equal quota the
difference is <0.5pt (DSv4F-Opt.md §5.12).

The spec string goes to stdout (shell-substitution friendly), stats to stderr.

Usage:
    SPEC=$(python3 tests/gen_placement.py /tmp/expert_hit_probs.csv \
           --max-experts 1579 --full-layers 0,1,2)
    EXTRA_ARGS="--kt-expert-placement-strategy custom \
      --kt-expert-placement-map $SPEC" DSPARK=1 ./run_dspark.sh
"""
import argparse
import csv
import sys
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="/tmp/expert_hit_probs.csv")
    ap.add_argument("--max-experts", type=int, required=True,
                    help="resident-expert budget for non-full layers")
    ap.add_argument("--full-layers", default="0,1,2",
                    help="comma-separated MoE layer ordinals placed fully on GPU")
    args = ap.parse_args()

    shares = defaultdict(dict)  # layer -> {expert: share}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            shares[int(r["layer"])][int(r["expert"])] = float(r["share_of_layer_pairs"])

    full = set()
    if args.full_layers.strip():
        full = {int(v) for v in args.full_layers.split(",")}
    for L in full:
        if L not in shares:
            ap.error(f"full layer {L} not present in {args.csv}")

    partial_layers = [L for L in sorted(shares) if L not in full]

    # Global greedy: top --max-experts experts by share across partial layers.
    pool = sorted(
        ((s, L, e) for L in partial_layers for e, s in shares[L].items()),
        reverse=True,
    )
    picked = pool[: args.max_experts]
    by_layer = defaultdict(list)
    for s, L, e in picked:
        by_layer[L].append(e)

    spec_parts = [f"{L}=F" for L in sorted(full)]
    for L in partial_layers:
        ids = sorted(by_layer.get(L, []))
        if ids:
            spec_parts.append(f"{L}={'-'.join(str(e) for e in ids)}")
    spec = ",".join(spec_parts)

    n_layers = len(shares)
    captured = len(full) + sum(s for s, _, _ in picked)
    cutoff = picked[-1][0] if picked else 0.0
    counts = {L: len(by_layer.get(L, [])) for L in partial_layers}
    print(
        f"# full layers: {sorted(full)}; partial budget: {args.max_experts} experts "
        f"across {len(partial_layers)} layers (min {min(counts.values())}, "
        f"max {max(counts.values())} per layer)\n"
        f"# captured routed mass: {100.0 * captured / n_layers:.1f}% of total "
        f"(cutoff share {cutoff:.5f})\n"
        f"# VRAM estimate: {(len(full) * 256 + len(picked)) * 12.6 / 1024:.1f} GiB "
        f"of expert weights (12.6 MiB/expert)",
        file=sys.stderr,
    )
    print(spec)


if __name__ == "__main__":
    main()
