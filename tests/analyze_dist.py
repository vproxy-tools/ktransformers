#!/usr/bin/env python3
"""Analyze /tmp/kt-distribute.txt (SIGUSR2 dump of KT expert routing).

Answers two placement questions per layer and in aggregate:
  1. contig28 — share of routed pairs captured by expert ids 0-27
     (what generate_uniform/hybrid residency actually catches today);
  2. top28   — share the 28 hottest experts of that layer catch
     (what a frequency-based placement could catch, same VRAM).

Usage: python3 tests/analyze_dist.py [/tmp/kt-distribute.txt] [--top 28]
Reads the TOTAL matrix (falls back to DELTA if TOTAL absent is impossible;
pass --section DELTA to analyze the window between two USR2 dumps instead).
"""
import argparse
import sys


def load_matrix(path: str, section: str):
    txt = open(path).read()
    marker = section if section != "TOTAL" else "TOTAL"
    if marker not in txt:
        print(f"section {marker} not found in {path}", file=sys.stderr)
        sys.exit(1)
    body = txt.split(marker, 1)[1]
    for stop in ("DELTA_SINCE_LAST_DUMP", "SUMMARY"):
        body = body.split(stop, 1)[0]
    rows = [[int(v) for v in ln.split()] for ln in body.splitlines()
            if ln and ln[0].isdigit()]
    if not rows:
        print("no matrix rows parsed", file=sys.stderr)
        sys.exit(1)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="/tmp/kt-distribute.txt")
    ap.add_argument("--section", default="TOTAL", choices=["TOTAL", "DELTA_SINCE_LAST_DUMP"])
    ap.add_argument("--top", type=int, default=28, help="resident experts per layer")
    args = ap.parse_args()

    mat = load_matrix(args.path, args.section)
    n_experts = len(mat[0])
    k = min(args.top, n_experts)
    print(f"{args.path} [{args.section}] layers={len(mat)} experts={n_experts} top={k}")
    print(f"{'layer':>5} {'pairs':>9} {'contig%':>8} {f'top{k}%':>8}  headroom")
    agg_c = agg_t = agg_p = 0
    hash_layers = {0, 1, 2}  # V4-Flash hash-MoE front layers
    for i, row in enumerate(mat):
        pairs = sum(row)
        if pairs == 0:
            print(f"{i:>5} {0:>9}   (no traffic)")
            continue
        c = sum(row[:k]) / pairs
        t = sum(sorted(row, reverse=True)[:k]) / pairs
        agg_c += sum(row[:k]); agg_t += sum(sorted(row, reverse=True)[:k]); agg_p += pairs
        tag = " hash" if i in hash_layers else ""
        print(f"{i:>5} {pairs:>9} {100*c:>7.1f}% {100*t:>7.1f}%  x{t/c:4.1f}{tag}")
    print("-" * 50)
    print(f"TOTAL {agg_p:>9} {100*agg_c/agg_p:>7.1f}% {100*agg_t/agg_p:>7.1f}%  "
          f"x{agg_t/max(agg_c,1e-9):4.1f}")
    print("\nheadroom = top28/contig28: how many times more routed mass a"
          " frequency-based\nplacement would put on GPU at identical VRAM.")


if __name__ == "__main__":
    main()
