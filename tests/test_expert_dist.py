#!/usr/bin/env python3
"""kt_ep_wrapper expert-dist tracking unit test (GPU + sglang import needed).

Validates the GPU-side counting math, the dump file format (TOTAL/DELTA/
SUMMARY), and that a disabled tracker (_DIST_WANTED False) never allocates
state. Run inside the repo venv:

    python3 tests/test_expert_dist.py

Mirrors what the real server does: _expert_dist_add per layer forward,
_expert_dist_capture_end zeroing, _expert_dist_write_dump on SIGUSR2.
"""
import os
import sys

os.environ["KT_EXPERT_DIST_TRACK"] = "1"

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "sglang", "python"))
from sglang.srt.layers.moe import kt_ep_wrapper as ktw  # noqa: E402

L, E = 8, 32


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU"

    ids = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 31], [5, 5, 5]], dtype=torch.int32, device="cuda"
    )
    for _ in range(3):  # 3 "forwards" of layer 2
        ktw._expert_dist_add(2, ids, L, E)
    assert ktw._DIST_STATE is not None, "state must be created when wanted"
    row = ktw._DIST_STATE["counts"][2].cpu()
    # per forward: expert0 x3, e1 x2, e2 x2, e3 x1, e5 x3, e31 x1; x3 forwards
    assert row[:6].tolist() == [9, 6, 6, 3, 0, 9], row[:6].tolist()
    assert int(row.sum()) == 36 and int(row[31]) == 3

    # capture-end must zero everything (dummy-capture pollution dropped)
    ktw._expert_dist_capture_begin()
    assert ktw._DIST_CAPTURE_DEPTH == 1
    ktw._expert_dist_capture_end()
    assert int(ktw._DIST_STATE["counts"].sum()) == 0
    assert ktw._DIST_STATE["last_dump"] is None

    # layer 0 full-layer style traffic + dump round trip
    ktw._expert_dist_add(0, torch.full((4, 6), 7, dtype=torch.int64, device="cuda"), L, E)
    ktw._expert_dist_write_dump()
    text1 = open(ktw._DIST_DUMP_PATH).read()
    assert "TOTAL" in text1 and "SUMMARY" in text1 and "l000 pairs=24" in text1
    assert "DELTA" not in text1, "first dump has no delta"

    ktw._expert_dist_add(0, torch.zeros((2, 6), dtype=torch.int64, device="cuda"), L, E)
    ktw._expert_dist_write_dump()
    text2 = open(ktw._DIST_DUMP_PATH).read()
    assert "DELTA_SINCE_LAST_DUMP" in text2
    sections = text2.split("DELTA_SINCE_LAST_DUMP")
    total_rows = [
        ln for ln in sections[0].splitlines()[5:] if ln and ln[0].isdigit()
    ]
    assert len(total_rows) == L, f"TOTAL matrix rows {len(total_rows)} != {L}"
    delta_rows = [
        ln for ln in sections[1].splitlines() if ln and ln[0].isdigit() and " " in ln
    ]
    assert len(delta_rows) == L
    assert delta_rows[0].split()[0] == "12", "delta row0 expert0 should gain 12"
    assert "l000 pairs=36" in text2, "TOTAL summary should show 24+12 pairs"

    # disabled mode: fresh module-level state must stay None
    ktw._DIST_STATE = None
    ktw._DIST_WANTED = False
    ktw._expert_dist_add(1, ids, L, E)
    assert ktw._DIST_STATE is None, "disabled tracker must not allocate"
    print("PASS")


if __name__ == "__main__":
    main()
