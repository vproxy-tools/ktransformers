"""Persistent hugepage weight cache helper (python side).

Mirrors kt-kernel/cpu_backend/hugepage_weights.hpp: resident MXFP4 expert
weights live in per-NUMA hugetlbfs files plus small ".done" markers on the
regular filesystem. Before loading a layer from safetensors, we ask this
module whether the C++ side would find every NUMA segment valid — if yes,
loading is skipped entirely and the C++ side just mmaps the hugepages.
"""

from __future__ import annotations

import ctypes
import hashlib
import os

_HUGETLBFS_MAGIC = 0x958458F6
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h = ((h ^ b) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def data_root() -> str:
    return os.environ.get("KT_HUGEPAGE_WEIGHT_DIR", "/dev/hugepages/kt_weights")


def meta_root() -> str:
    return os.environ.get("KT_HUGEPAGE_WEIGHT_META_DIR", "/var/lib/kt-hugepage-weights")


def enabled() -> bool:
    if os.environ.get("KT_HUGEPAGE_WEIGHTS", "1") == "0":
        return False
    try:
        st = os.statvfs(data_root())
        if st.f_bsize == 0:
            return False
        # f_bsize of statvfs on hugetlbfs reports the huge page size; use
        # statfs via ctypes for the filesystem magic.
        class StatFs(ctypes.Structure):
            _fields_ = [("type", ctypes.c_long)] + [
                (n, ctypes.c_long) for n in ("bsize", "blocks", "bfree", "bavail", "files", "ffree", "fsid0", "fsid1", "namelen", "frsize", "flags", "spare0", "spare1", "spare2", "spare3", "spare4")
            ]

        libc = ctypes.CDLL(None, use_errno=True)
        buf = StatFs()
        if libc.statfs(ctypes.c_char_p(data_root().encode()), ctypes.byref(buf)) != 0:
            return False
        return buf.type & 0xFFFFFFFF == _HUGETLBFS_MAGIC
    except OSError:
        return False


def ensure_model_stamp(weight_path: str) -> str:
    """Set KT_HP_MODEL_STAMP (read by the C++ side) once per process.

    The stamp ties the hugepage cache to a specific checkpoint directory: any
    change of path or config.json content invalidates every marker.
    """
    stamp = os.environ.get("KT_HP_MODEL_STAMP")
    if stamp:
        return stamp
    cfg = os.path.join(weight_path, "config.json")
    h = hashlib.sha256()
    h.update(os.path.realpath(weight_path).encode())
    if os.path.isfile(cfg):
        with open(cfg, "rb") as f:
            h.update(f.read())
    stamp = h.hexdigest()[:32]
    os.environ["KT_HP_MODEL_STAMP"] = stamp
    return stamp


def _parse_marker(path: str) -> dict | None:
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    if not lines or lines[0] != "KTHP1":
        return None
    out = {}
    for line in lines[1:]:
        k, _, v = line.partition("=")
        out[k] = v
    return out


def layer_key(layer_idx: int, num_experts: int, hidden: int, inter_tp: int, group_size: int, bits: int) -> str:
    return f"L{layer_idx}_E{num_experts}_H{hidden}_I{inter_tp}_g{group_size}_b{bits}"


def python_fingerprint(stamp: str, backend_tag: str, key: str) -> int:
    s = f"kp1|{stamp}|{backend_tag}|{key}"
    return fnv1a64(s.encode())


def map_hash(map_tensor) -> int:
    """Fingerprint the physical_to_logical map exactly like the C++ side:
    FNV-1a over expert_num little-endian uint64 entries."""
    import torch

    t = map_tensor
    if t is None:
        return 0
    t = t.to(torch.int64).contiguous().cpu()
    return fnv1a64(t.numpy().tobytes())


def check_reusable(
    layer_idx: int,
    num_experts: int,
    hidden: int,
    inter_tp: int,
    group_size: int,
    bits: int,
    backend_tag: str,
    map_tensor,
) -> bool:
    """True when every NUMA node has a valid marker + data segment for this layer."""
    if not enabled():
        return False
    stamp = os.environ.get("KT_HP_MODEL_STAMP", "")
    key = layer_key(layer_idx, num_experts, hidden, inter_tp, group_size, bits)
    pfp = f"{python_fingerprint(stamp, backend_tag, key):x}"
    mhash = f"{map_hash(map_tensor):016x}"
    meta = meta_root()
    try:
        nodes = [d for d in os.listdir(meta) if d.startswith("node")]
    except OSError:
        return False
    if not nodes:
        return False
    for nd in nodes:
        m = _parse_marker(os.path.join(meta, nd, key + ".done"))
        if m is None:
            return False
        if m.get("pfp") != pfp or m.get("map", "").split("=")[-1] != mhash:
            return False
        try:
            offset, size = int(m["offset"]), int(m["size"])
        except (KeyError, ValueError):
            return False
        data_file = os.path.join(data_root(), nd, "weights.bin")
        try:
            if os.path.getsize(data_file) < offset + size:
                return False
        except OSError:
            return False
    return True
