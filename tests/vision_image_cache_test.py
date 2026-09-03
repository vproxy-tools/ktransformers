#!/usr/bin/env python3
"""Offline unit tests for the dsv4-vision URL image disk cache.

Covers sglang/srt/multimodal/processors/deepseek_v4_vision.py:
  1. http(s) download is cached on disk keyed by sha256(url-sans-query) —
     second fetch of the same URL is served from disk without touching the
     network; presigned URLs differing only in the query share one entry
     (the download itself still uses the full URL).
  2. _IMAGE_MAX_BYTES cap aborts oversized downloads (no partial tmp left).
  3. Startup size trim: over-cap cache drops oldest files first; legacy
     pid-* dirs are removed outright.
  4. max_mb=0 disables the cache (always downloads, writes nothing).

Run:  .venv/bin/python tests/vision_image_cache_test.py
(no server needed)
"""

import io
import os
import sys
import tempfile

# Must be set before the processor module is imported: _IMAGE_CACHE_BASE is
# resolved at import time.
_TEST_CACHE_BASE = tempfile.mkdtemp(prefix="dsv4-imgcache-test-")
os.environ["SGLANG_DSV4_IMAGE_CACHE_DIR"] = _TEST_CACHE_BASE
os.environ["SGLANG_DSV4_IMAGE_MAX_BYTES"] = str(1 << 20)  # 1 MiB for the test

from sglang.srt.multimodal.processors import deepseek_v4_vision as proc  # noqa: E402

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class FakeResponse:
    def __init__(self, payload, chunk=65536):
        self._buf = io.BytesIO(payload)
        self._chunk = chunk

    def read(self, n=-1):
        return self._buf.read(min(n, self._chunk) if n > 0 else n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def cached_files():
    return [f for f in os.listdir(proc._IMAGE_CACHE_BASE)
            if os.path.isfile(os.path.join(proc._IMAGE_CACHE_BASE, f))
            and ".tmp." not in f]


def main():
    url = "https://example.com/pic.png"
    payload = b"\x89PNG-fake-bytes" * 100

    calls = {"n": 0}
    real_urlopen = proc.urlopen

    def fake_urlopen(u, timeout=None):
        calls["n"] += 1
        assert u == url and timeout == proc._IMAGE_FETCH_TIMEOUT
        return FakeResponse(payload)

    proc.urlopen = fake_urlopen
    try:
        # 1. miss -> download, hit -> disk
        first = proc.load_image_bytes({"url": url})
        check("download returns payload", first == payload)
        check("cache file written", len(cached_files()) == 1,
              repr(cached_files()))
        second = proc.load_image_bytes({"url": url})
        check("second fetch served from cache",
              second == payload and calls["n"] == 1,
              f"urlopen calls={calls['n']}")

        # 1b. query string excluded from the cache key (S3 presigned URLs
        #     re-sign per request; same object must hit the same entry)
        s3_base = "https://s3.example.com/bucket/pic.png"
        signed1 = s3_base + "?X-Amz-Signature=aaa&X-Amz-Expires=3600"
        signed2 = s3_base + "?X-Amz-Signature=bbb&X-Amz-Expires=7200"
        q_calls = []

        def signed_urlopen(u, timeout=None):
            q_calls.append(u)
            return FakeResponse(b"S3:" + u.encode())

        proc.urlopen = signed_urlopen
        r1 = proc.load_image_bytes({"url": signed1})
        check("presigned: download uses the FULL url (query kept)",
              q_calls == [signed1] and r1 == b"S3:" + signed1.encode(),
              repr(q_calls))
        r2 = proc.load_image_bytes({"url": signed2})
        check("presigned: re-signed query hits the same cache entry",
              r2 == r1 and len(q_calls) == 1, f"calls={q_calls}")
        check("presigned: bare url hits too",
              proc.load_image_bytes({"url": s3_base}) == r1
              and len(q_calls) == 1)
        proc.urlopen = fake_urlopen
        # keep later steps' file-count assertions intact
        os.remove(os.path.join(proc._IMAGE_CACHE_BASE,
                               proc._cache_key(s3_base)))

        # 2. size cap
        def big_urlopen(u, timeout=None):
            return FakeResponse(b"x" * (2 << 20))
        proc.urlopen = big_urlopen
        try:
            proc.load_image_bytes({"url": "https://example.com/big.png"})
            check("oversize download rejected", False, "no exception")
        except ValueError:
            check("oversize download rejected", True)
        leftover = [f for f in os.listdir(proc._IMAGE_CACHE_BASE)
                    if ".tmp." in f]
        check("no partial tmp file left", not leftover, repr(leftover))
        proc.urlopen = fake_urlopen

        # 3. startup size trim: over cap -> oldest (mtime) removed first,
        #    legacy pid-* dirs removed outright
        existing = cached_files()
        assert len(existing) == 1
        old = os.path.join(proc._IMAGE_CACHE_BASE, existing[0])
        new = os.path.join(proc._IMAGE_CACHE_BASE, "f" * 64)
        with open(new, "wb") as f:
            f.write(b"y" * 4096)
        legacy = os.path.join(proc._IMAGE_CACHE_BASE, "pid-12345")
        os.makedirs(legacy)
        open(os.path.join(legacy, "stale"), "w").close()
        now = 1_800_000_000
        os.utime(old, (now - 200, now - 200))  # oldest
        os.utime(new, (now - 100, now - 100))
        proc._enforce_cache_size_limit(0)  # 0 via direct call = trim all
        check("over-cap trim removed oldest file", not os.path.exists(old))
        check("legacy pid-* dir removed outright", not os.path.exists(legacy))
        # new file size 4096 > 0 limit -> also removed
        check("trim to 0 emptied cache", cached_files() == [],
              repr(cached_files()))

        # under cap -> nothing removed
        with open(old, "wb") as f:
            f.write(b"z" * 1024)
        proc._enforce_cache_size_limit(1)  # 1 MiB >> 1 KiB
        check("under-cap trim keeps files", cached_files() == [existing[0]])

        # 4. max_mb=0 disables cache
        proc._image_cache_max_mb = 0
        before = calls["n"]
        got = proc.load_image_bytes({"url": url})
        check("cache disabled: always downloads",
              got == payload and calls["n"] == before + 1)
        check("cache disabled: nothing written",
              len(cached_files()) == 1,  # only the file from step 3 remains
              repr(cached_files()))
    finally:
        proc.urlopen = real_urlopen

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
