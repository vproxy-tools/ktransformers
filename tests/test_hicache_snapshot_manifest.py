"""Pure-logic unit test for the HiCache snapshot manifest (no GPU required).

Run: .venv/bin/python tests/test_hicache_snapshot_manifest.py
"""

import importlib.util
import json
import os
import tempfile
import unittest

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "third_party",
    "sglang",
    "python",
    "sglang",
    "srt",
    "mem_cache",
    "hicache_snapshot.py",
)
_spec = importlib.util.spec_from_file_location("hicache_snapshot", _MODULE_PATH)
hicache_snapshot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hicache_snapshot)

SnapshotNode = hicache_snapshot.SnapshotNode
write_manifest = hicache_snapshot.write_manifest
read_manifest = hicache_snapshot.read_manifest


class TestManifestRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        nodes = [
            SnapshotNode(parent=None, token_ids=[1, 2, 3, 4]),
            SnapshotNode(parent=0, token_ids=[5, 6, 7, 8]),
            SnapshotNode(parent=1, token_ids=list(range(9, 9 + 64))),
            SnapshotNode(parent=0, token_ids=[100, 200]),
        ]
        with tempfile.TemporaryDirectory() as d:
            manifest_path = write_manifest(
                d, page_size=64, model="test-model", nodes=nodes
            )
            self.assertEqual(
                manifest_path, os.path.join(d, hicache_snapshot.MANIFEST_FILENAME)
            )
            meta, loaded = read_manifest(d)

        self.assertEqual(meta["version"], hicache_snapshot.MANIFEST_VERSION)
        self.assertEqual(meta["page_size"], 64)
        self.assertEqual(meta["model"], "test-model")
        self.assertEqual(loaded, nodes)

    def test_empty_tree(self):
        with tempfile.TemporaryDirectory() as d:
            write_manifest(d, page_size=1, model=None, nodes=[])
            meta, loaded = read_manifest(d)
        self.assertEqual(meta["page_size"], 1)
        self.assertIsNone(meta["model"])
        self.assertEqual(loaded, [])

    def test_atomic_write_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as d:
            write_manifest(d, page_size=1, model=None, nodes=[])
            self.assertEqual(os.listdir(d), [hicache_snapshot.MANIFEST_FILENAME])


class TestManifestValidation(unittest.TestCase):
    def _write_raw(self, d, manifest):
        with open(os.path.join(d, hicache_snapshot.MANIFEST_FILENAME), "w") as f:
            json.dump(manifest, f)

    def test_bad_version(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_raw(d, {"version": 999, "page_size": 1, "nodes": []})
            with self.assertRaises(ValueError):
                read_manifest(d)

    def test_bad_page_size(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_raw(d, {"version": 1, "page_size": 0, "nodes": []})
            with self.assertRaises(ValueError):
                read_manifest(d)

    def test_parent_must_precede_child(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_raw(
                d,
                {
                    "version": 1,
                    "page_size": 1,
                    "nodes": [
                        {"parent": 1, "token_ids": [1]},  # parent after child
                        {"parent": None, "token_ids": [2]},
                    ],
                },
            )
            with self.assertRaises(ValueError):
                read_manifest(d)

    def test_parent_out_of_range(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_raw(
                d,
                {
                    "version": 1,
                    "page_size": 1,
                    "nodes": [{"parent": 7, "token_ids": [1]}],
                },
            )
            with self.assertRaises(ValueError):
                read_manifest(d)

    def test_empty_token_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_raw(
                d,
                {
                    "version": 1,
                    "page_size": 1,
                    "nodes": [{"parent": None, "token_ids": []}],
                },
            )
            with self.assertRaises(ValueError):
                read_manifest(d)

    def test_write_rejects_invalid_parent(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_manifest(
                    d,
                    page_size=1,
                    model=None,
                    nodes=[SnapshotNode(parent=0, token_ids=[1])],
                )
            self.assertEqual(os.listdir(d), [])

    def test_missing_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                read_manifest(d)


if __name__ == "__main__":
    unittest.main()
