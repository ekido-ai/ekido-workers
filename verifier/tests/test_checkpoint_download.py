import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from gpuchain_verifier.worker import Verifier


class CheckpointDownloads(unittest.TestCase):
    def test_unexpected_file_names_are_refused_before_any_download(self):
        for name in ("../escaped.py", "nested/file", r"..\escaped.py", r"C:\escaped.py", "/escaped.py", "extra.safetensors"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                verifier = Verifier("http://coordinator", "0x" + "a" * 64, "cpu", 512)
                verifier._download = Mock()
                files = {n: {"sha256": "a" * 64, "bytes": 8} for n in ("trainable.safetensors", "optimizer.safetensors", name)}
                with self.assertRaises(ValueError):
                    verifier._fetch_checkpoint({"files": files}, Path(tmp))
                verifier._download.assert_not_called()
