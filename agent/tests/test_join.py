import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eth_account import Account

from ekido_agent.join import default_key_file, detect_device, http_base, join_document, load_or_create_key
from ekido_core.signing import sign_document, verify_document


class JoinTest(unittest.TestCase):
    def test_the_signed_document_verifies_against_the_wallet(self):
        key = "0x" + "7" * 64
        wallet = Account.from_key(key).address
        document = join_document("A", wallet, "0.002", "cpu", 4096, now=1_800_000_000)
        self.assertEqual(document["kind"], "gpuchain.provider_join.v1")
        self.assertTrue(verify_document(document, sign_document(document, key)[1], wallet))

    def test_a_new_key_file_is_created_once_and_then_reused(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROVIDER_WALLET_KEY", None)
            path = Path(tmp) / "nested" / "provider.key"
            first = load_or_create_key(path)
            self.assertTrue(path.exists())
            self.assertEqual(load_or_create_key(path), first)

    def test_a_key_from_before_the_rename_is_used_where_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.assertEqual(default_key_file(home), home / ".ekido" / "provider.key")
            earlier = home / ".gpuchain" / "provider.key"
            earlier.parent.mkdir()
            earlier.write_text("0x" + "9" * 64)
            self.assertEqual(default_key_file(home), earlier)
            current = home / ".ekido" / "provider.key"
            current.parent.mkdir()
            current.write_text("0x" + "a" * 64)
            self.assertEqual(default_key_file(home), current)

    def test_the_environment_key_wins_over_the_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PROVIDER_WALLET_KEY": "0x" + "8" * 64}):
            self.assertEqual(load_or_create_key(Path(tmp) / "provider.key"), "0x" + "8" * 64)
            self.assertFalse((Path(tmp) / "provider.key").exists())

    def test_cpu_is_used_when_asked_and_gets_a_declared_capacity(self):
        self.assertEqual(detect_device("cpu", None), ("cpu", 4096))
        self.assertEqual(detect_device("cpu", 2048), ("cpu", 2048))

    def test_the_http_address_follows_the_socket_address(self):
        self.assertEqual(http_base("wss://x.trycloudflare.com/"), "https://x.trycloudflare.com")
        self.assertEqual(http_base("ws://10.0.0.5:8000"), "http://10.0.0.5:8000")


if __name__ == "__main__":
    unittest.main()
