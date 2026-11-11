import unittest
import io
from unittest.mock import patch

from eth_abi import encode
from eth_utils import keccak

from gpuchain_verifier.trusted_chain import ChainReadError, JOB_TUPLE, _call_rpc, read_chain_context


class TrustedChain(unittest.TestCase):
    def setUp(self):
        self.contract = "0x" + "1" * 40
        self.job_id = "0x" + "2" * 64
        self.signer = "0x" + "3" * 40
        self.other = "0x" + "4" * 40
        self.rpc_chain = "0x7a69"
        self.state = 2

    def rpc(self, url, method, params):
        if method == "eth_chainId": return self.rpc_chain
        data = params[0]["data"]
        selector = data[2:10]
        if selector == keccak(text="getJob(bytes32)")[:4].hex():
            row = (self.signer, 1800000000, self.state, 7, 3, 0, 0, False, 5_000_000, 0, 0, bytes.fromhex("a" * 64), bytes.fromhex("b" * 64), bytes(32), [], [], self.signer)
            return "0x" + encode([JOB_TUPLE], [row]).hex()
        if selector == keccak(text="quorumOf(uint32)")[:4].hex():
            return "0x" + encode(["uint256"], [1]).hex()
        if selector == keccak(text="isVerifier(uint32,address)")[:4].hex():
            return "0x" + encode(["bool"], [data[-40:].lower() == self.signer[2:]]).hex()
        raise AssertionError("unexpected RPC method")

    def read(self):
        with patch("gpuchain_verifier.trusted_chain._call_rpc", side_effect=self.rpc):
            return read_chain_context("http://local-rpc", 31337, self.contract, self.job_id, [self.other, self.signer, self.signer])

    def test_reads_budget_hashes_and_only_allowlisted_distinct_verifiers(self):
        facts = self.read()
        self.assertEqual((facts["budget"], facts["spec_hash"], facts["policy_hash"]), ("5", "a" * 64, "b" * 64))
        self.assertEqual((facts["verifiers"], facts["quorum"]), ([self.signer], 1))

    def test_rejects_wrong_chain_and_inactive_job(self):
        self.rpc_chain = "0x1"
        with self.assertRaises(ChainReadError): self.read()
        self.rpc_chain, self.state = "0x7a69", 5
        with self.assertRaises(ChainReadError): self.read()

    def test_rejects_malformed_local_configuration_before_rpc(self):
        with patch("gpuchain_verifier.trusted_chain._call_rpc") as rpc:
            with self.assertRaises(ChainReadError):
                read_chain_context("http://local-rpc", 31337, "invalid", self.job_id, [self.signer])
            rpc.assert_not_called()

    def test_oversized_rpc_response_is_refused(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"x" * (2 * 1024 * 1024 + 1))):
            with self.assertRaises(ChainReadError):
                _call_rpc("http://local-rpc", "eth_chainId", [])
