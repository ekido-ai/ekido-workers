import unittest

from ekido_runtime.probe import probe
from ekido_runtime.synthetic import exec_config, synthetic_spec


def spec():
    s = synthetic_spec()
    s["template"] = "synthetic-linear/v1"
    return s


class Probe(unittest.TestCase):
    def test_measures_a_working_config(self):
        result = probe({"spec": spec(), "exec": exec_config(8), "device": "cpu", "timed_steps": 2})
        self.assertTrue(result["ok"])
        self.assertFalse(result["oom"])
        self.assertGreater(result["steps_per_s"], 0)
        self.assertEqual(result["device"], "cpu")

    def test_bad_config_is_an_error_not_an_oom(self):
        bad = dict(exec_config(8), accumulation=3)
        result = probe({"spec": spec(), "exec": bad, "device": "cpu"})
        self.assertFalse(result["ok"])
        self.assertFalse(result["oom"])
        self.assertIn("InvalidExecConfig", result["error"])

    def test_workload_outside_allowlist_refused(self):
        result = probe({"spec": spec() | {"template": "evil/v1"}, "exec": exec_config(8), "device": "cpu"})
        self.assertFalse(result["ok"])
        self.assertIn("allowlist", result["error"])


if __name__ == "__main__":
    unittest.main()


class ProbeDataFolder(unittest.TestCase):
    """The probe must hand the job's folder to the workload: llm-lora/v1 reads dataset.jsonl from it (found on 24 Nov,
    when a real provider refused every LLM job with "needs the run folder that holds dataset.jsonl")."""

    def test_the_data_folder_reaches_the_workload(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from unittest import mock

        from ekido_runtime import probe as probe_module

        seen = {}

        def fake_build(spec, exec_config, device, *, with_data=True, data_dir=None):
            seen["data_dir"] = data_dir
            raise ValueError("stop here")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(probe_module, "build_workload", fake_build):
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"spec": {"template": "llm-lora/v1", "optimizer": {}}, "exec": {"amp": "fp32"}, "device": "cpu"}), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["probe", str(config), str(Path(tmp) / "result.json")]):
                probe_module.main()
            self.assertEqual(Path(seen["data_dir"]), Path(tmp).resolve())
            self.assertFalse(json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))["ok"])
