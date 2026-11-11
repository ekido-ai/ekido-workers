import unittest

from gpuchain_runtime.probe import probe
from gpuchain_runtime.synthetic import exec_config, synthetic_spec


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
