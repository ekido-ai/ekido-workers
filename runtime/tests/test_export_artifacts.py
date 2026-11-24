import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ekido_runtime.export_artifacts import export_lock, record_gguf


def gguf_bytes():
    return struct.pack('<4sIQQ', b'GGUF', 3, 1, 1) + b'synthetic-test-only'


class ExportArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'export.json').write_text(json.dumps({'format': 'safetensors'}), encoding='utf-8')
        self.file = self.root / 'model-Q8_0.gguf'
        self.file.write_bytes(gguf_bytes())

    def test_records_hash_size_and_relative_path(self):
        entry = record_gguf(self.root, self.file, 'Q8_0')
        self.assertEqual(entry['sha256'], hashlib.sha256(gguf_bytes()).hexdigest())
        self.assertEqual(entry['bytes'], len(gguf_bytes()))
        self.assertEqual(entry['quantization'], 'Q8_0')
        self.assertEqual(entry['path'], 'model-Q8_0.gguf')
        self.assertEqual(json.loads((self.root / 'export.json').read_text(encoding='utf-8'))['artifacts'][entry['path']], entry)

    def test_truncated_bad_header_and_bad_quantization_do_not_update_metadata(self):
        initial = (self.root / 'export.json').read_bytes()
        for bad in (b'GGUF', b'NOT-A-GGUF', struct.pack('<4sIQQ', b'GGUF', 999, 1, 1),
                    struct.pack('<4sIQQ', b'GGUF', 3, 0, 1)):
            with self.subTest(bad=bad):
                self.file.write_bytes(bad)
                with self.assertRaises(ValueError):
                    record_gguf(self.root, self.file, 'Q8_0')
                self.assertEqual((self.root / 'export.json').read_bytes(), initial)
        self.file.write_bytes(gguf_bytes())
        with self.assertRaises(ValueError):
            record_gguf(self.root, self.file, '../bad')
        self.assertEqual((self.root / 'export.json').read_bytes(), initial)

    def test_outside_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            file = Path(other) / 'other.gguf'
            file.write_bytes(gguf_bytes())
            with self.assertRaisesRegex(ValueError, 'inside'):
                record_gguf(self.root, file, 'Q8_0')

    def test_failed_metadata_replace_does_not_publish_partial_record(self):
        initial = (self.root / 'export.json').read_bytes()
        with mock.patch('ekido_runtime.export_artifacts.os.replace', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                record_gguf(self.root, self.file, 'Q8_0')
        self.assertEqual((self.root / 'export.json').read_bytes(), initial)
        self.assertEqual(list(self.root.glob('.export-*')), [])

    def test_competing_exporter_is_rejected_and_lock_released(self):
        with export_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, 'already has a writer'):
                with export_lock(self.root):
                    pass
        with export_lock(self.root):
            self.assertEqual(record_gguf(self.root, self.file, 'Q8_0')['format'], 'gguf')


if __name__ == '__main__':
    unittest.main()
