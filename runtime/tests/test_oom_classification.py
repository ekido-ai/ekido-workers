"""CUDA libraries sometimes raise allocation failures as RuntimeError."""
import unittest

from ekido_runtime.oom import is_memory_exhaustion


class MemoryExhaustionTest(unittest.TestCase):
    def test_allocator_failures(self):
        for message in ('CUDA out of memory', 'CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate',
                        'CUDNN_STATUS_ALLOC_FAILED', 'CUDA_ERROR_OUT_OF_MEMORY'):
            with self.subTest(message=message):
                self.assertTrue(is_memory_exhaustion(message))

    def test_other_failures_are_not_oom(self):
        for message in ('CUBLAS_STATUS_INVALID_VALUE', 'CUDA invalid device ordinal',
                        'failed to allocate input because shape is invalid', ''):
            with self.subTest(message=message):
                self.assertFalse(is_memory_exhaustion(message))
