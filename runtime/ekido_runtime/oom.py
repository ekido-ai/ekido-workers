"""Identify allocator failures that CUDA libraries report as generic errors."""


def is_memory_exhaustion(message: str) -> bool:
    message = message.casefold()
    return "out of memory" in message or any(status in message for status in (
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "cuda_error_out_of_memory",
    ))
