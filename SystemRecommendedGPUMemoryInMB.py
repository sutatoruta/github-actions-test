#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx",
# ]
# ///

import mlx.core as mx


def system_recommended_max_gpu_memory_mb() -> float:
    """
    Returns the maximum GPU memory size (in MB) that can be allocated,
    as set by the system (not necessarily the physical maximum).
    Raises an exception if Metal is not available.

    Returns:
        float: Memory size in MB as set by the system.
    Raises:
        RuntimeError: If Metal is not available.
    """
    if not mx.metal.is_available():
        raise RuntimeError("Metal is not available on this system.")

    device_info = mx.metal.device_info()
    size_in_bytes = device_info.get("max_recommended_working_set_size", 0)
    size_in_mb = float(size_in_bytes) / (1024 * 1024)
    return size_in_mb


if __name__ == "__main__":
    size_mb = int(system_recommended_max_gpu_memory_mb())
    print(size_mb)
