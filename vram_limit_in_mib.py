#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx",
# ]
# ///

import subprocess

import mlx.core as mx


def system_recommended_vram_limit_mib() -> float:
    """
    Returns the maximum GPU memory size (in MiB) that can be allocated,
    as set by the system (not necessarily the physical maximum).
    Raises an exception if Metal is not available.

    Returns:
        float: VRAM limit size in MiB as set by the system.
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
    recommended_size_mib = int(system_recommended_vram_limit_mib())
    current_wired_limit_mib = int(
        subprocess.run(
            "sysctl -n iogpu.wired_limit_mb".split(),
            text=True,
            capture_output=True,
        ).stdout.strip()
    )

    print(max(recommended_size_mib, current_wired_limit_mib))
