"""Probe the current Python/packaged runtime, not just the installed GPU driver."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    reason: str
    name: str = ""
    memory_gb: float = 0.0
    cuda_version: str = ""
    torch_version: str = ""
    detail: str = ""
    index: int = 0


def _nvidia_name() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ""
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return ", ".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_cuda(index: int = 0) -> CudaStatus:
    """Run a tiny GPU calculation to distinguish detection from usable CUDA."""
    try:
        import torch
    except Exception as exc:
        return CudaStatus(False, "torch_missing", detail=str(exc), index=index)
    version = str(torch.__version__)
    cuda = str(torch.version.cuda or "")
    if not cuda:
        return CudaStatus(False, "cpu_build", name=_nvidia_name(), torch_version=version, index=index)
    try:
        if not torch.cuda.is_available():
            return CudaStatus(False, "unavailable", name=_nvidia_name(), cuda_version=cuda,
                              torch_version=version, index=index)
        properties = torch.cuda.get_device_properties(index)
        # A driver can enumerate a card even when this runtime cannot execute its kernels.
        value = torch.ones(1, device=f"cuda:{index}").add(1).item()
        if value != 2:
            raise RuntimeError("CUDA calculation returned an unexpected result")
        return CudaStatus(True, "ready", properties.name, properties.total_memory / 1024**3,
                          cuda, version, index=index)
    except Exception as exc:
        return CudaStatus(False, "runtime_error", cuda_version=cuda, torch_version=version,
                          detail=str(exc), index=index)


def describe_cuda(status: CudaStatus) -> str:
    if status.available:
        return f"CUDA {status.cuda_version} ready: {status.name} ({status.memory_gb:.1f} GB)"
    if status.reason == "cpu_build":
        gpu = f" Detected GPU: {status.name}." if status.name else ""
        return "This is a CPU-only PyTorch runtime; use a CUDA build for GPU inference." + gpu
    if status.reason == "unavailable":
        return "CUDA runtime is installed, but no usable CUDA device was found. Check the NVIDIA driver."
    if status.reason == "torch_missing":
        return "PyTorch could not be loaded: " + status.detail
    return "CUDA initialization failed: " + status.detail


def resolve_device(device: Optional[str] = None) -> str:
    requested = (device or "auto").lower().strip()
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and not re.fullmatch(r"cuda(?::\d+)?", requested):
        raise ValueError("Device must be auto, cpu, cuda, or cuda:N")
    index = int(requested.partition(":")[2] or "0")
    status = detect_cuda(index)
    if status.available:
        return f"cuda:{index}"
    if requested == "auto":
        return "cpu"
    raise RuntimeError(describe_cuda(status) + " Select auto or cpu to use the CPU.")
