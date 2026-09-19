"""Device detection must reflect executable CUDA, not just a card in the machine."""
import sys
from types import SimpleNamespace

import pytest

from autoosu import devices


def fake_torch(monkeypatch, *, cuda="13.0", available=True, broken=False):
    def ones(*args, **kwargs):
        if broken:
            raise RuntimeError("kernel image is unavailable")
        return SimpleNamespace(add=lambda n: SimpleNamespace(item=lambda: 2))
    module = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=cuda), ones=ones,
                             cuda=SimpleNamespace(is_available=lambda: available,
                             get_device_properties=lambda i: SimpleNamespace(name=f"GPU {i}", total_memory=8*1024**3)))
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setattr(devices, "_nvidia_name", lambda: "NVIDIA GPU")


def test_cpu_build_is_not_no_gpu(monkeypatch):
    fake_torch(monkeypatch, cuda=None)
    result = devices.detect_cuda()
    assert result.reason == "cpu_build" and result.name == "NVIDIA GPU"
    assert devices.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="CPU-only"):
        devices.resolve_device("cuda")


def test_available_gpu_executes_probe(monkeypatch):
    fake_torch(monkeypatch)
    result = devices.detect_cuda(1)
    assert result.available and result.memory_gb == 8 and result.name == "GPU 1"
    assert devices.resolve_device("cuda:1") == "cuda:1"


def test_enumeration_does_not_guarantee_kernel_compatibility(monkeypatch):
    fake_torch(monkeypatch, broken=True)
    result = devices.detect_cuda()
    assert not result.available and result.reason == "runtime_error"
    assert devices.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="kernel image"):
        devices.resolve_device("cuda")


def test_cuda_runtime_without_available_device(monkeypatch):
    fake_torch(monkeypatch, available=False)
    assert devices.detect_cuda().reason == "unavailable"
    assert devices.resolve_device(None) == "cpu"


def test_explicit_cpu_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(devices, "detect_cuda", lambda *args: pytest.fail("CPU must not probe CUDA"))
    assert devices.resolve_device("cpu") == "cpu"


@pytest.mark.parametrize("device", ["gpu", "cuda:-1", "cuda:x"])
def test_invalid_device(device):
    with pytest.raises(ValueError, match="Device must"):
        devices.resolve_device(device)
