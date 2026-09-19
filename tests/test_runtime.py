import json
from pathlib import Path
from threading import Event

import pytest

from autoosu import __version__, runtime
from autoosu.devices import CudaStatus


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOOSU_RUNTIME_HOME', str(tmp_path/'runtime'))
    return runtime.runtime_root()


def test_active_runtime_is_app_owned_and_versioned(owned):
    python = owned/'envs/gpu/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.touch()
    manifest = owned/'active.json'
    manifest.write_text(json.dumps(dict(python=str(python), app_version=__version__)))
    assert runtime.active_python() == python
    manifest.write_text(json.dumps(dict(python=str(python), app_version='old')))
    assert runtime.active_python() is None
    other = owned/'system-python.exe'
    other.touch()
    manifest.write_text(json.dumps(dict(python=str(other), app_version=__version__)))
    assert runtime.active_python() is None


def fake_installer(monkeypatch):
    monkeypatch.setattr(runtime, '_nvidia_name', lambda: 'GPU')
    monkeypatch.setattr(runtime, 'ensure_uv', lambda *a: Path('uv'))
    monkeypatch.setattr(runtime, 'create_environment', lambda *a: None)
    monkeypatch.setattr(runtime, 'run_logged', lambda *a: None)
    monkeypatch.setattr(runtime, '_source_for_install', lambda: (Path('source'), True))


def test_failed_verification_preserves_active_runtime(owned, monkeypatch):
    owned.mkdir(parents=True)
    manifest = owned/'active.json'
    original = '{"previous":"working runtime"}'
    manifest.write_text(original)
    fake_installer(monkeypatch)
    monkeypatch.setattr(runtime, 'probe_python', lambda _: CudaStatus(False, 'runtime_error', detail='kernel failed'))
    with pytest.raises(RuntimeError, match='kernel failed'):
        runtime.install_runtime(log=lambda _: None)
    assert manifest.read_text() == original


def test_activate_only_after_cuda_verified(owned, monkeypatch):
    fake_installer(monkeypatch)
    def probe(python):
        assert not (owned/'active.json').exists()
        return CudaStatus(True, 'ready', 'GPU', 8, '13.0', 'test')
    monkeypatch.setattr(runtime, 'probe_python', probe)
    result = runtime.install_runtime(log=lambda _: None)
    manifest = json.loads((owned/'active.json').read_text())
    assert manifest['python'] == str(result.python)
    assert manifest['cuda']['available'] is True


def test_cancel_does_not_activate(owned):
    stop = Event()
    stop.set()
    with pytest.raises(runtime.SetupCancelled):
        runtime.install_runtime(cancel=stop)
    assert not (owned/'active.json').exists()


def test_minor_link_failure_uses_owned_interpreter(owned, monkeypatch):
    checks = iter([None, owned/'python/cpython-3.12.14/python.exe'])
    monkeypatch.setattr(runtime, '_owned_python', lambda: next(checks))
    calls = []
    def run(args, log, cancel):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('Missing expected target directory for Python minor version link')
    monkeypatch.setattr(runtime, 'run_logged', run)
    runtime.create_environment(Path('uv'), owned/'envs/new', lambda _: None)
    assert '--managed-python' in calls[0]
    assert '--no-python-downloads' in calls[1]
    assert str(owned/'python/cpython-3.12.14/python.exe') == str(calls[1][3])


def test_child_environment_does_not_leak_python_paths(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'unexpected')
    monkeypatch.setenv('PYTHONHOME', 'unexpected')
    monkeypatch.setenv('_PYI_ARCHIVE_FILE', 'parent.exe')
    env = runtime.child_environment()
    assert 'PYTHONPATH' not in env and 'PYTHONHOME' not in env and '_PYI_ARCHIVE_FILE' not in env
    assert env['AUTOOSU_MANAGED_WORKER'] == '1'
