import numpy as np
import pytest

from debug_viz import boot


@pytest.fixture(autouse=True)
def _isolate_pid_file(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "_PID_FILE", tmp_path / "server.pid")
    monkeypatch.setattr(boot, "_BROWSER_OPENED", True, raising=False)
    yield


@pytest.fixture
def gray_2d():
    rng = np.random.default_rng(0)
    return rng.random((40, 60), dtype=np.float32)


@pytest.fixture
def gray_with_nan(gray_2d):
    arr = gray_2d.copy().astype(np.float64)
    arr[0, 0] = np.nan
    arr[5, 5] = np.nan
    return arr


@pytest.fixture
def rgb_3d():
    rng = np.random.default_rng(1)
    return rng.random((20, 30, 3), dtype=np.float32)


@pytest.fixture
def mask_2d():
    arr = np.zeros((20, 20), dtype=bool)
    arr[5:10, 5:10] = True
    return arr


@pytest.fixture
def big_2d():
    rng = np.random.default_rng(2)
    return rng.random((1200, 1500), dtype=np.float32)
