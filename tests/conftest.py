import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture
def anyio_backend():
    return "asyncio"
