"""Integration test for the FastAPI research service (via TestClient).

Requires the ``api`` extra (fastapi, httpx). Skipped automatically if the
dependency is absent so the unit suite stays lean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the apps/api package importable inside the test session.
APPS_API = Path(__file__).resolve().parents[2] / "apps" / "api"
if str(APPS_API) not in sys.path:
    sys.path.insert(0, str(APPS_API))

try:
    from fastapi.testclient import TestClient  # noqa: E402

    HAVE_API = True
except Exception:  # noqa: BLE001
    HAVE_API = False

pytestmark = pytest.mark.skipif(not HAVE_API, reason="fastapi not installed (api extra)")


@pytest.fixture(scope="module")
def client():
    from alphaforge_api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.slow
def test_run_then_serve(client):
    r = client.post(
        "/research/run",
        json={"start": "2019-01-01", "end": "2024-12-31", "report_dir": "research/reports"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["backtest"] is not None
    assert body["risk"] is not None
    assert body["brinson"] is not None

    assert client.get("/backtest").status_code == 200
    assert client.get("/attribution").status_code == 200
    assert client.get("/report").status_code == 200
    assert client.get("/briefing").status_code == 200
