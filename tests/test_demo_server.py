"""Test API HTTP — dùng service giả để không đụng model thật (nặng, cần GPU).

Quy tắc: server chỉ là controller mỏng, nên test ở đây kiểm tra **định tuyến và
ánh xạ lỗi**, không kiểm tra thuật toán. Thuật toán đã có test riêng
(`test_demo_lexer.py`, `test_demo_aligner.py`).

`TestClient` chỉ chạy được khi đã cài `fastapi` + `httpx`; nếu chưa cài thì bỏ
qua bộ test này (importorskip) thay vì làm cả đống test vỡ.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.demo.server import create_app  # noqa: E402
from src.demo.service import InputTooLarge  # noqa: E402

SAMPLE_ALIGNMENT = {
    "mode": "abstract",
    "stats": {
        "n_tokens_a": 5, "n_tokens_b": 5,
        "n_matched_a": 5, "n_matched_b": 5,
        "coverage_a": 1.0, "coverage_b": 1.0,
        "n_blocks": 1,
    },
    "blocks": [{"id": 0, "a": [0, 1], "b": [0, 1], "size": 5}],
    "a": {"lines": [{"n": 1, "tokens": [{"t": "int", "k": "type", "b": 0}]}]},
    "b": {"lines": [{"n": 1, "tokens": [{"t": "int", "k": "type", "b": 0}]}]},
}

MODEL_OK = {
    "model_id": "M1", "label": "CodeBERT", "ok": True,
    "score": 0.9, "threshold": 0.5, "verdict": "clone",
    "latency_ms": 1.1, "detail": {},
}


class FakeService:
    """Triển khai đúng hợp đồng mà `create_app` dùng, không cần model thật."""

    def __init__(self):
        self.registry = SimpleNamespace(
            config={
                "demo": {"limits": {"max_code_chars": 20000, "max_lines": 400}},
                "alignment": {"n_block_colors": 6},
                "runtime": {"device": "cpu"},
                "data": {"pairs_dir": None},
            },
            preload=lambda ids: {model_id: "ready" for model_id in ids},
        )
        self._raise: dict[str, Exception] = {}
        self.library_enabled = True

    def raise_next(self, key: str, exc: Exception) -> None:
        self._raise[key] = exc

    def status(self) -> dict:
        return {
            "limits": {"max_code_chars": 20000, "max_lines": 400, "n_block_colors": 6},
            "models": [
                {"model_id": "M1", "label": "CodeBERT", "state": "ready",
                 "threshold": 0.5, "test_metrics": {"f1": 0.9}},
            ],
        }

    def analyze(self, code_a, code_b, model_ids=None, mode="abstract", min_block_tokens=0):
        if "analyze" in self._raise:
            raise self._raise.pop("analyze")
        return {
            "alignment": SAMPLE_ALIGNMENT,
            "models": [dict(MODEL_OK)],
            "meta": {
                "mode": mode,
                "n_lines_a": 1,
                "n_lines_b": 1,
                "n_block_colors": 6,
            },
        }

    def real_pairs(self, split="test", limit=None):
        if "real_pairs" in self._raise:
            raise self._raise.pop("real_pairs")
        return [{
            "pair_id": "test-pos-101-1", "split": split, "label": 1,
            "code_a_id": "p-a", "code_b_id": "p-b",
            "problem_a": "101", "problem_b": "101",
        }]

    def load_real(self, pair_id, split="test"):
        if "load_real" in self._raise:
            raise self._raise.pop("load_real")
        if pair_id == "missing":
            raise KeyError(f"không thấy {pair_id}")
        return {
            "id": pair_id, "code_a": "int a;", "code_b": "int b;",
            "title": "t", "expected": "clone", "note": "", "origin": "poj104",
        }

    def library_files(self):
        if "library_files" in self._raise:
            raise self._raise.pop("library_files")
        if not self.library_enabled:
            return {"enabled": False, "directory": None, "files": []}
        return {
            "enabled": True,
            "directory": "/fake/library",
            "n_files": 2,
            "max_files": 300,
            "max_chars": 20000,
            "default_threshold": 0.5,
            "files": [
                {"path": "a.c", "name": "a.c", "size": 11, "ext": ".c", "n_lines": 1},
                {"path": "b.c", "name": "b.c", "size": 11, "ext": ".c", "n_lines": 1},
            ],
        }

    def read_library_file(self, rel_path):
        if "read_library_file" in self._raise:
            raise self._raise.pop("read_library_file")
        if rel_path in {"missing.c", "../secret.txt"}:
            raise KeyError(f"không có file {rel_path!r} trong library")
        return {"path": rel_path, "name": rel_path, "code": "int x;",
                "n_lines": 1, "size": 7, "truncated": False}

    def scan_library(self, threshold=None):
        if "scan_library" in self._raise:
            raise self._raise.pop("scan_library")
        if not self.library_enabled:
            raise KeyError("chưa khai data.library_dir")
        return {
            "n_files": 2, "threshold": threshold if threshold is not None else 0.5,
            "mode": "moss", "k": 50, "w": 100, "duration_ms": 0.4,
            "pairs": [{
                "a": "a.c", "b": "b.c", "similarity": 0.96,
                "n_fp_a": 4, "n_fp_b": 4,
            }],
        }


@pytest.fixture()
def client():
    return TestClient(create_app(FakeService()))


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Semantic Code Clone Detection" in response.text


def test_static_css_served(client):
    response = client.get("/static/css/app.css")
    assert response.status_code == 200
    assert ".app-header" in response.text


def test_status_returns_limits_and_models(client):
    data = client.get("/api/status").json()
    assert data["limits"]["max_code_chars"] == 20000
    assert data["models"][0]["model_id"] == "M1"
    assert data["models"][0]["test_metrics"]["f1"] == 0.9


def test_builtin_samples_are_list(client):
    samples = client.get("/api/samples").json()
    assert isinstance(samples, list) and len(samples) >= 3
    assert all("code_a" in s and "code_b" in s for s in samples)


def test_analyze_happy_path(client):
    data = client.post("/api/analyze", json={
        "code_a": "int a;", "code_b": "int b;", "mode": "abstract",
    }).json()
    assert data["alignment"]["stats"]["n_blocks"] == 1
    assert data["models"][0]["verdict"] == "clone"
    assert data["meta"]["mode"] == "abstract"


def test_analyze_empty_code_is_rejected(client):
    response = client.post("/api/analyze", json={"code_a": "", "code_b": "int b;"})
    assert response.status_code == 422


def test_analyze_too_large_is_413(client, monkeypatch):
    client = TestClient(create_app(FakeService()))
    service = client.app.state.service
    service.raise_next("analyze", InputTooLarge("quá dài"))
    response = client.post("/api/analyze", json={"code_a": "int a;", "code_b": "int b;"})
    assert response.status_code == 413
    assert "quá dài" in response.json()["detail"]


def test_analyze_invalid_mode_is_400(client):
    client = TestClient(create_app(FakeService()))
    service = client.app.state.service
    service.raise_next("analyze", ValueError("mode 'x' không hợp lệ"))
    response = client.post("/api/analyze", json={
        "code_a": "int a;", "code_b": "int b;", "mode": "x",
    })
    assert response.status_code == 400


def test_real_pairs_empty_without_data(client):
    client = TestClient(create_app(FakeService()))
    service = client.app.state.service
    service.raise_next("real_pairs", OSError("chưa có dữ liệu POJ-104"))
    response = client.get("/api/samples/real")
    assert response.status_code == 503
    assert "POJ-104" in response.json()["detail"]


def test_real_pair_missing_is_404(client):
    response = client.get("/api/samples/real/missing")
    assert response.status_code == 404


def test_real_pair_happy(client):
    data = client.get("/api/samples/real/test-pos-101-1").json()
    assert data["code_a"] == "int a;"
    assert data["origin"] == "poj104"


def test_preload_without_models_is_allowed(client):
    data = client.post("/api/preload", json={"models": ["M1"]}).json()
    assert data["M1"] == "ready"


def test_library_lists_files(client):
    data = client.get("/api/library").json()
    assert data["enabled"] is True
    assert data["n_files"] == 2
    assert data["files"][0]["path"] == "a.c"


def test_library_disabled_still_json(client):
    client = TestClient(create_app(FakeService()))
    service = client.app.state.service
    service.library_enabled = False
    data = client.get("/api/library").json()
    assert data["enabled"] is False
    assert data["files"] == []


def test_library_read_file(client):
    data = client.get("/api/library/file", params={"path": "a.c"}).json()
    assert data["code"] == "int x;"
    assert data["truncated"] is False


def test_library_read_missing_is_404(client):
    response = client.get("/api/library/file", params={"path": "missing.c"})
    assert response.status_code == 404


def test_library_read_traversal_is_blocked(client):
    response = client.get("/api/library/file", params={"path": "../secret.txt"})
    assert response.status_code == 404


def test_library_scan_returns_pairs(client):
    data = client.post("/api/library/scan", params={"threshold": 0.7}).json()
    assert data["pairs"][0]["similarity"] == 0.96
    assert data["threshold"] == 0.7
    assert data["mode"] == "moss"


def test_library_scan_uses_default_threshold(client):
    data = client.post("/api/library/scan").json()
    assert data["threshold"] == 0.5


def test_library_scan_disabled_is_400(client):
    client = TestClient(create_app(FakeService()))
    service = client.app.state.service
    service.library_enabled = False
    response = client.post("/api/library/scan")
    assert response.status_code == 400


def test_library_scan_value_error_is_400(client):
    client = TestClient(create_app(FakeService()))
    service = client.app.state.service
    service.raise_next("scan_library", ValueError("ngưỡng không hợp lệ"))
    response = client.post("/api/library/scan", params={"threshold": 1.5})
    assert response.status_code == 400