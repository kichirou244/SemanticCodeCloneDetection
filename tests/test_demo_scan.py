"""Quét zip upload: validate, ngưỡng, xếp hạng — stub scorer để nhanh deterministic."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from src.demo.scorers.base import ScoreResult
from src.demo.service import DemoService

A = "int f() { return 1; }"
B = "int f() { return 1; }"
C = "int g() { return 2; }"


def make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, code in files.items():
            z.writestr(name, code)
    return buf.getvalue()


class StubScorer:
    """Scorer giả: điểm tra bảng theo cặp (không cần model thật)."""

    def __init__(self, model_id, results_dir, table):
        self.model_id = model_id
        self.results_dir = Path(results_dir)
        self._table = table

    def score(self, a, b):
        s = self._table.get((a, b), self._table.get((b, a), 0.0))
        return ScoreResult(ok=True, score=s, threshold=None, verdict="unknown")


class StubRegistry:
    def __init__(self, config, scorers):
        self.config = config
        self.root = Path(".").resolve()
        self._scorers = {s.model_id: s for s in scorers}

    def selected(self, model_ids=None):
        if not model_ids:
            return [self._scorers[k] for k in sorted(self._scorers)]
        return [self._scorers[m] for m in model_ids if m in self._scorers]


def make_service(tmp_path, specs, scan_cfg=None):
    """specs: {mid: (threshold_frozen, score_table)}."""
    scorers = []
    for mid, (thr, table) in specs.items():
        rdir = tmp_path / mid
        rdir.mkdir(exist_ok=True)
        (rdir / "threshold.json").write_text(json.dumps({"threshold": thr}))
        scorers.append(StubScorer(mid, rdir, table))
    cfg = {"scan": {"max_files": 20, "max_bytes": 10485760,
                    "max_file_chars": 20000, **(scan_cfg or {})}}
    return DemoService(StubRegistry(cfg, scorers))


def test_upload_rejects_non_zip(tmp_path):
    svc = make_service(tmp_path, {"M": (0.5, {})})
    try:
        svc.upload_zip(b"khong phai zip", "x.zip")
    except ValueError as e:
        assert "zip" in str(e).lower()
        return
    raise AssertionError("phải ValueError")


def test_upload_skips_non_code_and_sanitizes(tmp_path):
    svc = make_service(tmp_path, {"M": (0.5, {})})
    data = make_zip({
        "a.cpp": A, "sub/b.h": B, "note.txt": "bo qua",
        "../evil.cpp": C, "bin": "\xff\xfe nhị phân",
    })
    out = svc.upload_zip(data, "t.zip")
    names = [f["name"] for f in out["files"]]
    assert "a.cpp" in names and "b.h" in names and "evil.cpp" in names
    assert ".." not in " ".join(names) and "/" not in " ".join(names)
    assert out["skipped"]["non_code"] >= 1


def test_upload_rejects_too_many_files(tmp_path):
    svc = make_service(tmp_path, {"M": (0.5, {})}, scan_cfg={"max_files": 3})
    data = make_zip({f"f{i}.cpp": A for i in range(4)})
    try:
        svc.upload_zip(data, "t.zip")
    except ValueError as e:
        assert "3" in str(e)
        return
    raise AssertionError("phải ValueError khi quá số file")


def test_upload_rejects_empty_zip(tmp_path):
    svc = make_service(tmp_path, {"M": (0.5, {})})
    try:
        svc.upload_zip(make_zip({"readme.txt": "x"}), "t.zip")
    except ValueError:
        return
    raise AssertionError("phải ValueError khi không có file code")


def test_scan_uses_frozen_thresholds_and_sorts(tmp_path):
    # M1 thr 0.5: A-B 0.9 keep; M2 thr 0.95: không ai đạt → nhưng M1 đạt là đủ giữ.
    specs = {"M1": (0.5, {(A, B): 0.9, (A, C): 0.1, (B, C): 0.2}),
             "M2": (0.95, {(A, B): 0.9, (A, C): 0.1, (B, C): 0.2})}
    svc = make_service(tmp_path, specs)
    up = svc.upload_zip(make_zip({"a.cpp": A, "b.cpp": B, "c.cpp": C}), "t.zip")
    res = svc.scan_upload(up["upload_id"])
    assert res["n_files"] == 3 and res["n_pairs"] == 3
    assert res["threshold_applied"] == {"M1": 0.5, "M2": 0.95}
    assert res["n_kept"] == 1
    row = res["pairs"][0]
    assert {row["a"], row["b"]} == {"a.cpp", "b.cpp"}
    assert row["scores"]["M1"]["verdict"] == "clone"
    assert row["scores"]["M2"]["verdict"] == "not_clone"


def test_scan_override_threshold_applies_to_all(tmp_path):
    specs = {"M1": (0.5, {(A, B): 0.9}), "M2": (0.95, {(A, B): 0.9})}
    svc = make_service(tmp_path, specs)
    up = svc.upload_zip(make_zip({"a.cpp": A, "b.cpp": B}), "t.zip")
    res = svc.scan_upload(up["upload_id"], threshold=0.95)
    assert res["threshold_applied"] == {"M1": 0.95, "M2": 0.95}
    assert res["n_kept"] == 0  # 0.9 < 0.95 cho cả hai
    res2 = svc.scan_upload(up["upload_id"], threshold=0.9)
    assert res2["n_kept"] == 1


def test_scan_single_file_has_no_pairs(tmp_path):
    svc = make_service(tmp_path, {"M1": (0.5, {})})
    up = svc.upload_zip(make_zip({"a.cpp": A}), "t.zip")
    res = svc.scan_upload(up["upload_id"])
    assert res["pairs"] == [] and res["n_pairs"] == 0


def test_scan_unknown_upload_id_raises_keyerror(tmp_path):
    svc = make_service(tmp_path, {"M1": (0.5, {})})
    try:
        svc.scan_upload("khong-co-id")
    except KeyError:
        return
    raise AssertionError("phải KeyError (→ 404)")


def test_cpg_batch_real_joern_three_files():
    """Joern thật: 1 lần chạy cho 3 file, chấm 3 cặp. Skip khi thiếu toolchain."""
    pytest.importorskip("torch_geometric")
    from scripts.build_cpg_export import joern_missing_reason, joern_or_die

    if joern_missing_reason(joern_or_die()):
        pytest.skip("cần Java+Joern")
    from src.demo.registry import ModelRegistry

    root = Path(__file__).resolve().parents[1]
    m4 = ModelRegistry.from_yaml(root / "configs" / "demo_web.yaml",
                                 root=root).get("M4")
    assert m4.ensure_loaded() is True, m4.status()["error"]
    tiny = "int f() { return 1; }"
    out = m4.score_all_pairs([("a", tiny), ("b", tiny), ("c", tiny)])
    assert len(out) == 3
    for (a, b), (score, detail) in out.items():
        assert score is not None, (a, b, detail)
        assert 0.0 <= score <= 1.0


def _client_with_stubs(tmp_path):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.demo.server import create_app

    specs = {"M1": (0.5, {(A, B): 0.9, (A, C): 0.1}),
             "M2": (0.9, {(A, B): 0.4, (A, C): 0.1})}
    svc = make_service(tmp_path, specs)
    return TestClient(create_app(service=svc))


def test_api_upload_then_scan_end_to_end(tmp_path):
    client = _client_with_stubs(tmp_path)
    r = client.post("/api/uploads", files={"file": ("t.zip", make_zip(
        {"a.cpp": A, "b.cpp": B, "c.cpp": C}), "application/zip")})
    assert r.status_code == 200, r.text
    up = r.json()
    assert len(up["files"]) == 3 and up["upload_id"]
    assert up["files"][0]["code"] == A
    r2 = client.post("/api/scan", json={"upload_id": up["upload_id"]})
    assert r2.status_code == 200, r2.text
    res = r2.json()
    assert res["n_kept"] == 1
    row = res["pairs"][0]
    assert {row["a"], row["b"]} == {"a.cpp", "b.cpp"}
    assert row["scores"]["M1"]["verdict"] == "clone"
    assert row["scores"]["M2"]["verdict"] == "not_clone"  # 0.4 < 0.9


def test_api_upload_rejects_non_zip(tmp_path):
    client = _client_with_stubs(tmp_path)
    r = client.post("/api/uploads",
                    files={"file": ("x.zip", b"nope", "application/zip")})
    assert r.status_code == 400


def test_api_scan_unknown_id_is_404(tmp_path):
    client = _client_with_stubs(tmp_path)
    r = client.post("/api/scan", json={"upload_id": "khong-co"})
    assert r.status_code == 404


def test_api_scan_override_threshold(tmp_path):
    client = _client_with_stubs(tmp_path)
    r = client.post("/api/uploads", files={"file": ("t.zip", make_zip(
        {"a.cpp": A, "b.cpp": B}), "application/zip")})
    uid = r.json()["upload_id"]
    assert client.post("/api/scan", json={
        "upload_id": uid, "threshold": 0.95}).json()["n_kept"] == 0
    assert client.post("/api/scan", json={
        "upload_id": uid, "threshold": 0.4}).json()["n_kept"] == 1
