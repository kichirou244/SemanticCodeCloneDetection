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

    def score_batch(self, items):
        return [(self.score(a, b).score, {}) for a, b in items]


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


def test_scan_events_stream_progress_then_result(tmp_path):
    """Refactor NDJSON: gom events phải bằng đúng kết quả bản đồng bộ, tiến độ
    đơn điệu tăng và kết thúc đúng tổng số cặp."""
    import json as _json

    svc = make_service(tmp_path, {"M1": (0.5, {(A, B): 0.9, (A, C): 0.1,
                                               (B, C): 0.2})})
    up = svc.upload_zip(make_zip({"a.cpp": A, "b.cpp": B, "c.cpp": C}), "t.zip")
    kinds, progresses, result = [], [], None
    for kind, payload in svc.scan_upload_events(up["upload_id"]):
        kinds.append(kind)
        if kind == "progress":
            progresses.append((payload["done"], payload["total"]))
        elif kind == "result":
            result = payload
    assert kinds[0] == "progress" and kinds[-1] == "result"
    assert progresses[-1] == (3, 3)
    assert all(b >= a for (_, a), (_, b) in
               zip([(0, 0)] + progresses, progresses))
    expect = svc.scan_upload(up["upload_id"])
    # duration_ms do thoi gian chay, khong so duoc — so phan con lai.
    result.pop("duration_ms", None)
    expect.pop("duration_ms", None)
    assert result == expect


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


def _read_ndjson(resp):
    """Doc body NDJSON cua /api/scan: gom progress + tra ve result."""
    import json as _json

    progresses, result = [], None
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        msg = _json.loads(line)
        if msg["event"] == "progress":
            progresses.append((msg["done"], msg["total"]))
        elif msg["event"] == "result":
            result = msg["data"]
    assert result is not None, resp.text[:300]
    return progresses, result


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
    _, res = _read_ndjson(r2)
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
    _, r95 = _read_ndjson(client.post("/api/scan", json={
        "upload_id": uid, "threshold": 0.95}))
    assert r95["n_kept"] == 0
    _, r40 = _read_ndjson(client.post("/api/scan", json={
        "upload_id": uid, "threshold": 0.4}))
    assert r40["n_kept"] == 1
    assert r40["threshold_applied"] == {"M1": 0.4, "M2": 0.4}


@pytest.mark.parametrize("mid", ["M1", "M2", "M3"])
def test_batched_matches_per_pair(tmp_path, mid):
    """Forward gop phai cho cung so voi tung cap (padding mask che phan dem).
    Can checkpoint that (1.5GB) — thieu thi skip."""
    pytest.importorskip("transformers")
    from src.demo.registry import ModelRegistry

    root = Path(__file__).resolve().parents[1]
    reg = ModelRegistry.from_yaml(root / "configs" / "demo_web.yaml", root=root)
    scorer = reg.get(mid)
    if not scorer.checkpoint.exists():
        pytest.skip("thieu checkpoint %s" % mid)
    pairs = [(A, B), (A, C), (B, C), (C, A)]
    per = [scorer.score_raw(a, b) for a, b in pairs]
    batched = scorer.score_batch(pairs)
    assert len(batched) == len(pairs)
    # Chống tương đương rỗng: model phải nạp được thật (không skip lặng).
    assert all(s is not None for s, _ in per), "model chua load?"
    assert all(s is not None for s, _ in batched), "batch chua chay?"
    for (s1, _), (s2, _) in zip(per, batched):
        assert abs(s1 - s2) < 1e-5, (s1, s2)


def test_moss_evidence_spans_are_within_bounds():
    from src.demo.scorers.moss_scorer import MossScorer

    # Fixture A/B/C qua ngan cho k=20 (0 k-gram) — dung ham that.
    long_a = ("int bubble_sort(int arr[], int n) { int i = 0; "
              "while (i < n) { arr[i] = arr[i] + 1; i = i + 1; } return n; }")
    long_b = long_a.replace("arr", "a").replace("bubble_sort", "sort")
    m = MossScorer("B1", "moss", Path("."), k=20, w=5)
    ev = m.evidence(long_a, long_b)
    assert ev["n_shared"] > 0 and ev["spans_a"] and ev["spans_b"]
    for s, e in ev["spans_a"]:
        assert 0 <= s < e <= len(long_a)
    for s, e in ev["spans_b"]:
        assert 0 <= s < e <= len(long_b)
    ev2 = m.evidence(A, "int main() { completely different zebra; }")
    assert ev2["n_shared"] == 0
    assert ev2["spans_a"] == [] and ev2["spans_b"] == []
    from src.demo.scorers.base import BaseScorer
    assert BaseScorer.evidence.__doc__ is not None


def test_evidence_for_returns_token_indices():
    """evidence_for() (fetch riêng, không nằm trong analyze): B1 có tok,
    model lạ thì KeyError (endpoint map thành 404)."""
    from src.demo.registry import ModelRegistry
    from src.demo.service import DemoService

    root = Path(__file__).resolve().parents[1]
    reg = ModelRegistry.from_yaml(root / "configs" / "demo_web.yaml", root=root)
    svc = DemoService(reg)
    long_a = ("int bubble_sort(int arr[], int n) { int i = 0; "
              "while (i < n) { arr[i] = arr[i] + 1; i = i + 1; } return n; }")
    long_b = long_a.replace("arr", "a").replace("bubble_sort", "sort")
    out = svc.evidence_for("B1", long_a, long_b)
    assert out["model_id"] == "B1"
    ev = out["evidence"]
    assert ev is not None and ev["tok_a"] and ev["tok_b"]
    assert all(isinstance(i, int) and i >= 0 for i in ev["tok_a"])
    try:
        svc.evidence_for("M9", long_a, long_b)
    except KeyError:
        return
    raise AssertionError("phải KeyError với id lạ (→ 404)")


def test_spans_to_indices_mapping_is_exact():
    from src.demo.alignment.lexer import lex_code
    from src.demo.service import _spans_to_indices

    code = "int ab = 1;\nint cd = 2;"
    lines = lex_code(code)
    flat = [t for line in lines for t in line]
    texts = [t.text for t in flat]
    i_ab = texts.index("ab")
    tok = flat[i_ab]
    s = code.split("\n")[0].index("ab")
    got = _spans_to_indices(lines, code, [(s, s + 2)])
    assert got == [i_ab], (got, i_ab)
    assert _spans_to_indices(lines, code, [(9999, 99999)]) == []


LONG_A = ("int bubble_sort(int arr[], int n) { int i = 0; "
          "while (i < n) { arr[i] = arr[i] + 1; i = i + 1; } return n; }")
LONG_B = LONG_A.replace("arr", "a").replace("bubble_sort", "sort")


def _reg():
    from src.demo.registry import ModelRegistry

    root = Path(__file__).resolve().parents[1]
    return ModelRegistry.from_yaml(root / "configs" / "demo_web.yaml", root=root)


@pytest.mark.parametrize("mid", ["M1", "M2"])
def test_transformer_evidence_spans_are_words(mid):
    """Saliency (gradient×input): spans nam trong code, khong rong tren clones."""
    reg = _reg()
    m = reg.get(mid)
    if not m.checkpoint.exists():
        pytest.skip("thieu checkpoint %s" % mid)
    ev = m.evidence(LONG_A, LONG_B)
    assert ev is not None and ev["kind"] == "saliency"
    assert ev["spans_a"] and ev["spans_b"]
    for s, e in ev["spans_a"]:
        assert 0 <= s < e <= len(LONG_A)
    for s, e in ev["spans_b"]:
        assert 0 <= s < e <= len(LONG_B)
    ev2 = m.evidence(LONG_A, LONG_B)
    assert ev["spans_a"] == ev2["spans_a"]


def test_astnn_evidence_mutual_best_statements():
    """Statement cosine mutual-best: spans hop le, on dinh, khong rong."""
    reg = _reg()
    m = reg.get("M3")
    if not m.checkpoint.exists():
        pytest.skip("thieu checkpoint M3")
    ev = m.evidence(LONG_A, LONG_B)
    assert ev is not None and ev["kind"] == "stmt"
    assert ev["n_shared"] > 0
    assert len(ev["spans_a"]) == len(ev["spans_b"]) == ev["n_shared"]
    for s, e in ev["spans_a"]:
        assert 0 <= s < e <= len(LONG_A)
    ev2 = m.evidence(LONG_A, LONG_B)
    assert ev["spans_a"] == ev2["spans_a"]


def test_evidence_endpoint_returns_mapped_spans():
    """POST /api/evidence: B1 có tok, id lạ 404."""
    from fastapi.testclient import TestClient

    from src.demo.server import create_app

    reg = _reg()
    from src.demo.service import DemoService

    client = TestClient(create_app(service=DemoService(reg)))
    r = client.post("/api/evidence",
                    json={"code_a": LONG_A, "code_b": LONG_B, "model_id": "B1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_id"] == "B1"
    assert body["evidence"]["tok_a"]
    r2 = client.post("/api/evidence",
                     json={"code_a": LONG_A, "code_b": LONG_B, "model_id": "M9"})
    assert r2.status_code == 404
