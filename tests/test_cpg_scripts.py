"""Test phần **không** phụ thuộc Joern: đọc samples, chia lô, tổng hợp benchmark.

`scripts/build_cpg_export.py` gọi Joern thật bằng CLI
`joern --script <file.sc> --param k=v` (đã đối chiếu docs.joern.io/interpreter),
nhưng phần điều phối của nó (gom lô, index ánh xạ tên file, resume, báo cáo) kiểm
được bằng một Joern **giả** nhận đúng các tham số đó — vì vậy lỗi ở tầng điều phối
không phải chờ tới Kaggle mới lộ ra.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_cpg_export import (
    benchmark,
    joern_or_die,
    read_samples,
    run_batch,
    run_smoke,
    write_batch,
)

# Joern giả: đúng hợp đồng CLI thật — nhận `--param srcDir/indexFile/outDir`,
# đọc index.tsv để ánh xạ tên file -> sample_id, ghi JSON + `.meta`.
FAKE_JOERN = r"""
import json, os, sys
from pathlib import Path

params = {}
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--param":
        k, _, v = argv[i + 1].partition("=")
        params[k] = v
        i += 2
    else:
        i += 1

if "srcFile" in params:            # chế độ smoke
    print("=== METHODS ===")
    print("  f | int f()")
    print("=== NODE LABELS ===")
    print("  METHOD, CALL")
    sys.exit(0)

src_dir = Path(params["srcDir"])
index = {}
for line in Path(params["indexFile"]).read_text(encoding="utf-8").splitlines():
    if line:
        fname, sid = line.split("\t", 1)
        index[fname] = sid
out = Path(params["outDir"])
assert src_dir.is_dir(), "srcDir khong ton tai - orchestrator chua gom lot"
out.mkdir(parents=True, exist_ok=True)
for fname, sid in index.items():
    if sid.endswith("-bad"):
        (out / (sid + ".json.error")).write_text("ValueError: fake parse error\n")
        continue
    n = 4 if len(sid) % 2 == 0 else 6
    nodes = [{"id": i, "label": "METHOD" if i == 0 else "CALL", "n_args": -1}
             for i in range(n)]
    edges = [{"src": 0, "dst": i, "label": "AST"} for i in range(1, n)]
    (out / (sid + ".json")).write_text(json.dumps(
        {"sample_id": sid, "nodes": nodes, "edges": edges}), encoding="utf-8")
    (out / (sid + ".meta")).write_text(
        f"n_methods_in_file\t1\nn_nodes\t{n}\nn_edges\t{n - 1}\nmethod_name\tf\n",
        encoding="utf-8")
"""

ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "scripts" / "build_cpg_export.py"


def _fake_joern(tmp_path: Path) -> list[str]:
    p = tmp_path / "fake_joern.py"
    p.write_text(FAKE_JOERN, encoding="utf-8")
    return [sys.executable, str(p)]


def test_read_samples_filters_split_and_limit(tmp_path):
    f = tmp_path / "samples.jsonl"
    f.write_text("\n".join(json.dumps({"sample_id": f"s{i}", "split": "train" if i < 3
                                       else "test", "code": f"f{i}();"})
                           for i in range(5)) + "\n", encoding="utf-8")
    assert len(read_samples(f, "train", 0)) == 3
    assert len(read_samples(f, "train", 2)) == 2
    assert read_samples(f, "test", 0)[0]["sample_id"] == "s3"


def test_write_batch_maps_filename_to_sample_id(tmp_path):
    """Joern chỉ trả về *tên file*; `sample_id` phải được ánh xạ ngược bằng index."""
    batch = tmp_path / "batch_0"
    index = write_batch([{"sample_id": "poj104-train-7", "code": "int f(){}"},
                         {"sample_id": "poj104-train-8", "code": "int g(){}"}], batch)
    rows = [line.split("\t") for line in index.read_text(encoding="utf-8").splitlines()]
    assert rows == [["00000.cpp", "poj104-train-7"], ["00001.cpp", "poj104-train-8"]]
    assert (batch / "00000.cpp").read_text(encoding="utf-8") == "int f(){}"


def test_run_batch_passes_real_cli_and_counts_ok(tmp_path):
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    samples = [{"sample_id": "a1", "code": "int f(){}"},
               {"sample_id": "a2-bad", "code": "???"}]
    n_ok, n_err = run_batch(samples, out, work, _fake_joern(tmp_path),
                            work / "batch_0", 60)
    assert (n_ok, n_err) == (1, 1)
    log = next(work.glob("joern_batch_*.log")).read_text(encoding="utf-8")
    assert "--script" in log and "--param" in log
    assert "export_cpg.sc" in log


def test_run_smoke_returns_joern_stdout(tmp_path):
    out = run_smoke("int f(){}", tmp_path, _fake_joern(tmp_path), 60)
    assert "METHODS" in out and "f | int f()" in out


def test_smoke_runs_without_dataset(tmp_path, monkeypatch):
    """`--smoke` phải chạy được khi CHƯA có `data/processed/...`.

    Lý do: smoke đứng trước bước dựng dataset trong notebook (dựng dataset =
    tải hàng trăm MB từ HuggingFace). Nếu smoke phụ thuộc dataset thì nó phải
    chạy sau, tức là tốn 100MB chỉ để kiểm tra 60 giây.
    """
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("experiment:\n  seed: 42\n"
                   "dataset:\n  processed: data/processed/poj104/samples.jsonl\n",
                   encoding="utf-8")
    # Cố tình trỏ `--joern` vào binary không tồn tại: nếu smoke chết ở chỗ Joern
    # thì nó đã đi qua được bước đọc dataset. Đây là cách kiểm thứ tự thực thi mà
    # không phụ thuộc Windows có lệnh `true` hay không.
    r = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--config", str(cfg), "--smoke",
         "--joern", "khong-ton-tai-joern-xyz", "--work-dir", str(tmp_path / "work")],
        capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert "samples.jsonl" not in r.stderr, r.stderr
    # Không phải FileNotFoundError mơ hồ: phải nói rõ đã thử những đường nào.
    assert "Khong tim thay Joern" in r.stdout
    assert "khong-ton-tai-joern-xyz" in r.stdout


def test_smoke_code_exercises_all_edge_kinds():
    """Snippet phải có nhánh điều khiển + lời gọi + gán — nếu không thì CPG ra
    vài node và không chứng minh được exporter dựng được cạnh data-flow."""
    from scripts.build_cpg_export import SMOKE_CODE

    assert "if" in SMOKE_CODE and "(" in SMOKE_CODE  # control structure + call
    assert "=" in SMOKE_CODE and "return" in SMOKE_CODE


def test_run_smoke_accepts_file(tmp_path):
    src = tmp_path / "real.cpp"
    src.write_text("int g(int x){return x+1;}", encoding="utf-8")
    out = run_smoke(src.read_text(encoding="utf-8"), tmp_path, _fake_joern(tmp_path), 60)
    assert "METHODS" in out


def test_benchmark_reports_rate_errors_and_meta(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    samples = [{"sample_id": "a1", "code": "int f(){}"},
               {"sample_id": "a2", "code": "int g(){return 2;}"}]
    (out / "a1.json").write_text("{}", encoding="utf-8")
    (out / "a1.meta").write_text("n_methods_in_file\t1\nn_nodes\t12\n", encoding="utf-8")
    (out / "a2.json.error").write_text("boom", encoding="utf-8")
    r = benchmark(out, samples, 10.0)
    assert r["n_ok"] == 1 and r["n_error"] == 1 and r["n_missing"] == 0
    assert r["functions_per_second"] == 0.1
    assert r["nodes_per_cpg_mean"] == 12.0
    assert r["files_with_multiple_methods"] == 0


def test_benchmark_counts_missing_as_failure(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    r = benchmark(out, [{"sample_id": "a1", "code": "x"}, {"sample_id": "a2", "code": "y"}], 5.0)
    assert r["n_missing"] == 2 and r["error_rate"] == 1.0


def test_benchmark_empty_export_dir(tmp_path):
    r = benchmark(tmp_path, [], 0.0)
    assert r["n_ok"] == 0 and r["functions_per_second"] == 0.0


def test_joern_or_die_prefers_env(monkeypatch):
    monkeypatch.setenv("JOERN_BIN", "/opt/joern/bin/joern")
    assert joern_or_die() == "/opt/joern/bin/joern"


def test_joern_or_die_falls_back_to_install_paths(monkeypatch, tmp_path):
    """Kaggle chạy dưới root: installer đặt binary ở /opt/joern, symlink ở
    /usr/local/bin — không phải ~/bin. Phải tìm được, không thì `FileNotFoundError`
    dù Joern vẫn nằm đúng chỗ (và tự tải lại 1,7GB)."""
    import scripts.build_cpg_export as mod

    fake = tmp_path / "joern"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv("JOERN_BIN", raising=False)
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    monkeypatch.setattr(mod, "JOERN_FALLBACK_PATHS", (tmp_path / "nope", fake))
    assert mod.joern_or_die() == str(fake)


def test_script_help_runs_without_joern():
    """`--help` phải chạy được kể cả khi máy không có Joern — nếu không, người
    dùng không đọc nổi các tuỳ chọn mà không cài gì."""
    r = subprocess.run([sys.executable, str(EXPORT_SCRIPT), "--help"],
                       capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "--batch-size" in r.stdout
    assert "--smoke" in r.stdout


def test_config_declares_blocked_status():
    """Config phải nói rõ đang bị chặn bởi gì, thay vì trông như sẵn sàng chạy."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "exp005_cpg_gnn.yaml").read_text(encoding="utf-8"))
    assert cfg["experiment"]["status"] == "BLOCKED_ON_JOERN"
    assert set(cfg["experiment"]["blocked_by"]) == {"java", "joern"}
    assert cfg["model"]["hidden_size"] % cfg["model"]["heads"] == 0


def test_toolchain_check_script_reports_missing_joern():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_toolchain.py")],
                       capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert "java" in payload["cpg"] and "joern" in payload["cpg"]
    assert payload["cpg"]["status"] in {"ready", "blocked_on_joern"}
    assert "Kaggle" in payload["cpg"]["where_to_install"]


@pytest.mark.parametrize("script", ["scripts/build_cpg_export.py"])
def test_scripts_are_syntactically_valid(script):
    compile((ROOT / script).read_text(encoding="utf-8"), script, "exec")


@pytest.mark.parametrize("script", ["scripts/joern/export_cpg.sc",
                                    "scripts/joern/joern_smoke.sc"])
def test_joern_scripts_use_param_interface(script):
    """Script Joern phải nhận đúng kênh `--param`; truyền sai kênh thì Joern
    chạy được nhưng nhận `null` và ghi rỗng — lỗi âm thầm khó nhất để phát hiện."""
    text = (ROOT / script).read_text(encoding="utf-8")
    assert "@main def exec(" in text
    assert "importCode(" in text
    assert "CPG_EXPORT_JOBS" not in text  # biến môi trường là cách cũ (joernpy)
