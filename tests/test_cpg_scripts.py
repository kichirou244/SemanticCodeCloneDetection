"""Test phần **không** phụ thuộc Joern: đọc samples, chia lô, tổng hợp benchmark.

`scripts/build_cpg_export.py` gọi Joern thật bằng CLI
`joern --script <file.sc> --param k=v` (đã đối chiếu docs.joern.io/interpreter),
nhưng phần điều phối của nó (gom lô, index ánh xạ tên file, resume, báo cáo) kiểm
được bằng một Joern **giả** nhận đúng các tham số đó — vì vậy lỗi ở tầng điều phối
không phải chờ tới Kaggle mới lộ ra.
"""

from __future__ import annotations

import json
import re
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
script = ""
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--param":
        k, _, v = argv[i + 1].partition("=")
        params[k] = v
        i += 2
    elif argv[i] == "--script":
        script = Path(argv[i + 1]).name
        i += 2
    else:
        i += 1
print("[fake] script=" + script)

if "srcFile" in params:            # chế độ smoke / probe
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
    assert "joern_smoke.sc" in out


def test_run_probe_uses_its_own_script_and_isolated_from_smoke(tmp_path):
    """Probe và smoke là hai script khác nhau trả lời hai câu khác nhau. Nếu orchestrator
    lỡ gọi nhầm script thì ta âm thầm hỏi sai câu và tưởng đã dò API."""
    from scripts.build_cpg_export import run_probe

    out = run_probe(_fake_joern(tmp_path), tmp_path, 60)
    assert "joern_probe.sc" in out
    assert "joern_smoke.sc" not in out
    assert (tmp_path / "probe.cpp").exists()
    assert (tmp_path / "joern_probe.log").exists()


def test_probe_script_never_guesses_a_step_name():
    """Bản chất của probe: nó dùng `getClass.getMethods` (reflection) chứ không gọi tên
    step bằng cú pháp. Nếu ai đó thêm `cpg.call.ref` trở lại vào đây thì probe lại có
    thể chết lúc compile — mất đúng thứ nó sinh ra để tránh. Vì vậy file probe bị
    khoá: chỉ được dùng step đã xác nhận ở `wanted`/docstring, không được gọi trực tiếp."""
    src = (ROOT / "scripts" / "joern" / "joern_probe.sc").read_text(encoding="utf-8")
    # Bỏ comment trước khi kiểm: ghi lại tên step đã hỏng là điều **cần** giữ (đó là
    # lý do probe tồn tại), chỉ cái gọi thật trong code mới nguy hiểm.
    code = "\n".join(line.split("//", 1)[0] for line in src.splitlines())
    # `wanted` là danh sách *tên* để hỏi qua reflection — hỏi tên không phải gọi tên.
    for banned in ("cpg.call.ref", ".reachingDef.l", ".astChildren", "cpg.all.help"):
        assert banned not in code, f"probe gọi trực tiếp {banned} — se lai compile-sai"


def test_export_cpg_only_uses_api_steps_already_scanned():
    """CHỐT: `export_cpg.sc` không được dùng step chưa từng được compiler xác nhận.

    Ba vòng Kaggle liên tiếp chết vì đoán tên step. Test này biến "đã từng chạy
    thật" thành điều kiện kiểm tra tự động: mọi `cpg.<type>.<step>` trong exporter
    phải có mặt trong `docs/joern_api_v4.json` do `api_scan.py` sinh ra. Thêm
    một step lạ ⇒ test đỏ ngay ở máy local, thay vì mất 15 phút trên Kaggle.

    Bảng `cpg.call.help` không thay thế được: `cpg.call.argument` chạy tốt mà
    không có trong bảng, còn `cpg.call.ref` bị compiler bác — nên "có trong help"
    không tương đương "hợp lệ".
    """
    api = ROOT / "docs" / "joern_api_v4.json"
    if not api.exists():
        pytest.skip("chua chay scripts/joern/api_scan.py tren Kaggle")
    results = json.loads(api.read_text(encoding="utf-8"))["results"]
    verified = {name for name, r in results.items() if r.get("ok")}

    src = (ROOT / "scripts" / "joern" / "export_cpg.sc").read_text(encoding="utf-8")
    code = "\n".join(line.split("//", 1)[0] for line in src.splitlines())
    used = {f"{t}.{s}" for t, s in
            re.findall(r"cpg\.(method|call|identifier|local|all)\.(\w+)", code)}
    unknown = sorted(used - verified)
    assert not unknown, (
        f"export_cpg.sc dung step chua duoc compiler xac nhan: {unknown}. "
        f"Chay `python scripts/joern/api_scan.py --only <ten>` tren Kaggle truoc.")


def test_api_scan_controls_are_the_ones_joern_really_rejected():
    """Bản đồ API chỉ đáng tin nếu nó còn bác đúng những thứ Joern v4 đã bác.

    Nếu ai đó "sửa" `call.ref` thành ứng viên `expect="ok"` thì script sẽ tự
    dâng thành công và báo cáo sai mà không ai hay. Danh sách đối chứng phải
    giữ đúng hai tên đã bị bác bởi compiler ở các lần chạy thật.
    """
    from scripts.joern.api_scan import CANDIDATES

    controls = {c["name"] for c in CANDIDATES if c["expect"] == "fail"}
    assert controls == {"call.ref", "local.reachingDef"}
    assert all(c["expect"] == "ok" for c in CANDIDATES if c["name"] not in controls)
    # Không được trùng tên: kết quả ghi theo tên, trùng là âm thầm ghi đè.
    names = [c["name"] for c in CANDIDATES]
    assert len(names) == len(set(names))


def test_api_scan_never_puts_more_than_one_step_in_a_run():
    """Scala compile cả file: một biểu thức hỏng làm mất trắng cả nhóm, kể cả
    những cái đã hỏng mà ta chỉ muốn hỏi. Một ứng viên = một tiến trình."""
    from scripts.joern.api_scan import CANDIDATES, build_script

    for c in CANDIDATES:
        s = build_script(c, "cpg_scan_test")
        assert s.count("@main") == 1
        assert c["expr"] in s
        if c.get("setup"):
            assert c["setup"] in s


def test_api_scan_warns_when_a_control_suddenly_passes(capsys):
    """Nếu Joern đổi tên API thì phải nói ra, không được im lặng báo cáo bản đồ
    sai. Đây là lỗi tự tin — loại lỗi tệ nhất với công cụ đo."""
    from scripts.joern.api_scan import CANDIDATES, _summarise

    data = {"results": {"call.ref": {"ok": True, "value": "7"},
                        "local.reachingDef": {"ok": False, "error": "not a member"}}}
    _summarise(data, [c for c in CANDIDATES if c["name"] in data["results"]],
               Path("docs/joern_api_v4.json"))
    out = capsys.readouterr().out
    assert "CANH BAO" in out and "call.ref" in out


def test_api_scan_error_hint_separates_arity_from_missing_member():
    """Sai *tên* và sai *số tham số* dẫn tới hai việc khác hẳn nhau: cái sau chỉ
    cần thêm tham số, cái trước phải đổi thiết kế. Lần quét đầu in cột lý do rỗng
    nên ta không suy ra được gì từ chữ "FAIL" — nên phải trích đúng dòng compiler.
    """
    from scripts.joern.api_scan import error_hint

    missing = "x\n[E008] ------------ E008 ------------\n[E008]  value ref is not a member of Iterator[Call]\n"
    assert "not a member" in error_hint(missing)
    arity = "[E008] missing argument list for method reachingDef(Boolean)\n"
    assert "missing argument" in error_hint(arity)
    # Không bao giờ trả chuỗi rỗng: chuỗi rỗng chính là lỗi lần trước.
    assert error_hint("$ python x.py\nrc=1\n")
    assert error_hint("")


def test_meta_stats_reports_which_edge_types_exist(tmp_path):
    """`n_edges` lớn không bảo chứng CPG có data-flow: một đồ thị đầy AST vẫn
    trông rất ổn. Phải trả lời được "loại cạnh nào có, loại nào vắng"."""
    from scripts.build_cpg_export import _meta_stats

    (tmp_path / "a.meta").write_text(
        "n_nodes\t10\nfile_name\tsrc_a.cpp\n"
        "edge_types\tAST=9;CFG=4\n"
        "edge_types_missing\tCALL;REF;TARG;PARAMETER_OUT;REACHING_DEF;EVAL_ORDER\n",
        encoding="utf-8")
    (tmp_path / "b.meta").write_text(
        "n_nodes\t20\nfile_name\tsrc_a.cpp\n"
        "edge_types\tAST=18;CFG=8;REACHING_DEF=3\n"
        "edge_types_missing\tCALL;REF;TARG;PARAMETER_OUT;EVAL_ORDER\n",
        encoding="utf-8")
    s = _meta_stats(tmp_path, ["a", "b"])
    assert s["edge_types_present"] == ["AST", "CFG", "REACHING_DEF"]
    assert s["edge_type_totals"]["AST"] == 27
    assert "CALL" in s["edge_types_never_produced"]
    # Chỉ 1/2 hàm có data-flow → tỉ lệ thiếu phải lộ ra, không được gộp mất.
    assert s["edge_types_missing_ratio"]["REACHING_DEF"] == 0.5
    assert s["nodes_per_cpg_mean"] == 15.0
    # Cùng một file sinh ra 2 hàm ⇒ ánh xạ sample_id của Python sai.
    assert s["files_with_multiple_methods"] == 1


def test_export_cpg_avoids_constructs_that_failed_to_compile():
    """V1 chết vì hai thứ: type pattern `io.joern.x2cpg.*` và traversal `m.l` trên
    node đơn. Cả hai đều không phải thứ Joern v4 chấp nhận."""
    raw = Path("scripts/joern/export_cpg.sc").read_text(encoding="utf-8")
    # Bỏ comment trước khi kiểm tra: header giải thích *vì sao* bỏ các construct
    # này thì chứa tên chúng, và test sẽ dính nhầm vào chính lời giải thích.
    src = "\n".join(l.split("//", 1)[0] for l in raw.splitlines())
    assert "io.joern.x2cpg" not in src
    assert not re.search(r"\bcase \w+: io\.joern", src)
    # `m.l` — biểu thức chạy trên node đơn; CPGQL chỉ nhận Iterator.
    assert not re.search(r"\bm\.l\b", src)
    # Cạnh chưa xác minh được thì không được viết vào: `.meta` phải kê tên.
    for kind in ("REF", "TARG", "REACHING_DEF", "EVAL_ORDER"):
        assert kind in src


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


def test_zero_throughput_gives_null_estimate_not_absurd_number(monkeypatch, tmp_path):
    """0 hàm ra JSON thì KHÔNG được xuất `est_hours_full_dataset` — trước đây
    chia cho 1e-9 ra 14441944444.44 giờ, một con số vô nghĩa trong báo cáo còn tệ
    hơn không có số."""
    import scripts.build_cpg_export as mod

    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    processed = tmp_path / "samples.jsonl"
    processed.write_text("\n".join(
        json.dumps({"sample_id": f"s{i}", "code": "int f(){}", "split": "train"})
        for i in range(4)) + "\n", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"experiment:\n  seed: 42\ndataset:\n  processed: {processed}\n",
                   encoding="utf-8")
    monkeypatch.setattr(mod, "run_batch", lambda *a, **k: (0, 0))
    monkeypatch.setattr(mod, "joern_missing_reason", lambda *_: "")
    monkeypatch.setattr(sys, "argv",
                        ["build_cpg_export.py", "--config", str(cfg), "--batch-size", "2",
                         "--out-dir", str(out), "--work-dir", str(work), "--joern", "x"])
    mod.main()
    report = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert report["n_ok"] == 0 and report["n_missing"] == 4
    assert report["est_hours_full_dataset"] is None
    assert "KHONG uoc luong duoc" in report["est_hours_full_dataset_note"]


def test_progress_rate_counts_success_not_dispatch(monkeypatch, tmp_path, capsys):
    """Dòng tiến độ phải tính `ham/giay` trên hàm ĐÃ RA JSON. Tính trên hàm đã
    gửi đi thì một lần chạy thất bại toàn bộ vẫn in ra "7.8 ham/giay" trông như
    chạy tốt — đúng thứ gây hiểu nhầm nhất."""
    import scripts.build_cpg_export as mod

    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    samples = [{"sample_id": f"s{i}", "code": "int f(){}"} for i in range(4)]
    processed = tmp_path / "samples.jsonl"
    processed.write_text("\n".join(
        json.dumps({**s, "split": "train"}) for s in samples) + "\n", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"experiment:\n  seed: 42\ndataset:\n  processed: {processed}\n",
                   encoding="utf-8")
    # Joern giả nhưng trả về 0 JSON — giống hệt trường hợp 0/500 trên Kaggle.
    monkeypatch.setattr(mod, "run_batch", lambda *a, **k: (0, 0))
    monkeypatch.setattr(mod, "joern_missing_reason", lambda *_: "")
    monkeypatch.setattr(sys, "argv",
                        ["build_cpg_export.py", "--config", str(cfg), "--batch-size", "2",
                         "--out-dir", str(out), "--work-dir", str(work), "--joern", "x"])
    mod.main()
    printed = capsys.readouterr().out
    assert "ham/giay THANH CONG" in printed
    assert "0.00 ham/giay THANH CONG" in printed
    assert "CANH BAO: 0/4" in printed


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
