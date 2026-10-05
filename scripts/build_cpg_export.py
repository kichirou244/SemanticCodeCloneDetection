"""Trích CPG hàng loạt bằng Joern + đo throughput thật.

    python scripts/build_cpg_export.py --smoke                 # 1 file, in output Joern
    python scripts/build_cpg_export.py --config configs/exp005_cpg_gnn.yaml \
        --split train --limit 500 --batch-size 100

Vì sao script này tồn tại (và vì sao phải chạy **trên Kaggle**, không cài ở máy
local): Joern cần JVM, còn máy dự án không có Java. Kaggle có sẵn JDK và quota
CPU miễn phí, nên toàn bộ phần Joern nằm trong một notebook
(`notebooks/exp005_cpg_extract_kaggle.ipynb`) chứ không cài gì lên máy.

Vì sao chia lô thay vì một tiến trình mỗi hàm: Joern khởi động JVM + scala-repl
mất **vài chục giây**, trong khi dựng CPG một hàm mất dưới 1 giây. 62.000 tiến
trình nghĩa là hàng tuần chỉ để khởi động JVM. Nên `--batch-size` file được gộp
vào MỘT thư mục rồi `scripts/joern/export_cpg.sc` chạy đúng một lần cho cả lô.

Đầu ra:
  - `artifacts/cpg_export/<sample_id>.json` — CPG từng hàm (xoá được, dựng lại được).
  - `artifacts/cpg_export/<sample_id>.meta` — số node/cạnh, tên hàm (kiểm chất lượng).
  - `artifacts/cpg_export/benchmark.json` — **số đo quyết định có làm M4 không**:
    số hàm/giây, tỉ lệ lỗi parse, ước lượng cho toàn bộ dataset.

Có `--resume` (mặc định bật): bỏ qua hàm đã có JSON. Lần chạy 4 giờ bị kill giữa
chừng thì chạy lại tiếp từ chỗ dừng, không mất tiến độ.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.seed import set_global_seed
from src.representations.cpg.schema import EDGE_LABELS

DEFAULT_OUT = "artifacts/cpg_export"
DEFAULT_WORK = "artifacts/cpg_work"
EXPORT_SCRIPT = Path(__file__).resolve().parent / "joern" / "export_cpg.sc"
SMOKE_SCRIPT = Path(__file__).resolve().parent / "joern" / "joern_smoke.sc"
PROBE_SCRIPT = Path(__file__).resolve().parent / "joern" / "joern_probe.sc"

# Nơi `joern-install.sh` hay đặt binary: `~/bin/joern` khi chạy bình thường,
# `/opt/joern` khi chạy dưới root (Kaggle). Tách thành hằng để test được.
JOERN_FALLBACK_PATHS = (Path.home() / "bin" / "joern", Path("/opt/joern/joern"),
                        Path("/usr/local/bin/joern"))

# Đoạn C cho smoke test: nhỏ, tự chứa, và **cố tình có cả ba loại cạnh** ta cần
# (AST + CALL + ARGUMENT) cùng một phép gán để có `reachingDef`. Một snippet quá
# tầm thường (chỉ `return 0;`) sẽ cho ra CPG 3 node và không chứng minh được
# exporter dựng được cạnh data-flow.
SMOKE_CODE = """#include <stdio.h>

int classify(int n) {
  int total = 0;
  for (int i = 0; i < n; i++) {
    if (i % 2 == 0) {
      total = total + i;
    } else {
      printf("%d\\n", total);
    }
  }
  return total;
}
"""


def _as_argv(joern_cmd: str | list[str]) -> list[str]:
    """Chuẩn hoá lệnh Joern thành argv.

    Nhận cả chuỗi lẫn list. Chuỗi thì tách bằng `shlex` với `posix` đúng theo
    platform (posix trên Windows ăn mất backslash: `D:\\Code` thành `D:Code`).
    Trên Windows, `.bat`/`.cmd` không chạy trực tiếp qua CreateProcess được
    nên phải bọc `cmd /d /c` — kể cả khi gọi bằng tên trần mà `which` trỏ tới
    file `.bat` (đúng trường hợp `joern` -> `joern.bat` trong PATH).
    """
    if isinstance(joern_cmd, (list, tuple)):
        argv = [str(c) for c in joern_cmd]
    else:
        argv = shlex.split(joern_cmd, posix=(sys.platform != "win32"))
    if sys.platform == "win32" and argv:
        exe = argv[0]
        needs_cmd = exe.lower().endswith((".bat", ".cmd"))
        if not needs_cmd and "." not in os.path.basename(exe):
            found = shutil.which(exe)
            needs_cmd = bool(found) and found.lower().endswith((".bat", ".cmd"))
        if needs_cmd:
            comspec = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
            argv = [comspec, "/d", "/c", *argv]
    return argv


def _param_value(text: str) -> str:
    """Bọc value `--param` trong ngoặc kép trên Windows.

    Launcher `.bat` của Joern tách tham số bằng `%n`/`shift`, mà `%n` cắt ở
    dấu `=` (đã chứng minh: `srcDir=C:\\x` tới mainargs chỉ còn `srcDir`).
    Bọc `"k=v"` thì `%n` giữ nguyên khối, JVM lột ngoặc khi parse argv và
    mainargs nhận đúng `k=v` (đã chứng minh bằng probe `PROBE_GOT` sạch).
    Linux giữ nguyên (không bọc) để đường Kaggle không đổi gì.
    """
    if sys.platform == "win32" and "=" in text:
        return f'"{text}"'
    return text


def joern_or_die() -> str:
    """Tìm lệnh Joern: `$JOERN_BIN` → `PATH` → các nơi installer hay đặt binary.

    Bản cài chính thức (`joern-install.sh`) đặt binary ở `~/bin/joern`, nhưng khi
    chạy dưới root (đúng trường hợp của Kaggle) nó đặt `/opt/joern` và tạo
    symlink trong `/usr/local/bin` — chỉ check một đường là fail giả rồi tải lại
    1,7GB cho nothing.
    """
    env = os.environ.get("JOERN_BIN")
    if env:
        return env
    found = shutil.which("joern")
    if found:
        return found
    for p in JOERN_FALLBACK_PATHS:
        if p.exists():
            return str(p)
    return "joern"


def joern_missing_reason(joern_cmd: str | list[str]) -> str:
    """Trả về lý do không tìm thấy Joern, hoặc chuỗi r���ng nếu tìm thấy.

    Cần hàm này vì `subprocess.run` với binary không tồn tại ném
    `FileNotFoundError: [WinError 2]` — không nói cho biết *đã thử những đường
    nào*. Trên Kaggle mà PATH lệch (chạy dưới root, binary ở `/opt/joern`) thì
    người dùng sẽ phải tự đoán.
    """
    exe = _as_argv(joern_cmd)[0]
    if exe == "joern":
        return "" if shutil.which("joern") else (
            "Khong tim thay lenh `joern` trong PATH")
    if Path(exe).exists():
        return ""
    return (f"Khong tim thay Joern tai {exe!r}. Da thu: $JOERN_BIN, PATH, "
            f"{', '.join(str(p) for p in JOERN_FALLBACK_PATHS)}. "
            f"Neu vua cai xong, chay: ls -l /usr/local/bin/joern /opt/joern/joern")


def read_samples(processed: Path, split: str | None, limit: int) -> list[dict]:
    rows = []
    with Path(processed).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if split and r.get("split") != split:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def write_batch(samples: list[dict], batch_dir: Path) -> Path:
    """Gom lô thành thư mục `*.cpp` + `index.tsv` (tên file -> sample_id).

    Joern chỉ thấy *tên file*, không thấy `sample_id` của dataset; nên phải ánh xạ
    ngược bằng file index, không đoán từ tên file (sample_id dạng `poj104-train-7`
    không suy ra được từ tên file Joern tự sinh).
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, r in enumerate(samples):
        fname = f"{i:05d}.cpp"
        (batch_dir / fname).write_text(r["code"], encoding="utf-8")
        lines.append(f"{fname}\t{r['sample_id']}")
    index = batch_dir / "index.tsv"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def _log(work_dir: Path, tag: str, cmd: list[str], proc: subprocess.CompletedProcess) -> None:
    (work_dir / f"{tag}.log").write_text(
        f"$ {' '.join(shlex.quote(c) for c in cmd)}\nrc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-20000:]}\n--- stderr ---\n{proc.stderr[-8000:]}",
        encoding="utf-8")


def run_smoke(code: str, work_dir: Path, joern_cmd: str, timeout_s: int) -> str:
    """Chạy `joern_smoke.sc` trên MỘT đoạn code và trả về stdout để đối chiếu tên step.

    Cố ý **không** đọc dataset: smoke test phải chạy được trước cả bước dựng
    dataset (mà dựng dataset nghĩa là tải hàng trăm MB từ HuggingFace). Một
    phép kiểm tra mà phải tải dữ liệu mới chạy được thì không phải smoke test.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    src = work_dir / "smoke.cpp"
    src.write_text(code, encoding="utf-8")
    cmd = _as_argv(joern_cmd) + ["--script", str(SMOKE_SCRIPT),
                                   "--param", _param_value(f"srcFile={src}")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    _log(work_dir, "joern_smoke", cmd, proc)
    return proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")


def run_probe(joern_cmd: str, work_dir: Path, timeout_s: int) -> str:
    """Chạy `joern_probe.sc` — dò API Joern v4 bằng reflection.

    Tách riêng khỏi `run_smoke` vì hai lần chạy này trả lời hai câu khác nhau và
    **fail theo hai kiểu khác nhau**:
      - probe hỏi "Joern v4 tên step này là gì" → dùng `getMethods`, không có tên
        nào sai được ⇒ chắc chắn compile.
      - smoke hỏi "bảng step của Joern in ra gì" (`cpg.call.help`) → dùng step
        `.help`, tên đúng chưa xác nhận nên vẫn có thể chết lúc compile.
    Chạy probe trước: nếu probe đã trả lời xong thì smoke chỉ còn là đối chiếu
    chéo, thất bại của nó không chặn gì cả.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    src = work_dir / "probe.cpp"
    src.write_text(SMOKE_CODE, encoding="utf-8")
    cmd = _as_argv(joern_cmd) + ["--script", str(PROBE_SCRIPT),
                                   "--param", _param_value(f"srcFile={src}")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    _log(work_dir, "joern_probe", cmd, proc)
    return proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")


def run_batch(samples: list[dict], out_dir: Path, work_dir: Path, joern_cmd: str,
              batch_dir: Path, timeout_s: int) -> tuple[int, int]:
    """Một lần khởi động Joern cho cả lô. Trả về (số OK, số lỗi)."""
    index = write_batch(samples, batch_dir)
    cmd = _as_argv(joern_cmd) + ["--script", str(EXPORT_SCRIPT),
           "--param", _param_value(f"srcDir={batch_dir}"),
           "--param", _param_value(f"indexFile={index}"),
           "--param", _param_value(f"outDir={out_dir}")]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        _log(work_dir, f"joern_batch_{batch_dir.name}", cmd, proc)
    except subprocess.TimeoutExpired as exc:
        (work_dir / f"joern_batch_{batch_dir.name}.log").write_text(
            f"TIMEOUT sau {timeout_s}s\n{(exc.stdout or b'').decode('utf-8', 'replace')[-5000:]}",
            encoding="utf-8")
        return 0, len(samples)
    n_ok = sum(1 for r in samples if (out_dir / f"{r['sample_id']}.json").exists())
    n_err = sum(1 for r in samples if (out_dir / f"{r['sample_id']}.json.error").exists())
    # Lô chết mà **không in lý do** là kiểu lỗi tốn nhiều thời gian nhất: biết
    # "0 hàm ra JSON" mà không biết vì sao thì phải đoán, mà đoán thì tốn một cả
    # vòng chạy Joern nữa. Nên in thẳng dòng compiler nói ngay tại đây.
    if n_ok == 0 and proc is not None:
        print(_failure_digest(proc, batch_dir))
    return n_ok, n_err


# Mẫu nhận diện dòng lỗi thật trong output Joern (E008 compile, exception
# runtime, frontend crash). Không khớp thì không phải lỗi — banner khởi động
# ("Invoking CPG generator", "Creating project"...) không bao giờ khớp.
_ERROR_PAT = re.compile(
    r"\[E\d+\]|error|exception|caused by|not a member|not found|cannot|"
    r"failed|fatal|killed|out of memory|at (scala|java|io\.joern|replpp|ammonite)\.",
    re.IGNORECASE,
)


def _failure_digest(proc: subprocess.CompletedProcess, batch_dir: Path) -> str:
    """Rút lỗi đầu tiên Joern thật sự báo ra.

    KHÔNG lấy "12 dòng đầu": lỗi compile Scala / exception runtime nằm SAU
    banner khởi động của Joern, nên 12 dòng đầu toàn là banner ("Invoking CPG
    generator", "Creating project"...) — đã chứng minh bằng một lô chết mà 12
    dòng in ra không chứa lỗi. Tìm dòng khớp mẫu lỗi trước, kèm 1 dòng ngữ
    cảnh; chỉ khi không có dòng nào khớp mới rơi về 12 dòng đầu.
    """
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    lines = [ln.rstrip() for ln in blob.splitlines() if ln.strip()]
    hits: list[str] = []
    for i, ln in enumerate(lines):
        if ln.startswith("$ "):
            continue
        if _ERROR_PAT.search(ln):
            if i > 0 and not lines[i - 1].startswith("$ ") \
                    and lines[i - 1] not in hits:
                hits.append(lines[i - 1])
            if ln not in hits:
                hits.append(ln)
            if len(hits) >= 12:
                break
    head = hits[:12] if hits else [
        ln for ln in lines if not ln.startswith("$ ")][:12]
    return ("\n[cpg-export] LÔ CHẾT mà không ra JSON: "
            f"{batch_dir.name} (rc={proc.returncode}).\n"
            "  Dưới đây là dòng lỗi Joern báo (không phải banner khởi động):\n    "
            + "\n    ".join(head)
            + f"\n  Log đầy đủ: artifacts/cpg_work/joern_batch_{batch_dir.name}.log")


def _meta_stats(out_dir: Path, sample_ids: list[str]) -> dict:
    """Đọc `.meta` để đo chất lượng CPG (n_nodes, node label, loại cạnh nào có)."""
    n_nodes, parsed = [], 0
    synthetic = 0
    edge_totals: dict[str, int] = {}
    label_totals: dict[str, int] = {}
    missing_seen: dict[str, int] = {}
    for sid in sample_ids:
        p = out_dir / f"{sid}.meta"
        if not p.exists():
            continue
        parsed += 1
        kv = dict(line.split("\t", 1) for line in
                  p.read_text(encoding="utf-8").splitlines() if "\t" in line)
        if int(kv.get("n_nodes", 0)) > 0:
            n_nodes.append(int(kv["n_nodes"]))
        # Hàm rơi về method GIẢ của Joern (`<global>`, `<includes>`…) = dấu hiệu
        # duy nhất còn lại cho "file này không dựng được method thật".
        #
        # ⚠ ĐÃ BỎ metric `files_with_multiple_methods` (đếm theo `file_name`):
        # nó **không thể đúng**. `write_batch` đặt tên lại `00000.cpp`… từ đầu mỗi
        # lô, nên `file_name` lặp đúng bằng số lô — đo được `100` trên lô 500 (5
        # lô) và `0` trên lô 20 (1 lô) **dù dữ liệu hoàn toàn tốt**. Đây là
        # báo động giả: nó kêu "100 hàm bị lấy sai" trên đúng lần chạy đã sửa xong
        # lỗi ghi đè, và âm thầm giấu lỗi thật ở các lần khác.
        if kv.get("method_name", "").startswith("<"):
            synthetic += 1
        # Cộng dồn histogram cạnh. `edge_types` là "loại nào có trong hàm này",
        # `edge_types_missing` là "loại nào vắng" — phải thấy cả hai, không được
        # chỉ nhìn tổng số cạnh: một CPG đầy AST mà không có data-flow vẫn có
        # `n_edges` rất lớn và trông rất ổn.
        for item in filter(None, kv.get("edge_types", "").split(";")):
            k, _, v = item.partition("=")
            if v.isdigit():
                edge_totals[k] = edge_totals.get(k, 0) + int(v)
        for item in filter(None, kv.get("node_labels", "").split(";")):
            k, _, v = item.partition("=")
            if v.isdigit():
                label_totals[k] = label_totals.get(k, 0) + int(v)
        for k in filter(None, kv.get("edge_types_missing", "").split(";")):
            missing_seen[k] = missing_seen.get(k, 0) + 1
    return {
        "n_meta_parsed": parsed,
        "nodes_per_cpg_mean": round(sum(n_nodes) / max(len(n_nodes), 1), 1),
        "nodes_per_cpg_p5": sorted(n_nodes)[int(0.05 * (len(n_nodes) - 1))] if n_nodes else 0,
        "n_synthetic_method_fallback": synthetic,
        "node_label_totals": dict(sorted(label_totals.items())),
        "edge_type_totals": edge_totals,
        "edge_types_present": sorted(edge_totals),
        "edge_types_never_produced": sorted(
            k for k in EDGE_LABELS if k not in edge_totals),
        "edge_types_missing_ratio": {k: round(v / max(parsed, 1), 3)
                                     for k, v in sorted(missing_seen.items())},
    }


def benchmark(out_dir: Path, samples: list[dict], seconds: float) -> dict:
    sizes = {r["sample_id"]: len(r["code"]) for r in samples}
    per_file = [Path(f).stat().st_size for f in sorted(out_dir.glob("*.json"))]
    ok = [s for s in samples if (out_dir / f"{s['sample_id']}.json").exists()]
    n_err = len(list(out_dir.glob("*.json.error")))
    chars = [sizes[s["sample_id"]] for s in ok] or [1]
    n_missing = max(len(samples) - len(ok) - n_err, 0)
    report = {
        "n_requested": len(samples),
        "n_ok": len(ok),
        "n_error": n_err,
        "n_missing": n_missing,
        "error_rate": round((n_err + n_missing) / max(len(samples), 1), 4),
        "fail_rate_note": "n_missing = hàm không ra JSON (parse fail / lô timeout) — "
                          "tính chung vào error_rate",
        "wall_seconds": round(seconds, 1),
        "functions_per_second": round(len(ok) / max(seconds, 1e-9), 3),
        "seconds_per_function": round(seconds / max(len(ok), 1), 3),
        "json_bytes_mean": int(sum(per_file) / max(len(per_file), 1)),
        "code_chars_mean": int(sum(chars) / len(chars)),
        "code_chars_p95": int(sorted(chars)[int(0.95 * (len(chars) - 1))]),
    }
    report.update(_meta_stats(out_dir, [s["sample_id"] for s in ok]))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp005_cpg_gnn.yaml")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = toàn bộ split")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="số hàm / 1 lần khởi động Joern (JVM khởi động tốn vài chục giây)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--work-dir", default=DEFAULT_WORK)
    ap.add_argument("--joern", default=joern_or_die(), help="lệnh Joern")
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="chạy joern_smoke.sc trên 1 đoạn C và in output "
                         "(không cần dataset — chạy cái này TRƯỚC mọi bước khác)")
    ap.add_argument("--smoke-file", default="",
                    help="file .cpp để smoke; mặc định dùng SMOKE_CODE nhúng sẵn")
    ap.add_argument("--probe", action="store_true",
                    help="chạy joern_probe.sc: dò API Joern v4 bằng reflection để "
                         "chốt tên step, KHÔNG cần dataset. Chạy cái này TRƯỚC khi "
                         "sửa export_cpg.sc — nó không thể fail vì tên step sai")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    set_global_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    out_dir, work_dir = Path(args.out_dir), Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    missing = joern_missing_reason(args.joern)
    if missing:
        print(f"[cpg-export] {missing}")
        raise SystemExit(1)

    if args.smoke:
        # Đặt TRƯỚC khi đọc dataset: smoke không được phụ thuộc
        # `data/processed/...` — nếu phụ thuộc thì nó phải chạy sau khi tải
        # dataset, tức là tốn hàng trăm MB để kiểm tra 60 giây.
        code = (Path(args.smoke_file).read_text(encoding="utf-8")
                if args.smoke_file else SMOKE_CODE)
        print(run_smoke(code, work_dir, args.joern, args.timeout_s))
        return

    if args.probe:
        # Cùng nguyên tắc với smoke: tự chứa, không đọc dataset. Chạy nó ngay sau
        # khi Joern cài xong để biết tên step thật trước khi đụng vào exporter.
        print(run_probe(args.joern, work_dir, args.timeout_s))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    processed = Path(cfg["dataset"]["processed"])
    samples = read_samples(processed, args.split, args.limit)
    if args.no_resume:
        for f in out_dir.glob("*.json"):
            f.unlink()
    todo = [s for s in samples if not (out_dir / f"{s['sample_id']}.json").exists()]
    print(f"[cpg-export] split={args.split} yeu cau={len(samples):,} "
          f"con lai={len(todo):,}", flush=True)
    if not todo:
        print("[cpg-export] khong con gi de lam — dung report cu")
        return

    t0 = time.perf_counter()
    n_ok = n_err = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        ok, err = run_batch(batch, out_dir, work_dir, args.joern,
                            work_dir / f"batch_{i:06d}", args.timeout_s)
        n_ok += ok
        n_err += err
        el = time.perf_counter() - t0
        done = i + len(batch)
        # `ham/giay` phải tính trên **hàm đã ra JSON**, không phải hàm đã gửi đi.
        # Trước đây dùng `done/elapsed` nên khi exporter hỏng cả lô, dòng log vẫn
        # in ra "7.81 ham/giay" trông như đang chạy tốt — đúng thứ gây hiểu nhầm
        # nhất: một lần chạy thất bại trông giống một lần chạy nhanh.
        rate = n_ok / max(el, 1e-9)
        print(f"[cpg-export] {done:,}/{len(todo):,} da gui  ok={n_ok:,} err={n_err}  "
              f"{el / 60:.1f} phut  ({rate:.2f} ham/giay THANH CONG)", flush=True)
        if n_ok == 0 and n_err == 0 and done == len(todo):
            print(f"[cpg-export] CANH BAO: 0/{len(todo):,} ham ra JSON. 90% khong phai "
                  f"loi du lieu — ma xem {work_dir / 'joern_batch_*.log'} "
                  f"(export_cpg.sc compile sai thi ca lo chet trong ~13s/lot).", flush=True)

    seconds = time.perf_counter() - t0
    report = benchmark(out_dir, samples, seconds)
    n_total = sum(1 for line in processed.open(encoding="utf-8") if line.strip())
    per_s = report["functions_per_second"] or 0.0
    n_batches = max(1, -(-len(todo) // args.batch_size))
    if per_s <= 0:
        # Trước đây chia cho 1e-9 nên lỗi 0/500 ra `est_hours_full_dataset =
        # 14441944444.44` giờ — một con số vô nghĩa trong báo cáo còn tệ hơn là
        # không có số. Không có throughput thì không có ước lượng.
        report["est_hours_full_dataset"] = None
        report["est_hours_full_dataset_note"] = (
            f"KHONG uoc luong duoc: n_ok=0 nen khong co ham/giay. Xem "
            f"{work_dir / 'joern_batch_*.log'}")
    else:
        report["est_hours_full_dataset"] = round(n_total / per_s / 3600, 2)
        report["est_hours_full_dataset_note"] = (
            f"{n_total:,} ham / {per_s:.3f} ham.s^-1 — uoc luong CPU, KHONG tinh gioi "
            f"han GPU va KHONG tinh thoi gian tai dataset")
    # Chi phí theo lô. ⚠ Đổi tên khỏi `jvm_startup_*` vì nó **không bao giờ đo
    # startup**: `seconds / n_batches` là thời gian trung bình của CẢ lô (JVM +
    # dựng CPG thật), còn `jvm_startup_hours_full_dataset` lại nhân con số đó với
    # số lô của *lần chạy này* chứ không phải của toàn dataset — nên lô 500 in ra
    # `0.03` giờ cho "full dataset" trong khi thực tế 520 lô x 22.5s ≈ 3.25 giờ.
    # Tên sai làm người đọc tưởng có thể tăng batch_size để tiết kiệm 0.03 giờ.
    batch_seconds_mean = seconds / max(n_batches, 1)
    report["batch_seconds_mean"] = round(batch_seconds_mean, 1)
    report["batches_full_dataset"] = -(-n_total // args.batch_size)
    report["batch_hours_full_dataset"] = round(
        report["batches_full_dataset"] * batch_seconds_mean / 3600, 2)
    # Tự kiểm tra nhất quán: hai đường ước lượng phải gần nhau. Lệch xa nghĩa là
    # throughput không đều giữa đầu lô và cuối lô (thường do lô cuối nhỏ hơn).
    report["estimate_crosscheck_ok"] = (
        report["est_hours_full_dataset"] is not None
        and abs(report["batch_hours_full_dataset"]
                - report["est_hours_full_dataset"]) < 0.25 * max(
                    report["est_hours_full_dataset"], 1e-9))
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
