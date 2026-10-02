"""DÒ API JOERN V4 BẰNG CHÍNH COMPILER LÀM ORACLE.

BỐI CẢNH (đọc trước khi sửa gì ở đây): ba lần chạy Kaggle liên tiếp đều chết ở
bước *compile* trước khi in được một dòng nào:

    1. `cpg.call.ref`           → value ref is not a member of Iterator[Call]
    2. `cpg.local.reachingDef`  → value reachingDef is not a member of Iterator[Local]
    3. `io.joern.x2cpg.X2CpgNode` → package không tồn tại trong v4

Mỗi lần mất ~15 phút vòng Kaggle để đổi đúng một tên. Nguyên nhân gốc không phải
"Joern khó" mà là **đoán tên API không có oracle nào kiểm tra giùm**. Bảng
`cpg.call.help` mà smoke in ra cũng không cứu được: nó **không đầy đủ** —
`cpg.call.argument` chạy tốt nhưng không có trong bảng `Call`, còn `ref` bị
compiler bác. Nên "có trong help" không tương đương "hợp lệ".

CÁCH LÀM Ở ĐÂY: hỏi chính compiler. Mỗi biểu thức nghi ngờ được đưa vào **một
`.sc` riêng**, chạy `joern --script` một lần, rồi đọc kết quả:
    - có dòng `VALUE ...`  → hợp lệ, kèm luôn **giá trị thật** (không chỉ yes/no)
    - không có, `rc != 0`   → không hợp lệ, kèm nguyên văn lỗi compiler

Chi phí là ~13s JVM khởi động mỗi cái, nhưng trả về một bản đồ API **vĩnh viễn**
lưu vào `docs/joern_api_v4.json` — sau đó `export_cpg.sc` và test của repo dựa
vào file đó thay vì vào trí nhớ của tôi.

Hai mục `expect="fail"` là **đối chứng**: chúng đã hỏng chắc chắn ở ba lần chạy
trước. Nếu script này báo chúng hợp lệ thì chính script này hỏng (ví dụ: joern
mới đã đổi tên), và ta phải biết ngay chứ không tin báo cáo của nó.

CÁCH DÙNG:
    python scripts/joern/api_scan.py --out docs/joern_api_v4.json
    python scripts/joern/api_scan.py --only astParent --verbose   # dò lại
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build_cpg_export import (  # noqa: E402
    SMOKE_CODE,
    joern_missing_reason,
)

OUT_DEFAULT = ROOT / "docs" / "joern_api_v4.json"

# `setup` chạy sau `importCode`; `expr` là biểu thức CPGQL cần kiểm tra hợp lệ.
# Chọn `cpg.method.ast.l.head` (một node cây cú pháp) làm `n` vì `.astParent` chỉ
# tồn tại trên node thuộc cây cú pháp — nếu lấy `cpg.all.head` có thể ra FILE hoặc
# NAMESPACE, rồi ta đoán nhầm "API không có" trong khi thực ra node sai loại.
CANDIDATES: list[dict] = [
    # --- truy vấn hàm: định danh, thứ tự, vị trí trong file -------------------
    {"name": "method.name", "expr": "cpg.method.name.l.size", "expect": "ok"},
    {"name": "method.fullName", "expr": "cpg.method.fullName.l.size", "expect": "ok"},
    {"name": "method.methodFullName", "expr": "cpg.method.methodFullName.l.size",
     "expect": "ok"},
    {"name": "method.file.name", "expr": "cpg.method.file.name.l.size", "expect": "ok"},
    {"name": "method.lineNumber", "expr": "cpg.method.lineNumber.l.size", "expect": "ok"},
    {"name": "method.signature", "expr": "cpg.method.signature.l.size", "expect": "ok"},

    # --- thân hàm ------------------------------------------------------------
    {"name": "method.body", "expr": "cpg.method.body.l.size", "expect": "ok"},
    {"name": "method.local", "expr": "cpg.method.local.l.size", "expect": "ok"},
    {"name": "method.call", "expr": "cpg.method.call.l.size", "expect": "ok"},

    # --- call graph: dựng cạnh CALL ----------------------------------------
    {"name": "method.callOut", "expr": "cpg.method.callOut.l.size", "expect": "ok"},
    {"name": "method.callIn", "expr": "cpg.method.callIn.l.size", "expect": "ok"},
    {"name": "call.callee", "expr": "cpg.call.callee.l.size", "expect": "ok"},
    {"name": "call.methodFullName", "expr": "cpg.call.methodFullName.l.size",
     "expect": "ok"},

    # --- cây cú pháp & CFG ----------------------------------------------------
    {"name": "method.cfgNext", "expr": "cpg.method.cfgNext.l.size", "expect": "ok"},
    {"name": "method.typeDecl", "expr": "cpg.method.typeDecl.l.size", "expect": "ok"},
    {"name": "call.target", "expr": "cpg.call.target.l.size", "expect": "ok"},
    {"name": "call.source", "expr": "cpg.call.source.l.size", "expect": "ok"},

    # --- data flow ------------------------------------------------------------
    {"name": "call.reachingDef", "expr": "cpg.call.reachingDef.l.size", "expect": "ok"},
    {"name": "identifier.reachingDef", "expr": "cpg.identifier.reachingDef.l.size",
     "expect": "ok"},

    # --- thuộc tính trên node đơn: exporter cần id/label khi ghi JSON ---------
    {"name": "node.id", "setup": "val n = cpg.method.ast.l.head", "expr": "n.id",
     "expect": "ok"},
    {"name": "node.label", "setup": "val n = cpg.method.ast.l.head", "expr": "n.label",
     "expect": "ok"},
    {"name": "node.astParent", "setup": "val n = cpg.method.ast.l.head",
     "expr": "Iterator(n).astParent.l.size", "expect": "ok"},
    {"name": "node.astChildren", "setup": "val n = cpg.method.ast.l.head",
     "expr": "Iterator(n).astChildren.l.size", "expect": "ok"},

    # === VÒNG 2 (sau lần quét đầu tiên) =====================================
    # Đây là những mắt xích exporter *bắt buộc* phải có. Chúng không nằm trong 25
    # ứng viên đầu vì lúc đó ta chưa biết `reachingDef` hỏng vì lý do gì — mà
    # "không có member" và "thiếu tham số" là hai việc khác hẳn nhau. Lần quét đầu
    # in ra cột lý do **rỗng** (bug của chính script này), nên không suy ra được
    # gì từ "FAIL"; phải hỏi lại với đúng dạng lời gọi.

    # Cạnh CFG là mắt xích số 1. Hai đường độc lập: (a) đi theo cạnh thật,
    # (b) nối các node theo thứ tự tự nhiên. Nếu (a) hỏng thì (b) vẫn dựng được
    # CFG xấp xỉ — và phải ghi rõ trong `.meta` rằng CFG là bản xấp xỉ.
    {"name": "node.cfgNext", "setup": "val n = cpg.method.cfgNode.l.head",
     "expr": "Iterator(n).cfgNext.l.size", "expect": "ok"},
    {"name": "cfgNode.sorted", "expr": "cpg.method.cfgNode.sorted.l.size", "expect": "ok"},
    {"name": "node.order", "setup": "val n = cpg.method.cfgNode.l.head",
     "expr": "Iterator(n).order.l.size", "expect": "ok"},

    # ARGUMENT + PARAMETER_OUT: exporter cần ghép cặp (call, đối số) và nhận biết
    # đối số nào là tham số hàm.
    {"name": "node.argument", "setup": "val cn = cpg.call.l.head",
     "expr": "Iterator(cn).argument.l.size", "expect": "ok"},

    # Cạnh CALL: callee nằm ngoài method này nên không có trong node set; cách duy
    # nhất để gắn cạnh nội bộ là tìm IDENTIFIER cùng tên trong thân hàm. Cần `name`
    # để khớp — và `code` để dựng EVAL_ORDER mà không cần tới `lineNumber`.
    {"name": "node.name", "setup": "val n = cpg.method.ast.l.head", "expr": "Iterator(n).name.l.size",
     "expect": "ok"},
    {"name": "node.code", "setup": "val n = cpg.method.ast.l.head", "expr": "Iterator(n).code.l.size",
     "expect": "ok"},

    # Data flow. `reachingDef` trong Joern nhận tham số `includeImplicit`, nên
    # dò cả ba biến thể. Đây là câu hỏi quyết định M4 còn giữ được cạnh data-flow
    # hay phải bỏ nó khỏi hợp đồng schema.
    {"name": "call.reachingDef(false)", "expr": "cpg.call.reachingDef(false).l.size",
     "expect": "ok"},
    {"name": "identifier.reachingDef(false)",
     "expr": "cpg.identifier.reachingDef(false).l.size", "expect": "ok"},
    {"name": "local.reachingDef(false)", "expr": "cpg.local.reachingDef(false).l.size",
     "expect": "ok"},

    # TARG/SOURCE: bảng help liệt kê `.target`/`.source` dưới `Call` nhưng gọi
    # trực tiếp thì hỏng — thử đường qua `.assignment` trước khi kết luận là bỏ.
    {"name": "call.assignment.target", "expr": "cpg.call.assignment.target.l.size",
     "expect": "ok"},
    {"name": "call.assignment.source", "expr": "cpg.call.assignment.source.l.size",
     "expect": "ok"},

    # === VÒNG 3 (sau 36 ứng viên) ============================================
    # Kết luận vòng 2: `reachingDef` **không tồn tại** — cả dạng không tham số lẫn
    # dạng `reachingDef(false)` đều FAIL với `E008 Not Found` (không phải "missing
    # argument list"). Nên nó không phải chuyện thiếu tham số; CPGQL của Joern v4
    # không có step này cho C++. Hệ quả: nếu chỉ dựa vào DSL thì M4 **không** lấy
    # được data-flow ⇒ phải bỏ REACHING_DEF khỏi hợp đồng (và M4 chỉ còn
    # AST+CFG, gần như trùng ASTNN).
    #
    # Nhưng dữ liệu thì CÓ: log của `ReachingDefPass` in "Number of definitions:
    # 22", tức cạnh đã nằm trong đồ thị, chỉ là không có step để đi tới nó.
    # OverflowDB cho phép đi theo cạnh thô. Nếu `cpg.graph` mở được thì ta dựng
    # được REACHING_DEF/REF từ đúng cạnh gốc — không phải đoán mò.
    {"name": "graph.nodes", "expr": "cpg.graph.l.size", "expect": "ok"},
    {"name": "graph.E", "expr": "cpg.graph.E.l.size", "expect": "ok"},

    # Câu hỏi quan trọng nhất của cả EXP-005: CPG này thực sự lưu những loại cạnh
    # nào. In ra **từ vựng cạnh** thay vì đếm — biết danh sách là biết ngay M4
    # có thể có những gì, thay vì mò tiếp từng step một.
    {"name": "graph.E.labels",
     "expr": 'cpg.graph.E.l.map(_.label).distinct.sorted.mkString(",")', "expect": "ok"},

    # Nếu vocabulary chứa REACHING_DEF thì lọc trực tiếp được, không cần DSL.
    {"name": "graph.E.reachingDef",
     "expr": 'cpg.graph.E.l.filter(_.label == "REACHING_DEF").size', "expect": "ok"},
    {"name": "graph.E.ref",
     "expr": 'cpg.graph.E.l.filter(_.label == "REF").size', "expect": "ok"},

    # `.inE`/`.outE` là đường đi cạnh gốc từ một node — nếu mở được thì không cần
    # `cpg.graph` nữa, mà gắn trực tiếp được cạnh data-flow cho từng node.
    {"name": "node.outE", "setup": "val n = cpg.method.ast.l.head",
     "expr": "Iterator(n).outE.l.size", "expect": "ok"},
    {"name": "node.inE", "setup": "val n = cpg.method.ast.l.head",
     "expr": "Iterator(n).inE.l.size", "expect": "ok"},

    # Cạnh CALL trong thân hàm: `node.name` đã FAIL nên không khớp được bằng tên.
    # Nhưng callee là Method (mà `method.name` thì OK) và identifier trong thân hàm
    # có `.code` (OK). Khớp bằng *text* là đường còn lại.
    {"name": "callee.name", "setup": "val cn = cpg.call.l.head",
     "expr": "Iterator(cn).callee.name.l.size", "expect": "ok"},

    # --- ĐỐI CHỨNG: hai cái này chắc chắn hỏng, đã bác ở các lần chạy trước ----
    # Giữ **nguyên tên** để api_scan nhận cache và không chạy lại (tiết kiệm 20s).
    {"name": "call.ref", "expr": "cpg.call.ref.l.size", "expect": "fail"},
    {"name": "local.reachingDef", "expr": "cpg.local.reachingDef.l.size", "expect": "fail"},
]

TEMPLATE = """// Sinh tự động bởi scripts/joern/api_scan.py — đừng sửa tay.
// Một biển thức mỗi lần chạy: nếu file này không compile thì mất trắng cả
// nhóm, đó là lý do mỗi ứng viên phải là một tiến trình riêng.
@main def exec(srcFile: String) = {{
  importCode(inputPath = srcFile, projectName = "{project}")
{setup}
  println("VALUE " + ({expr}))
}}
"""


def error_hint(blob: str) -> str:
    """Dòng lỗi compiler đáng đọc nhất, bỏ tiếng ồn tiền tố `[E008]`/đường dẫn."""
    for raw in blob.splitlines():
        line = raw.strip()
        if not line or line.startswith("$ "):
            continue
        low = line.lower()
        if any(h in low for h in _HINTS):
            return line[:160]
    return blob.strip().splitlines()[0][:160] if blob.strip() else "?"


def build_script(cand: dict, project: str) -> str:
    setup = cand.get("setup")
    body = f"  {setup}\n" if setup else ""
    return TEMPLATE.format(project=project, setup=body, expr=cand["expr"])


def run_one(joern_cmd: list[str], cand: dict, src: Path, sc_path: Path,
            timeout_s: int, project: str) -> dict:
    sc_path.write_text(build_script(cand, project), encoding="utf-8")
    cmd = joern_cmd + ["--script", str(sc_path), "--param", f"srcFile={src}"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -1, "", f"TIMEOUT sau {timeout_s}s"
    m = re.search(r"^VALUE (.*)$", out, re.M)
    # Chỉ coi là hợp lệ khi VỪA compile được VỪA in ra VALUE. `rc==0` một mình
    # không đủ: joern có thể trả 0 dù script không chạy hết.
    ok = bool(m) and rc == 0
    res = {"expr": cand["expr"], "ok": ok, "seconds": round(time.perf_counter() - t0, 1)}
    if ok:
        res["value"] = m.group(1).strip()
    else:
        blob = err or out
        res["error"] = blob.strip()[:300]
        res["error_hint"] = error_hint(blob)
    return res


# Sai *tên* và sai *số tham số* là hai lỗi khác hẳn nhau: `reachingDef` tồn tại
# nhưng cần `(includeImplicit)` thì chỉ cần thêm tham số, còn không có member thì
# phải đổi hẳn thiết kế. Trộn hai cái này thành một dòng trống thì ta không phân
# biệt được, nên phải trích đúng dòng compiler nói.
_HINTS = ("not a member", "missing argument", "too many arguments", "type mismatch",
          "not found", "does not exist", "value ", "cannot resolve")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joern", default="joern")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--only", default="", help="lọc theo tên (substring)")
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--refresh", action="store_true", help="dò lại cả mục đã có kết quả")
    args = ap.parse_args()

    missing = joern_missing_reason(args.joern)
    if missing:
        raise SystemExit(f"[api-scan] {missing}")

    cands = [c for c in CANDIDATES if args.only in c["name"]]
    if not cands:
        raise SystemExit(f"[api-scan] khong co ung vien khop --only {args.only!r}")

    out_path = Path(args.out)
    data = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "joern": "v4", "scanned_at": None, "results": {}}
    data.setdefault("results", {})

    work = ROOT / "artifacts" / "api_scan"
    work.mkdir(parents=True, exist_ok=True)
    src = work / "scan.cpp"
    src.write_text(SMOKE_CODE, encoding="utf-8")
    sc_path = work / "one.sc"
    joern_cmd = shlex.split(args.joern, posix=(sys.platform != "win32"))

    print(f"[api-scan] {len(cands)} ung vien x ~{args.timeout_s}s timeout "
          f"(moi lan khoi dong JVM la mot lan chay)")
    for i, cand in enumerate(cands, 1):
        name = cand["name"]
        if name in data["results"] and not args.refresh:
            print(f"  [{i:2}/{len(cands)}] {name:24s} (co roi, dung --refresh)")
            continue
        res = run_one(joern_cmd, cand, src, sc_path, args.timeout_s, f"cpg_scan_{i}")
        data["results"][name] = res
        # Ghi sau MỖI ứng viên: quét này tốn vài phút, không được mất trắng vì
        # một lần timeout ở giữa chừng.
        data["scanned_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        flag = "OK  " if res["ok"] else "FAIL"
        print(f"  [{i:2}/{len(cands)}] {name:24s} {flag} {res.get('value', '')}"
              f"  ({res['seconds']}s)")

    _summarise(data, cands, out_path)


def _summarise(data: dict, cands: list[dict], out_path: Path) -> None:
    results = data["results"]
    print("\n=== HOP LE (dung duoc trong export_cpg.sc) ===")
    for c in cands:
        r = results.get(c["name"])
        if r and r["ok"]:
            print(f"  {c['name']:24s} = {r.get('value')}")
    print("\n=== KHONG HOP LE (cam dung, tach ly do) ===")
    for c in cands:
        r = results.get(c["name"])
        if r and not r["ok"]:
            # `error_hint` là phần quan trọng nhất: "not a member" = đổi hẳn tên,
            # "missing argument" = chỉ thiếu tham số. Hai việc khác hẳn nhau.
            print(f"  {c['name']:24s} {r.get('error_hint') or '(khong doc loi compiler)'}")

    # Đối chứng hỏng ⇒ bản đồ này đáng nghi, phải nói ra chứ không im lặng.
    bad_control = [c["name"] for c in cands
                   if c["expect"] == "fail" and results.get(c["name"], {}).get("ok")]
    bad_expect = [c["name"] for c in cands
                  if c["expect"] == "ok" and c["name"] in results
                  and not results[c["name"]]["ok"]]
    if bad_control:
        print(f"\nCANH BAO: {bad_control} trong danh sach doi chung lai HOP LE. "
              f"Khong tin bao cao nay — co the joern da doi ten API, hoac script "
              f"scan bi hong.")
    if bad_expect:
        print(f"\nCANH BAO: {bad_expect} khong hop le. Xa loi compiler o "
              f"{out_path} truoc khi dung.")
    print(f"\n[api-scan] ket qua: {out_path}")


if __name__ == "__main__":
    main()
