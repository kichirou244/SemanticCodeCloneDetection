"""Client cho MOSS thật (moss.stanford.edu) — baseline B1 của báo cáo.

KHÁC `src/models/baseline/moss.py` (bản winnowing tự viết, chạy offline): module
này **nộp code lên service Stanford** rồi đọc HTML report trả về, tức đúng MOSS
gốc. Dùng gói `mosspy` (client chính thức của cộng đồng, giao thức socket
`moss <userid>` + ngôn ngữ), không tự viết lại giao thức.

Cần MOSS user id (đăng ký qua email moss@moss.stanford.edu). Id **không** hardcode
trong repo — đọc từ biến môi trường `MOSS_USER_ID` (hoặc truyền thẳng vào client).

Hai bước MOSS trả về:

1. `index.html` — bảng `File1 (pct1%) | File2 (pct2%) | lines matched`. Chỉ liệt
   kê các cặp MOSS coi là có khớp; cặp không xuất hiện coi như 0.
2. `matchN-{0,1}.html` — hai phía file, vùng khớp bọc trong
   `<font color="#FF0000">…</font>`. Giữ lại để lấy evidence (tô vàng trong demo).

Điểm một cặp = `max(pct1, pct2) / 100` (một bên bị sao chép nhiều là đủ; nhất
quán với cách JPlag/MOSS được đọc). Cách gộp này ghi trong `run_metadata.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SERVER = "moss.stanford.edu"
DEFAULT_PORT = 7690
#: MOSS: "cc" = C/C++ (dùng cho POJ-104). Xem `mosspy.Moss.languages`.
DEFAULT_LANGUAGE = "cc"
#: Ngưỡng nội bộ: MOSS chỉ hiện cặp có khớp; `maxmatches`/`show` để đủ cho N².
DEFAULT_MAX_MATCHES = 1000
DEFAULT_SHOW = 1000

_ENV_USER_ID = "MOSS_USER_ID"
#: Id mặc định của nhóm (đã đăng ký với Stanford). Biến môi trường trên
#: được ưu tiên nếu có — giữ được cho máy khác mà không sửa code.
_DEFAULT_USER_ID = "954365299"

#: `saveWebPage` trả URL tuyệt đối + thẻ IN HOA; `download_report` trả tương đối.
#: Bắt cả hai nên parse được từ nguồn nào cũng chạy.
_MATCH_ANCHOR_RE = re.compile(
    r'<a href="([^"]*?match\d+\.html)">(.*?) \((\d+)%\)</a>', re.DOTALL | re.IGNORECASE
)


class MossServiceError(RuntimeError):
    """Lỗi gọi MOSS (thiếu id, mạng, service từ chối)."""


def user_id_from_env() -> int:
    """Đọc MOSS user id từ môi trường, fallback id mặc định của nhóm."""
    raw = os.environ.get(_ENV_USER_ID, "").strip() or _DEFAULT_USER_ID
    if not raw:
        raise MossServiceError(
            f"thiếu MOSS user id — set biến môi trường {_ENV_USER_ID} "
            "(đăng ký qua email moss@moss.stanford.edu)"
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise MossServiceError(f"{_ENV_USER_ID}={raw!r} không phải số") from exc


@dataclass(frozen=True)
class MossMatch:
    """Một cặp MOSS báo khớp. `pct_*` là % mỗi phía bị khớp (0..100)."""

    name_a: str
    name_b: str
    pct_a: float
    pct_b: float
    url: str = ""

    def score(self) -> float:
        return max(self.pct_a, self.pct_b) / 100.0


def parse_index_html(html: str) -> list[MossMatch]:
    """Bóc bảng kết quả từ `index.html`.

    Mỗi cặp = hai anchor `matchN.html` liên tiếp (`File1` rồi `File2`). Lọc theo
    đúng pattern anchor nên bỏ qua các link header (How to read/Tips/...) an toàn.
    """
    anchors = _MATCH_ANCHOR_RE.findall(html)
    out: list[MossMatch] = []
    for i in range(0, len(anchors) - 1, 2):
        (url_a, name_a, pct_a), (url_b, name_b, pct_b) = anchors[i], anchors[i + 1]
        out.append(MossMatch(name_a.strip(), name_b.strip(),
                             float(pct_a), float(pct_b), _basename(url_a)))
    return out


def _decode(text: str) -> str:
    """Giải mã HTML entity trong tên hiển thị (mặc dù tên tự sinh đã sạch)."""
    import html as _html

    return _html.unescape(text)


def _basename(url: str) -> str:
    return url.rstrip("/").split("/")[-1] if url else ""


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(s for s in spans if s[1] > s[0])
    merged: list[list[int]] = []
    for s, e in ordered:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def extract_matched_blocks(page_html: str) -> list[str]:
    """Bóc các vùng khớp trong trang `matchN-side.html`.

    MOSS bọc mỗi vùng khớp trong `<font color="#FF0000">…</font>`; trả về text
    của từng vùng (đã bỏ thẻ, giải mã entity, bỏ ảnh marker). Không dùng để tái
    dựng code gốc vì MOSS chèn dòng marker + dòng trống — phải định vị lại trên
    code gốc bằng `locate_in_code`.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(page_html, "html.parser")
    pre = soup.find("pre")
    if pre is None:
        return []
    for tag in pre.find_all(["img"]):
        tag.decompose()
    blocks: list[str] = []
    for font in pre.find_all("font"):
        if str(font.get("color", "")).upper() != "#FF0000":
            continue
        blocks.append(font.get_text())
    return blocks


def locate_in_code(blocks: list[str], code: str) -> list[tuple[int, int]]:
    """Tìm từng block khớp trong code gốc → span ký tự `[start, end)`.

    Bỏ dòng rỗng (artifact của MOSS) rồi `find` nguyên khối trong code. Tìm từ
    vị trí duyệt trước để giữ thứ tự; thất bại thì thử lại từ đầu, còn không thì
    span một dòng đầu cho đỡ mất hẳn. Gộp span chồng lấn ở cuối.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        body = "\n".join(lines)
        pos = code.find(body, cursor)
        if pos < 0:
            pos = code.find(body)
        if pos < 0:
            first = lines[0]
            pos = code.find(first)
            if pos < 0:
                continue
            spans.append((pos, pos + len(first)))
            cursor = pos + len(first)
            continue
        spans.append((pos, pos + len(body)))
        cursor = pos + len(body)
    return _merge_spans(spans)


@dataclass
class MossReport:
    """Kết quả một lần nộp MOSS: URL + các cặp khớp + nơi lưu trang chi tiết."""

    url: str
    matches: list[MossMatch]
    names: list[str]
    report_dir: Path | None = None
    _by_pair: dict[frozenset, MossMatch] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for m in self.matches:
            self._by_pair.setdefault(frozenset((m.name_a, m.name_b)), m)

    def score(self, name_a: str, name_b: str) -> float:
        """Điểm 0..1 của một cặp; 0 nếu MOSS không báo khớp."""
        m = self._by_pair.get(frozenset((name_a, name_b)))
        return m.score() if m else 0.0

    def evidence(self, name_a: str, name_b: str, code_a: str, code_b: str) -> dict | None:
        """Span ký tự vùng khớp cho cặp, lấy từ trang `matchN-*.html`.

        Trả `None` nếu cặp không khớp hoặc chưa tải report chi tiết.
        """
        m = self._by_pair.get(frozenset((name_a, name_b)))
        if not m or self.report_dir is None:
            return None
        stem = m.url[:-5] if m.url.endswith(".html") else m.url  # "matchN"
        page_a = self.report_dir / f"{stem}-0.html"
        page_b = self.report_dir / f"{stem}-1.html"
        if not page_a.exists() or not page_b.exists():
            return None
        ba = extract_matched_blocks(page_a.read_text(encoding="utf-8", errors="replace"))
        bb = extract_matched_blocks(page_b.read_text(encoding="utf-8", errors="replace"))
        sa = locate_in_code(ba, code_a)
        sb = locate_in_code(bb, code_b)
        # Trang matchN-0 là File1 (name_a), matchN-1 là File2 (name_b) — theo MOSS.
        if m.name_a != name_a:
            sa, sb = sb, sa
        return {
            "spans_a": [list(s) for s in sa],
            "spans_b": [list(s) for s in sb],
            "n_shared": len(sa) + len(sb),
            "kind": "moss",
            "note": f"MOSS: {m.pct_a:.0f}% / {m.pct_b:.0f}% khớp",
        }


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name)[:180] or "file"


class MossServiceClient:
    """Nộp một tập file lên MOSS và trả `MossReport` (có cache trên đĩa).

    Cache theo hash nội dung tập file: chạy lại cùng dữ liệu không tốn thêm
    lượt nộp (MOSS giới hạn tần suất, và demo bấm đi bấm lại một cặp).
    """

    def __init__(
        self,
        user_id: int | None = None,
        language: str = DEFAULT_LANGUAGE,
        max_matches: int = DEFAULT_MAX_MATCHES,
        show: int = DEFAULT_SHOW,
        cache_dir: Path | None = None,
        timeout_s: float = 120.0,
    ):
        self.user_id = user_id if user_id is not None else user_id_from_env()
        self.language = language
        self.max_matches = max_matches
        self.show = show
        self.cache_dir = cache_dir
        self._timeout_s = timeout_s

    # ------------------------------------------------------------------ nộp
    def submit(self, files: dict[str, str], with_details: bool = False) -> MossReport:
        """Nộp `files = {tên_hiển_thị: code}`; trả report đã parse.

        `with_details=True` thì tải luôn các trang `matchN-*.html` để lấy evidence.
        """
        if not files:
            return MossReport(url="", matches=[], names=[])
        cache_file = self._cache_path(files, with_details)
        if cache_file and cache_file.exists():
            return self._load_cache(cache_file)

        import tempfile

        import mosspy

        names = sorted(files)
        with tempfile.TemporaryDirectory(prefix="moss_submit_") as tmp:
            work = Path(tmp)
            m = mosspy.Moss(self.user_id, self.language)
            m.setIgnoreLimit(self.max_matches)
            m.setNumberOfMatchingFiles(self.show)
            for i, name in enumerate(names):
                path = work / f"{i:06d}_{_safe_name(name)}.cpp"
                # MOSS không nhận byte NUL (POJ-104 có file dính) — thay bằng space.
                path.write_text(files[name].replace("\x00", " "), encoding="utf-8")
                m.addFile(str(path), _safe_name(name))
            url = self._send(m)
            index_html = self._fetch(url)
            matches = parse_index_html(index_html)
            report_dir = None
            if with_details and matches:
                report_dir = work / "report"
                try:
                    mosspy.download_report(url, str(report_dir), connections=8,
                                           log_level=30)
                    # giữ lại ngoài temp: chuyển sang cache dir nếu có
                    report_dir = self._persist_details(cache_file, report_dir, matches)
                except Exception:  # noqa: BLE001 — thiếu details không sập, chỉ mất evidence
                    report_dir = None
            report = MossReport(url=url, matches=matches, names=names,
                                report_dir=report_dir)
        if cache_file:
            self._save_cache(cache_file, report)
        return report

    def _send(self, m) -> str:
        """Gọi `mosspy.Moss.send` với timeout socket (mosspy không set timeout)."""
        prev = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self._timeout_s)
        try:
            url = m.send(lambda *_a: None)
        except Exception as exc:  # noqa: BLE001
            raise MossServiceError(f"nộp MOSS thất bại: {type(exc).__name__}: {exc}") from exc
        finally:
            socket.setdefaulttimeout(prev)
        if not url:
            raise MossServiceError("MOSS không trả URL report")
        return url

    @staticmethod
    def _fetch(url: str) -> str:
        from urllib.request import urlopen

        with urlopen(url, timeout=60) as resp:  # noqa: S310 — URL do MOSS trả
            return resp.read().decode("utf-8", errors="replace")

    # --------------------------------------------------------------- cache
    def _cache_path(self, files: dict[str, str], with_details: bool) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256()
        h.update(f"lang={self.language}|m={self.max_matches}|d={int(with_details)}\n".encode())
        for name in sorted(files):
            h.update(name.encode("utf-8"))
            h.update(b"\x00")
            h.update(files[name].encode("utf-8", errors="replace"))
            h.update(b"\x00")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{h.hexdigest()[:24]}.json"

    def _persist_details(self, cache_file: Path | None, tmp_dir: Path,
                         matches: list[MossMatch]) -> Path:
        if cache_file is None:
            return tmp_dir
        dest = cache_file.with_suffix(".d")
        if dest.exists():
            return dest
        import shutil

        shutil.copytree(tmp_dir, dest, dirs_exist_ok=True)
        return dest

    def _save_cache(self, path: Path, report: MossReport) -> None:
        path.write_text(json.dumps({
            "url": report.url,
            "names": report.names,
            "matches": [[m.name_a, m.name_b, m.pct_a, m.pct_b, m.url]
                        for m in report.matches],
            "report_dir": str(report.report_dir) if report.report_dir else None,
        }, ensure_ascii=False), encoding="utf-8")

    def _load_cache(self, path: Path) -> MossReport:
        data = json.loads(path.read_text(encoding="utf-8"))
        matches = [MossMatch(a, b, pa, pb, u)
                   for a, b, pa, pb, u in data["matches"]]
        rdir = Path(data["report_dir"]) if data.get("report_dir") else None
        return MossReport(url=data["url"], matches=matches, names=data["names"],
                          report_dir=rdir)
