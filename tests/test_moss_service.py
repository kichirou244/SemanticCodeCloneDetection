"""Test client MOSS SERVICE (moss.stanford.edu) — offline với fixture report thật.

Fixture lấy từ một lần nộp 10 file demo thật (2026-10-07); nằm trong repo nên bộ
test không cần mạng. Chạy:
    py -m pytest tests/test_moss_service.py -q
"""

from pathlib import Path

import pytest

from src.models.baseline.moss_service import (
    MossReport,
    extract_matched_blocks,
    locate_in_code,
    parse_index_html,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "moss_report"


@pytest.fixture(scope="module")
def report_index() -> str:
    return (FIX / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def report_index_save() -> str:
    return (FIX / "index_save.htm").read_text(encoding="utf-8")


def test_parse_downloaded_report(report_index):
    ms = parse_index_html(report_index)
    assert len(ms) == 11
    pair = [m for m in ms if m.name_a == "spit0035_24T1020331_355172.cpp"]
    assert pair
    m = pair[0]
    assert m.name_b == "spit0035_24t1020331dd60a2f4e2be4f9a_355188.cpp"
    assert m.pct_a == 98 and m.pct_b == 98
    assert m.score() == pytest.approx(0.98)
    assert m.url == "match3.html"


def test_parse_savewebpage_upper_absolute(report_index_save):
    """Bản `saveWebPage` dùng thẻ IN HOA + URL tuyệt đối — vẫn phải parse được."""
    ms = parse_index_html(report_index_save)
    assert len(ms) == 11
    m = ms[0]
    assert m.score() == pytest.approx(0.98)
    assert m.url == "match0.html"  # chỉ lấy basename, không lấy domain


def test_parse_ignores_header_links(report_index):
    # Nếu regex bắt nhầm link header (How to read / Tips...) sẽ ra cặp ảo.
    for m in parse_index_html(report_index):
        assert ".cpp" in m.name_a and ".cpp" in m.name_b
        assert 0 <= m.pct_a <= 100 and 0 <= m.pct_b <= 100


def test_parse_bad_html_returns_empty():
    assert parse_index_html("<html>no rows</html>") == []


def test_blocks_match_pages():
    a = (FIX / "match3-0.html").read_text(encoding="utf-8")
    b = (FIX / "match3-1.html").read_text(encoding="utf-8")
    ba = extract_matched_blocks(a)
    bb = extract_matched_blocks(b)
    assert len(ba) == 1 and len(bb) == 1
    # Marker/blank artifacts phải bị loại bằng cách tìm lại trong code gốc.
    assert "using namespace std;" in ba[0]
    assert bb[0] == ba[0]


def test_locate_in_code_maps_artifacts_away():
    code_a = (FIX / "355172.cpp").read_text(encoding="utf-8")
    code_b = (FIX / "355188.cpp").read_text(encoding="utf-8")
    ba = extract_matched_blocks((FIX / "match3-0.html").read_text(encoding="utf-8"))
    bb = extract_matched_blocks((FIX / "match3-1.html").read_text(encoding="utf-8"))
    sa = locate_in_code(ba, code_a)
    sb = locate_in_code(bb, code_b)
    assert len(sa) == 1 and len(sb) == 1
    s, e = sa[0]
    assert code_a[s:e].startswith("using namespace std;")
    assert e <= len(code_a)
    t, u = sb[0]
    assert code_b[t:u] == code_a[s:e]  # hai file giống hệt nên vùng khớp bằng nhau


def test_report_score_and_evidence():
    code_a = (FIX / "355172.cpp").read_text(encoding="utf-8")
    code_b = (FIX / "355188.cpp").read_text(encoding="utf-8")
    ms = parse_index_html((FIX / "index.html").read_text(encoding="utf-8"))
    rep = MossReport(url="http://x/results/1/abc", matches=ms,
                     names=[m.name_a for m in ms] + [ms[-1].name_b],
                     report_dir=FIX)
    assert rep.score(code_a_id := "spit0035_24T1020331_355172.cpp",
                     "spit0035_24t1020331dd60a2f4e2be4f9a_355188.cpp") == pytest.approx(0.98)
    ev = rep.evidence(code_a_id, "spit0035_24t1020331dd60a2f4e2be4f9a_355188.cpp",
                      code_a, code_b)
    assert ev is not None
    assert len(ev["spans_a"]) == 1 and len(ev["spans_b"]) == 1
    assert "MOSS: 98% / 98%" in ev["note"]


def test_report_score_unknown_pair_is_zero():
    ms = parse_index_html((FIX / "index.html").read_text(encoding="utf-8"))
    rep = MossReport(url="u", matches=ms, names=[])
    assert rep.score("x.cpp", "y.cpp") == 0.0
    assert rep.evidence("x.cpp", "y.cpp", "a", "b") is None