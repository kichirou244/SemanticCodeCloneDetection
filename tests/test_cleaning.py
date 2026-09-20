"""Test cleaning: reasons đầy đủ, không drop thầm lặng."""

from src.data.cleaning.cleaner import CleaningConfig, DatasetCleaner
from src.data.loaders.base import CodeSample


def _s(sid, code="int main(){return 0;}", prob="p1", lang="cpp", meta=None):
    return CodeSample(sid, code, prob, lang, metadata=meta)


def test_reasons():
    samples = [
        _s("ok1"),
        _s("empty", code="   "),
        _s("short", code="x"),
        _s("noprob", prob="?"),
        _s("ok1-dup", code="other code here!!!"),
    ]
    samples[-1].sample_id = "ok1"  # trùng submission id
    res = DatasetCleaner(CleaningConfig(expected_language="cpp")).clean(samples)
    assert len(res.kept) == 1
    reasons = {r["reason"] for r in res.removed}
    assert reasons == {"empty_code", "too_short", "missing_problem", "duplicate_submission_id"}


def test_status_allowlist_reports_but_defaults_keep_all():
    samples = [_s("a", meta={"status": "Accepted"}), _s("b", meta={"status": "Wrong"})]
    res = DatasetCleaner().clean(samples)
    assert len(res.kept) == 2  # mặc định không lọc status
    res2 = DatasetCleaner(CleaningConfig(status_allowlist=["Accepted"])).clean(samples)
    assert len(res2.kept) == 1 and res2.removed[0]["reason"] == "status_excluded:Wrong"
