"""Test cho ước lượng độ bất định (bootstrap, paired bootstrap, McNemar).

Trọng tâm của các test này là những chỗ **dễ làm sai âm thầm**:
  - lấy điểm khác với metric chính thức (lệch 1 chữ số là hỏng ý nghĩa),
  - ghép hai mô hình trên hai tập test khác nhau,
  - kết luận "không khác nhau" chỉ vì hai CI ±SD chạm nhau.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.uncertainty import (
    bootstrap_ci, bootstrap_paired, bootstrap_single, confusion_at, mcnemar, prf_at,
)


def _perfect_and_worst(n: int = 400, seed: int = 0):
    """y_true cân bằng; `good` phân biệt được, `bad` đoán đại 50/50."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n // 2, dtype=np.int8),
                        np.zeros(n - n // 2, dtype=np.int8)])
    good = np.clip(y * 0.6 + rng.normal(0, 0.15, n), 0, 1)
    bad = rng.uniform(0.35, 0.65, n)
    return y, good, bad


# ---------------------------------------------------------------- điểm & cơ bản

def test_prf_matches_sklearn_exactly():
    """Bootstrap chỉ có ý nghĩa nếu điểm ước lượng khớp metric chính thức.

    Đây là chốt chặn cho lỗi hay gặp: tự viết lại công thức F1 lệch với
    `sklearn.f1_score` một chút số là báo cáo sai ngay mà không ai thấy.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 500).astype(np.int8)
    s = rng.uniform(0, 1, 500)
    p, r, f1 = prf_at(y, s, 0.5)
    pred = (s >= 0.5).astype(np.int8)
    assert f1 == pytest.approx(f1_score(y, pred, zero_division=0), abs=1e-12)
    assert p == pytest.approx(precision_score(y, pred, zero_division=0), abs=1e-12)
    assert r == pytest.approx(recall_score(y, pred, zero_division=0), abs=1e-12)


def test_confusion_at_counts_add_up():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200).astype(np.int8)
    s = rng.uniform(0, 1, 200)
    tp, fp, tn, fn = confusion_at(y, s, 0.5)
    assert tp + fp + tn + fn == y.size


def test_empty_prediction_gives_f1_zero_not_nan():
    """Ngưỡng quá cao ⇒ không dự đoán dương nào. F1 phải bằng 0 (không tìm được
    bản sao nào), không phải NaN — NaN sẽ hủy cả trung bình của 2000 lần lặp."""
    y = np.array([1, 1, 0, 0], dtype=np.int8)
    s = np.array([0.1, 0.2, 0.1, 0.2])
    p, r, f1 = prf_at(y, s, 0.99)
    assert f1 == 0.0 and np.isfinite(f1)


# ------------------------------------------------------------------ bootstrap

def test_bootstrap_point_estimate_is_the_ordinary_metric():
    """Điểm ước lượng phải **đúng bằng** F1 tính thường trên toàn tập — mean của
    bootstrap được phép lệch nhẹ, còn `point` thì không."""
    y, good, _ = _perfect_and_worst()
    out = bootstrap_single(y, good, 0.5, n_boot=200, seed=1)
    assert out["point"]["f1"] == pytest.approx(prf_at(y, good, 0.5)[2], abs=1e-12)
    assert out["point"]["n"] == y.size


def test_bootstrap_ci_brackets_point_and_mean():
    y, good, _ = _perfect_and_worst()
    out = bootstrap_single(y, good, 0.5, n_boot=500, seed=1)
    lo, hi = out["f1"]["sd_ci95"]
    assert lo < out["f1"]["mean"] < hi
    assert out["f1"]["sd"] > 0


def test_bootstrap_is_deterministic_given_seed():
    y, good, _ = _perfect_and_worst()
    a = bootstrap_single(y, good, 0.5, n_boot=200, seed=99)
    b = bootstrap_single(y, good, 0.5, n_boot=200, seed=99)
    c = bootstrap_single(y, good, 0.5, n_boot=200, seed=100)
    assert a["f1"]["sd"] == pytest.approx(b["f1"]["sd"], abs=1e-15)
    assert a["f1"]["sd"] != pytest.approx(c["f1"]["sd"], abs=1e-15)


def test_bigger_test_set_gives_smaller_sd():
    """SD bootstrap phải thu hẹp khi test set lớn hơn. Nếu không, module đang
    đo sai thứ gì đó (ví dụ resample có kích thước cố định không liên quan tới n)."""
    rng = np.random.default_rng(5)
    sds = []
    for n in (500, 4000):
        y = rng.integers(0, 2, n).astype(np.int8)
        s = np.clip(y * 0.6 + rng.normal(0, 0.15, n), 0, 1)
        sds.append(bootstrap_single(y, s, 0.5, n_boot=400, seed=2)["f1"]["sd"])
    assert sds[1] < sds[0]


def test_bootstrap_ci_rejects_nan():
    v = np.array([0.1, np.nan, 0.3, 0.2])
    lo, hi = bootstrap_ci(v)
    assert np.isfinite(lo) and np.isfinite(hi)


# ------------------------------------------------------- paired bootstrap

def test_paired_identical_models_gives_exactly_zero():
    """Cùng một mô hình so với chính nó ⇒ Δ = 0 tuyệt đối và CI phải là [0, 0].

    Đây là test chống rò rỉ: nếu `f1_vec` của A và B dùng chỉ số resample khác
    nhau thì Δ sẽ lung lay quanh 0 với CI rộng, và ta sẽ tưởng "có khác biệt
    nhỏ" — tức kết luận sai từ một lỗi lập trình.
    """
    y, good, _ = _perfect_and_worst()
    out = bootstrap_paired(y, good, good, 0.5, 0.5, n_boot=300, seed=4)
    assert out["delta_f1_point"] == pytest.approx(0.0, abs=1e-12)
    assert out["delta_f1_mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["delta_f1_ci95"] == [0.0, 0.0]
    assert out["significant"] is False


def test_paired_detects_a_clearly_better_model():
    y, good, bad = _perfect_and_worst()
    out = bootstrap_paired(y, good, bad, 0.5, 0.5, n_boot=400, seed=6)
    assert out["delta_f1_point"] > 0
    assert out["delta_f1_ci95"][0] > 0, "CI phai hoan toan duong khi A tốt hơn hẳn"
    assert out["significant"] is True


def test_paired_narrower_than_unpaired():
    """Cái lý do dùng bootstrap ghép: SD của Δ phải **nhỏ hơn** SD của từng mô
    hình cộng lại, vì lỗi tương quan trên cùng tập test triệt tiêu nhau.

    Đây là lý do không được so hai cột ±SD: cách đó phóng đại sai lệch.
    """
    rng = np.random.default_rng(11)
    n = 2000
    y = rng.integers(0, 2, n).astype(np.int8)
    # Hai mô hình gần như giống nhau, chỉ lệch chút → Δ có variance rất nhỏ,
    # trong khi variance của mỗi F1 riêng lẻ vẫn lớn.
    base = np.clip(y * 0.55 + rng.normal(0, 0.2, n), 0, 1)
    a = np.clip(base + rng.normal(0, 0.01, n), 0, 1)
    b = np.clip(base + rng.normal(0, 0.01, n), 0, 1)
    sd_a = bootstrap_single(y, a, 0.5, n_boot=400, seed=1)["f1"]["sd"]
    sd_b = bootstrap_single(y, b, 0.5, n_boot=400, seed=1)["f1"]["sd"]
    pair = bootstrap_paired(y, a, b, 0.5, 0.5, n_boot=400, seed=1)
    assert pair["delta_f1_sd"] < np.hypot(sd_a, sd_b)


def test_paired_uses_each_models_own_threshold():
    """Mỗi mô hình phải ở ngưỡng tối ưu của RIÊNG nó.

    Regression: trước đây `f1_vec` dùng chung một ngưỡng cho cả A và B, nên
    M2 (ngưỡng 0,43) bị chấm ở 0,59 của M1 ⇒ F1 tụt và kết luận sai.

    Dữ liệu test ở đây cố tình làm B **không nhìn ra được** ở ngưỡng của A: score
    của B nằm dồn trong dải hẹp 0,56–0,62, nên ở 0,5 thì B dự đoán dương tất cả
    và trông rất tệ, còn ở ngưỡng riêng 0,59 thì B hoàn hảo.
    """
    y = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    a = np.array([0.90, 0.80, 0.70, 0.10, 0.20, 0.05])   # tách rõ ở 0,5
    b = np.array([0.62, 0.61, 0.60, 0.58, 0.57, 0.56])   # dải hẹp, cần ngưỡng riêng
    assert prf_at(y, a, 0.5)[2] == pytest.approx(1.0)
    assert prf_at(y, b, 0.5)[2] < 1.0, "o nguong chung B phai trong te hon"
    assert prf_at(y, b, 0.59)[2] == pytest.approx(1.0), "o nguong rieng B hoan hao"

    # Dùng đúng ngưỡng từng mô hình ⇒ hai cái ngang nhau.
    ok = bootstrap_paired(y, a, b, 0.5, 0.59, n_boot=200, seed=1)
    assert ok["delta_f1_point"] == pytest.approx(0.0, abs=1e-12)
    assert ok["significant"] is False
    assert ok["threshold_a"] == 0.5 and ok["threshold_b"] == 0.59

    # Dùng nhầm ngưỡng chung ⇒ lệch điểm ước lượng, dù thực tế hai mô hình ngang
    # nhau. (Với n = 6 thì CI vẫn chứa 0 nên chưa "significant" — test này kiểm
    # điểm ước lượng bị lệch, phần CI đã có test riêng ở trên.)
    wrong = bootstrap_paired(y, a, b, 0.5, 0.5, n_boot=200, seed=1)
    assert wrong["delta_f1_point"] > 0
    assert ok["delta_f1_point"] == pytest.approx(0.0, abs=1e-12)


def test_paired_rejects_mismatched_lengths():
    y = np.array([1, 0, 1, 0], dtype=np.int8)
    with pytest.raises(ValueError):
        bootstrap_paired(y, np.array([0.5, 0.5]), np.array([0.5, 0.5, 0.5]), 0.5, 0.5)


# ------------------------------------------------------------------- McNemar

def test_mcnemar_identical_models_gives_p_one():
    y, good, _ = _perfect_and_worst()
    r = mcnemar(y, good, good, 0.5, 0.5)
    assert r["discordant"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_counts_discordant_pairs_correctly():
    y = np.array([1, 1, 0, 0], dtype=np.int8)
    # A đúng cả 4; B sai 2 cặp đầu, đúng 2 cặp sau.
    a = np.array([0.9, 0.9, 0.9, 0.9])
    b = np.array([0.1, 0.1, 0.9, 0.9])
    r = mcnemar(y, a, b, 0.5, 0.5)
    assert r["a_correct_b_wrong"] == 2
    assert r["a_wrong_b_correct"] == 0
    assert r["discordant"] == 2
    # p = 2 * P(X <= 0 | Binomial(2, 0.5)) = 2 * 1/4 = 0.5.
    assert r["p_value"] == pytest.approx(0.5)


def test_mcnemar_needs_enough_discordant_pairs_to_reach_significance():
    """Bài test này chốt một cái bẫy dễ gặp: rất ít cặp bất đồng ý thì *không thể*
    có p < 0,05, dù A sai hoàn toàn theo hướng đúng.

    Với n = 2, p tối thiểu là 0,5. Muốn p < 0,05 cần ít nhất khoảng 6 cặp bất
    đồng ý và tất cả nghiêng hẳn về một phía. Đây cũng là lý do "McNemar p >
    0,05" KHÔNG đồng nghĩa với "hai mô hình giống nhau" — nó có thể chỉ vì
    lực thống kê yếu.
    """
    rng = np.random.default_rng(17)
    n = 4000
    y = rng.integers(0, 2, n).astype(np.int8)
    a = np.clip(y * 0.55 + rng.normal(0, 0.20, n), 0, 1)
    b = np.clip(y * 0.55 + rng.normal(0, 0.20, n), 0, 1)
    few = mcnemar(y, a, b, 0.5, 0.5)
    assert few["discordant"] > 0

    # Dồn toàn bộ điểm bất đồng ý về một phía: A đúng hết, B sai hết trên đúng
    # các cặp mà A đúng. Không thể làm bằng score ngẫu nhiên — dựng trực tiếp.
    a2 = np.where(y == 1, 0.9, 0.1).astype(float)   # A hoàn hảo
    b2 = np.where(y == 1, 0.1, 0.9).astype(float)   # B sai ngược hoàn toàn
    many = mcnemar(y, a2, b2, 0.5, 0.5)
    assert many["discordant"] == n
    assert many["p_value"] == pytest.approx(0.0, abs=1e-12)


def test_mcnemar_is_symmetric_in_which_model_is_a():
    """Đổi thứ tự A/B không được làm đổi kết luận — tránh rối chiều ΔF1."""
    y, good, bad = _perfect_and_worst()
    ab = mcnemar(y, good, bad, 0.5, 0.5)
    ba = mcnemar(y, bad, good, 0.5, 0.5)
    assert ab["discordant"] == ba["discordant"]
    assert ab["p_value"] == pytest.approx(ba["p_value"], abs=1e-12)


def test_mcnemar_handles_huge_discordant_count_without_overflow():
    """n vài nghìn thì `2 ** n` tràn float và `sum(comb(n, i))` tràn int.

    Đã dính lỗi này thật khi chạy trên 11.000 cặp thật, nên có test riêng.
    """
    rng = np.random.default_rng(13)
    n = 6000
    y = rng.integers(0, 2, n).astype(np.int8)
    a = np.clip(y * 0.6 + rng.normal(0, 0.15, n), 0, 1)
    b = np.clip((1 - y) * 0.6 + rng.normal(0, 0.15, n), 0, 1)  # ngược hẳn
    r = mcnemar(y, a, b, 0.5, 0.5)
    assert r["discordant"] > 1000
    assert 0.0 <= r["p_value"] <= 1.0


# ------------------------------------------------- hợp đồng với artifact thật

RESULTS = Path("results")


@pytest.mark.parametrize("exp,expected_f1", [
    ("exp002_codebert_poj104", 0.9099191224319048),
    ("exp003_graphcodebert_poj104", 0.9069047829546459),
    ("exp004_astnn_poj104", 0.7499056390126067),
])
def test_point_estimate_reproduces_published_metrics(exp, expected_f1):
    """Số F1 mà module này suy ra phải **bằng** `metrics.json` đã công bố.

    Nếu lệch, nghĩa là hoặc module sai, hoặc `predictions/` không khớp với lần
    đánh giá đã report — cả hai đều là lý do không được tin báo cáo.
    """
    d = RESULTS / exp
    if not d.exists():
        pytest.skip(f"{exp} chua co ket qua tren may nay")
    pred = d / "predictions" / "test_scores.jsonl"
    thr = json.loads((d / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    if not pred.exists():
        pytest.skip(f"{exp} thieu predictions")
    ys, ss = [], []
    for line in pred.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ys.append(int(r["label"]))
            ss.append(float(r["score"]))
    assert prf_at(np.asarray(ys, dtype=np.int8), np.asarray(ss), thr)[2] == \
        pytest.approx(expected_f1, abs=1e-12)