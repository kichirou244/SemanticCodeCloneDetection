"""Ước lượng độ bất định của số đo trên tập test (mean ± SD, CI, paired test).

Vấn đề module này giải
---------------------
Mỗi mô hình chạy một lần thì chỉ có **một** con số F1, và một con số thì không
lấy được độ lệch chuẩn. Muốn báo ``mean ± SD`` phải có nhiều lần đo, mà có
**hai** kiểu lặp khác nhau, đo hai thứ khác nhau:

1. **Lặp theo hạt giống** (train lại N lần, seed khác nhau) → SD đo *độ bất định
   của thuật toán train*. Đắt: M1/M2 mỗi seed ~7h GPU.
2. **Bootstrap trên tập test** (model đã train, lấy mẫu lại có hoàn lại các cặp
   test) → SD đo *độ bất định do tập test chỉ hữu hạn*. Rẻ: dùng được ngay từ
   ``predictions/test_scores.jsonl``.

Hai đại lượng này **không thay thế nhau** và phải ghi rõ khi báo cáo. Module này
làm loại 2.

Vì sao KHÔNG so sánh bằng cách đặt hai cột ``mean ± SD`` cạnh nhau
----------------------------------------------------------------
Tất cả mô hình đều được đánh giá trên **cùng** 11.000 cặp test, nên sai sót của
chúng tương quan: cặp nào khó thì mọi mô hình đều sai ở cặp đó. So sánh hai
khoảng ± độc lập sẽ **phóng đại** độ bất định và dễ kết luận nhầm "không khác
nhau".

Cách đúng là **paired bootstrap**: mỗi lần resample, tính ΔF1 giữa hai mô hình
trên **cùng bộ cặp được lấy mẫu**, rồi lấy CI của ΔF1. Vì lỗi triệt tiêu nhau,
SD của Δ nhỏ hơn nhiều so với SD của từng mô hình — và đây chính là cách trả lời
câu "chênh 0.003 F1 có ý nghĩa không" mà không cần train thêm một epoch nào.

Ngưỡng quyết định: **ghim sẵn**, lấy từ ``threshold.json`` (đã chọn trên
validation). Tuyệt đối không dò lại ngưỡng trong từng lần bootstrap — làm vậy là
rò rỉ nhãn test và CI sẽ hẹp lại một cách giả tạo.

Phương pháp
-----------
Bootstrap không tham số: CI theo phân vị (percentile). Với n = 11.000 và F1 ~ 0.91
thì phân phối bootstrap của F1 khá đối xứng, percentile đủ dùng; nếu sau này
muốn chính xác hơn thì thêm BCa.
"""

from __future__ import annotations

import numpy as np

# Ngưỡng bỏ qua cảnh báo "mean of empty slice" khi một lần resample không có cặp
# dương nào (xảy ra với xác suất cực nhỏ ở tập cân bằng, nhưng không phải 0 về lý
# thuyết). Giá trị trả về khi đó là nan và bị loại khỏi tổng hợp.
_EPS = 1e-12


def confusion_at(y_true: np.ndarray, y_score: np.ndarray,
                 threshold: float) -> tuple[int, int, int, int]:
    """TP, FP, TN, FN tại một ngưỡng. Nhanh hơn sklearn cho bootstrap (B=2000 lần)."""
    y_pred = y_score >= threshold
    pos = y_true == 1
    tp = int(np.sum(pos & y_pred))
    fp = int(np.sum(~pos & y_pred))
    tn = int(np.sum(~pos & ~y_pred))
    fn = int(np.sum(pos & ~y_pred))
    return tp, fp, tn, fn


def prf_at(y_true: np.ndarray, y_score: np.ndarray,
           threshold: float) -> tuple[float, float, float]:
    """(precision, recall, F1) tại ngưỡng. F1 = 2TP/(2TP+FP+FN), đúng khi cả hai
    vế bằng 0 thì F1 = 0 (không có dự đoán dương nào ⇒ không có bản sao nào tìm
    được ⇒ coi như 0 để một lần resample rỗng không làm méo trung bình)."""
    tp, fp, _, fn = confusion_at(y_true, y_score, threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > _EPS else 0.0
    return precision, recall, f1


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """CI theo phân vị của một mẫu các giá trị bootstrap."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(v, 100.0 * alpha / 2.0))
    hi = float(np.percentile(v, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_single(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
                     n_boot: int = 2000, seed: int = 42,
                     alpha: float = 0.05) -> dict:
    """mean ± SD của P/R/F1 của MỘT mô hình, theo bootstrap trên tập test.

    Ngưỡng giữ cố định ở mọi lần resample.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=float)
    n = y_true.size
    rng = np.random.default_rng(seed)

    point_p, point_r, point_f1 = prf_at(y_true, y_score, threshold)
    # Lưu sẵn TP/FP/FN rồi tính F1 bằng công thức đóng — nhanh hơn gọi prf_at
    # trong vòng lặp. F1 = 2TP / (2TP + FP + FN).
    tp = np.empty(n_boot); fp = np.empty(n_boot); fn = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        t = y_true[idx]; s = y_score[idx]
        p = s >= threshold
        pos = t == 1
        tp[b] = np.sum(pos & p)
        fp[b] = np.sum(~pos & p)
        fn[b] = np.sum(pos & ~p)

    denom_p = tp + fp
    denom_r = tp + fn
    prec = np.where(denom_p > 0, tp / np.maximum(denom_p, _EPS), 0.0)
    rec = np.where(denom_r > 0, tp / np.maximum(denom_r, _EPS), 0.0)
    f1 = np.where(prec + rec > _EPS, 2.0 * prec * rec / np.maximum(prec + rec, _EPS), 0.0)

    lo_f1, hi_f1 = bootstrap_ci(f1, alpha)
    out = {
        "point": {"precision": point_p, "recall": point_r, "f1": point_f1,
                  "threshold": threshold, "n": int(n)},
        "n_boot": int(n_boot), "seed": int(seed), "alpha": alpha,
    }
    for name, arr in (("precision", prec), ("recall", rec), ("f1", f1)):
        out[name] = {"mean": float(np.mean(arr)), "sd": float(np.std(arr, ddof=1)),
                     "sd_ci95": [lo_f1, hi_f1] if name == "f1"
                     else list(bootstrap_ci(arr, alpha))}
    return out


def bootstrap_paired(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                     threshold_a: float, threshold_b: float,
                     n_boot: int = 2000, seed: int = 42,
                     alpha: float = 0.05) -> dict:
    """Bootstrap **ghép** cho ΔF1 = F1(A) − F1(B).

    Cùng một chỉ số resample được dùng cho cả A và B trong mỗi lần lặp. Đây là
    điểm mấu chốt: lỗi của hai mô hình trên cùng một cặp test là tương quan, nên
    hiệu chênh lệch nhỏ hơn nhiều so với khi so hai mô hình độc lập.

    Mỗi mô hình dùng **ngưỡng của riêng nó** (đều đã chọn và ghim trên validation).
    Dùng chung một ngưỡng sẽ trộn lẫn "chất lượng mô hình" với "chất lượng ngưỡng",
    và có thể khiến một mô hình bị đánh giá sai ở ngưỡng không phải của nó.

    ``significant`` = CI 95% của ΔF1 **không chứa 0**.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if score_a.shape != score_b.shape:
        raise ValueError("paired bootstrap: A va B phai cung do dai")
    n = y_true.size
    rng = np.random.default_rng(seed)

    def f1_vec(scores: np.ndarray, idx: np.ndarray, thr: float) -> np.ndarray:
        t = y_true[idx]
        p = scores[idx] >= thr
        pos = t == 1
        tp = np.sum(pos & p, axis=1).astype(float)
        fp = np.sum(~pos & p, axis=1).astype(float)
        fn = np.sum(pos & ~p, axis=1).astype(float)
        prec = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, _EPS), 0.0)
        rec = np.where((tp + fn) > 0, tp / np.maximum(tp + fn, _EPS), 0.0)
        return np.where(prec + rec > _EPS,
                        2.0 * prec * rec / np.maximum(prec + rec, _EPS), 0.0)

    idx = rng.integers(0, n, size=(n_boot, n))
    f1_a = f1_vec(score_a, idx, threshold_a)
    f1_b = f1_vec(score_b, idx, threshold_b)
    delta = f1_a - f1_b

    point_a = prf_at(y_true, score_a, threshold_a)[2]
    point_b = prf_at(y_true, score_b, threshold_b)[2]
    lo, hi = bootstrap_ci(delta, alpha)
    return {
        "delta_f1_point": point_a - point_b,
        "delta_f1_mean": float(np.mean(delta)),
        "delta_f1_sd": float(np.std(delta, ddof=1)),
        "delta_f1_ci95": [lo, hi],
        "threshold_a": threshold_a, "threshold_b": threshold_b,
        "significant": bool(lo > 0.0 or hi < 0.0),
        "n_boot": int(n_boot), "seed": int(seed), "alpha": alpha,
        "note": ("CI cua delta F1 theo bootstrap ghep tren cung mot tap test, "
                 "moi mo hinh o nguong da ghim cua no. Khong chua 0 => khac biet "
                 "co y nghia o muc alpha."),
    }


def mcnemar(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
            threshold_a: float, threshold_b: float) -> dict:
    """McNemar's test — bài kiểm định kinh điển cho so sánh hai bộ dự đoán trên
    CÙNG tập test (kiểu ghép).

    Quan tâm hai loại cặp mà hai mô hình **bất đồng ý**: A đúng/B sai (b) và
    A sai/B đúng (c). Với giả thuyết không có khác biệt, b và c phải phân bố
    như nhau (Binomial(n, 0.5)). Chỉ nhìn tổng số ``b + c`` không đủ — phải so
    ``b`` với ``c``.

    Mỗi mô hình dùng ngưỡng đã ghim của riêng nó, khớp với ``bootstrap_paired``.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    correct_a = (np.asarray(score_a, dtype=float) >= threshold_a) == (y_true == 1)
    correct_b = (np.asarray(score_b, dtype=float) >= threshold_b) == (y_true == 1)
    b = int(np.sum(correct_a & ~correct_b))   # A đúng, B sai
    c = int(np.sum(~correct_a & correct_b))   # A sai, B đúng
    n = b + c
    # Chính xác hai phía (exact binomial), không cần xấp xỉ.
    if n == 0:
        p_value = 1.0
    else:
        # KHÔNG dùng `2.0 ** n`: với n vài nghìn cặp bất đồng ý thì float tràn,
        # và `sum(comb(...))` cũng vượt int. `binomtest` chuẩn hoá bằng log-gamma.
        from scipy.stats import binomtest
        p_value = float(binomtest(min(b, c), n, 0.5).pvalue)
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "discordant": n,
            "p_value": p_value,
            "threshold_a": threshold_a, "threshold_b": threshold_b,
            "note": ("p_value < 0.05 => hai mo hinh sai khac nhau co y nghia. "
                     "Dung cung nguoi bao cao nay thay cho so hai cot +-SD.")}