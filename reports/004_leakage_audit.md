# 004 — Leakage audit (KẾT LUẬN POJ-104: official FAIL, problem PASS)

## Strategy
- Split theo problem identity trước mọi augmentation/transformation.
- Kiểm tra: problem overlap (=0), exact + normalized code overlap xuyên split
  (`src/data/leakage/checks.py`, `python scripts/check_leakage.py`).
- Phase sau bổ sung: near-duplicate, pair xuyên split, augmentation leakage.

## Results (POJ-104, chạy 2026-09-20)
- **Official split: FAILED** — problem overlap train×val 1, val×test 1;
  exact code overlap train×val 500, val×test 500 → `passed: false`.
- **problem_split: PASSED** — mọi overlap 0 → `passed: true`.
- Kết luận: **bác official split POJ-104** (DEC-007).

## Results (CodeNet, mẫu streaming 20k split small, 2026-09-20)
- Official check vacuous (1 split) — không ý nghĩa, bỏ qua.
- **problem_split: FAILED ở code overlap train×test = 3** (problem overlap 0).
  Cùng source xuất hiện ở problems khác nhau → problem split không đủ.
- Kết luận: **dedup cross-problem trước split** (DEC-010).

## Observations
500 mẫu giống hệt nhau xuyên train/val và val/test làm metric test bị thổi
phồng nếu dùng split chính thức — đây là finding đáng đưa vào paper (mục
dataset/threats). Cần điều tra problem ID cụ thể ở phase sau (nghi problems
65/81 loại 1000-mẫu).

## Limitations
Near-duplicate chưa implement ở INIT (ghi nhận, không coi là đạt full).

## Next Actions
1. Điều tra problem ID gây overlap official split (phase pair generation).
2. Chạy leakage CodeNet bản streaming sau khi có mẫu 20k.
3. Experiment nào fail → INVALIDATED, giữ report.
