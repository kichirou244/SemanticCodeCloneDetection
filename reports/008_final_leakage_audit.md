# 008 — Final leakage audit (POJ-104 + CodeNet pilot: PASSED)

## Phase
Phase 1 — Steps 18–19

## Date
2026-09-20

## Checks (`run_pair_leakage_audit`, fail → build exit 1)
- problem / sample / exact code / normalized code / pair (vô thứ tự) overlap.

## Result (POJ-104 build, từ output thật)
problem 0/0/0, sample 0/0/0, exact code 0/0/0, normalized 0/0/0,
pair 0/0/0 → **passed: true**. Dataset dùng được cho EXP-001.

## Result (CodeNet pilot 200k small, từ output thật)
Mọi overlap 0 → **passed: true** (sau fix exact = byte-identical;
cặp chỉ khác whitespace nội bộ là near-dup, ghi limitation).
Manifest: `data/manifests/codenet_python.json` — rebuild với allowlist
(2026-09-20) số không đổi, leakage vẫn passed. **Phase 1 khép.**

## Bugfix trong phase (paper trail)
Exact check từng dùng `fingerprint` collapse-whitespace, không nhất quán với
`CodeNormalizer` (bảo thủ vì indent Python mang semantics) → sửa thành
byte-identical (`leakage_prevention.md` 1.1.1).

## Limitations
- Near-duplicate và augmentation leakage chưa có (phase sau).
- CodeNet pilot (200k small), chưa full.

## Next Actions
Rebuild CodeNet pilot với allowlist → khép Phase 1 → EXP-001.
