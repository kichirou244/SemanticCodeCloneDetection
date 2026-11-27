# A1 → A2 → A3: hậu xử lý M1/M2 DiffMul

- Validation: 3,000 cặp | Test: 5,000 cặp
- Protocol: validation split (bcb_validation.parquet, 3000 pairs); 5-fold OOF trên validation, chốt ngưỡng quét F1 trên toàn validation
- chỉ đánh giá cuối; Không tune ngưỡng/trọng số trên test

## A1 — Calibration & ngưỡng OOF

| Biến thể | Method | NLL | ECE | Thresh (val) | Thresh OOF | Val OOF F1 | Test F1 (cal) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| best | isotonic | 0.0810 | 0.0000 | 0.45 | - | 0.9754 | 0.9583 |
| best | isotonic | 0.1031 | 0.0000 | 0.31 | - | 0.9630 | 0.9650 |

So sánh các phương pháp calibration (trên cùng validation):

| Biến thể | Method | NLL | ECE | OOF F1 | Ngưỡng |
| :--- | :--- | ---: | ---: | ---: | ---: |
| best | none | 0.1039 | 0.0104 | 0.9747 | 0.11 |
| best | temperature | 0.1039 | 0.0139 | 0.9751 | 0.12 |
| best | platt | 0.1035 | 0.0175 | 0.9763 | 0.14 |
| best | isotonic | 0.0810 | 0.0000 | 0.9754 | 0.45 |
| best | none | 0.1315 | 0.0280 | 0.9630 | 0.46 |
| best | temperature | 0.1287 | 0.0305 | 0.9630 | 0.45 |
| best | platt | 0.1239 | 0.0177 | 0.9630 | 0.59 |
| best | isotonic | 0.1031 | 0.0000 | 0.9630 | 0.31 |

## A2 — SWA (best + last)

- **m1**: best và last cùng epoch, bỏ qua SWA → chọn `m1_best`
- **m2**: best và last cùng epoch, bỏ qua SWA → chọn `m2_best`

## A3 — Ensemble M1 + M2

- Trọng số validation: M1=0.45, M2=0.55 | ngưỡng=0.45
- Baseline (m1_best): F1=0.9583 (CI95 0.9523–0.9639), P=0.9759, R=0.9412, FP=58, FN=147
- **Ensemble**: F1=0.9636 (CI95 0.9580–0.9687), P=0.9793, R=0.9484, FP=50, FN=129, ROC-AUC=0.9769
- McNemar (baseline vs ensemble): p=0.0003059, discordant=50, significant=True
