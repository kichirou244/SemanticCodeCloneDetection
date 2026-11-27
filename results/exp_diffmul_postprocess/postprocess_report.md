# A1 → A2 → A3: hậu xử lý M1/M2 DiffMul

- Validation: 3,000 cặp | Test: 5,000 cặp
- Protocol: validation split (bcb_validation.parquet, 3000 pairs); 5-fold OOF trên validation, chốt ngưỡng quét F1 trên toàn validation
- chỉ đánh giá cuối; Không tune ngưỡng/trọng số trên test

## A1 — Calibration & ngưỡng OOF

| Biến thể | Method | NLL | ECE | Thresh (val) | Thresh OOF | Val OOF F1 | Test F1 (cal) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| best | isotonic | 0.0810 | 0.0000 | 0.45 | - | 0.9754 | 0.9583 |
| best | isotonic | 0.0888 | 0.0000 | 0.45 | - | 0.9700 | 0.9631 |
| swa | isotonic | 0.0886 | 0.0000 | 0.34 | - | 0.9692 | 0.9619 |

So sánh các phương pháp calibration (trên cùng validation):

| Biến thể | Method | NLL | ECE | OOF F1 | Ngưỡng |
| :--- | :--- | ---: | ---: | ---: | ---: |
| best | none | 0.1039 | 0.0104 | 0.9747 | 0.11 |
| best | temperature | 0.1039 | 0.0139 | 0.9751 | 0.12 |
| best | platt | 0.1035 | 0.0175 | 0.9763 | 0.14 |
| best | isotonic | 0.0810 | 0.0000 | 0.9754 | 0.45 |
| best | none | 0.1112 | 0.0129 | 0.9700 | 0.14 |
| best | temperature | 0.1111 | 0.0178 | 0.9697 | 0.15 |
| best | platt | 0.1089 | 0.0177 | 0.9706 | 0.20 |
| best | isotonic | 0.0888 | 0.0000 | 0.9700 | 0.45 |
| swa | none | 0.1147 | 0.0143 | 0.9689 | 0.06 |
| swa | temperature | 0.1141 | 0.0197 | 0.9682 | 0.07 |
| swa | platt | 0.1130 | 0.0215 | 0.9685 | 0.09 |
| swa | isotonic | 0.0886 | 0.0000 | 0.9692 | 0.34 |

## A2 — SWA (best + last)

- **m1**: best và last cùng epoch, bỏ qua SWA → chọn `m1_best`
- **m2**: best OOF F1=0.9700 → SWA OOF F1=0.9692 → chọn `m2_best`

## A3 — Ensemble M1 + M2

- Trọng số validation: M1=0.65, M2=0.35 | ngưỡng=0.32
- Baseline (m1_best): F1=0.9583 (CI95 0.9523–0.9639), P=0.9759, R=0.9412, FP=58, FN=147
- **Ensemble**: F1=0.9609 (CI95 0.9549–0.9662), P=0.9737, R=0.9484, FP=64, FN=129, ROC-AUC=0.9838
- McNemar (baseline vs ensemble): p=0.02266, discordant=24, significant=True
