# Tối ưu protocol chọn ngưỡng/trọng số

- Validation: 10,000 cặp từ `validation_shard_0.parquet (first 5k pos + 5k neg, shuffle seed 42)` | Test: 5,000 cặp
- OOF: 10 fold | chi danh gia cuoi; khong tune ngưỡng/trong so tren test

| Hệ thống | Ngưỡng | Val OOF F1 | Test F1 | P | R | FP/FN | CI95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| m2_single | 0.40 | 0.9597 | **0.9649** | 0.9778 | 0.9524 | 54/119 | [0.9593, 0.9700] |
| m1_single | 0.41 | 0.9756 | **0.9593** | 0.9713 | 0.9476 | 70/131 | [0.9532, 0.9650] |
| ensemble_avg | 0.30 | 0.9757 | **0.9604** | 0.9710 | 0.9500 | 71/125 | [0.9544, 0.9658] |
| stacking | 0.49 | 0.9727 | **0.9614** | 0.9833 | 0.9404 | 40/149 | [0.9557, 0.9667] |

- McNemar `stacking_vs_m2`: p=0.08051, discordant=74, significant=False
- McNemar `ensemble_vs_m2`: p=0.005918, discordant=65, significant=True