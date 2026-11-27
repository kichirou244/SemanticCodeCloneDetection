# Thư mục Kịch bản Thực thi (`scripts/`)

Các kịch bản thực thi trực tiếp từ dòng lệnh phục vụ toàn bộ pipeline của đề tài:

```text
scripts/
├── evaluate_bcb_5000_test.py    # Đánh giá 10 biến thể trên 5.000 cặp Test BigCloneBench (Java)
├── evaluate_coder_benchmark.py  # Đánh giá trên 400 cặp C/C++ thực tế CODER HUSC
├── prepare_coder_benchmark.py   # Chuẩn hóa & tạo tập test phân tầng CODER HUSC
├── crawl_coder_husc.py          # Thu thập dữ liệu bài nộp sinh viên từ CODER HUSC
├── download_bcb_subsets.py      # Tải dữ liệu BigCloneBench từ CodeXGLUE
├── build_bcb_features.py        # Trích xuất đặc trưng AST & DFG cho BigCloneBench
├── fast_native_cpg.py           # Parser đồ thị CPG đa thành phần tốc độ cao
├── build_cpg_export.py          # Xuất đồ thị CPG chuẩn hóa bằng Joern
├── env_report.py                # Báo cáo thông tin phần cứng GPU, CUDA và môi trường
├── postprocess_diffmul.py       # A1 calibration + A2 SWA + A3 ensemble cho M1/M2 DiffMul
├── average_swa_snapshots.py     # Trung bình các snapshot SWA do bcb_runner lưu (opt-in)
├── generate_postprocess_nb.py   # Tạo notebook Kaggle chạy pipeline hậu xử lý
├── monitor_postprocess_kernel.py# Theo dõi & tải kết quả kernel Kaggle về results/
├── generate_b5b6_nb.py          # Tạo notebook Kaggle train lại M1/M2 với focal + augment + SupCon
└── monitor_b5b6_kernel.py       # Theo dõi & tải kết quả kernel B5/B6
```

## Hậu xử lý M1/M2 DiffMul (A1 → A2 → A3)

Chạy trên Kaggle GPU (xem `kaggle_deploy/postprocess_diffmul/`):

```bash
python scripts/generate_postprocess_nb.py
kaggle kernels push -p kaggle_deploy/postprocess_diffmul
python scripts/monitor_postprocess_kernel.py
```

Chạy local khi đã có checkpoint và parquet:

```bash
python scripts/postprocess_diffmul.py \
  --val-file data/raw/bigclonebench/bcb_validation.parquet \
  --test-file data/raw/bigclonebench/bcb_test.parquet \
  --out-dir results/exp_diffmul_postprocess
```

Ngưỡng và trọng số ensemble luôn chốt trên validation/OOF; test chỉ dùng để đánh giá
cuối kèm bootstrap CI và McNemar. Để bật lưu SWA snapshot cho lần train sau, thêm vào config:

```yaml
swa:
  enabled: true
  start_epoch: 3
  dir: swa
```

## B5 + B6 (huấn luyện lại)

Config mẫu: `configs/tuning/exp_m1_b5b6.yaml`, `exp_m2_b5b6.yaml`.

```bash
python scripts/generate_b5b6_nb.py
kaggle kernels push -p kaggle_deploy/b5b6
python scripts/monitor_b5b6_kernel.py
```

- `training.loss: {type: focal, gamma: 2.0, alpha: 0.5}` — giảm trọng số easy negatives.
- `augmentation` — xóa comment + đổi tên identifier nhất quán trong từng hàm (giữ nhãn).
- `supcon` — thêm loss tương phản giữa `ha`/`hb` của cặp clone (temperature 0.07).
