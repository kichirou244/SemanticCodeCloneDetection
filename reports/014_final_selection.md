# Báo cáo 014 — Đánh giá tổng hợp & chọn phương pháp cho pipeline cuối (EXP-FINAL-001)

| Mục | Nội dung |
| :--- | :--- |
| **Ngày** | 2026-10-17 |
| **Phạm vi** | "Đóng sổ" giai đoạn huấn luyện: đánh giá mọi run đã hoàn thành, chọn phương pháp tốt nhất theo protocol, dựng pipeline cuối (eval → chọn → inference). |
| **Dataset** | POJ-104 (C/C++), test 11.000 cặp / validation 10.000 cặp |
| **Config** | `configs/final_pipeline.yaml` |
| **Code** | `src/evaluation/leaderboard.py` · `scripts/final_pipeline.py` · `tests/test_final_pipeline.py` |
| **Artifact** | `results/final_pipeline/{leaderboard.json,leaderboard.md,selection.json}` |

---

## 1. Vì sao dừng train và chuyển sang đánh giá

Chín run đã có đủ `metrics.json` + `threshold.json` + `predictions/`. Bản 512 token
đã cho thấy trần thực dụng (F1 ≥ 0,94), phần cải thiện còn lại nằm trong nhiễu
giữa các bản. Train thêm epoch không còn trả về thông tin mới, nhưng lại tốn quota
GPU và làm số liệu báo cáo phải sửa lại. Bước có giá trị hơn là **đánh giá có kỷ
luật**: chọn phương pháp bằng một luật rõ ràng, lượng hoá độ bất định, rồi đóng
băng thành **một pipeline chạy lại được**.

## 2. Luật chọn (và vì sao là luật này)

- **Chọn theo `validation.f1`, không theo test.** Test đã dùng để báo cáo; chọn
  model bằng test là tự khen và làm con số test mất ý nghĩa (DEC-022).
- **Ngưỡng quyết định đóng băng** từ `threshold.json` của từng run (đã chọn trên
  validation). Pipeline không bao giờ dò lại ngưỡng trên test — dò lại là rò rỉ
  nhãn test và làm CI hẹp giả tạo (xem `docs/evaluation_protocol.md`).
- **Chỉ `role: candidate` mới được chọn.** Baseline MOSS không thể "thắng" nhưng
  vẫn nằm trong bảng để đối chiếu.
- **So sánh cặp có kiểm định:** winner vs á quân so trên *chính tập validation* bằng
  McNemar (exact binomial) + **paired bootstrap** CI 95% của ΔF1 — tái dùng nguyên
  `src/evaluation/uncertainty.py`, không viết lại.

## 3. Bảng xếp hạng (chọn theo val F1)

| # | Vai trò | PP | Val F1 | Test P | Test R | Test F1 | Test AUC | Ngưỡng |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Ứng viên | **M5-512** | **0,9379** | 0,9612 | 0,9420 | 0,9515 | 0,9878 | **0,010** |
| 2 | Ứng viên | M2-512 | 0,9337 | 0,9515 | 0,9585 | **0,9550** | **0,9912** | 0,660 |
| 3 | Ứng viên | M1-512 | 0,9238 | 0,9470 | 0,9351 | 0,9410 | 0,9868 | 0,210 |
| 4 | Ứng viên | M1-256 | 0,9200 | 0,9309 | 0,8898 | 0,9099 | 0,9729 | 0,590 |
| 5 | Ứng viên | M2-256 | 0,9165 | 0,8826 | 0,9325 | 0,9069 | 0,9713 | 0,430 |
| 6 | Ứng viên | M5-256 | 0,9150 | 0,9050 | 0,9216 | 0,9133 | 0,9699 | 0,370 |
| 7 | Ứng viên | M3 | 0,8570 | 0,7598 | 0,8938 | 0,8214 | 0,8893 | 0,420 |
| 8 | Ứng viên | M4 | 0,8562 | 0,7232 | 0,8871 | 0,7968 | 0,8823 | 0,470 |
| 9 | Baseline | B0 (MOSS local) | 0,7293 | 0,6458 | 0,7218 | 0,6817 | 0,7311 | 0,0121 |
| 10 | Baseline | B1 (MOSS service) | 0,0000 | 1,0000 | 0,0016 | 0,0033 | — | 0,010 |

Sinh tự động bằng `python scripts/final_pipeline.py select`.

## 4. Kết quả: người thắng theo protocol **chưa đủ vững để tin**

Theo luật (val F1), **M5-512 thắng**. Nhưng pipeline in ra bốn cảnh báo, và cả bốn
đều đúng — đây là phần quan trọng nhất của báo cáo:

1. **Ngưỡng của M5-512 rất mỏng (0,010 < 0,05).** Verdict nhạy với nhiễu hiệu
   chỉnh xác suất; cùng một model, chỉnh lệch nhỏ ở đầu ra là đổi nhãn hàng loạt.
2. **Chênh val F1 đúng +0,0043 (< 0,01) so với á quân M2-512.** Rất sát.
3. **Hai kiểm định không nhất quán** (trên validation, 10.000 cặp):
   - McNemar: `p = 0,0814` ⇒ *chưa* có ý nghĩa ở mức 0,05.
   - Paired bootstrap: `ΔF1 = +0,0043`, `CI95 [ +0,0004, +0,0083 ]` ⇒ CI **vừa chạm**
     là không chứa 0.

   Hai kiểm định đo hai thứ khác nhau (tỉ lệ cặp bất đồng ý vs độ lớn ΔF1) nên có
   thể bất đồng; khi đó pipeline gắn nhãn `mixed` chứ không chọn bừa một cái để kết
   luận cho gọn.
4. **Theo test F1 thì M2-512 (0,9550) lại hơn M5-512 (0,9515)** và AUC cũng cao
   hơn (0,9912 vs 0,9878) — dù test không dùng để chọn.

**Quyết định (chốt 2026-10-17):** pipeline cuối dùng **M2-512** cho bước suy luận.
`select` vẫn báo M5-512 là winner theo protocol (kèm đủ cảnh báo), nhưng
`inference.default_model` đặt thẳng `M2_512`: AUC cao nhất, ngưỡng khoẻ (0,66), và
**không cần JVM (Joern) lúc suy luận**. M5-512 vẫn gọi được khi cần bằng
`--model M5_512`.

## 5. Pipeline cuối đã dựng

```
python scripts/final_pipeline.py select
    # đọc artifact → xếp hạng val F1 → McNemar + paired bootstrap → winner
    # ghi results/final_pipeline/{leaderboard.json,leaderboard.md,selection.json}
    # KHÔNG cần torch/GPU/Joern

python scripts/final_pipeline.py detect --a a.cpp --b b.cpp            # model = M2_512 (default)
python scripts/final_pipeline.py detect --a a.cpp --b b.cpp --model M5_512 --json
    # cặp file → {score, threshold, verdict, latency}; ngưỡng đọc từ threshold.json
```

`select` chạy trên CPU trong vài giây. `detect` nạp model thật: M2-512 (mặc định)
cần `transformers` + checkpoint; M5-512 cần thêm Java/Joern để dựng CPG.

## 6. Tái lập

```bash
python scripts/final_pipeline.py select
pytest tests/test_final_pipeline.py -q        # 14 test, không cần GPU
```

## 7. Hạn chế (ghi để không tự huyễn)

- ΔF1 giữa top-2 nằm trong nhiễu ⇒ **không được viết "hybrid thắng"**. Chỉ được
  viết: "theo val F1, M5-512 nhỉnh hơn nhưng khác biệt chưa vững; M2-512 mạnh hơn
  ở test F1 và AUC".
- Bootstrap CI ở đây đo biến thiên do **kích thước test/val hữu hạn**, không đo
  biến thiên qua lần train (muốn cái sau phải train nhiều seed — đắt). Xem
  `src/evaluation/uncertainty.py`.
- Cảnh báo ngưỡng mỏng dùng ngưỡng heuristic 0,05; nó là *cờ để xem lại*, không
  phải một kiểm định.

## 8. Lịch sử thay đổi

| No. | Ngày | Nội dung |
| :--- | :--- | :--- |
| 1 | 2026-10-17 | Khởi tạo: EXP-FINAL-001 (eval + chọn + inference), bảng xếp hạng 10 run, McNemar + paired bootstrap trên validation, khuyến nghị M2-512 cho vận hành. |
