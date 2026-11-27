# Phân tích lỗi M1/M2 DiffMul kèm độ tương tự cosine

Cosine được tính giữa hai vector CLS `hA`, `hB` (trước tầng tương tác DiffMul): `cos(hA, hB)`. Giá trị càng gần 1 nghĩa là hai hàm càng gần nhau trong không gian biểu diễn.

Cột `prob` là xác suất dự đoán cặp là clone, **khác với cosine**. Cách tính:

```python
# src/models/transformer_pair.py :: TransformerPairClassifier.forward()
ha = encoder(inputs_a).last_hidden_state[:, 0, :]   # vector CLS của hàm A
hb = encoder(inputs_b).last_hidden_state[:, 0, :]   # vector CLS của hàm B
diff = torch.abs(ha - hb)
prod = ha * hb                                      # Hadamard product
pair = torch.cat([diff, prod], dim=-1)              # 1536 chiều (2 x 768)
logits = head(pair).squeeze(-1)                     # Linear->Tanh->Dropout->Linear

# src/experiments/bcb_runner.py / scripts/analyze_diffmul_errors.py
prob = torch.sigmoid(logits)                        # giá trị nằm trong [0, 1]
pred = (prob >= threshold).astype(int)              # M1: 0.12 | M2: 0.14
```

Lúc huấn luyện dùng `nn.BCEWithLogitsLoss()` nên sigmoid nằm bên trong loss; lúc inference mới lấy `sigmoid(logits)` thành cột `prob`. Ngưỡng được chọn từ F1 trên tập validation. `prob` là score chưa hiệu chỉnh (not calibrated) — không hiểu là xác suất thống kê, chỉ dùng để so ngưỡng. **Cosine không tham gia quyết định phân loại.**

## M1 CodeBERT DiffMul

- Checkpoint: `D:\Edu\NLP\SemanticCodeCloneDetection\results\exp_m1_diffmul\best_checkpoint.pt`
- Tập test: `results\bcb_5k_test_pairs_kaggle.jsonl`
- Ngưỡng: `0.12000000000000001`
- Kết quả: TP=2353, TN=2442, FP=58, FN=147, F1=0.9583, Accuracy=0.9590

### Thống kê cosine theo nhóm dự đoán

| Nhóm | Số cặp | Cosine trung bình | Trung vị | Q25 | Q75 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP | 2353 | 0.9706 | 0.9872 | 0.9779 | 0.9925 | 0.1477 | 1.0000 |
| TN | 2442 | 0.0416 | -0.0385 | -0.1042 | 0.0154 | -0.3358 | 1.0000 |
| FP | 58 | 0.5948 | 0.6874 | 0.3368 | 0.8683 | 0.0345 | 0.9788 |
| FN | 147 | -0.0468 | -0.0360 | -0.1300 | -0.0126 | -0.3319 | 0.9581 |

Tương quan Pearson giữa cosine và xác suất dự đoán: toàn bộ = 0.9031, non-clone = 0.3023, clone = 0.9772.

### False Positive (dự đoán Clone, nhãn Non-clone) — tổng 58 ca

| # | Pair ID | id1 | id2 | prob | cosine | Tên hàm A | Tên hàm B |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 1633 | 8983517 | 22625683 | 0.9883 | 0.9740 | `getUser` | `downloadModel` |
| 2 | 476 | 21821404 | 160705 | 0.9873 | 0.9190 | `getWebPage` | `downloadFile` |
| 3 | 335 | 21998571 | 23246123 | 0.9863 | 0.9637 | `readTwitterFead` | `createDialogArea` |
| 4 | 1160 | 5142039 | 8815137 | 0.9839 | 0.9788 | `uncompress` | `copyToDir` |
| 5 | 2458 | 19584877 | 5142039 | 0.9839 | 0.9766 | `encodeFileToFile` | `uncompress` |
| 6 | 1518 | 5142039 | 7087108 | 0.9834 | 0.9735 | `uncompress` | `doGet` |
| 7 | 2195 | 8747840 | 5142039 | 0.9819 | 0.9670 | `copy` | `uncompress` |
| 8 | 766 | 3809087 | 20073619 | 0.9814 | 0.8432 | `getWave` | `getPagina` |
| 9 | 2149 | 3401153 | 20886320 | 0.9766 | 0.8685 | `getHTML` | `search` |
| 10 | 2167 | 20210699 | 3631989 | 0.9751 | 0.8787 | `getTicketsForQueue` | `run` |
| 11 | 1380 | 16353490 | 5989666 | 0.9727 | 0.9069 | `executePost` | `postXml` |
| 12 | 2841 | 1169642 | 12724876 | 0.9722 | 0.9402 | `run` | `loadURL` |
| 13 | 1986 | 15971794 | 7846688 | 0.9697 | 0.8092 | `getWave` | `postData` |
| 14 | 1791 | 17202158 | 2736184 | 0.9668 | 0.7072 | `readZoneIDs` | `getURLContent` |
| 15 | 1860 | 9210168 | 20210699 | 0.9663 | 0.9020 | `run` | `getTicketsForQueue` |
| 16 | 1020 | 13596891 | 7908169 | 0.9658 | 0.8464 | `doTransfer` | `execute` |
| 17 | 2448 | 13596891 | 18974466 | 0.9653 | 0.8575 | `doTransfer` | `copyFile` |
| 18 | 1204 | 22844377 | 12724876 | 0.9648 | 0.8440 | `scrapeForIsbns` | `loadURL` |
| 19 | 377 | 14617444 | 2465747 | 0.9644 | 0.8491 | `readUNI` | `postRequest` |
| 20 | 390 | 1122585 | 2199604 | 0.9644 | 0.8259 | `copyImageFile` | `unzipEntry` |

*(Chỉ hiển thị 20/58 ca; danh sách đầy đủ trong `m1_errors.csv`.)*

### False Negative (dự đoán Non-clone, nhãn Clone) — tổng 147 ca

| # | Pair ID | id1 | id2 | prob | cosine | Tên hàm A | Tên hàm B |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 15372 | 7044153 | 7143591 | 0.0154 | -0.2667 | `sendExceptionToServer` | `getWebByUrl` |
| 2 | 16321 | 7044153 | 5989666 | 0.0157 | -0.3144 | `sendExceptionToServer` | `postXml` |
| 3 | 15229 | 7044153 | 7846688 | 0.0161 | -0.2887 | `sendExceptionToServer` | `postData` |
| 4 | 7792 | 5125848 | 13563706 | 0.0162 | 0.3249 | `combineJs` | `doGet` |
| 5 | 7614 | 807346 | 22503685 | 0.0167 | 0.7796 | `URL` | `readGeoParserResult` |
| 6 | 13395 | 17580775 | 9526031 | 0.0176 | 0.7372 | `runInternal` | `register` |
| 7 | 12472 | 10690321 | 23246123 | 0.0189 | -0.2481 | `readData` | `createDialogArea` |
| 8 | 16310 | 13596891 | 22875895 | 0.0190 | -0.2703 | `doTransfer` | `readVersion` |
| 9 | 2369 | 13596891 | 3401153 | 0.0194 | -0.2705 | `doTransfer` | `getHTML` |
| 10 | 6407 | 13596891 | 20028790 | 0.0195 | -0.2763 | `doTransfer` | `setBundleInfoName` |
| 11 | 13924 | 6470716 | 19739421 | 0.0195 | -0.0482 | `main` | `main` |
| 12 | 6370 | 13596891 | 12883117 | 0.0198 | -0.2745 | `doTransfer` | `HashMap` |
| 13 | 6313 | 13596891 | 5989666 | 0.0198 | -0.2755 | `doTransfer` | `postXml` |
| 14 | 15578 | 10690321 | 22401746 | 0.0199 | -0.2030 | `readData` | `sendRequest` |
| 15 | 4376 | 20855053 | 807346 | 0.0203 | -0.2837 | `init` | `URL` |
| 16 | 8396 | 4599372 | 14758866 | 0.0204 | -0.0409 | `readFixString` | `getResourceAsStream` |
| 17 | 13538 | 9210168 | 807346 | 0.0204 | -0.3073 | `run` | `URL` |
| 18 | 1052 | 13596891 | 16931472 | 0.0204 | -0.2676 | `doTransfer` | `handler` |
| 19 | 8817 | 15241397 | 22536033 | 0.0205 | -0.0477 | `doGet` | `setPayload` |
| 20 | 13925 | 17580775 | 20073619 | 0.0206 | -0.2710 | `runInternal` | `getPagina` |

*(Chỉ hiển thị 20/147 ca; danh sách đầy đủ trong `m1_errors.csv`.)*

## M2 GraphCodeBERT DiffMul

- Checkpoint: `D:\Edu\NLP\SemanticCodeCloneDetection\results\exp_m2_diffmul\best_checkpoint.pt`
- Tập test: `results\bcb_5k_test_pairs_kaggle.jsonl`
- Ngưỡng: `0.14`
- Kết quả: TP=2363, TN=2456, FP=44, FN=137, F1=0.9631, Accuracy=0.9638

### Thống kê cosine theo nhóm dự đoán

| Nhóm | Số cặp | Cosine trung bình | Trung vị | Q25 | Q75 | Min | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP | 2363 | 0.9458 | 0.9668 | 0.9466 | 0.9798 | 0.2210 | 1.0000 |
| TN | 2456 | 0.1459 | 0.0442 | 0.0086 | 0.1327 | -0.1290 | 1.0000 |
| FP | 44 | 0.6170 | 0.5909 | 0.3874 | 0.8150 | 0.1645 | 0.9825 |
| FN | 137 | 0.1568 | 0.0992 | 0.0383 | 0.2260 | -0.0302 | 0.8839 |

Tương quan Pearson giữa cosine và xác suất dự đoán: toàn bộ = 0.9097, non-clone = 0.2901, clone = 0.9497.

### False Positive (dự đoán Clone, nhãn Non-clone) — tổng 44 ca

| # | Pair ID | id1 | id2 | prob | cosine | Tên hàm A | Tên hàm B |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 1160 | 5142039 | 8815137 | 0.9912 | 0.9725 | `uncompress` | `copyToDir` |
| 2 | 2195 | 8747840 | 5142039 | 0.9907 | 0.9825 | `copy` | `uncompress` |
| 3 | 2458 | 19584877 | 5142039 | 0.9907 | 0.9743 | `encodeFileToFile` | `uncompress` |
| 4 | 1518 | 5142039 | 7087108 | 0.9902 | 0.9622 | `uncompress` | `doGet` |
| 5 | 970 | 8588992 | 364438 | 0.9785 | 0.8106 | `perform` | `convert` |
| 6 | 2520 | 18989018 | 16142024 | 0.9517 | 0.7602 | `doSetInput` | `copyFile` |
| 7 | 2103 | 18989018 | 22801734 | 0.9502 | 0.7593 | `doSetInput` | `setup` |
| 8 | 243 | 21921000 | 9210168 | 0.9438 | 0.8571 | `parse` | `run` |
| 9 | 377 | 14617444 | 2465747 | 0.8857 | 0.8477 | `readUNI` | `postRequest` |
| 10 | 1821 | 3401153 | 18097962 | 0.8613 | 0.9303 | `getHTML` | `getVersion` |
| 11 | 2550 | 11952735 | 6682825 | 0.8511 | 0.6035 | `copyFile` | `unknown` |
| 12 | 80 | 7006052 | 6682825 | 0.8457 | 0.6031 | `copyFileTo` | `unknown` |
| 13 | 2841 | 1169642 | 12724876 | 0.8394 | 0.8283 | `run` | `loadURL` |
| 14 | 2036 | 20024612 | 21921000 | 0.8374 | 0.9067 | `parse` | `parse` |
| 15 | 335 | 21998571 | 23246123 | 0.8110 | 0.7479 | `readTwitterFead` | `createDialogArea` |
| 16 | 1204 | 22844377 | 12724876 | 0.7959 | 0.7943 | `scrapeForIsbns` | `loadURL` |
| 17 | 1961 | 18097962 | 7846688 | 0.7778 | 0.9137 | `getVersion` | `postData` |
| 18 | 1380 | 16353490 | 5989666 | 0.7769 | 0.9127 | `executePost` | `postXml` |
| 19 | 1785 | 15822137 | 12783713 | 0.6768 | 0.5489 | `unknown` | `sendErrorMessage` |
| 20 | 1092 | 17791385 | 15822137 | 0.6353 | 0.5201 | `copyFile` | `unknown` |

*(Chỉ hiển thị 20/44 ca; danh sách đầy đủ trong `m2_errors.csv`.)*

### False Negative (dự đoán Non-clone, nhãn Clone) — tổng 137 ca

| # | Pair ID | id1 | id2 | prob | cosine | Tên hàm A | Tên hàm B |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 14718 | 5836744 | 10212189 | 0.0082 | 0.0511 | `read` | `getXML` |
| 2 | 8252 | 5836744 | 742465 | 0.0085 | 0.0412 | `read` | `doVersionCheck` |
| 3 | 16301 | 5836744 | 19109981 | 0.0094 | 0.0857 | `read` | `startScript` |
| 4 | 4376 | 20855053 | 807346 | 0.0150 | 0.1938 | `init` | `URL` |
| 5 | 8351 | 6890417 | 5836744 | 0.0154 | 0.2501 | `main` | `read` |
| 6 | 7614 | 807346 | 22503685 | 0.0155 | 0.2772 | `URL` | `readGeoParserResult` |
| 7 | 8704 | 804637 | 8754809 | 0.0160 | -0.0173 | `readAndRewrite` | `launch` |
| 8 | 9256 | 807346 | 12883117 | 0.0160 | 0.1133 | `URL` | `HashMap` |
| 9 | 12707 | 536614 | 8754809 | 0.0160 | -0.0173 | `readAndRewrite` | `launch` |
| 10 | 14128 | 13783898 | 8754809 | 0.0167 | -0.0242 | `encodeFileToFile` | `launch` |
| 11 | 13538 | 9210168 | 807346 | 0.0173 | 0.1060 | `run` | `URL` |
| 12 | 2981 | 18693224 | 8754809 | 0.0173 | 0.0089 | `runDynusT` | `launch` |
| 13 | 3277 | 884867 | 8754809 | 0.0180 | -0.0245 | `main` | `launch` |
| 14 | 1635 | 16851955 | 2324868 | 0.0182 | -0.0302 | `testTrainingBackprop` | `actionPerformed` |
| 15 | 4151 | 14758866 | 22264586 | 0.0183 | 0.0541 | `getResourceAsStream` | `copyFiles` |
| 16 | 2382 | 17729554 | 8754809 | 0.0184 | -0.0266 | `clonarFichero` | `launch` |
| 17 | 5736 | 15241397 | 6470716 | 0.0187 | 0.7731 | `doGet` | `main` |
| 18 | 2226 | 2324868 | 21488518 | 0.0188 | -0.0217 | `actionPerformed` | `createNew` |
| 19 | 13455 | 2324868 | 4599372 | 0.0188 | -0.0230 | `actionPerformed` | `readFixString` |
| 20 | 11482 | 7945594 | 5836744 | 0.0189 | 0.8399 | `fileDownload` | `read` |

*(Chỉ hiển thị 20/137 ca; danh sách đầy đủ trong `m2_errors.csv`.)*
