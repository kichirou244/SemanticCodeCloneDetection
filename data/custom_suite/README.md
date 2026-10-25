# Bo test custom_suite (rieng biet voi custom_hard)

6 bai rieng biet, moi bai 1 base + cac bien the clone + 1 non-clone kho
(cung skeleton, khac ngu nghia) + cap cross-problem. Tong **89 cap**:
**25 clone + 64 non-clone** (non-clone gap 4 lan ban dau 16 cap).

| Bai | Base | Bien the clone | Non-clone kho |
| --- | ---- | -------------- | ------------- |
| P1 sum 1..n | `p1_base_sum.cpp` (for+cin/cout) | `p1_v_while/dowhile/scanf/combo` | `p1_n_fact.cpp` (giai thua) |
| P2 UCLN | `p2_base_gcd.cpp` (while+cin/cout) | `p2_v_for/dowhile/scanf/combo` | `p2_n_digitsum.cpp` (tong chu so) |
| P3 max mang | `p3_base_max.cpp` (for+cin/cout) | `p3_v_while/dowhile/scanf/combo` | `p3_n_sum.cpp` (tong mang) |
| P4 SNT | `p4_base_prime.cpp` (for+cin/cout) | `p4_v_while/dowhile/scanf/combo` | `p4_n_square.cpp` (chinh phuong) |
| P5 DSLK + OOP | `p5_base_list.cpp` (struct + 4 ham) | `p5_v_class/while/scanf/combo` | `p5_n_palindrome.cpp` (kiem tra doi xung) |
| P6 DSLK + OOP **dai** | `p6_base_list.cpp` (struct + chen co thu tu) | `p6_v_class/while/scanf/combo` | `p6_n_filter.cpp` (them cuoi + xoa lon hon nguong + dem le) |

- Quy uoc bien the: `while` = doi for/while + doi ten bien; `scanf` = scanf/printf
  + doi ten; `combo` = loop + scanf + dao dieu kien phu dinh + bien thua/dead code
  (kho nhat).
- **P5 (them theo feedback: code dai, co ham, cau truc DSLK, huong doi tuong)**:
  moi file 59–70 dong, `struct Node` + cac ham
  `pushBack/sumList/printList/freeList`. Bien the `v_class` cua P5 = **cung hanh
  vi** nhung viet bang `class IntList` + phuong thuc (thay cho `dowhile`) →
  cap `P5_struct_class` dung de do case "struct vs class gan nhu giong nhau";
  `p5_n_palindrome.cpp` dung khung `class` + `Node` tuong tu nhung **khac ngu
  nghia** (kiem tra day doi xung thay vi tinh tong) → non-clone kho.
- **P6 (nhom code dai, 83–93 dong)** — thay nhom P6 cu 126–148 dong (bi cat 60–70 %
  token, xem lai lich su git): base `p6_base_list.cpp` = `struct Node` + cac ham
  `insertSorted/removeDuplicates/sumList/printList/freeList`, doc $n$ so, chen co
  thu tu, xoa trung, in lai va in tong. Bien the `v_class` = **cung hanh vi**
  viet bang `class IntList` (constructor + `~IntList()` goi `clear()` + phuong
  thuc) → cap `P6_struct_class`. `p6_n_filter.cpp` = cung khung `Node`/danh sach
  nhung **khac ngu nghia** (them vao cuoi, xoa phan tu lon hon nguong, dem le,
  khong sap xep) → non-clone kho; khac biet nam o cac ham giua file, khong chi o
  cuoi file.
- Manifest: `pairs.jsonl` — 25 clone (label 1) + **30 nonclone kho** + **34 cross-problem**
  (label 0, tong 64 non-clone) = **89 cap**. Cap P6: `P6_struct_class`,
  `P6_while`, `P6_scanf`, `P6_combo`, `P6_nonclone_filter`, `X_P5P6`,
  `X_P5P6_class`.

## Mo rong non-clone (2026-10-25) — 16 → 64 cap

Bo cu chi co 16 non-clone, qua it so voi thuc te (phan lon cap dua vao he
thong khong phai clone), va de danh gia sai vi mot vai mau kho. Script
`scripts/build_custom_suite_negatives.py` sinh them **48 cap non-clone** theo
hai luat tat dinh, **khong sua cap clone cu**:

| Nguon | So cap | Cach sinh |
| --- | ---: | --- |
| Non-clone kho (cung bai) | 24 | file `pX_n_*.cpp` vs **tat ca** file clone cua cung bai (base + 4 bien the); bo sung 4 cap/bai cho 6 bai |
| Cross-problem base-base | 6 | 6 cap base con thieu trong 15 cap giua 6 bai |
| Cross-problem cung kieu bien doi | 18 | cung `while`/`dowhile`/`scanf`/`combo` nhung khac bai (khac problem ⇒ chac chan non-clone) |
| **Tong moi** | **48** | 24 nonclone kho + 24 nonclone de |

Ket qua: `nonclone_kho` 6 → **30**, `nonclone_de` 10 → **34**, tong non-clone
**16 → 64 (gap 4,0 lan)**; ti le non-clone/clone = 64/25 ≈ 2,56. Cac cap moi
duoc sinh lai bang:

```bash
python scripts/build_custom_suite_negatives.py --check   # chi kiem tra
python scripts/build_custom_suite_negatives.py --write   # ghi them (idempotent)
```

`pairs.jsonl` giu nguyen toan bo cap cu (ke ca `pair_id`), chi append cap moi.
Test `tests/test_custom_suite_manifest.py` chan cap trung, sai nhan cross-bai,
va generator chay lai khong them gi.

## Do do dai token (truncation 512)

Ca 6 file P6 (83–93 dong) van vuot 512 subword token (do bang tokenizer
GraphCodeBERT), nen M1/M2/M5 (`max_length: 512`) chi doc duoc phan dau file.
Gioi han 512 la **cung kien truc** cua CodeBERT/GraphCodeBERT (embedding vi tri
toi da 514), khong the noi cua so ma khong huan luyen lai:

| File | Dong | Token | Bi cat khi max_length=512 |
| ---- | ---- | ----- | ------------------------- |
| `p6_v_scanf.cpp` | 83 | 818 | 37,4 % |
| `p6_base_list.cpp` | 84 | 822 | 37,7 % |
| `p6_v_class.cpp` | 86 | 1016 | 49,6 % |
| `p6_v_combo.cpp` | 87 | 858 | 40,3 % |
| `p6_n_filter.cpp` | 88 | 878 | 41,7 % |
| `p6_v_while.cpp` | 93 | 927 | 44,8 % |

Do: `workspace/token_lengths.json` (do voi `add_special_tokens=True`, giong so
P5: `p5_base_list` 521, `p5_v_scanf` 520, `p5_v_class` 683 token).

## Chay test

```bash
python scripts/score_custom_suite.py --models B1        # baseline MOSS, nhanh
python scripts/score_custom_suite.py --models B1 M1     # + CodeBERT
python scripts/score_custom_suite.py                   # full theo demo_web.yaml
python scripts/score_custom_suite.py --models M1 M2 M3 M4 M5 --only-new --merge
    # chi cham cap CHUA co trong workspace/custom_suite_scores.json (sau khi
    # them dong vao pairs.jsonl); M4/M5 chay Joern 1 lan cho toan bo file
python scripts/score_custom_suite.py --pairs P6_struct_class P6_nonclone_filter \
    --models B1 M1 M2 M3 M4 M5 --merge                 # chi cham cap moi, giu cap cu

python scripts/summarize_custom_suite.py               # mean ± SD + F1 bootstrap
```

Luu y: may khong khai `MOSS_USER_ID` hop le thi B1 tu chay winnowing local (khac
so MOSS that). Muon ep che do offline ma khong sua code, dat
`MOSS_USER_ID=local` — client that bai ngay va roi ve nhanh local.

Ket qua: `workspace/custom_suite_scores.json`.
Tong hop: `workspace/custom_suite_summary.{json,md,tex}` — bang diem trung binh
± do lech chuan theo nhom, F1 bootstrap, va bang "common" tren giao cac cap co
du diem cho moi mo hinh (B1/MOSS that chi co 41 cap loi cu; M1–M5 co 89 cap).
Ky vong: clone de ~ diem cao, combo thap hon, nonclone kho + cross ~ diem thap.

## Evidence cho bao cao

```bash
python scripts/evidence_custom_suite.py --pairs P6_struct_class P6_nonclone_filter
```

Ra `workspace/demo_suite_<pair>.tex` (5 model M1..M5, giong demo_clone5.tex)
+ `workspace/demo_suite_moss_<pair>.tex` (MOSS server)
+ `workspace/demo_suite_evidence.json` (tich luy, khong ghi de cap cu).
Code dai (>48 dong) tu dong dung `\scriptsize`; tep dai nhat cua nhom P6
(93 dong) dung `\tiny` va chi hien 90 dong dau, phan con lai danh dau bang
dong `\ldots` de hinh van vua mot trang A4 (`fig_layout`).
