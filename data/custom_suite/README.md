# Bo test custom_suite (rieng biet voi custom_hard)

5 bai rieng biet, moi bai 1 base + 4 bien the clone + 1 non-clone kho
(cung skeleton, khac ngu nghia) + 8 cap cross-problem.

| Bai | Base | Bien the clone | Non-clone kho |
| --- | ---- | -------------- | ------------- |
| P1 sum 1..n | `p1_base_sum.cpp` (for+cin/cout) | `p1_v_while/dowhile/scanf/combo` | `p1_n_fact.cpp` (giai thua) |
| P2 UCLN | `p2_base_gcd.cpp` (while+cin/cout) | `p2_v_for/dowhile/scanf/combo` | `p2_n_digitsum.cpp` (tong chu so) |
| P3 max mang | `p3_base_max.cpp` (for+cin/cout) | `p3_v_while/dowhile/scanf/combo` | `p3_n_sum.cpp` (tong mang) |
| P4 SNT | `p4_base_prime.cpp` (for+cin/cout) | `p4_v_while/dowhile/scanf/combo` | `p4_n_square.cpp` (chinh phuong) |
| P5 DSLK + OOP | `p5_base_list.cpp` (struct + 4 ham) | `p5_v_class/while/scanf/combo` | `p5_n_palindrome.cpp` (kiem tra doi xung) |

- Quy uoc bien the: `while` = doi for/while + doi ten bien; `dowhile` = do-while
  + doi ten; `scanf` = scanf/printf + doi ten; `combo` = loop + scanf + dao
  dieu kien phu dinh + bien thua/dead code (kho nhat).
- **P5 (them theo feedback: code dai, co ham, cau truc DSLK, huong doi tuong)**:
  moi file 59–70 dong (P1–P4 chi 11–17 dong), `struct Node` + cac ham
  `pushBack/sumList/printList/freeList`. Bien the `v_class` cua P5 = **cung hanh
  vi** nhung viet bang `class IntList` + phuong thuc (thay cho `dowhile`) →
  cap `P5_struct_class` dung de do case "struct vs class gan nhu giong nhau";
  `p5_n_palindrome.cpp` dung khung `class` + `Node` tuong tu nhung **khac ngu
  nghia** (kiem tra day doi xung thay vi tinh tong) → non-clone kho.
- Manifest: `pairs.jsonl` — 21 clone (label 1) + 5 nonclone kho (label 0)
  + 8 cross-problem (label 0) = 34 cap.

## Chay test

```bash
python scripts/score_custom_suite.py --models B1        # baseline MOSS, nhanh
python scripts/score_custom_suite.py --models B1 M1     # + CodeBERT
python scripts/score_custom_suite.py                   # full theo demo_web.yaml
python scripts/score_custom_suite.py --pairs P5_struct_class P5_nonclone_palindrome \
    --models B1 M1 M2 M3 M4 M5 --merge                 # chi cham cap moi, giu cap cu
```

Ket qua: `workspace/custom_suite_scores.json`.
Ky vong: clone de ~ diem cao, combo thap hon, nonclone kho + cross ~ diem thap.

## Evidence cho bao cao

```bash
python scripts/evidence_custom_suite.py --pairs P5_struct_class P5_nonclone_palindrome
```

Ra `workspace/demo_suite_<pair>.tex` + `workspace/demo_suite_moss_<pair>.tex`
+ `workspace/demo_suite_evidence.json`. Code dai (>48 dong) tu dong dung
`\scriptsize` cho minipage de hinh vua mot trang A4.
