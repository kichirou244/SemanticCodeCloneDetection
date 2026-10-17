# Prompt vẽ kiến trúc pipeline M2 (GraphCodeBERT) — bản số liệu thật

Mục đích: vẽ **kiến trúc suy luận của M2** theo chiều ngang trái → phải, với **đúng**
đầu vào, đúng tham số và đúng kết quả mà repo đang chạy.

Mọi con số trong file này lấy trực tiếp từ repo (xem §1 và §7 để kiểm chứng).

---

## 1. Số liệu thật của M2 (dùng để điền vào hình)

| Hạng mục | Giá trị thật | Nguồn |
| :--- | :--- | :--- |
| Model | `microsoft/graphcodebert-base` (RoBERTa-base: 12 layer, hidden 768) | `configs/demo_web.yaml` |
| Class | `TransformerPairClassifier` (siamese dùng chung trọng số) | `src/models/transformer_pair.py` |
| Run | **EXP-003 bản 512** (warm-start từ checkpoint 256, 2 epoch) | `results/exp003_graphcodebert_poj104_512/` |
| `max_length` | **512** token | `configs/demo_web.yaml` |
| Ngưỡng quyết định | **0,66** (chọn trên validation, đóng băng) | `results/exp003_graphcodebert_poj104_512/threshold.json` |
| Checkpoint demo | `results/exp003_graphcodebert_poj104_512/best_checkpoint.pt` | `configs/demo_web.yaml` |
| Head | `[hA ; hB ; |hA − hB|]` (3 × 768 = 2304) → Linear → Tanh → Dropout 0.1 → Linear → 1 logit → sigmoid | `src/models/transformer_pair.py` |

## 2. Đầu vào thật (2 đoạn code của cặp P1_combo)

`data/custom_suite/p1_base_sum.cpp` (A) vs `data/custom_suite/p1_v_combo.cpp` (B) —
GT = **clone** (Type-3/4: đổi vòng lặp, đổi IO, đổi tên biến, thêm biến thừa).

**File A — `p1_base_sum.cpp`**
```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, sum = 0;
    cin >> n;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    cout << sum << endl;
    return 0;
}
```

**File B — `p1_v_combo.cpp`**
```cpp
#include <cstdio>
int main() {
    int N;
    scanf("%d", &N);
    int acc = 0;
    int step = 1;
    int unused = 999;
    (void)unused;
    int k = 1;
    while (!(k > N)) {
        acc = acc + k;
        k = k + step;
    }
    printf("%d\n", acc);
    return 0;
}
```

## 3. Kết quả thật của M2 cho cặp này

| | |
| :--- | :--- |
| `score` (P(clone)) | **1,000** |
| `threshold` | **0,66** |
| `verdict` | **clone** (đúng) |
| Số token mỗi phía | 148 — không bị cắt ở `max_length = 512` |

**Highlight thật** (cặp từ tương ứng do M2 chọn, mutual-best cosine, tô cùng màu):

| Cặp # | Màu | File A | File B |
| :--- | :--- | :--- | :--- |
| **6** | **xanh ngọc `#A8E0E0`** | **`cin`**, `1` | **`scanf`**, `1` |
| 3 | vàng chanh `#F7E08B` | `{`, `sum`, `endl` | `{`, `acc`, `"%d\n"` |
| 0 | hồng `#F4B6C2` | `n`, `+=`, `;` | `N`, `+`, `;` |
| 1 | xanh dương `#B5D8F7` | `int`, `++)`, `{` | `int`, `))`, `{` |
| 2 | xanh lá `#B8E6B8` | `main`, `;`, `return` | `main`, `;`, `return` |
| 4 | tím `#D9C2F0` | `#include <bits/stdc++.h>`, `}`, `sum` | `#include <cstdio>`, `}`, `acc` |
| 5 | cam `#F5C99B` | `()`, `}` | `()`, `}` |
| 7 | nâu nhạt `#E8C39E` | `int`, `sum` | `int`, `acc` |

Ba cặp đáng chú ý để nhìn ra "model hiểu ngữ nghĩa chứ không so chữ":

- `cin` ↔ `scanf` (cặp 6) — hai cách đọc vào hoàn toàn khác cú pháp.
- `sum` ↔ `acc` (cặp 4 và 7) — biến đổi tên.
- `endl` ↔ `"%d\n"` (cặp 3) — cùng việc in kết quả, khác cách viết.

Token **vàng mustard `#EFD9A0`** = token model dựa vào để cho điểm nhưng **không ghép được cặp**:
A → `using`, `namespace`, `std`, `;`(dòng 2), `>>`, `n`(dòng 5), `cout`, `<<`;
B → `step`, `unused`, `999`, `(void)`, `=`, `k`, `+`, `step`, `printf`.

Đây chính là bảng màu mà báo cáo `[17 - 10]` đang dùng cho các hình highlight — dùng lại
đúng mã màu này thì hình mới khớp các hình còn lại trong báo cáo.

---

## 4. Prompt chính (GPT-image / DALL·E 3 / Ideogram / Gemini)

Copy nguyên khối:

```text
A high-quality, professional academic architecture diagram of "M2: GraphCodeBERT
Semantic Code Clone Detection — Inference Pipeline". Wide landscape format 16:9,
pure white background, modern flat vector infographic style, publication-ready.

LAYOUT: strictly horizontal left-to-right flow of six rounded modules connected by
clean indigo (#4338CA) arrows with sharp geometric heads. Module headers are small
bold uppercase badges (01..06).

MODULE 01 — "INPUT PAIR"
Two small monospace code cards side by side, labelled "A · p1_base_sum.cpp" and
"B · p1_v_combo.cpp". Render exactly these lines in a code editor style
(syntax: keywords indigo, strings brown, plain text dark slate):

A:  int n, sum = 0;  |  cin >> n;  |  for (int i = 1; i <= n; i++) {  |
    sum += i;  |  }  |  cout << sum << endl;

B:  int N;  |  scanf("%d", &N);  |  int acc = 0;  |  int k = 1;  |
    while (!(k > N)) {  |  acc = acc + k;  |  k = k + step;  |  }  |
    printf("%d\n", acc);

Sub-label: "C++ · cùng bài toán tính tổng 1..n · GT = clone"

MODULE 02 — "TOKENIZE (max_length = 512)"
A small strip showing a RoBERTa BPE token sequence with special tokens:
"<s> int n , sum = 0 ; cin >> n ; for ( int i … </s>"
Sub-label: "AutoTokenizer · vocab 50265 · cắt ở 512 token · cặp này chỉ 148 token"

MODULE 03 — "SHARED SIAMESE ENCODER"
One central neural-network block with two inputs entering it from the left and two
outputs leaving on the right, clearly showing the SAME block is used for both sides
(a subtle "shared weights" double-arrow between the two entry points).
Block label: "GraphCodeBERT (RoBERTa-base, 12 layers, hidden 768, ~125M params)".
Inside: rows of stacked attention layers drawn as thin horizontal slabs.
Outputs: two dense 768-dim vectors labelled "hA" and "hB" (draw them as small
vertical vector bars).

MODULE 04 — "PAIR HEAD"
A concatenation block showing "[ hA ; hB ; |hA − hB| ]" with the dimension
"3 × 768 = 2304", flowing into a small stack labelled
"Linear → Tanh → Dropout 0.1 → Linear" and then a single logit node.

MODULE 05 — "PREDICTION"
A probability gauge/arc filled almost completely, with a large value "P(clone) = 1.000"
under it. Below, a pill badge with an emerald check: "CLONE · ngưỡng 0.66".
Sub-label: "sigmoid(logit) · ngưỡng chọn trên validation, đóng băng"

MODULE 06 — "TOKEN-LEVEL EXPLANATION"
Two small code cards side by side, same snippets as module 01, now with coloured
token highlights forming matched pairs across the two cards. Vertical curved
connector lines tie each pair together. Use exactly these colours:
  - indigo/cyan  #A8E0E0 pair: "cin" (A)  ↔  "scanf" (B)   [emphasise this pair]
  - lemon        #F7E08B pair: "sum" "endl" (A) ↔ "acc" "%d\n" (B)
  - pink         #F4B6C2 pair: "n" "+=" (A) ↔ "N" "+" (B)
  - blue         #B5D8F7 pair: "int" (A) ↔ "int" (B)
  - green        #B8E6B8 pair: "main" ";" "return" (A) ↔ "main" ";" "return" (B)
  - violet       #D9C2F0 pair: "#include" "}" "sum" (A) ↔ "#include" "}" "acc" (B)
  - orange       #F5C99B pair: "()" "}" (A) ↔ "()" "}" (B)
  - light brown  #E8C39E pair: "int" "sum" (A) ↔ "int" "acc" (B)
Tokens that matter to the score but have no partner are shaded mustard #EFD9A0.
Sub-label: "saliency |grad × embedding| + ghép từ mutual-best theo cosine"

STYLE: flat vector, minimal, no gradients, no 3D, no drop shadows, no extra icons,
no fake filler text, no watermark, no signature. Background pure white #FFFFFF.
Modules are soft light cards (#F8FAFC) with a thin #CBD5E1 outline and rounded
corners. Accent colours: indigo #4338CA for arrows and headers, emerald #10B981 for
the positive verdict, cyan/violet for the highlighted token pairs.
Vietnamese diacritics must be complete and correct. No text may overflow its module.
```

## 5. Prompt rút gọn (Midjourney / model hay sai chính tả)

Chỉ nên dùng khi định **chèn chữ bằng LaTeX sau**, vì ảnh sẽ sai code:

```text
academic deep learning architecture diagram, horizontal left-to-right, six modules:
two C++ code cards, tokenizer strip, one siamese Transformer encoder labelled
GraphCodeBERT shared by both inputs, pair head "[hA; hB; |hA-hB|]" MLP, probability
gauge 1.000 with emerald CLONE badge, and two code cards with matching tokens
colour-coded across the pair. Flat vector, pure white background, indigo and emerald
accents, crisp thin lines, publication quality --ar 16:9 --style raw --v 7
```

## 6. ⚠️ Lưu ý quan trọng — pipeline này KHÔNG có nhánh DFG

Prompt cũ vẽ thêm nhánh "Data Flow Graph (DFG) / graph-guided attention". **Nhánh đó không
tồn tại trong implementation của repo**:

- `src/models/transformer_pair.py` chỉ truyền `input_ids` + `attention_mask` (đầu ra của
  tokenizer) vào `AutoModel` — **không** truyền `position_ids` / cạnh DFG / graph mask.
- Nghĩa là M2 của ta chạy GraphCodeBERT như một **encoder văn bản thuần** (giống hệt M1 về
  code path, khác đúng bộ trọng số pretrained). Đây là chủ ý của DEC-016/DEC-020 để so sánh
  RQ1 "M1 vs M2" không lẫn khác biệt implementation.

⇒ Trong hình **không nên** vẽ nhánh DFG. Nếu muốn nhắc, chỉ nên ghi một dòng nhỏ kiểu
"chỉ dùng nhánh text; DFG của GraphCodeBERT không được cấp vào" — nếu không nói gì thì
hình sẽ bị hiểu là mô hình có dùng đồ thị luồng dữ liệu, tức là sai.

## 7. Bản 256 chỉ để đối chiếu — không dùng vẽ hình

Repo vẫn giữ run 256 (`results/exp003_graphcodebert_poj104/`, ngưỡng 0,43) để đối chiếu
trong bảng huấn luyện/test. **Không dùng nó cho hình:** demo và báo cáo đã chốt dùng bản
512 (DEC-031), nên `configs/demo_web.yaml` khai `max_length: 512` và trỏ thẳng vào
`results/exp003_graphcodebert_poj104_512/`.

Nếu vì lý do nào đó phải vẽ theo bản 256 thì đổi 3 chỗ — `max_length = 256`, ngưỡng
`0,43`, checkpoint `results/exp003_graphcodebert_poj104/best_checkpoint.pt` — nhưng hình
khi đó sẽ không khớp số trong báo cáo.

## 8. Ghi chú khi chèn vào báo cáo

- Tỉ lệ 16:9 hoặc 2:1; chèn bằng
  `\includegraphics[width=\textwidth]{figures/m2_architecture.png}`.
- Nhờ AI vẽ ~15 dòng code chính xác là việc rất dễ sai. Cách chắc ăn: cho AI vẽ **khung**,
  rồi bạn dán code + tô màu bằng LaTeX/PowerPoint chồng lên.
- Bảng màu ở §3 khớp đúng bảng màu trong báo cáo `[17 - 10]` (hình highlight P1_combo),
  nên hai chỗ sẽ nhìn như cùng một hệ.
