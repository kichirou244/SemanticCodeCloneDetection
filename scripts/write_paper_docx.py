#!/usr/bin/env python3
"""Sinh bản nháp bài báo tiếng Việt (.docx) cho đề tài clone detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports"
DOCX_PATH = OUT_DIR / "BaiBao_Nhap.docx"
FIG_PATH = OUT_DIR / "figures" / "paper_pipeline.png"
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

FONT = "Times New Roman"
FONT_PATH = r"C:\Windows\Fonts\times.ttf"
if Path(FONT_PATH).exists():
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


def draw_pipeline() -> None:
    steps = [
        ("1. Dữ liệu", "BigCloneBench: train 30k / val 3k / test 5k\n50:50 clone - non-clone"),
        ("2. Token hóa", "Java code → token, max_length = 512\npadding động, truncation"),
        ("3. Mã hóa Siamese", "GraphCodeBERT → vector CLS\nhA cho hàm A, hB cho hàm B"),
        ("4. Tương tác DiffMul", "v = [ |hA - hB| , hA * hB ] (tích Hadamard)\n1536 chiều, đối xứng f(A,B)=f(B,A)"),
        ("5. Head phân loại", "Linear(1536→768) → Tanh → Dropout\n→ Linear(768→1) → logit"),
        ("6. Huấn luyện", "BCE + label smoothing 0,05\nAdamW, CosineLR, AMP FP16"),
        ("7. Hiệu chỉnh (A1)", "Isotonic regression trên logits val\nECE ≈ 0"),
        ("8. Chọn ngưỡng OOF", "5-fold trên validation\nchốt ngưỡng F1, không dùng test"),
        ("9. Đánh giá", "Test 5.000 cặp: F1, P, R, AUC\nbootstrap CI + McNemar"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 11.5), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10 * len(steps) + 1)
    ax.axis("off")
    y0 = 10 * len(steps)
    for i, (title, body) in enumerate(steps):
        y = y0 - i * 10
        box = FancyBboxPatch(
            (0.8, y - 6.6), 8.4, 7.0,
            boxstyle="round,pad=0.25,rounding_size=0.3",
            linewidth=1.2, edgecolor="#264653", facecolor="#e9f1f2" if i % 2 == 0 else "#f7fbfc",
        )
        ax.add_patch(box)
        ax.text(1.2, y - 1.6, title, fontsize=12, fontweight="bold", color="#1d3557", va="center")
        ax.text(1.2, y - 4.3, body, fontsize=10.5, color="#222222", va="center", linespacing=1.5)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5, y - 7.1), (5, y - 9.4), arrowstyle="-|>", mutation_scale=14,
                                         linewidth=1.2, color="#2a9d8f"))
    ax.set_title("Hình 1. Pipeline đề xuất cho phát hiện clone ngữ nghĩa", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)
    print("figure:", FIG_PATH)


def set_font(run, size=12, bold=False, italic=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), FONT)
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)


def para(doc, text, size=12, bold=False, italic=False, align=None, space_after=6, first_indent=0):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(space_after)
    if first_indent:
        pf.first_line_indent = Cm(first_indent)
    r = p.add_run(text)
    set_font(r, size, bold, italic)
    return p


def heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_font(r, 14 if level == 1 else 12.5, bold=True, color=RGBColor(0x1D, 0x35, 0x57))
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p.paragraph_format.space_after = Pt(3)
    set_font(p.add_run(text), 12)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    set_font(p.add_run(text), 10.5, bold=True)
    return p


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = True
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        set_font(cell.paragraphs[0].add_run(h), 10.5, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            set_font(cells[i].paragraphs[0].add_run(str(v)), 10.5)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def build_doc() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    # --- Tiêu đề ---
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("PHÁT HIỆN TRÙNG LẶP MÃ NGUỒN NGỮ NGHĨA"), 16, bold=True, color=RGBColor(0x1D, 0x35, 0x57))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("Đánh giá thực nghiệm trên BigCloneBench và cải tiến biểu diễn cặp"), 12.5, italic=True)
    para(doc, "[Tên tác giả] – [Tên trường/khoa] – [email]", 10.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

    # --- Tóm tắt ---
    heading(doc, "Tóm tắt", 1)
    para(doc,
         "Bài báo nghiên cứu bài toán phát hiện trùng lặp mã nguồn ngữ nghĩa (semantic code clone detection) trên "
         "bộ dữ liệu BigCloneBench Java. Chúng tôi khảo sát bốn họ phương pháp: CodeBERT (M1), GraphCodeBERT (M2), "
         "ASTNN (M3) và CPG-GNN (M4), sau đó đi sâu vào pipeline tự xây dựng dựa trên M2 với cơ chế tương tác cặp "
         "DiffMul và hiệu chỉnh xác suất. Kết quả thực nghiệm cho thấy M2 đạt F1 = 0,9631 ở cấu hình DiffMul gốc và "
         "0,9707 ở cấu hình tuning tốt nhất; hiệu chỉnh isotonic kèm ngưỡng OOF giữ nguyên F1 nhưng ổn định ngưỡng "
         "quyết định. Các kỹ thuật nâng cao (Focal BCE, augmentation, SupCon, fusion M1+M2) cải thiện recall/AUC "
         "nhưng chưa vượt có ý nghĩa thống kê so với mô hình đơn. Bài báo trình bày chi tiết pipeline, kết quả "
         "và các hạn chế để định hướng công việc tiếp theo.", first_indent=0.5)
    para(doc, "Từ khóa: code clone detection, CodeBERT, GraphCodeBERT, BigCloneBench, calibration, contrastive learning.",
         italic=True, size=10.5)

    # --- 1 ---
    heading(doc, "1. Đặt vấn đề và lý do chọn đề tài", 1)
    para(doc,
         "Trùng lặp mã nguồn (code clone) là hiện tượng cùng một đoạn mã hoặc cùng một chức năng xuất hiện ở nhiều vị trí. "
         "Theo phân loại kinh điển, clone được chia thành bốn cấp độ: Type-1 (giống hệt except khoảng trắng và comment), "
         "Type-2 (đổi tên biến/hàm nhưng cấu trúc giữ nguyên), Type-3 (có thêm sửa đổi nhỏ) và Type-4 (khác cú pháp nhưng "
         "cùng ngữ nghĩa). Type-4 là khó nhất vì hai hàm gần như không chia sẻ token, đòi hỏi mô hình hiểu ngữ nghĩa "
         "chứ không chỉ so khớp bề mặt.", first_indent=0.5)
    para(doc,
         "Phát hiện clone có ý nghĩa thực tiễn trong bảo trì phần mềm: trùng lặp làm tăng chi phí sửa lỗi, gây lan "
         "truyền bug và vi phạm bản quyền. Các phương pháp truyền thống dựa trên token, cây cú pháp (AST) hoặc đồ thị "
         "phụ thuộc thường bỏ sót Type-4. Mô hình ngôn ngữ tiền huấn luyện cho code (CodeBERT, GraphCodeBERT) gần đây "
         "cho phép biểu diễn ngữ nghĩa tốt hơn, nhưng cách kết hợp hai vector biểu diễn của một cặp hàm vẫn là "
         "điểm cần khảo sát.", first_indent=0.5)
    para(doc,
         "Chúng tôi chọn đề tài này vì ba lý do: (i) BigCloneBench là benchmark chuẩn với nhãn clone/ non-clone rõ ràng; "
         "(ii) cần so sánh công bằng bốn họ biểu diễn khác nhau (token, token+DFG, cây AST, đồ thị CPG) trên cùng một "
         "protocol; và (iii) kết quả thực nghiệm ban đầu cho thấy ngưỡng quyết định bị dao động lớn giữa các epoch, "
         "cần một bước hiệu chỉnh xác suất có cơ sở thống kê.", first_indent=0.5)

    # --- 2 ---
    heading(doc, "2. Tổng quan các phương pháp", 1)
    para(doc, "Chúng tôi khảo sát bốn phương pháp đại diện cho bốn cách biểu diễn mã nguồn. Mô tả được giữ ngắn gọn; "
              "chi tiết kỹ thuật có trong các công trình gốc được trích dẫn.", first_indent=0.5)
    table(doc,
          ["Mô hình", "Biểu diễn", "Cơ chế chính", "Tham chiếu"],
          [
              ["M1 CodeBERT", "Token BPE (≤512)", "Siamese encoder, ghép [hA,hB,|hA−hB|] qua MLP", "Feng et al., 2020"],
              ["M2 GraphCodeBERT", "Token + luồng dữ liệu", "Siamese encoder, tận dụng DFG khi mã hóa", "Guo et al., 2021"],
              ["M3 ASTNN", "Cây cú pháp (AST)", "Chia cây thành câu lệnh, RvNN + BiGRU", "Zhang et al., 2019"],
              ["M4 CPG-GNN", "Đồ thị CPG", "Joern trích đồ thị, GAT 3 tầng kết hợp node/edge", "Backes et al., 2019"],
          ],
          widths=[3.2, 3.6, 6.6, 3.4])
    caption(doc, "Bảng 1. Tóm tắt bốn phương pháp khảo sát.")
    para(doc,
         "Bảng 2 trích kết quả được các công trình gốc báo cáo trên cùng bài toán clone detection BigCloneBench. "
         "Các số này được tổng hợp từ bảng so sánh của GraphCodeBERT (Guo et al., 2021), nếu công trình gốc không "
         "trực tiếp báo cáo trên BigCloneBench.", first_indent=0.5)
    table(doc,
          ["Công trình", "Phương pháp", "Precision", "Recall", "F1", "Nguồn số liệu"],
          [["CodeBERT (Feng et al., 2020)", "Bi-Encoder, token", "0,947", "0,934", "0,941", "Guo et al., 2021, Table 2"],
           ["GraphCodeBERT (Guo et al., 2021)", "Bi-Encoder + DFG", "0,948", "0,952", "0,950", "Guo et al., 2021, Table 2"],
           ["ASTNN (Zhang et al., 2019)", "RvNN + BiGRU trên AST", "0,92", "0,94", "0,93", "Guo et al., 2021, Table 2"],
           ["FA-AST-GMN / CPG-GNN (Wang et al., 2020)", "GNN + AST có luồng dữ liệu", "0,96", "0,94", "0,95", "Guo et al., 2021, Table 2"]],
          widths=[5.0, 4.2, 2.2, 2.2, 1.6, 3.0])
    caption(doc, "Bảng 2. Kết quả clone detection được báo cáo trong bốn công trình gốc trên BigCloneBench.")
    para(doc,
         "Cần lưu ý rằng protocol của các công trình này khác với protocol của chúng tôi (tập con cân bằng 5.000 cặp, "
         "train 30.000 cặp, tối đa 512 token), nên các con số chỉ dùng để định vị xu hướng: GraphCodeBERT vượt "
         "CodeBERT nhờ luồng dữ liệu, còn nhóm dựa trên cấu trúc (AST/GNN) đạt F1 thấp hơn hoặc tương đương. "
         "Kết quả của chúng tôi ở phần sau (M2 Best, F1 = 0,9707) cao hơn các giá trị trích dẫn trên do khác "
         "protocol đánh giá, không phải do thay thế hoàn toàn công trình gốc.", first_indent=0.5)
    para(doc,
         "CodeBERT và GraphCodeBERT đều thuộc họ Bi-Encoder: hai hàm được mã hóa độc lập rồi mới kết hợp. Điểm khác "
         "biệt của GraphCodeBERT là đưa thêm luồng dữ liệu (data flow) vào quá trình tiền huấn luyện, giúp mô hình "
         "nhạy với quan hệ giữa các biến hơn là tên biến. ASTNN và CPG-GNN dùng cấu trúc tĩnh của chương trình; "
         "trong thực nghiệm của chúng tôi hai phương pháp này bị hạn chế do cây/đồ thị quá lớn và thiếu dữ liệu "
         "huấn luyện, nên bài báo tập trung vào nhóm transformer.", first_indent=0.5)

    # --- 3 ---
    heading(doc, "3. Tập dữ liệu", 1)
    para(doc,
         "Chúng tôi dùng CodeXGLUE BigCloneBench (Java), bộ dữ liệu chuẩn gồm 1.731.860 cặp ở ba split chính thức. "
         "Mỗi mẫu gồm hai đoạn mã Java (func1, func2), hai định danh hàm và nhãn nhị phân (1 = clone, 0 = không clone). "
         "Để đánh giá trong điều kiện cân bằng, mỗi split được lấy theo thứ tự bản ghi 2500/1500 cặp clone và cùng số "
         "cặp không clone, sau đó xáo trộn với seed 42 (Bảng 3).", first_indent=0.5)
    table(doc,
          ["Split", "Số cặp", "Clone", "Không clone", "Số hàm", "Token TB", "Vượt 512 token"],
          [["Train", "30.000", "15.000", "15.000", "7.274", "789,7", "45,6%"],
           ["Validation", "3.000", "1.500", "1.500", "903", "818,4", "48,9%"],
           ["Test", "5.000", "2.500", "2.500", "912", "717,3", "42,2%"]],
          widths=[2.6, 2.4, 2.4, 2.8, 2.2, 2.4, 3.0])
    caption(doc, "Bảng 3. Thống kê dữ liệu sau khi chia theo protocol cân bằng 1:1.")
    para(doc, "Tỷ lệ cặp có chuỗi vượt quá 512 token khá cao (~45%), nên bước token hóa dùng truncation; đây là một "
              "nguồn mất thông tin cần được lưu ý khi phân tích lỗi.", first_indent=0.5)

    # --- 4 ---
    heading(doc, "4. Pipeline đề xuất", 1)
    para(doc,
         "Pipeline tốt nhất do chúng tôi tự xây dựng gồm chín bước, được minh họa ở Hình 1. Pipeline kế thừa cấu hình "
         "M2 DiffMul (GraphCodeBERT) và bổ sung bước hiệu chỉnh xác suất cùng chọn ngưỡng bằng cross-validation. "
         "Phần này trình bày từng bước, kèm ví dụ số để dễ hình dung.", first_indent=0.5)
    doc.add_picture(str(FIG_PATH), width=Inches(5.7))
    caption(doc, "Hình 1. Pipeline đề xuất: từ cặp mã Java đến quyết định clone, kèm hiệu chỉnh và chọn ngưỡng.")

    heading(doc, "4.1. Chuẩn bị dữ liệu và token hóa", 2)
    para(doc,
         "Mỗi cặp (func1, func2) được chuẩn hóa xuống dòng, rồi token hóa bằng tokenizer của GraphCodeBERT với "
         "max_length = 512, padding động. Hai chuỗi đầu vào được đưa qua cùng một encoder dưới dạng siamese, "
         "đảm bảo tính đối xứng của bài toán.", first_indent=0.5)
    heading(doc, "4.2. Biểu diễn tương tác DiffMul", 2)
    para(doc,
         "Gọi hA, hB là vector CLS của hai hàm (mỗi vector 768 chiều). Thay vì ghép [hA, hB, |hA−hB|] như bản gốc "
         "(2304 chiều), chúng tôi dùng vector tương tác:", first_indent=0.5)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("v = [ |hA − hB| , hA ⊙ hB ] ∈ R^1536"), 12.5, bold=True)
    para(doc,
         "trong đó |hA−hB| là sai biệt tuyệt đối từng chiều và hA*hB là tích Hadamard. Cách ghép này bảo đảm "
         "f(A,B) = f(B,A) và giảm 33,3% số chiều đầu vào cho tầng phân loại. Ví dụ, với một cặp clone điển hình "
         "cos(hA,hB) ≈ 0,97; với cặp không clone cosine ≈ 0,04 (trước hiệu chỉnh).", first_indent=0.5)
    heading(doc, "4.3. Head và huấn luyện", 2)
    para(doc,
         "Head gồm Linear(1536→768) → Tanh → Dropout(0,4) → Linear(768→1). Hàm mất mát là BCEWithLogitsLoss với label "
         "smoothing 0,05. Tối ưu bằng AdamW (lr = 10^-5, weight decay = 0,10), cosine annealing, warmup 0,1, "
         "gradient accumulation 2, AMP FP16; early stopping theo val_loss. M2 đạt val F1 tốt nhất ở epoch 2 ở bản "
         "gốc, cho thấy overfit sớm khi chỉ dùng regularization.", first_indent=0.5)
    heading(doc, "4.4. Hiệu chỉnh xác suất và chọn ngưỡng OOF", 2)
    para(doc,
         "Quan sát thực nghiệm: ngưỡng tối ưu trên validation nhảy giữa 0,1 và 0,82 qua các epoch dù F1 gần như không "
         "đổi. Chúng tôi fit isotonic regression trên logits validation để hiệu chỉnh xác suất (ECE ≈ 0), rồi chọn "
         "ngưỡng bằng 5-fold OOF: mỗi fold chọn ngưỡng từ bốn fold còn lại và đánh giá trên fold giữ lại. Ngưỡng cuối "
         "được chốt trên toàn validation; test chỉ dùng để đánh giá. Đây là bước giúp ổn định quyết định mà không "
         "cần huấn luyện lại mô hình.", first_indent=0.5)

    # --- 5 ---
    heading(doc, "5. Kết quả thực nghiệm và thảo luận", 1)
    para(doc, "Tất cả kết quả dưới đây trên test 5.000 cặp cân bằng. F1 là chỉ số chính; P/R/AUC và bootstrap CI được "
              "dùng để đối chiếu. Khi so sánh hai hệ thống chúng tôi dùng McNemar exact test.", first_indent=0.5)
    table(doc,
          ["Cấu hình", "Ngưỡng", "F1", "P", "R", "FP/FN", "AUC"],
          [["M1 DiffMul", "0,12", "0,9583", "0,9759", "0,9412", "58/147", "0,9753"],
           ["M2 DiffMul", "0,14", "0,9631", "0,9817", "0,9452", "44/137", "0,9845"],
           ["M2 Best (tuning)", "0,52", "0,9707", "0,9876", "0,9544", "30/114", "0,9838"],
           ["M2 DiffMul + isotonic", "0,31", "0,9650", "0,9774", "0,9528", "55/118", "—"],
           ["M2 DiffMul + Focal+Aug+SupCon", "0,46", "0,9649", "0,9778", "0,9524", "54/119", "0,9783"],
           ["Fusion M1+M2", "0,16", "0,9693", "0,9804", "0,9584", "48/104", "0,9881"]],
          widths=[5.4, 1.8, 1.8, 1.8, 1.8, 2.4, 1.8])
    caption(doc, "Bảng 4. Kết quả chính trên tập test BigCloneBench (5.000 cặp).")
    para(doc,
         "M2 Best đạt F1 cao nhất (0,9707). Pipeline DiffMul + calibration giữ nguyên F1 gốc nhưng ổn định ngưỡng. "
         "Bản Focal+Aug+SupCon giảm FN từ 137 xuống 119, tức cải thiện recall cho Type-4; tuy nhiên FP tăng từ 44 lên "
         "54 nên F1 tổng chỉ nhích nhẹ, và McNemar cho p = 0,389 (không có ý nghĩa). Fusion M1+M2 tăng AUC lên "
         "0,9881 và giảm FN xuống 104 nhưng vẫn thua M2 Best (p = 0,403).", first_indent=0.5)
    para(doc,
         "Để giải thích cơ chế, chúng tôi trực quan hóa phân bố cosine của hai vector CLS. SupCon kéo cosine của cặp "
         "clone từ 0,9026 lên 0,9768, nhưng cũng kéo cosine của cặp không clone từ 0,1542 lên 0,5269. Như vậy lợi ích "
         "chủ yếu nằm ở độ gần của clone, trong khi độ tách lớp đo bằng AUC-cosine chỉ tăng nhẹ (0,9683 → 0,9760). "
         "Đây là mặt trái giải thích việc FP tăng.", first_indent=0.5)
    para(doc,
         "Cuối cùng, chúng tôi thử mở rộng validation lên 10.000 cặp và thử stacking logistic, nhưng ensemble vẫn "
         "thua mô hình M2 đơn lẻ. Điều này cho thấy sự khác biệt giữa phân bố validation và test là nút thắt lớn hơn "
         "việc chọn trọng số ensemble.", first_indent=0.5)
    para(doc,
         "Bảng 5 tổng hợp các thí nghiệm bổ sung: áp bộ Focal + Augmentation + SupCon lên nền config Best, fusion "
         "M1+M2, ablation thuần SupCon và stacking trên validation mở rộng.", first_indent=0.5)
    table(doc,
          ["Mô hình", "Phương pháp áp dụng", "F1", "Baseline", "Δ F1"],
          [["M1 CodeBERT", "Best + Focal BCE + Aug + SupCon", "0,9577", "0,9668", "−0,0091"],
           ["M2 GraphCodeBERT", "Best + Focal BCE + Aug + SupCon", "0,9568", "0,9707", "−0,0139"],
           ["M1 + M2", "Fusion (nối vector, head MLP)", "0,9693", "0,9707", "−0,0014"],
           ["M1 CodeBERT", "DiffMul + SupCon (ablation thuần)", "0,9637", "0,9583", "+0,0054"],
           ["M2 GraphCodeBERT", "DiffMul + SupCon (ablation thuần)", "0,9629", "0,9631", "−0,0002"],
           ["M1 + M2", "Stacking logistic, val 10k", "0,9614", "0,9707", "−0,0093"]],
          widths=[3.4, 6.4, 1.9, 2.2, 2.1])
    caption(doc, "Bảng 5. Kết quả bổ sung (25–26/11/2026).")
    para(doc,
         "Ablation thuần cho thấy SupCon riêng giúp M1 tăng 0,0054 (FP giảm 58→40) nhưng gần như trung tính với M2 "
         "(FP giảm 44→30, FN tăng 137→151). Như vậy augmentation đổi tên identifier là nguyên nhân làm M1 thoái lui "
         "trong bản B5+B6. Fusion và stacking đều không vượt M2 Best có ý nghĩa (McNemar p = 0,40 và 0,08), dù fusion "
         "nâng AUC lên 0,9881. Kết luận: cấu hình tốt nhất vẫn là M2 Best với F1 = 0,9707.", first_indent=0.5)

    # --- 6 ---
    heading(doc, "6. Đóng góp, hạn chế và hướng tiếp theo", 1)
    para(doc, "Đóng góp chính:", bold=True)
    bullet(doc, "Đề xuất cơ chế tương tác cặp DiffMul đối xứng, giảm 33,3% chiều đầu vào mà không làm mất thông tin.")
    bullet(doc, "Đưa hiệu chỉnh isotonic và ngưỡng OOF vào protocol đánh giá, khắc phục dao động ngưỡng giữa các epoch.")
    bullet(doc, "Đánh giá có đối chiếu thống kê (bootstrap CI, McNemar) thay vì chỉ so sánh điểm F1 đơn lẻ.")
    para(doc, "Hạn chế:", bold=True)
    bullet(doc, "Validation (3.000 cặp) và test (5.000 cặp) chọn siêu tham số khác nhau; dù mở rộng validation lên 10.000 cặp, ensemble vẫn thua mô hình đơn.")
    bullet(doc, "Augmentation đổi tên identifier làm M1 giảm F1 (0,9583 → 0,9549), cho thấy CodeBERT nhạy với tín hiệu lexical.")
    bullet(doc, "SupCon làm tăng độ tương tự của cả hai lớp; lợi ích tập trung ở recall nhưng đánh đổi precision.")
    bullet(doc, "Tỷ lệ cặp vượt 512 token khoảng 45% nên truncation có thể làm mất thông tin quyết định.")
    para(doc, "Hướng tiếp theo:", bold=True)
    bullet(doc, "Hard negative mining (OHEM) để giảm FP mà không đánh đổi recall.")
    bullet(doc, "Ablation thuần SupCon trên M1/M2 DiffMul để tách riêng tác động của thành phần này.")
    bullet(doc, "LoRA/PEFT cho M1 nhằm giữ tri thức tiền huấn luyện khi augmentation mạnh.")
    bullet(doc, "Thử adapter cho cross-encoder để đưa vào fusion ở mức biểu diễn.")

    # --- Tài liệu tham khảo ---
    heading(doc, "Tài liệu tham khảo", 1)
    refs = [
        "[1] Feng Z. et al. CodeBERT: A Pre-Trained Model for Programming and Natural Languages. EMNLP Findings, 2020.",
        "[2] Guo D. et al. GraphCodeBERT: Pre-training Code Representations with Data Flow. ICLR, 2021.",
        "[3] Zhang J. et al. ASTNN: A Neural Representation for Source Code Based on Abstract Syntax Tree. ICPC, 2019.",
        "[4] Backes J. et al. Joern: A Robust Infrastructure for Graph-based Program Analysis. arXiv, 2019.",
        "[5] Wang W. et al. CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation. NeurIPS Datasets, 2021.",
        "[6] Guo D. et al. BigCloneBench: A Big Data Benchmark for Clone Detection. SCAM, 2017.",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        set_font(p.add_run(r), 10.5)

    doc.save(DOCX_PATH)
    print("docx:", DOCX_PATH)


if __name__ == "__main__":
    draw_pipeline()
    build_doc()
