"""Tests EXP-003 / M2 GraphCodeBERT (CPU) — chủ yếu là **bất biến công bằng** (§36).

EXP-003 dùng *đúng* code path với EXP-002 (`transformer_pair_runner`). Nếu ai đó
tách runner hoặc đổi hyperparameter, RQ1 ("GraphCodeBERT có tốt hơn CodeBERT
không?") mất ý nghĩa — mọi khác biệt phải đến từ pretrained weights. Test ở
đây khóa lại điều đó bằng cấu hình, không bằng lời hứa.
"""

import pytest
import torch
import yaml
from pathlib import Path

from src.experiments import exp002_codebert, exp003_graphcodebert
from src.models.transformer_pair import (
    TransformerPairClassifier, collate_pair_batch, load_tokenizer,
)

CB_ID = "microsoft/codebert-base"
GCB_ID = "microsoft/graphcodebert-base"
CFG2 = Path("configs/exp002_codebert.yaml")
CFG3 = Path("configs/exp003_graphcodebert.yaml")


@pytest.fixture(scope="module")
def cfgs():
    return (yaml.safe_load(CFG2.read_text(encoding="utf-8")),
            yaml.safe_load(CFG3.read_text(encoding="utf-8")))


# ------------------------------------------------------- fairness (DEC-016)
def test_runner_is_shared_code_path():
    """EXP-002 và EXP-003 phải dùng CHUNG runner, có CÙNG public API.

    Bản sao là nguồn sai lệch âm thầm kinh điển: sửa một bên, quên bên kia, rồi
    kết luận sai rằng GraphCodeBERT tốt hơn vì code khác nhau. Ngược lại, nếu
    một wrapper export nhiều hơn bên kia thì notebook/test dùng `import exp002`
    sẽ hỏng khi đổi sang exp003 — cũng là một kiểu sai lệch âm thầm.
    """
    assert set(exp002_codebert.__all__) == set(exp003_graphcodebert.__all__)
    for fn in exp002_codebert.__all__:
        assert getattr(exp003_graphcodebert, fn) is getattr(exp002_codebert, fn), (
            f"{fn} phải là CÙNG đối tượng function, không phải bản sao")
    # và cả hai phải là re-export, không phải định nghĩa lại
    import src.experiments.transformer_pair_runner as shared

    assert exp002_codebert.train_model is shared.train_model
    assert exp003_graphcodebert.train_model is shared.train_model


def test_configs_differ_only_in_model_identity(cfgs):
    """Cấu hình M2 phải khớp M1 mọi trường dùng chung, chỉ khác `hf_id`/tên."""
    c2, c3 = cfgs
    for section in ("dataset", "tokenizer", "training", "early_stopping",
                    "inference", "threshold", "evaluation", "error_analysis"):
        assert c2[section] == c3[section], (
            f"Mục `{section}` lệch giữa EXP-002 và EXP-003 → phá vỡ RQ1")
    assert c2["model"]["hf_id"] == CB_ID
    assert c3["model"]["hf_id"] == GCB_ID
    # phần khác biệt duy nhất ngoài identity: tên thí nghiệm + thư mục output
    assert c2["model"]["id"] == "M1" and c3["model"]["id"] == "M2"
    assert c2["output"]["directory"] != c3["output"]["directory"]


def test_shared_hyperparameters_match_paper_protocol(cfgs):
    """Ghim protocol dùng chung: 3 epoch, max_length 256, warmup 10%, patience 2.

    Đây là các con số mà báo cáo sẽ trích; test để chúng không trôi.
    """
    _, c3 = cfgs
    tr = c3["training"]
    assert tr["epochs"] == 3 and tr["batch_size"] == 16
    assert tr["gradient_accumulation_steps"] == 2
    assert tr["warmup_ratio"] == 0.1
    assert c3["tokenizer"]["max_length"] == 256
    assert c3["tokenizer"]["truncation"] is True
    assert c3["early_stopping"]["enabled"] is True
    assert c3["early_stopping"]["patience"] == 2
    assert c3["threshold"]["method"] == "validation_f1"
    assert c3["model"]["pair_architecture"] == "concat_hA_hB_absdiff_mlp"
    assert c3["model"]["dropout"] == 0.1


def test_output_dirs_are_separate(cfgs):
    """Hai thí nghiệm không được ghi đè artifact của nhau."""
    c2, c3 = cfgs
    assert c2["output"]["directory"] != c3["output"]["directory"]
    assert "exp002" in c2["output"]["directory"]
    assert "exp003" in c3["output"]["directory"]


def test_dataset_is_identical(cfgs):
    """Cùng dataset, cùng pairs — nếu lệch thì so sánh không còn ý nghĩa."""
    c2, c3 = cfgs
    assert c2["dataset"] == c3["dataset"]
    assert c3["dataset"]["language"] == "cpp"
    assert c2["training"]["seed"] == c3["training"]["seed"]


# ----------------------------------------------------------------- model
def test_graphcodebert_loads_with_correct_dims():
    m = TransformerPairClassifier(GCB_ID)
    assert m.hidden_size == 768
    assert m.param_count() > 100_000_000


def test_same_architecture_same_param_count():
    """CodeBERT và GraphCodeBERT cùng kiến trúc ⇒ cùng số tham số.

    Nếu số tham số lệch, nghĩa là đã lỡ tay đổi head/encoder ở một bên, và mọi
    so sánh F1 sau đó là so sánh hai mô hình khác nhau chứ không phải hai
    pretrained weights.
    """
    a = TransformerPairClassifier(CB_ID).param_count()
    b = TransformerPairClassifier(GCB_ID).param_count()
    assert a == b, f"CodeBERT {a:,} != GraphCodeBERT {b:,}"
    assert a == 126_416_641, a


def test_graphcodebert_forward_backward_finite():
    torch.manual_seed(0)
    m = TransformerPairClassifier(GCB_ID)
    tok = load_tokenizer(GCB_ID)
    ba, bb, y, _ = collate_pair_batch(
        tok, ["int add(int a,int b){return a+b;}", "void f(){}"],
        ["int sum(int x,int y){return x+y;}", "void f(){}"], [1, 0], 32,
        torch.device("cpu"))
    logits = m(ba, bb)
    assert logits.shape == (2,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_tokenizer_matches_codebert_on_same_codes():
    """Hai mô hình dùng **cùng** vocab ⇒ tokenization phải giống hệt.

    Nếu khác, khác biệt F1 sẽ pha trộn khác biệt tokenizer với khác biệt weights.
    """
    cb, gcb = load_tokenizer(CB_ID), load_tokenizer(GCB_ID)
    codes = ["int add(int a,int b){return a+b;}", "int f(int n){int r=0;for(int i=0;i<n;i++)r+=i;return r;}"]
    a, _, _, _ = collate_pair_batch(cb, codes, codes, [1, 1], 256, torch.device("cpu"))
    b, _, _, _ = collate_pair_batch(gcb, codes, codes, [1, 1], 256, torch.device("cpu"))
    assert torch.equal(a["input_ids"], b["input_ids"])


def test_codebert_alias_still_works():
    """`src.models.codebert_pair` là alias của model generic — EXP-002 còn dùng."""
    from src.models.codebert_pair import CodeBERTPairClassifier

    assert CodeBERTPairClassifier is TransformerPairClassifier
