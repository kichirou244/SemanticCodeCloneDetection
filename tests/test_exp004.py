"""Tests EXP-004 / M3 ASTNN (CPU): abstraction, compile_tree, RvNN, runner (§36).

Khác test EXP-002: ASTNN chỉ 428k tham số, train từ scratch và **không cần mạng**,
nên test dùng model thật thay vì mock — mọi lỗi kiến trúc đều bị bắt.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.experiments.exp004_astnn import (
    build_stores, cache_root, evaluate_model, preflight, run_batches, train_model,
)
from src.models.astnn.model import ASTNNPairClassifier, to_device
from src.representations.ast.extract import (
    LEAF_ID, LEAF_NUM, LEAF_STR, N_LEAF, N_STMT, collate_trees, compile_tree,
    extract_tree, get_parser,
)
from src.representations.ast.store import ASTTreeStore

CODES = {
    "add": "int add(int x,int y){return x+y;}",
    "ifelse": "int f(int a,int b){int s=0;if(a>b){s=a;}else{s=b;}return s;}",
    "loop": "void g(){int q=1;while(q<9){q++;}}",
    "cls": "class K{public:int v;K(){v=0;}int get(){return v;}};",
    "globals": "int x;double y;",
    "empty": "",
    "nest": "int h(int n){int r=1;for(int k=2;k<=n;k++){r*=k;}return r;}",
}


def _compiled(code: str, p=None) -> dict:
    p = p or get_parser("cpp")
    return compile_tree(extract_tree(code, p, "cpp"))


def _batches(codes, device=None):
    device = device or torch.device("cpu")
    p = get_parser("cpp")
    t = [_compiled(c, p) for c in codes]
    return to_device(collate_trees(t), device)


# ------------------------------------------------------------------ abstraction
def test_leaf_abstraction_has_only_id_num_str():
    """Bài gốc chỉ giữ 3 lớp lá: ID / NUM / STR (đã loại tên hàm, tên biến, toán tử)."""
    t = extract_tree(CODES["ifelse"], get_parser("cpp"), "cpp")
    assert set(np.unique(t["label"][t["is_leaf"] == 1]).tolist()) <= {LEAF_ID, LEAF_NUM, LEAF_STR}
    assert 0 <= t["label"][t["is_leaf"] == 1].min()
    assert t["label"][t["is_leaf"] == 1].max() < N_LEAF


def test_extract_is_deterministic():
    p = get_parser("cpp")
    a = extract_tree(CODES["loop"], p, "cpp")
    b = extract_tree(CODES["loop"], p, "cpp")
    for k in ("label", "parent", "is_leaf"):
        assert np.array_equal(a[k], b[k]), k


def test_language_gate():
    """Thư viện giữ hỗ trợ Python (DEC-019 chỉ huỷ *thí nghiệm* Python, không
    xoá năng lực), nhưng ngôn ngữ lạ phải bị chặn và config EXP-004 phải ghim C++."""
    with pytest.raises(ValueError):
        get_parser("java")
    cfg = yaml.safe_load(Path("configs/exp004_astnn.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["language"] == "cpp"
    assert cfg["dataset"]["language"] == "cpp"
    assert cfg["model"]["type"] == "astnn"
    # ASTNN không được có max_length (nếu có nghĩa là ai đó cấu hình nhầm truncate)
    assert "tokenizer" not in cfg


def test_empty_and_minimal_code_still_yield_a_tree():
    """Code rỗng phải ra ≥1 statement, nếu không `collate` sẽ vỡ với max=0."""
    for code in (CODES["empty"], CODES["globals"]):
        t = _compiled(code)
        assert t["s_label"].shape[0] >= 1
        assert t["top"].shape[0] >= 1


# ----------------------------------------------------------------- compile_tree
def test_compile_tree_statement_space_invariants():
    """Mọi chỉ số phải nằm trong [0, S) và cây phải là DFS pre-order."""
    for code in CODES.values():
        t = _compiled(code)
        s = int(t["n_stmt"])
        assert s >= 1
        assert s == t["s_label"].shape[0] == t["s_parent"].shape[0] == t["s_depth"].shape[0]
        # DFS pre-order: cha luôn có chỉ số nhỏ hơn (trừ gốc tự tham chiếu)
        assert np.all(t["s_parent"][1:] < np.arange(1, s))
        # gốc tự tham chiếu làm sentinel; KHÔNG node nào khác tự tham chiếu
        assert t["s_parent"][0] == 0
        non_root = np.arange(1, s)
        assert np.all(t["s_parent"][non_root] != non_root)
        # depth = depth(cha) + 1, và tăng dần theo thứ tự DFS
        assert t["s_depth"][0] == 0
        assert np.all(t["s_depth"][1:] == t["s_depth"][t["s_parent"][1:]] + 1)
        # nhãn trong [0, N_STMT)
        assert t["s_label"].min() >= 0 and t["s_label"].max() < N_STMT
        # top là con của gốc, không chứa gốc
        # top là con của gốc. `top = [0]` CHỈ hợp lệ khi cây chỉ có 1 node
        # (code rỗng) — lúc đó fallback dùng chính gốc làm chuỗi cấp 1.
        if s > 1:
            assert 0 not in set(t["top"].tolist())
        assert np.all(t["s_parent"][t["top"]] == 0)
        # cặp (statement, lớp lá) hợp lệ (code rỗng thì không có cặp nào)
        if t["anc_s"].shape[0]:
            assert t["anc_s"].min() >= 0 and t["anc_s"].max() < s
            assert t["anc_c"].min() >= 0 and t["anc_c"].max() < N_LEAF
            assert t["anc_s"].shape == t["anc_c"].shape


def test_level_ptr_partitions_s_order():
    """`s_level_ptr` phải phủ đúng s_order và theo depth giảm dần."""
    for code in CODES.values():
        t = _compiled(code)
        ptr = t["s_level_ptr"]
        assert ptr[0] == 0 and ptr[-1] == t["s_label"].shape[0]
        assert np.all(np.diff(ptr) >= 0)
        for j in range(len(ptr) - 1):
            sl = t["s_order"][ptr[j]: ptr[j + 1]]
            if sl.size:
                assert t["s_depth"][sl].min() == t["s_depth"][sl].max()
        # thứ tự là depth giảm dần
        assert np.all(np.diff(t["s_depth"][t["s_order"]]) <= 0)


# ------------------------------------------------------------------ collate
def test_collate_shifts_indices_by_offset():
    """Ghép N cây: mọi chỉ số của cây thứ i phải cộng offset của cây đó.

    Nếu quên offset, `s_parent`/`anc_s` của code này sẽ trỏ sang statement của
    code khác — vẫn chạy, vẫn ra số, nhưng sai hoàn toàn. Đây là loại lỗi
    im lặng nguy hiểm nhất nên phải có test.
    """
    keys = ("add", "ifelse", "loop", "cls")
    p = get_parser("cpp")
    trees = [_compiled(CODES[k], p) for k in keys]
    b = collate_trees(trees)

    assert int(b["batch_size"]) == len(keys)
    stmt_off = anc_off = 0
    for t in trees:
        n = int(t["n_stmt"])
        m = len(t["anc_s"])
        assert np.array_equal(b["s_label"][stmt_off: stmt_off + n], t["s_label"])
        assert np.array_equal(b["s_depth"][stmt_off: stmt_off + n], t["s_depth"])
        # s_parent đã offset: cha nằm trong [stmt_off, stmt_off+n)
        par = b["s_parent"][stmt_off: stmt_off + n]
        assert par.min() >= stmt_off and par.max() < stmt_off + n
        # anc_s cũng phải nằm trong cửa sổ statement của chính cây đó
        if m:
            a = b["anc_s"][anc_off: anc_off + m]
            assert a.min() >= stmt_off and a.max() < stmt_off + n
            assert np.array_equal(a, t["anc_s"] + stmt_off)
        anc_off += m
        stmt_off += n
    assert stmt_off == b["s_label"].shape[0]
    assert anc_off == b["anc_s"].shape[0]


def test_collate_top_mask_marks_real_entries():
    p = get_parser("cpp")
    cs = [CODES["add"], CODES["cls"], CODES["nest"]]
    b = collate_trees([_compiled(c, p) for c in cs])
    m = b["top_mask"]
    assert m.shape[0] == 3
    # trong mỗi hàng, mask phải là 1 liên tục từ đầu
    for row in m:
        v = row.astype(bool)
        assert np.all(v == np.cumprod(v)), "top_mask phải đánh dấu tiền tố liên tục"
        assert v.any()


# ---------------------------------------------------------------------- RvNN
def _reference_rvnn(rv, b):
    """Cài đặt tham chiếu độc lập: đệ quy từng node, không vectorize.

    Mục đích: bảo đảm `VectorizedRvNN` đúng *toán học*, không chỉ nhanh. Nếu ai đó
    đổi công thức RvNN mà quên cập nhật bản vectorize, test này đỏ.
    """
    s = int(b["s_label"].shape[0])
    x = torch.zeros(s, rv.hidden, dtype=rv.W_s.weight.dtype)
    for i in range(int(b["anc_s"].shape[0])):
        v, c = int(b["anc_s"][i]), int(b["anc_c"][i])
        x[v] = x[v] + rv.W_x(rv.leaf_emb.weight[c])
    x = x + rv.W_s(rv.stmt_emb(b["s_label"]))

    children: list[list[int]] = [[] for _ in range(s)]
    for v in range(1, s):
        p = int(b["s_parent"][v])
        if p != v:
            children[p].append(v)

    out: list[torch.Tensor | None] = [None] * s

    def rec(v: int) -> torch.Tensor:
        acc = torch.zeros(rv.hidden, dtype=x.dtype)
        for ch in children[v]:
            acc = acc + rec(ch)
        out[v] = torch.tanh(x[v] + rv.W_h(acc))
        return out[v]

    # Batch gộp N code ⇒ N gốc, mỗi gốc tự tham chiếu (`s_parent[v] == v`).
    # Chỉ gọi rec(0) sẽ bỏ sót các code còn lại.
    for v in range(s):
        if out[v] is None:
            rec(v)
    assert all(t is not None for t in out)
    return torch.stack([t for t in out if t is not None])


def test_vectorized_rvnn_matches_reference_loop():
    """Invariant cốt lõi của M3: vectorize phải khớp bản đệ quy."""
    torch.manual_seed(0)
    rv = ASTNNPairClassifier(200).rvnn.eval()
    for key in ("ifelse", "nest", "cls", "loop"):
        b = _batches([CODES[key], CODES["add"]])
        fast = rv(b)
        ref = _reference_rvnn(rv, b)
        assert fast.shape == ref.shape == (int(b["s_label"].shape[0]), 200)
        assert torch.allclose(fast, ref, atol=1e-5), (
            key, float((fast - ref).abs().max()))


def test_encode_side_keeps_batch_dimension():
    """Regression: `nn.GRU` với PackedSequence trả 2-tuple, không phải
    `(output, (h_n, h_c))`. Unpack 3 tầng cắt mất chiều batch → encode ra (H,)
    thay vì (B, H). Test này đỏ nếu ai đó viết lại chỗ đó."""
    m = ASTNNPairClassifier(200).eval()
    b = _batches([CODES["nest"], CODES["add"], CODES["cls"], CODES["loop"]])
    enc = m.encode_side(b)
    assert enc.shape == (4, 200), enc.shape
    assert m.enc_dim == 200


def test_forward_backward_finite_and_shaped():
    torch.manual_seed(0)
    m = ASTNNPairClassifier(200)
    ba, bb = _batches(["ifelse", "nest", "cls"]), _batches(["loop", "add", "empty"])
    logits = m(ba, bb)
    assert logits.shape == (3,), logits.shape
    y = torch.tensor([1.0, 0.0, 1.0])
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss).item()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "không có tham số nào nhận gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_identical_pair_scores_high_identical_differs():
    """Smoke sanity: pair trùng nhau phải cho logit cao hơn pair khác nhau
    (mô hình chưa train vẫn phải có tín hiệu qua head, nếu không mô hình chết)."""
    torch.manual_seed(0)
    m = ASTNNPairClassifier(200).eval()
    with torch.no_grad():
        same = m(_batches([CODES["add"]]), _batches([CODES["add"]]))
        diff = m(_batches([CODES["add"]]), _batches([CODES["cls"]]))
    assert same.shape == (1,) and diff.shape == (1,)
    assert torch.isfinite(same).all() and torch.isfinite(diff).all()


def test_param_count_matches_published_size():
    """~428k tham số — bằng chứng đây là mô hình nhỏ, không phải transformer."""
    m = ASTNNPairClassifier(200)
    assert m.param_count() == 428_001, m.param_count()


# ------------------------------------------------------------------- store/cache
def test_store_roundtrip_is_lossless(tmp_path):
    """Cache .npz phải cho kết quả collation Y HÉT sau khi nạp lại."""
    store = ASTTreeStore.build(CODES, verbose=False)
    ids = ["add", "ifelse", "cls", "nest"]
    a = store.collate(ids)
    path = tmp_path / "t.npz"
    store.save(path)
    again = ASTTreeStore.build(CODES, cache_path=path, verbose=False)
    b = again.collate(ids)
    assert again.ids == store.ids
    for k in a:
        if k == "batch_size":
            assert int(a[k]) == int(b[k])
        else:
            assert a[k].dtype == b[k].dtype, k
            assert np.array_equal(a[k], b[k]), k


def test_cache_path_separates_smoke_from_full_run(tmp_path):
    """Regression: nếu smoke và train dùng chung cache path thì smoke (vài trăm
    code) sẽ ghi đè cache train (41k code) và lần train sau âm thầm chạy trên
    vài trăm code — không có cảnh báo, chỉ có kết quả sai."""
    cfg = {"preprocessing": {"cache_dir": str(tmp_path), "max_code_chars": 20000}}
    full = cache_root(cfg, "train")
    smoke = cache_root(cfg, "train", "smoke256")
    check = cache_root(cfg, "train", "diagcheck32")
    assert full != smoke != check
    assert len({full, smoke, check}) == 3
    # đổi max_code_chars cũng phải tách cache (đổi cap = đổi feature)
    cfg2 = {"preprocessing": {"cache_dir": str(tmp_path), "max_code_chars": 5000}}
    assert cache_root(cfg2, "train") != full


def test_store_deterministic_across_builds():
    s1 = ASTTreeStore.build(CODES, verbose=False)
    s2 = ASTTreeStore.build(CODES, verbose=False)
    assert np.array_equal(s1["ifelse"]["s_label"], s2["ifelse"]["s_label"])
    assert np.array_equal(s1["ifelse"]["anc_s"], s2["ifelse"]["anc_s"])


# ------------------------------------------------------------------- runner
def _tiny_dataset(tmp_path):
    proc = tmp_path / "samples.jsonl"
    rows = []
    for split, pre, pid in (("train", "r", "p1"), ("validation", "v", "p2"), ("test", "t", "p3")):
        for i, code in enumerate(CODES.values()):
            rows.append({"sample_id": f"{pre}{i}", "code": code, "problem_id": pid,
                         "split": split})
    with open(proc, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    for split, pre in (("train", "r"), ("validation", "v"), ("test", "t")):
        with open(pairs_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for pid, a, b, lab in ((f"{pre}0", f"{pre}0", f"{pre}0", 1),
                                   (f"{pre}1", f"{pre}0", f"{pre}3", 0),
                                   (f"{pre}2", f"{pre}1", f"{pre}4", 1),
                                   (f"{pre}3", f"{pre}2", f"{pre}5", 0)):
                f.write(json.dumps({"pair_id": pid, "code_a_id": a, "code_b_id": b,
                                    "problem_a": "px", "problem_b": "py",
                                    "label": lab, "split": split}) + "\n")
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps({"dataset": "tiny", "leakage_passed": True}), encoding="utf-8")
    return proc, pairs_dir, mp


def _cfg(tmp_path, proc, pairs_dir, mp, epochs=2, bs=2):
    return {
        "_config_path": str(tmp_path / "cfg.yaml"),
        "dataset": {"manifest": str(mp), "pairs_dir": str(pairs_dir), "processed": str(proc)},
        "model": {"id": "M3", "type": "astnn", "language": "cpp", "hidden_size": 32},
        "preprocessing": {"max_code_chars": 20000, "cache_dir": str(tmp_path / "cache")},
        "training": {"seed": 42, "epochs": epochs, "batch_size": bs, "learning_rate": 1e-3,
                     "weight_decay": 0.0, "warmup_ratio": 0.1,
                     "gradient_accumulation_steps": 1, "max_grad_norm": 5.0,
                     "dropout": 0.2, "mixed_precision": False},
        "inference": {"pair_batch_size": 2},
        "threshold": {"method": "validation_f1", "grid_step": 0.1},
        "error_analysis": {"top_k": 5},
        "early_stopping": {"enabled": False, "patience": 2},
    }


def test_train_model_writes_all_artifacts_every_epoch(tmp_path):
    """DEC-018: flush sau TỪNG epoch. `ast_stats.json` thay `truncation.json`
    vì ASTNN không truncate."""
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    cfg = _cfg(tmp_path, proc, pairs_dir, mp, epochs=1)
    (tmp_path / "cfg.yaml").write_text("x: 1", encoding="utf-8")
    out = tmp_path / "out"
    res = train_model(cfg, out, torch.device("cpu"))
    for name in ("training_history.csv", "run_metadata.json", "ast_stats.json",
                 "last_checkpoint.pt", "best_checkpoint.pt"):
        assert (out / name).exists(), name
    assert not list(out.glob("checkpoint_epoch*.pt"))
    assert res["epochs_completed"] == 1
    assert res["ast_stats"]["train"]["truncation"].startswith("none")
    rows = (out / "training_history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2 and "n_stmt_train" in rows[0] and "global_step" in rows[0]


def test_resume_continues_and_does_not_restart(tmp_path):
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    (tmp_path / "cfg.yaml").write_text("x: 1", encoding="utf-8")
    out = tmp_path / "out"
    cfg = _cfg(tmp_path, proc, pairs_dir, mp, epochs=2)
    first = train_model(cfg, out, torch.device("cpu"), epochs=1)
    assert first["epochs_completed"] == 1
    second = train_model(cfg, out, torch.device("cpu"), epochs=2, resume=True)
    assert [int(r["epoch"]) for r in second["history"]] == [0, 1]
    assert int(second["history"][-1]["global_step"]) > int(first["history"][-1]["global_step"])


def test_resume_without_checkpoint_fails_fast(tmp_path):
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    cfg = _cfg(tmp_path, proc, pairs_dir, mp, epochs=1)
    with pytest.raises(FileNotFoundError):
        train_model(cfg, tmp_path / "empty", torch.device("cpu"), resume=True)


def test_time_budget_stops_gracefully(tmp_path):
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    (tmp_path / "cfg.yaml").write_text("x: 1", encoding="utf-8")
    out = tmp_path / "out"
    cfg = _cfg(tmp_path, proc, pairs_dir, mp, epochs=5)
    res = train_model(cfg, out, torch.device("cpu"), time_budget_s=0.001)
    assert res["stopped_reason"] == "time_budget"
    assert res["epochs_completed"] < 5
    assert (out / "best_checkpoint.pt").exists()


def test_evaluate_model_pipeline(tmp_path):
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    m = ASTNNPairClassifier(32, 0.2)
    torch.save({"model_state": m.state_dict()}, out / "best_checkpoint.pt")
    cfg = _cfg(tmp_path, proc, pairs_dir, mp)
    summary = evaluate_model(cfg, out, torch.device("cpu"))
    for name in ("metrics.json", "metrics.csv", "threshold.json", "confusion_matrix.json",
                 "error_cases.jsonl", "model_stats.json", "EVAL_COMPLETE.json"):
        assert (out / name).exists(), name
    assert (out / "predictions" / "test_scores.jsonl").exists()
    assert set(summary["test"]) >= {"precision", "recall", "f1"}
    stats = json.loads((out / "model_stats.json").read_text(encoding="utf-8"))
    assert stats["truncation"] == "none" and stats["param_count"] == m.param_count()


def test_evaluate_missing_checkpoint_fails_fast(tmp_path):
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileNotFoundError):
        evaluate_model(_cfg(tmp_path, proc, pairs_dir, mp), out, torch.device("cpu"))


def test_run_batches_reports_stmt_count_and_zero_grad_at_end(tmp_path):
    """`n_stmt` thay cho `n_trunc`; cuối pass phải zero_grad để resume sạch."""
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    cfg = _cfg(tmp_path, proc, pairs_dir, mp)
    store = build_stores(cfg, proc, ["validation"],
                         {"validation": _refs(pairs_dir, "validation")},
                         tag="unittest")["validation"]
    m = ASTNNPairClassifier(32, 0.2)
    refs = _refs(pairs_dir, "validation")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss, s, y, n_stmt, opt_steps = run_batches(m, store, refs, 2, torch.device("cpu"), opt)
    assert np.isfinite(loss) and s.shape == y.shape == (len(refs),)
    assert set(np.unique(y).tolist()) == {0, 1}
    assert n_stmt > 0 and opt_steps >= 1
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in m.parameters())


def _refs(pairs_dir, split):
    from src.experiments.transformer_pair_runner import load_pair_refs

    return load_pair_refs(pairs_dir / f"{split}.jsonl")


def test_preflight_reports_no_truncation_and_cuda_or_fails(tmp_path):
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "train.jsonl").write_text('{"pair_id": "x"}\n', encoding="utf-8")
    cfg = {
        "model": {"id": "M3", "type": "astnn", "language": "cpp", "hidden_size": 200},
        "training": {"batch_size": 32, "gradient_accumulation_steps": 1,
                     "mixed_precision": False},
        "threshold": {"method": "validation_f1"},
        "dataset": {"pairs_dir": str(pairs_dir)},
    }
    if torch.cuda.is_available():
        info = preflight(cfg, torch.device("cuda"))
        assert info["max_length"] == "n/a (không truncate)"
        assert info["param_count"] == 428_001
    else:
        with pytest.raises(RuntimeError):
            preflight(cfg, torch.device("cuda"))


def test_atomic_write_leaves_no_partial_file(tmp_path):
    from src.experiments.transformer_pair_runner import _atomic_write_text

    p = tmp_path / "a.json"
    _atomic_write_text(p, json.dumps({"v": 1}))
    _atomic_write_text(p, json.dumps({"v": 2}))
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 2}
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------------ notebook
NB4 = Path("notebooks/exp004_astnn_kaggle.ipynb")


def _nb_cells(p: Path) -> list[dict]:
    return json.loads(p.read_text(encoding="utf-8"))["cells"]


def _nb_code_cells(p: Path) -> list[dict]:
    return [c for c in _nb_cells(p) if c["cell_type"] == "code"]


def _nb_code(p: Path) -> str:
    return "\n".join("".join(c["source"]) for c in _nb_code_cells(p))


def test_exp004_notebook_is_self_contained():
    """Notebook EXP-004 phải tự chạy được: clone, cài đủ dependency, build dataset.

    Rủi ro riêng của EXP-004: `tree-sitter-cpp` không có sẵn trên Kaggle, thiếu nó
    thì cell train chết ở bước build cache AST — chết *sau* khi đã dùng hết phần
    lớn thời gian? Không, chết ngay, nhưng người dùng phải debug giữa phiên.
    """
    assert NB4.exists(), "thiếu notebook Kaggle cho EXP-004"
    t = _nb_code(NB4)
    assert "tree-sitter-cpp" in t, "thiếu cài tree-sitter-cpp (parser C++ cho AST cache)"
    assert "scripts/build_dataset.py" in t
    assert "get_parser('cpp')" in t, "phải kiểm tra parser C++ nạp được trước khi train"
    assert "pip install -q" in t
    # KHÔNG được cài torch (Kaggle đã có bản CUDA; cài lại sẽ kéo bản CPU)
    assert not re.search(r"pip install[^\n]*\btorch\b", t), "cài torch sẽ phá tốc độ train"


def test_exp004_notebook_evaluates_on_kaggle_not_locally():
    """Evaluate chạy trên Kaggle (~1 phút) rồi mới đóng gói — không tải về chạy lại.

    Quyết định của người dùng (2026-09-28): chạy ở Kaggle rồi tải xuống thôi, khỏi
    tải checkpoint về máy. Thiếu cell evaluate ⇒ người dùng phải tải model về chạy.
    """
    cells = _nb_code_cells(NB4)
    i_eval = next((i for i, c in enumerate(cells) if "--mode evaluate" in "".join(c["source"])), None)
    i_pack = next((i for i, c in enumerate(cells) if "tar --exclude" in "".join(c["source"])), None)
    assert i_eval is not None, "thiếu cell chạy evaluate"
    assert i_pack is not None, "thiếu cell đóng gói"
    assert i_eval < i_pack, "phải evaluate xong rồi mới đóng gói"
    t = _nb_code(NB4)
    assert "KEEP_WEIGHTS = False" in t, "mặc định không tải checkpoint về"
    assert "export_eval_weights" in t
    assert "--exclude='*.pt'" not in t


def test_exp004_runner_accepts_lightweight_checkpoint(tmp_path):
    """M3 cũng phải evaluate được từ `best_weights_only.pt` (đường tải 0,5GB
    dùng chung cho mọi mô hình, không có ngoại lệ riêng cho ASTNN)."""
    from src.experiments.transformer_pair_runner import export_eval_weights, load_eval_weights

    d = tmp_path / "out"
    d.mkdir()
    m = ASTNNPairClassifier(200, 0.2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sum(p.sum() for p in m.parameters()).backward()
    opt.step()
    torch.save({"model_state": m.state_dict(), "optimizer_state": opt.state_dict(),
                "epoch": 2, "best_f1": 0.4,
                "rng": {"numpy": np.random.get_state()}}, d / "best_checkpoint.pt")
    info = export_eval_weights(d)
    assert info["to_mb"] < info["from_mb"]
    m2 = ASTNNPairClassifier(200, 0.2)
    m2.load_state_dict(load_eval_weights(d / "best_weights_only.pt", "cpu"))
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
