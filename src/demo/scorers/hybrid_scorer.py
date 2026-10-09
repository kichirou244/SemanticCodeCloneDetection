"""M5 — Hybrid GraphCodeBERT + CPG-GNN (EXP-006).

Mỗi phía cần CẢ hai biểu diễn: text qua GraphCodeBERT, graph qua CPG-GNN —
đúng `collate_pair` của `exp006_hybrid` (tokenize_side + store.collate), không
viết đường collate thứ hai. Vì cần Joern như M4, class kế thừa `CpgScorer` để
tái dùng `extract_graphs` dùng chung + được `service.py` batch all-pairs tự
động (nhánh `isinstance(s, (CpgScorer, MossScorer))`) — thêm M5 không sửa
service hay server, chỉ thêm khối YAML.

Riêng evidence: saliency (gradient × input) trên NHÁNH TEXT, nhánh graph giữ
cố định — cùng họ kỹ thuật với M1/M2 nên vệt vàng đọc giống nhau ("model dựa
vào token nào để cho điểm"), thay vì trộn hai tín hiệu khác đơn vị (saliency
token + attention node) thành một màu khó đọc.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.build_cpg_export import joern_missing_reason, joern_or_die
from src.demo.scorers.base import resolve_device
from src.demo.scorers.cpg_scorer import CpgScorer, extract_graphs
from src.demo.scorers.transformer_scorer import _build_encoder
from src.experiments.transformer_pair_runner import load_eval_weights
from src.models.hybrid.model import HybridPairClassifier
from src.models.transformer_pair import load_tokenizer, tokenize_side

#: Trần cache đồ thị theo hash code — quét zip 10 file chỉ Joern 1 lần, các
#: chunk sau trúng cache. Quá trần thì xóa hết làm lại (demo ngắn, đơn giản
#: hơn LRU mà đủ dùng).
_GRAPH_CACHE_LIMIT = 512


class HybridScorer(CpgScorer):
    """Bọc checkpoint hybrid EXP-006: chấm pair bằng text + CPG."""

    def __init__(
        self,
        model_id: str,
        label: str,
        results_dir: Path,
        checkpoint: Path,
        hf_id: str = "microsoft/graphcodebert-base",
        max_length: int = 256,
        hidden_size: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        max_nodes: int = 1500,
        device: str = "auto",
        jvm_xmx: str = "2g",
    ):
        super().__init__(
            model_id, label, results_dir, checkpoint,
            hidden_size=hidden_size, num_layers=num_layers, heads=heads,
            dropout=dropout, max_nodes=max_nodes, device=device,
            jvm_xmx=jvm_xmx,
        )
        self.hf_id = hf_id
        self.max_length = max_length
        self._tokenizer = None
        self._graph_cache: dict[str, tuple] = {}

    def checkpoint_hint(self) -> str:
        return (f"{self.hf_id} + gat{self.num_layers}x{self.heads} · "
                f"{self.checkpoint.name} · max_length={self.max_length}")

    def _load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"thiếu checkpoint {self.checkpoint} — chạy notebook train M5 "
                "(EXP-006) trước"
            )
        joern_cmd = joern_or_die()
        missing = joern_missing_reason(joern_cmd)
        if missing:
            raise RuntimeError(
                f"{missing}. M5 cần Java + Joern để dựng CPG lúc chấm điểm "
                "(xem notebooks/exp005_cpg_extract_kaggle.ipynb)"
            )
        import torch  # noqa: F401 — cần torch trước khi dựng model

        from src.models.cpg_gnn.model import CPGEncoder

        device = resolve_device(self.device)
        text_enc = _build_encoder(self.hf_id)
        graph_enc = CPGEncoder(self.hidden_size, self.num_layers,
                               self.heads, self.dropout)
        model = HybridPairClassifier(
            hf_id=self.hf_id, text_encoder=text_enc, graph_encoder=graph_enc)
        model.load_state_dict(load_eval_weights(self.checkpoint, device))
        model.to(device).eval()
        self._model = model
        self._tokenizer = load_tokenizer(self.hf_id)
        self._device = device
        self._joern_cmd = [joern_cmd, f"-J-Xmx{self.jvm_xmx}"]

    # ------------------------------------------------------------------ graph
    @staticmethod
    def _code_key(code: str) -> str:
        return hashlib.sha1(code.encode("utf-8")).hexdigest()

    def _graphs_for_codes(self, codes: list[str]) -> list[tuple]:
        """Đồ thị cho nhiều code, Joern 1 lần cho số code chưa gặp.

        Trả list cùng thứ tự `codes`, mỗi phần tử `(graph|None, raw|None,
        stats)`. Cache theo hash nội dung nên quét zip (file lặp lại qua các
        chunk) chỉ trả giá JVM đúng một lần.
        """
        keys = [self._code_key(c) for c in codes]
        missing_idx = [i for i, k in enumerate(keys)
                       if k not in self._graph_cache]
        if missing_idx:
            graphs = extract_graphs(
                [codes[i] for i in missing_idx],
                self._joern_cmd, self.max_nodes)
            for i, g in zip(missing_idx, graphs):
                if len(self._graph_cache) >= _GRAPH_CACHE_LIMIT:
                    self._graph_cache.clear()
                self._graph_cache[keys[i]] = g
        return [self._graph_cache[k] for k in keys]

    # ------------------------------------------------------------------- chấm
    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        out = self.score_all_pairs([("a", code_a), ("b", code_b)])
        score, detail = out[("a", "b")]
        if score is None:
            raise RuntimeError(detail.get("error") or "M5 không chấm được cặp")
        return score, detail

    def score_batch(
        self, items: list[tuple[str, str]]
    ) -> list[tuple[float | None, dict]]:
        """Forward gộp N cặp: tokenize 2N code một lần, Joern 1 lần cho code lạ.

        Service chia chunk rồi gọi hàm này nên tiến độ vẫn báo đều; cache đồ
        thị giữ số lần khởi động JVM ở mức tối thiểu trên toàn lần quét.
        """
        import torch

        from src.models.cpg_gnn.model import to_device
        from src.representations.cpg.graph import collate_pyg

        if not self.ensure_loaded():
            return [(None, {"error": self._error or "load failed"})
                    for _ in items]
        assert (self._model is not None and self._tokenizer is not None
                and self._device is not None)
        codes = [t for ab in items for t in ab]
        graphs = self._graphs_for_codes(codes)
        batch, _ = tokenize_side(self._tokenizer, codes, self.max_length)
        input_ids, attn = batch["input_ids"], batch["attention_mask"]
        out = []
        with torch.no_grad():
            for i in range(len(items)):
                ga, _, _ = graphs[2 * i]
                gb, _, _ = graphs[2 * i + 1]
                if ga is None or gb is None:
                    out.append((None, {"error": "Joern không ra JSON"}))
                    continue
                ta = {"input_ids": input_ids[2 * i: 2 * i + 1].to(self._device),
                      "attention_mask": attn[2 * i: 2 * i + 1].to(self._device)}
                tb = {"input_ids": input_ids[2 * i + 1: 2 * i + 2].to(self._device),
                      "attention_mask": attn[2 * i + 1: 2 * i + 2].to(self._device)}
                ba = to_device(collate_pyg([ga]), self._device)
                bb = to_device(collate_pyg([gb]), self._device)
                prob = torch.sigmoid(
                    self._model(ta, ba, tb, bb)).item()
                n_a = int(attn[2 * i].sum())
                n_b = int(attn[2 * i + 1].sum())
                out.append((float(prob), {
                    "n_tokens": max(n_a, n_b),
                    "truncated": bool(n_a >= self.max_length
                                      or n_b >= self.max_length),
                    "n_nodes_a": int(ga.n_nodes),
                    "n_nodes_b": int(gb.n_nodes),
                    "device": str(self._device),
                    "batched": True,
                }))
        return out

    def score_all_pairs(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], tuple[float | None, dict]]:
        """Cùng hợp đồng với `CpgScorer.score_all_pairs` để service batch M5
        như M4: MỘT lần Joern cho mọi file, rồi ghép cặp từ đồ thị đã có."""
        names = [name for name, _ in items]
        keys = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        by_name = dict(items)
        got = self.score_batch(
            [(by_name[a], by_name[b]) for a, b in keys])
        return {k: v for k, v in zip(keys, got)}

    # ---------------------------------------------------------------- evidence
    def evidence(self, code_a: str, code_b: str) -> dict | None:
        """Token nhánh text ảnh hưởng mạnh tới logit (gradient × input).

        Nhánh graph mã hóa trước (không grad) rồi giữ cố định — vệt vàng là
        "nhánh text của hybrid dựa vào đâu để quyết", đọc giống M1/M2. Giữ
        token trên trung bình, gom subword theo từ.
        """
        if not self.ensure_loaded():
            return None
        assert (self._model is not None and self._tokenizer is not None
                and self._device is not None)
        import torch

        from src.models.cpg_gnn.model import to_device
        from src.representations.cpg.graph import collate_pyg

        model = self._model
        was_training = model.training
        model.eval()
        frozen = [p for p in model.parameters() if p.requires_grad]
        try:
            for p in frozen:
                p.requires_grad_(False)
            enc = self._tokenizer(
                [code_a, code_b], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping").tolist()
            word_ids = [enc.word_ids(batch_index=i) for i in range(2)]
            inputs = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                graphs = self._graphs_for_codes([code_a, code_b])
                h_graphs = []
                for (g, _, _), code in zip(graphs, (code_a, code_b)):
                    if g is None:
                        raise RuntimeError("Joern không ra đồ thị cho evidence")
                    gb = to_device(collate_pyg([g]), self._device)
                    h_graphs.append(model.graph_encoder(gb))
            with torch.enable_grad():
                emb = model.text_encoder.embeddings(
                    input_ids=inputs["input_ids"])
                emb = emb.detach().requires_grad_(True)
                out = model.text_encoder(
                    inputs_embeds=emb,
                    attention_mask=inputs["attention_mask"])
                cls = out.last_hidden_state[:, 0, :]
                h = torch.cat([cls, torch.cat(h_graphs, dim=0)], dim=-1)
                ha, hb = h[0:1], h[1:2]
                pair = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
                logit = model.head(pair).squeeze(-1)
                logit.backward()
                grad = emb.grad
                assert grad is not None
                sal = (emb.detach() * grad.detach()).abs().sum(-1).cpu()
            out_spans = []
            for side in range(2):
                by_word: dict[int, list] = {}
                off, wid = offsets[side], word_ids[side]
                for i, (w, (s, e)) in enumerate(zip(wid, off)):
                    if w is None or e <= s:
                        continue
                    by_word.setdefault(w, []).append(
                        (float(sal[side, i]), s, e))
                scored = [(max(w for w, _, _ in v),
                           min(s for _, s, _ in v), max(e for _, _, e in v))
                          for v in by_word.values()]
                if not scored:
                    out_spans.append([])
                    continue
                mean = sum(w for w, _, _ in scored) / len(scored)
                out_spans.append([[s, e] for w, s, e in scored if w > mean])
        finally:
            for p in frozen:
                p.requires_grad_(True)
            model.train(was_training)
        # Tương ứng từng từ trên nhánh text, cùng cách làm M1/M2.
        # Nhánh graph chỉ vào điểm số, không vào cặp này.
        pairs = self._word_pairs(code_a, code_b)
        return {"spans_a": out_spans[0], "spans_b": out_spans[1],
                "n_shared": len(out_spans[0]) + len(out_spans[1]),
                "kind": "saliency",
                "pairs": pairs,
                "note": "token nhánh text (GraphCodeBERT) ảnh hưởng mạnh "
                        "tới logit, nhánh graph giữ cố định"}

    def _word_pairs(self, code_a: str, code_b: str) -> list[dict]:
        """Cặp từ tương ứng mutual-best trên encoder nhánh text."""
        import torch

        from src.demo.correspondence import mutual_best, word_vectors

        assert (self._model is not None and self._tokenizer is not None
                and self._device is not None)
        model = self._model
        was_training = model.training
        model.eval()
        try:
            enc = self._tokenizer(
                [code_a, code_b], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping").tolist()
            word_ids = [enc.word_ids(batch_index=i) for i in range(2)]
            inputs = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                h = model.text_encoder(**inputs).last_hidden_state.cpu()
        finally:
            model.train(was_training)
        va, sa = word_vectors(h[0], word_ids[0], offsets[0])
        vb, sb = word_vectors(h[1], word_ids[1], offsets[1])
        return [{"a": sa[i], "b": sb[j], "cos": round(c, 4)}
                for i, j, c in mutual_best(va, vb)]
