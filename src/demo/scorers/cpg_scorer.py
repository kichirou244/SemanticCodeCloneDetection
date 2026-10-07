"""M4 — CPG-GNN. Khác 4 scorer còn lại ở đúng một điểm: input phải qua Joern.

Vì Joern cần JVM (khởi động + parse vài chục giây cho 2 hàm), `_infer` dựng
batch 2 file tạm rồi gọi đúng `run_batch` của `scripts/build_cpg_export` —
không viết lại đường export thứ hai, nên CPG demo thấy là CPG train đã thấy.
Đồ thị đọc bằng `ingest.parse_cpg_file` (đã có test), collate hai phía riêng
bằng `collate_pyg` (gộp chung một batch sẽ chấm nhầm cặp — xem chú thích tương
tự trong `astnn_scorer.py`).

Máy không có Java/Joern (đúng trường hợp máy dự án, DEC-023) thì `_load` báo
hỏng với lý do rõ ràng — trang demo vẫn chạy với 4 model còn lại, M4 hiện
trạng thái lỗi chứ không kéo sập request.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.build_cpg_export import (
    joern_missing_reason,
    joern_or_die,
    run_batch,
    write_batch,
)
from src.demo.scorers.base import BaseScorer, resolve_device
from src.representations.cpg.ingest import parse_cpg_file


class CpgScorer(BaseScorer):
    def __init__(
        self,
        model_id: str,
        label: str,
        results_dir: Path,
        checkpoint: Path,
        hidden_size: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        max_nodes: int = 1500,
        device: str = "auto",
        jvm_xmx: str = "2g",
    ):
        """`jvm_xmx` giới hạn heap JVM khi dựng CPG demo (mặc định 2g là thừa
        cho vài file nhỏ; Joern mặc định đòi ~8g và OOM trên máy RAM chật)."""
        super().__init__(model_id, label, results_dir, device)
        self.checkpoint = checkpoint
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.max_nodes = max_nodes
        self.jvm_xmx = jvm_xmx
        self._model = None
        self._device = None
        self._joern_cmd: str | list[str] = ""

    def checkpoint_hint(self) -> str:
        return f"{self.checkpoint.name} · gat{self.num_layers}x{self.heads}"

    def _load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"thiếu checkpoint {self.checkpoint} — chạy notebook train M4 "
                "(bước 2) trước"
            )
        # Kiểm tra môi trường TRƯỚC import nặng: mỗi máy thiếu thứ khác nhau
        # (máy này thiếu Java, máy khác có thể thiếu torch_geometric) và thông
        # báo phải chỉ đúng thứ thiếu — fail fast thay vì lỗi import khó đọc.
        joern_cmd = joern_or_die()
        missing = joern_missing_reason(joern_cmd)
        if missing:
            raise RuntimeError(
                f"{missing}. M4 cần Java + Joern để dựng CPG lúc chấm điểm "
                "(xem notebooks/exp005_cpg_extract_kaggle.ipynb)"
            )
        # Import nặng để trong `_load` (lazy): máy không có torch_geometric
        # vẫn import được registry/server, M4 chỉ báo trạng thái lỗi.
        import torch

        from src.models.cpg_gnn.model import CPGGNNPairClassifier

        device = resolve_device(self.device)
        model = CPGGNNPairClassifier(self.hidden_size, self.num_layers,
                                     self.heads, self.dropout)
        state = torch.load(self.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        self._model = model
        self._device = device
        # JVM flag đi KÈM binary (dạng list): `-J*` được joern.bat forward
        # thẳng tới JVM. Không nhét vào chuỗi lệnh (shlex sẽ cắt sai).
        self._joern_cmd = [joern_cmd, f"-J-Xmx{self.jvm_xmx}"]

    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        out = self.score_all_pairs([("a", code_a), ("b", code_b)])
        score, detail = out[("a", "b")]
        if score is None:
            raise RuntimeError(detail.get("error") or "Joern không ra đồ thị")
        return score, detail

    def _extract_graphs(
        self, items: list[tuple[str, str]]
    ) -> dict[str, tuple[object | None, list | None, dict]]:
        """Chung cho score và evidence: Joern 1 lần → mỗi tên một
        `(graph|None, raw_nodes|None, stats)`. Raw nodes giữ thứ tự JSON gốc
        để map ngược node graph → dòng code (qua `stats["order"]`)."""
        samples = [
            {"code": code, "sample_id": f"scan_{i}"}
            for i, (_, code) in enumerate(items)
        ]
        out: dict[str, tuple[object | None, list | None, dict]] = {}
        with tempfile.TemporaryDirectory(prefix="cpg_scan_") as tmp:
            work = Path(tmp)
            batch_dir = work / "batch"
            out_dir = work / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            write_batch(samples, batch_dir)
            timeout_s = max(300, 15 * len(items))
            run_batch(samples, out_dir, work, self._joern_cmd,
                      batch_dir, timeout_s)
            import json as _json

            for (name, _), s in zip(items, samples):
                path = out_dir / f"{s['sample_id']}.json"
                if not path.exists():
                    out[name] = (None, None, {})
                    continue
                g, st = parse_cpg_file(path, self.max_nodes)
                raw = _json.loads(path.read_text(encoding="utf-8"))["nodes"]
                out[name] = (g, raw, st)
        return out

    def score_all_pairs(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], tuple[float | None, dict]]:
        """Chấm all-pairs với MỘT lần chạy Joern: dựng CPG mỗi file một lần rồi
        ghép cặp từ đồ thị đã có. Gọi `_infer` từng cặp ở đây sẽ là N² lần khởi
        động JVM (mỗi lần hàng chục giây)."""
        import torch

        from src.models.cpg_gnn.model import to_device
        from src.representations.cpg.graph import collate_pyg

        names = [name for name, _ in items]
        keys = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        if not self.ensure_loaded():
            err = self._error or "load failed"
            return {k: (None, {"error": err}) for k in keys}
        assert self._model is not None and self._device is not None
        extracted = self._extract_graphs(items)
        results: dict[tuple[str, str], tuple[float | None, dict]] = {}
        with torch.no_grad():
            for a, b in keys:
                ga, _, _ = extracted[a]
                gb, _, _ = extracted[b]
                if ga is None or gb is None:
                    results[(a, b)] = (None, {"error": "Joern không ra JSON"})
                    continue
                ba = to_device(collate_pyg([ga]), self._device)
                bb = to_device(collate_pyg([gb]), self._device)
                prob = torch.sigmoid(self._model(ba, bb)).item()
                results[(a, b)] = (float(prob), {
                    "n_nodes_a": int(ga.n_nodes),
                    "n_nodes_b": int(gb.n_nodes),
                    "device": str(self._device),
                })
        return results

    def evidence(self, code_a: str, code_b: str) -> dict | None:
        """Dòng code được GAT chú ý nhất (attention nhận được, tầng cuối).

        Chạy encoder lại từng phía, lấy attention trung bình các heads ở tầng
        GAT cuối, cộng dồn theo node đích. Node nào vượt trung bình thì giữ,
        map về dòng code qua `stats["order"]` + field `line` của exporter
        (JSON không có `line` — vd dữ liệu cũ — thì trả spans rỗng trung thực
        thay vì đoán). Đây là "GAT nhìn đâu", KHÔNG phải "đoạn nào khớp".
        """
        if not self.ensure_loaded():
            return None
        import torch

        from src.models.cpg_gnn.model import to_device
        from src.representations.cpg.graph import collate_pyg

        assert self._model is not None and self._device is not None
        extracted = self._extract_graphs([("a", code_a), ("b", code_b)])
        enc = self._model.encoder
        out_spans: dict[str, list] = {}
        with torch.no_grad():
            for side, code in (("a", code_a), ("b", code_b)):
                g, raw, st = extracted[side]
                if g is None or raw is None:
                    out_spans[side] = []
                    continue
                b = to_device(collate_pyg([g]), self._device)
                x = enc.node_emb(b.node_type) + enc.feat_lin(b.x)
                x = enc.in_norm(x)
                alpha_last = None
                for layer in enc.layers:
                    e = layer.edge_emb(b.edge_type)
                    h, (_ei, alpha) = layer.conv(
                        x, b.edge_index, e, return_attention_weights=True)
                    alpha_last = (alpha, _ei)
                    x = x + layer.dropout(layer.norm(h))
                n = int(b.node_type.shape[0])
                recv = torch.zeros(n, device=self._device)
                if alpha_last is not None and n > 0:
                    a_mat, ei = alpha_last
                    recv.index_add_(0, ei[1], a_mat.mean(dim=1))
                mean = float(recv.mean().item()) if n else 0.0
                order = st.get("order") or list(range(len(raw)))
                lines = code.split("\n")
                starts = _line_starts(lines)
                spans = []
                for j in range(n):
                    if float(recv[j].item()) <= mean:
                        continue
                    ln = (raw[order[j]].get("line")
                          if order[j] < len(raw) else None)
                    if not isinstance(ln, int) or ln < 1 or ln > len(lines):
                        continue
                    s = starts[ln - 1]
                    spans.append([s, s + len(lines[ln - 1])])
                out_spans[side] = spans
        total = 0
        final: dict[str, list] = {}
        for side in ("a", "b"):
            # Gộp span trùng (nhiều node salient cùng dòng → highlight 1 lần):
            # dict.fromkeys giữ thứ tự, khử trùng chính xác.
            uniq = list(dict.fromkeys(map(tuple, out_spans[side])))
            final[side] = [list(s) for s in uniq]
            total += len(final[side])
        return {"spans_a": final["a"], "spans_b": final["b"],
                "n_shared": total,
                "kind": "gat",
                "note": "dòng được GAT chú ý trên trung bình (tầng cuối)"}


def _line_starts(lines: list[str]) -> list[int]:
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    return starts
