"""Rút `encoder.*` từ checkpoint M2/M4 về bản nhẹ và đẩy lên Hugging Face Hub.

Vì sao cần: repo `.gitignore` mọi `*.pt` nên checkpoint không thể push lên
GitHub. EXP-003 output trên Kaggle thường **không** chứa checkpoint (mặc định
`KEEP_WEIGHTS=False` → chỉ gói metrics + predictions). Cách tin cậy nhất để
EXP-006 warm start (DEC-029): export **bản encoder thuần** ở máy local, đẩy lên
một Hugging Face repo (private nếu cần), rồi notebook `exp006_hybrid_kaggle.ipynb`
(cell 7) pull thẳng về bằng `HF_TOKEN` đã khai báo ở cell 1.

File sinh ra (mặc định `warm_start/`):
    m2_text_encoder.pt   ~0.5 GB  (GraphCodeBERT encoder, khóa `encoder.*`)
    m4_graph_encoder.pt  vài MB   (CPG-GNN encoder, khóa `encoder.*`)

Cách dùng (push thường dùng nhất, cần `HF_TOKEN` trong env):
    python scripts/export_hybrid_warm_start.py \
        --m2 results/exp003_graphcodebert_poj104/best_checkpoint.pt \
        --m4 results/exp005_cpg_gnn_poj104/best_checkpoint.pt \
        --out warm_start \
        --hf-repo Kichirou24/SemanticCodeCloneDetection

Bước chuẩn bị 1 lần trên HF: tạo repo model (không cần file gì) với tên khớp
`--hf-repo`; muốn private thì chọn Private (notebook vẫn pull được nhờ HF_TOKEN).
Chỉ giữ khóa bắt đầu bằng `encoder.` — khớp với `warm_start_text` /
`warm_start_graph` trong `src/models/hybrid/model.py`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def extract_encoder_state(ckpt: dict) -> dict:
    """Lọc `model_state` còn lại khóa `encoder.*` của checkpoint M2/M4."""
    model_state = ckpt.get("model_state") or ckpt
    return {k: v for k, v in model_state.items()
            if k.startswith("encoder.") and hasattr(v, "numel")}


def export(src: str, dst: str) -> dict:
    src_p, dst_p = Path(src), Path(dst)
    if not src_p.exists():
        raise FileNotFoundError(f"Thiếu checkpoint: {src_p} — hãy chỉ đúng path local")
    ckpt = torch.load(src_p, map_location="cpu", weights_only=False)
    enc = extract_encoder_state(ckpt)
    if not enc:
        raise ValueError(f"{src_p} không có khóa `encoder.*` — sai file?")
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_suffix(".pt.tmp")
    torch.save({"model_state": enc}, tmp)
    tmp.replace(dst_p)
    stats = {
        "src": str(src_p),
        "dst": str(dst_p),
        "params": sum(int(t.numel()) for t in enc.values()),
        "mb": round(sum(t.numel() for t in enc.values()) * 4 / 2**20, 2),
        "keys": len(enc),
    }
    print(json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m2", required=True,
                    help="checkpoint M2 (results/exp003_graphcodebert_poj104/best_checkpoint.pt)")
    ap.add_argument("--m4", required=True,
                    help="checkpoint M4 (results/exp005_cpg_gnn_poj104/best_checkpoint.pt)")
    ap.add_argument("--out", default="warm_start", help="thư mục đầu ra (default: warm_start)")
    ap.add_argument("--hf-repo", default="",
                    help="Hugging Face repo id (vd Kichirou24/SemanticCodeCloneDetection) "
                         "để đẩy 2 file encoder lên sau khi export; bỏ trống thì chỉ export local")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                    help="Hugging Face token (mặc định dùng env HF_TOKEN)")
    args = ap.parse_args()

    export(args.m2, str(Path(args.out) / "m2_text_encoder.pt"))
    export(args.m4, str(Path(args.out) / "m4_graph_encoder.pt"))

    if args.hf_repo:
        if not args.token:
            raise SystemExit("Cần HF_TOKEN khi dùng --hf-repo (đặt env HF_TOKEN hoặc dùng --token)")
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise SystemExit("Thiếu huggingface_hub — chạy: pip install huggingface_hub") from e
        api = HfApi(token=args.token)
        for fname in ("m2_text_encoder.pt", "m4_graph_encoder.pt"):
            src = Path(args.out) / fname
            url = api.upload_file(path_or_fileobj=str(src),
                                  path_in_repo=fname,
                                  repo_id=args.hf_repo,
                                  repo_type="model")
            print(f"HF uplink {fname} -> {url}")
        print(f"Xong. Trong notebook set cell 7: HF_WARM_REPO = '{args.hf_repo}' "
              "(nếu thuộc user/repo khác mặc định) rồi chạy lại cell 7.")
    else:
        print("Xong. Upload 2 file trong "
              f"{Path(args.out).resolve()} lên một Hugging Face repo (hoặc chạy lại "
              "với --hf-repo <user>/<repo-name>) — notebook exp006_hybrid_kaggle.ipynb "
              "cell 7 sẽ pull theo tên file.")


if __name__ == "__main__":
    main()