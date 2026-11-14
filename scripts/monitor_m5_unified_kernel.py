"""Giám sát tiến độ Kernel M5 Unified (Cấp độ 1, 2, 3 trên 30.000 cặp) trên Kaggle."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
import urllib3
import requests

urllib3.disable_warnings()

from kagglesdk.kaggle_http_client import KaggleHttpClient

orig_init = KaggleHttpClient._init_session
def patched_init(self):
    orig_init(self)
    if self._session is not None:
        self._session.verify = False
KaggleHttpClient._init_session = patched_init

orig_requests_get = requests.get
def patched_requests_get(*args, **kwargs):
    kwargs["verify"] = False
    return orig_requests_get(*args, **kwargs)
requests.get = patched_requests_get

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

sys.stdout.reconfigure(line_buffering=True)

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-m5-unified-all-levels-30k"
OUTPUT_DIR = Path("results/exp_m5_unified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_status(api: KaggleApi) -> tuple[str, str]:
    res = api.kernels_status(KERNEL_SLUG)
    if hasattr(res, "to_dict"):
        d = res.to_dict()
        status = str(d.get("status", "")).upper()
        failure_msg = str(d.get("failure_message", d.get("failureMessage", "")))
    else:
        status = str(getattr(res, "status", "UNKNOWN")).upper()
        failure_msg = str(getattr(res, "failure_message", getattr(res, "failureMessage", "")))

    if "COMPLETE" in status:
        norm_status = "COMPLETE"
    elif "RUNNING" in status:
        norm_status = "RUNNING"
    elif "QUEUED" in status:
        norm_status = "QUEUED"
    elif "ERROR" in status or "FAIL" in status:
        norm_status = "ERROR"
    elif "CANCEL" in status:
        norm_status = "CANCELLED"
    else:
        norm_status = status
    return norm_status, failure_msg


def download_outputs(api: KaggleApi) -> list[str]:
    print("\n" + "=" * 80)
    print(f">>> Đang tải toàn bộ kết quả từ Kaggle: {KERNEL_SLUG} <<<")
    print("=" * 80)
    owner, slug = KERNEL_SLUG.split("/")
    saved_files = []

    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        print(f"Tìm thấy {len(resp.files)} file đầu ra trên Kaggle:")

        for f in resp.files:
            fname = f.file_name
            url = f.url
            if not url:
                continue
            dest = OUTPUT_DIR / fname
            print(f"  -> Đang tải {fname}...", end="", flush=True)
            r = requests.get(url, verify=False, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as out_f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_f.write(chunk)
            print(f" Xong ({dest.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
            saved_files.append(fname)

            if "m5_unified_all_levels" in fname:
                shutil.copy2(dest, OUTPUT_DIR / "m5_unified_all_levels_test_metrics.json")

    return saved_files


def main():
    api = KaggleApi()
    api.authenticate()
    print(f"Bắt đầu theo dõi Kernel: {KERNEL_SLUG}...")

    t0 = time.time()
    last_status = ""

    while True:
        try:
            status, failure_msg = check_status(api)
        except Exception as e:
            print(f"Lỗi kiểm tra trạng thái: {e}")
            time.sleep(20)
            continue

        elapsed = time.time() - t0
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Trạng thái thay đổi: {status} (sau {elapsed/60:.1f} phút)")
            last_status = status
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Đang chạy... ({status}) - {elapsed/60:.1f} phút trôi qua", flush=True)

        if status == "COMPLETE":
            print(f"\n>>> KERNEL ĐÃ HOÀN TẤT THÀNH CÔNG! (Tổng thời gian: {elapsed/60:.1f} phút) <<<")
            saved = download_outputs(api)
            print(f"Đã lưu {len(saved)} files vào {OUTPUT_DIR}/")

            # Đọc và hiển thị bảng đối chứng
            metrics_file = OUTPUT_DIR / "m5_unified_all_levels_test_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                print("\n" + "=" * 80)
                print(">>> BẢNG ĐỐI CHỨNG TỔNG HỢP TOÀN BỘ CÁC CẤP ĐỘ M5 (30.000 CẶP) <<<")
                print("=" * 80)
                print(f"{'Mô hình / Cấp độ':<38} | {'F1-Score':<8} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'FP':<4} | {'FN':<4}")
                print("-" * 85)
                def pr(name, m):
                    print(f"{name:<38} | {m['f1']:>6.2f}% | {m['accuracy']:>6.2f}% | {m['precision']:>7.2f}% | {m['recall']:>6.2f}% | {m['fp']:>4} | {m['fn']:>4}")

                if "m4_ep_baseline" in summary: pr("Đơn lẻ M4-EP (CPG-GNN)", summary["m4_ep_baseline"])
                if "m2_dr_baseline" in summary: pr("Đơn lẻ M2-DR (GraphCodeBERT)", summary["m2_dr_baseline"])
                if "level1" in summary and "linear_blend" in summary["level1"]:
                    pr("Cấp độ 1: Linear Blend (M2+M4)", summary["level1"]["linear_blend"])
                if "level1" in summary and "confidence_gating" in summary["level1"]:
                    pr("Cấp độ 1: Confidence Gating", summary["level1"]["confidence_gating"])
                if "level2_30k" in summary and "m5_concat_30k" in summary["level2_30k"]:
                    pr("Cấp độ 2: M5-Concat (30.000 cặp)", summary["level2_30k"]["m5_concat_30k"])
                if "level2_30k" in summary and "m5_gated_30k" in summary["level2_30k"]:
                    pr("Cấp độ 2: M5-Gated (30.000 cặp)", summary["level2_30k"]["m5_gated_30k"])
                if "level3_30k" in summary:
                    pr("Cấp độ 3: Deep Graph-Trans (30k)", summary["level3_30k"])
                print("=" * 85)
            break

        if status in ("ERROR", "CANCELLED"):
            print(f"\n>>> KERNEL THẤT BẠI! Trạng thái: {status} <<<")
            if failure_msg:
                print(f"Thông báo lỗi: {failure_msg}")
            sys.exit(1)

        time.sleep(30)


if __name__ == "__main__":
    main()
