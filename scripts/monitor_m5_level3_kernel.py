"""Giám sát tiến độ Kernel M5 Level 3 trên Kaggle và tự động tải kết quả khi hoàn tất."""

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

# Đảm bảo flush tức thì mọi output ra log
sys.stdout.reconfigure(line_buffering=True)

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-hybrid-level3-deep-graph-transformer"
OUTPUT_DIR = Path("results/exp_m5_level3")
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
    
    # Chuẩn hóa trạng thái
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
    print("\n" + "=" * 75)
    print(f">>> Đang tải toàn bộ kết quả từ Kaggle: {KERNEL_SLUG} <<<")
    print("=" * 75)
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

            # Nếu là file test_metrics.json hoặc training_summary.json thì tạo bản copy chuẩn tên
            if "test_metrics" in fname and fname != "test_metrics.json":
                shutil.copy2(dest, OUTPUT_DIR / "test_metrics.json")
            if "training_summary" in fname and fname != "training_summary.json":
                shutil.copy2(dest, OUTPUT_DIR / "training_summary.json")
            if "best_checkpoint" in fname and fname != "best_checkpoint.pt":
                shutil.copy2(dest, OUTPUT_DIR / "best_checkpoint.pt")

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
            time.sleep(15)
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
            print(f"Đã lưu thành công {len(saved)} file vào {OUTPUT_DIR}/")

            # Đọc và hiển thị test metrics
            metrics_file = OUTPUT_DIR / "test_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                print("\n" + "=" * 75)
                print(">>> KẾT QUẢ ĐÁNH GIÁ M5 LEVEL 3 (DEEP GRAPH-TRANSFORMER) <<<")
                print("=" * 75)
                opt = metrics.get("metrics_optimal_threshold", {})
                print(f"  * F1-Score : {opt.get('f1')}%")
                print(f"  * Precision: {opt.get('precision')}%")
                print(f"  * Recall   : {opt.get('recall')}%")
                print(f"  * Accuracy : {opt.get('accuracy')}%")
                print(f"  * ROC-AUC  : {opt.get('roc_auc')}%")
                print(f"  * Ma trận nhầm lẫn: TP={opt.get('tp')}, FP={opt.get('fp')}, FN={opt.get('fn')}, TN={opt.get('tn')}")
                print(f"  * Ngưỡng tối ưu: {metrics.get('best_threshold')}")
                print(f"  * Thời gian train: {metrics.get('train_time_hours')} giờ")
                print("=" * 75)
            break

        if status in ("ERROR", "CANCELLED"):
            print(f"\n>>> KERNEL THẤT BẠI! Trạng thái: {status} <<<")
            if failure_msg:
                print(f"Thông báo lỗi: {failure_msg}")
            sys.exit(1)

        time.sleep(25)


if __name__ == "__main__":
    main()
