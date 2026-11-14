"""Giám sát tiến độ Kernel M5-Pro Phase 3 (Cross-Attention & Contrastive trên 30.000 cặp) trên Kaggle."""

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

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-m5-pro-phase-3-contrastive-30k"
OUTPUT_DIR = Path("results/exp_m5_pro_phase3")
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

            if "test_metrics" in fname:
                shutil.copy2(dest, OUTPUT_DIR / "m5_pro_phase3_test_metrics.json")

    return saved_files


def main():
    print("=" * 80)
    print(f">>> BẮT ĐẦU THEO DÕI KERNEL M5-PRO PHASE 3 TRÊN KAGGLE: {KERNEL_SLUG} <<<")
    print("=" * 80)

    api = KaggleApi()
    api.authenticate()

    start_time = time.time()
    last_status = None
    poll_count = 0

    while True:
        poll_count += 1
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)

        status, failure_msg = check_status(api)

        if status != last_status:
            print(f"[{mins:02d}:{secs:02d}] Trạng thái chuyển đổi: {status}")
            last_status = status
        else:
            print(f"[{mins:02d}:{secs:02d}] Kiểm tra lần {poll_count}: {status}")

        if status == "COMPLETE":
            print(f"\n>>> KERNEL ĐÃ HOÀN TẤT THÀNH CÔNG SAU {mins} PHÚT {secs} GIÂY! <<<")
            saved = download_outputs(api)
            print(f"Đã lưu trữ {len(saved)} tệp vào {OUTPUT_DIR}")
            break
        elif status in ["ERROR", "CANCELLED"]:
            print(f"\n[LỖI] Kernel kết thúc với trạng thái: {status}")
            if failure_msg:
                print(f"Nguyên nhân: {failure_msg}")
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
