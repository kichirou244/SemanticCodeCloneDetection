"""Theo dõi tiến trình Kernel SOTA Pool DFG trên Kaggle và tự động tải kết quả về."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import sys
import tarfile
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monitor_sota_pool_dfg")

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-sota-pool-dfg-30k"
OUT_DIR = Path("results/exp_sota_pool_dfg")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print("=" * 80)
    print(f">>> BẮT ĐẦU THEO DÕI KERNEL SOTA POOL DFG TRÊN KAGGLE: {KERNEL_SLUG} <<<")
    print("=" * 80)

    last_status = None
    start_time = time.time()
    poll_count = 0

    while True:
        poll_count += 1
        elapsed_sec = int(time.time() - start_time)
        mm, ss = divmod(elapsed_sec, 60)
        time_str = f"[{mm:02d}:{ss:02d}]"

        try:
            status_obj = api.kernels_status(KERNEL_SLUG)
            if hasattr(status_obj, "status"):
                status = status_obj.status.name if hasattr(status_obj.status, "name") else str(status_obj.status)
            elif isinstance(status_obj, dict):
                status = str(status_obj.get("status", "UNKNOWN"))
            else:
                status = str(status_obj)

            if hasattr(status_obj, "failure_message"):
                failure_msg = status_obj.failure_message
            elif isinstance(status_obj, dict):
                failure_msg = status_obj.get("failureMessage", None)
            else:
                failure_msg = None
        except Exception as e:
            status = "API_ERROR"
            failure_msg = str(e)

        if status != last_status:
            print(f"{time_str} Trạng thái chuyển đổi: {status}")
            last_status = status
        else:
            if poll_count % 2 == 0:
                print(f"{time_str} Kiểm tra lần {poll_count}: {status}")

        if status in ("COMPLETE", "COMPLETED"):
            print("\n" + "=" * 80)
            print(">>> KERNEL ĐÃ HOÀN TẤT THÀNH CÔNG! ĐANG TẢI KẾT QUẢ VỀ... <<<")
            print("=" * 80)

            api.kernels_output(KERNEL_SLUG, path=str(OUT_DIR))
            print(f"Đã tải toàn bộ kết quả vào: {OUT_DIR}")

            bundle_file = OUT_DIR / "bcb_sota_pool_dfg_bundle.tar.gz"
            if bundle_file.exists():
                print(f"Giải nén bundle: {bundle_file.name} ({bundle_file.stat().st_size / 1024 / 1024:.2f} MB)...")
                with tarfile.open(bundle_file, "r:gz") as tar:
                    tar.extractall(OUT_DIR)
                print("Đã giải nén bundle thành công!")

            metrics_file = OUT_DIR / "all_sota_pool_dfg_test_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                print("\n>>> KẾT QUẢ KIỂM THỬ SOTA POOL DFG TRÊN 5.000 MẪU BIGCLONEBENCH TEST <<<")
                print(json.dumps(metrics, indent=2))
            break

        elif status in ("ERROR", "CANCELLED", "FAILED"):
            print(f"\n[LỖI] Kernel kết thúc với trạng thái: {status}")
            if failure_msg:
                print(f"Chi tiết lỗi: {failure_msg}")
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
