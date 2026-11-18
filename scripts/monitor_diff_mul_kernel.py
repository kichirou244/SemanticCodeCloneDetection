"""Theo dõi tiến trình Kernel M1 & M2 DiffMul Paired trên Kaggle và tự động tải kết quả về."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import shutil
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
logger = logging.getLogger("monitor_diff_mul")

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-tuning-m1-m2-diffmul-paired"
OUT_BASE = Path("results")


def main():
    api = KaggleApi()
    api.authenticate()

    print("=" * 80)
    print(f">>> BẮT ĐẦU THEO DÕI KERNEL TRÊN KAGGLE: {KERNEL_SLUG} <<<")
    print("=" * 80)

    last_status = None
    start_time = time.time()
    poll_count = 0

    download_dir = Path("results/diff_mul_download")
    download_dir.mkdir(parents=True, exist_ok=True)

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

            api.kernels_output(KERNEL_SLUG, path=str(download_dir))
            print(f"Đã tải toàn bộ kết quả vào: {download_dir}")

            # Phân bổ file kết quả vào results/exp_m1_diffmul và results/exp_m2_diffmul
            m1_dir = OUT_BASE / "exp_m1_diffmul"
            m2_dir = OUT_BASE / "exp_m2_diffmul"
            m1_dir.mkdir(parents=True, exist_ok=True)
            m2_dir.mkdir(parents=True, exist_ok=True)

            for f in download_dir.glob("*"):
                name = f.name
                if name.startswith("exp_m1_diffmul_"):
                    clean_name = name.replace("exp_m1_diffmul_", "")
                    shutil.copy2(f, m1_dir / clean_name)
                    print(f"  [DISPATCH M1] {clean_name}")
                elif name.startswith("exp_m2_diffmul_"):
                    clean_name = name.replace("exp_m2_diffmul_", "")
                    shutil.copy2(f, m2_dir / clean_name)
                    print(f"  [DISPATCH M2] {clean_name}")

            # In kết quả đánh giá nếu có
            if (m1_dir / "test_metrics.json").exists():
                print("\n>>> KẾT QUẢ TEST M1 (CodeBERT DiffMul): <<<")
                print((m1_dir / "test_metrics.json").read_text(encoding="utf-8"))

            if (m2_dir / "test_metrics.json").exists():
                print("\n>>> KẾT QUẢ TEST M2 (GraphCodeBERT DiffMul): <<<")
                print((m2_dir / "test_metrics.json").read_text(encoding="utf-8"))

            break

        elif status in ("ERROR", "CANCELLED", "FAILED"):
            print(f"\n[LỖI] Kernel kết thúc với trạng thái: {status}")
            if failure_msg:
                print(f"Chi tiết lỗi: {failure_msg}")

            # Dù có lỗi, vẫn thử tải kết quả về phòng trường hợp M1 đã xong trước đó
            try:
                print("Đang thử tải kết quả đã lưu trữ phòng ngừa...")
                api.kernels_output(KERNEL_SLUG, path=str(download_dir))
                for f in download_dir.glob("exp_m1_diffmul_*"):
                    m1_dir = OUT_BASE / "exp_m1_diffmul"
                    m1_dir.mkdir(parents=True, exist_ok=True)
                    clean_name = f.name.replace("exp_m1_diffmul_", "")
                    shutil.copy2(f, m1_dir / clean_name)
                    print(f"  [RECOVERED M1] {clean_name}")
            except Exception as rec_err:
                print(f"Không thể tải kết quả dở dang: {rec_err}")
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
