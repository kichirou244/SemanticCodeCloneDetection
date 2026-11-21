"""Theo dõi tiến trình 2 Kernel M1 & M2 DiffMul chạy song song trên Kaggle và tự động tải kết quả về."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
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
logger = logging.getLogger("monitor_both_diff_mul")

def get_kaggle_username() -> str:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            return json.loads(kaggle_json.read_text(encoding="utf-8")).get("username", "onngctrnchu")
        except Exception:
            pass
    return "onngctrnchu"

UNAME = get_kaggle_username()

KERNELS = {
    "M1": {
        "slug": f"{UNAME}/bcb-tuning-m1-diffmul",
        "out_dir": Path("results/exp_m1_diffmul"),
        "download_dir": Path("results/download_m1_diffmul"),
        "done": False,
    },
    "M2": {
        "slug": f"{UNAME}/bcb-tuning-m2-diffmul",
        "out_dir": Path("results/exp_m2_diffmul"),
        "download_dir": Path("results/download_m2_diffmul"),
        "done": False,
    },
}


def check_kernel(api: KaggleApi, name: str, info: dict) -> None:
    slug = info["slug"]
    dl_dir = info["download_dir"]
    out_dir = info["out_dir"]
    dl_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        status_obj = api.kernels_status(slug)
        if hasattr(status_obj, "status"):
            status = status_obj.status.name if hasattr(status_obj.status, "name") else str(status_obj.status)
        elif isinstance(status_obj, dict):
            status = str(status_obj.get("status", "UNKNOWN"))
        else:
            status = str(status_obj)

        fail_msg = getattr(status_obj, "failure_message", None) or getattr(status_obj, "failureMessage", None)
    except Exception as e:
        status = "API_ERROR"
        fail_msg = str(e)

    logger.info("[%s] %s -> Status: %s", name, slug, status)

    if status in ("COMPLETE", "COMPLETED"):
        logger.info(">>> [%s] ĐÃ HOÀN TẤT THÀNH CÔNG! ĐANG TẢI KẾT QUẢ VỀ... <<<", name)
        api.kernels_output(slug, path=str(dl_dir), force=True)

        for f in dl_dir.rglob("*"):
            if f.is_file() and f.suffix in (".pt", ".json", ".csv"):
                clean_name = f.name
                for pfx in ("exp_m1_diffmul_", "exp_m2_diffmul_", "m1_diff_mul_", "m2_diff_mul_"):
                    if clean_name.startswith(pfx):
                        clean_name = clean_name.replace(pfx, "")
                dest = out_dir / clean_name
                shutil.copy2(f, dest)
                logger.info("  [%s] Saved: %s", name, clean_name)

        if (out_dir / "test_metrics.json").exists():
            logger.info(">>> KẾT QUẢ TEST %s: <<<\n%s", name, (out_dir / "test_metrics.json").read_text(encoding="utf-8"))

        info["done"] = True

    elif status in ("ERROR", "CANCELLED", "FAILED", "CANCEL_ACKNOWLEDGED"):
        logger.error("[%s] KẾT THÚC VỚI LỖI: %s (Chi tiết: %s)", name, status, fail_msg)
        try:
            api.kernels_output(slug, path=str(dl_dir), force=True)
            for f in dl_dir.rglob("*"):
                if f.is_file() and f.suffix in (".pt", ".json", ".csv"):
                    shutil.copy2(f, out_dir / f.name)
        except Exception:
            pass
        info["done"] = True


def main():
    api = KaggleApi()
    api.authenticate()

    print("=" * 80)
    print(">>> BẮT ĐẦU THEO DÕI SONG SONG 2 KERNEL TRÊN KAGGLE <<<")
    print(f"M1: {KERNELS['M1']['slug']}")
    print(f"M2: {KERNELS['M2']['slug']}")
    print("=" * 80)

    start_time = time.time()
    while not all(info["done"] for info in KERNELS.values()):
        elapsed_min = int((time.time() - start_time) / 60)
        print(f"\n--- [Thời gian đã theo dõi: {elapsed_min} phút] ---")

        for name, info in KERNELS.items():
            if not info["done"]:
                check_kernel(api, name, info)

        if all(info["done"] for info in KERNELS.values()):
            print("\n" + "=" * 80)
            print(">>> CẢ 2 MÔ HÌNH ĐÃ HOÀN TẤT THÀNH CÔNG! <<<")
            print("=" * 80)
            break

        time.sleep(45)


if __name__ == "__main__":
    main()
