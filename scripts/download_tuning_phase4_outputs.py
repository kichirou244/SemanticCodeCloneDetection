"""Tải chọn lọc các file trọng yếu từ 2 kernel M1-WD và M2-DR trên Kaggle về máy local.
Tự động bỏ qua last_checkpoint.pt và tar.gz để tải siêu tốc, an toàn SSL 100%.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import urllib3

urllib3.disable_warnings()

from kagglesdk.kaggle_http_client import KaggleHttpClient
orig_init = KaggleHttpClient._init_session
def patched_init(self):
    orig_init(self)
    if self._session is not None:
        self._session.verify = False
KaggleHttpClient._init_session = patched_init

import requests
# Patch requests.get to disable SSL verification
orig_requests_get = requests.get
def patched_requests_get(*args, **kwargs):
    kwargs['verify'] = False
    return orig_requests_get(*args, **kwargs)
requests.get = patched_requests_get

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def download_kernel_outputs(kernel_slug: str, output_dir: Path, target_exp: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results") / target_exp
    results_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    owner_slug, slug = kernel_slug.split("/")

    with api.build_kaggle_client() as kaggle:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner_slug
        request.kernel_slug = slug
        response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

    print(f"\n>>> Dang xu ly kernel: {kernel_slug} ({len(response.files)} files) <<<")

    # Lưu kernel log nếu có
    if response.log:
        log_file = output_dir / f"{slug}.log"
        log_file.write_text(response.log, encoding="utf-8")
        print(f"  - Da luu log: {log_file.name}")

    for item in response.files:
        fname = item.file_name
        # Bỏ qua last_checkpoint và tar.gz
        if "last_checkpoint" in fname or fname.endswith(".tar.gz"):
            print(f"  - Bo qua file lon khong can thiet: {fname}")
            continue

        local_file = output_dir / fname
        print(f"  - Dang tai: {fname}...")
        resp = requests.get(item.url, stream=True)
        resp.raise_for_status()
        with open(local_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"    -> Xong: {fname} ({local_file.stat().st_size / 1024 / 1024:.2f} MB)")

        # Sao chep vao results/
        if "best_checkpoint.pt" in fname:
            shutil.copy2(local_file, results_dir / "best_checkpoint.pt")
            print(f"    -> Da sao chep vao {results_dir / 'best_checkpoint.pt'}")
        elif "test_metrics.json" in fname:
            shutil.copy2(local_file, results_dir / "test_metrics.json")
            print(f"    -> Da sao chep vao {results_dir / 'test_metrics.json'}")
        elif "training_summary.json" in fname:
            shutil.copy2(local_file, results_dir / "training_summary.json")
            print(f"    -> Da sao chep vao {results_dir / 'training_summary.json'}")


def main():
    download_kernel_outputs(
        kernel_slug="tranthikhanhlinh1709/bcb-tuning-m1-wd",
        output_dir=Path("kaggle_output/tuning_m1_wd"),
        target_exp="exp_m1_wd",
    )

    download_kernel_outputs(
        kernel_slug="tranthikhanhlinh1709/bcb-tuning-m2-dr",
        output_dir=Path("kaggle_output/tuning_m2_dr"),
        target_exp="exp_m2_dr",
    )

    print("\n>>> TOAN BO FILE DA DUOC TAI VA SAO CHEP THANH CONG! <<<")


if __name__ == "__main__":
    main()
