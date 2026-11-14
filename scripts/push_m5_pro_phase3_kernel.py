"""Đẩy Kernel M5-Pro Phase 3 (Cross-Attention & Contrastive SOTA trên 30.000 cặp) lên Kaggle."""

from __future__ import annotations

import sys
from pathlib import Path
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


def main():
    deploy_dir = Path("kaggle_deploy/m5_pro_phase3")
    assert deploy_dir.exists(), f"Không tìm thấy thư mục {deploy_dir}"

    print(">>> Đang kết nối Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    user = api.get_config_value("username")
    print(f"Đã xác thực tài khoản Kaggle: {user}")

    print(f">>> Đang đẩy kernel từ {deploy_dir} lên Kaggle...")
    res = api.kernels_push(str(deploy_dir))
    print(f"Kết quả đẩy kernel: {res}")

    kernel_slug = "tranthikhanhlinh1709/bcb-m5-pro-phase-3-contrastive-30k"
    status = api.kernels_status(kernel_slug)
    print(f"Trạng thái ban đầu của Kernel: {status}")


if __name__ == "__main__":
    main()
