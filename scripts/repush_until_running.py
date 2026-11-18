"""Kiểm tra và tự động re-push nếu kernel bị kẹt ở trạng thái QUEUED cho đến khi chuyển sang RUNNING."""

import subprocess
import time

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-tuning-m1-m2-diffmul-paired"
DEPLOY_DIR = "kaggle_deploy/tuning_m1_m2_diff_mul"


def get_status() -> str:
    cmd = ["kaggle", "kernels", "status", KERNEL_SLUG]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    return out


def push_kernel() -> str:
    cmd = ["kaggle", "kernels", "push", "-p", DEPLOY_DIR]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()


def main():
    print(f"Bắt đầu theo dõi và tự động push cho kernel: {KERNEL_SLUG}")
    attempt = 1

    while True:
        status_line = get_status()
        print(f"[{time.strftime('%H:%M:%S')}] Kiểm tra: {status_line}")

        if "RUNNING" in status_line:
            print("\n" + "=" * 70)
            print(">>> KERNEL ĐÃ CHUYỂN SANG TRẠNG THÁI RUNNING THÀNH CÔNG! <<<")
            print("=" * 70)
            break

        if "COMPLETE" in status_line:
            print("\n>>> KERNEL ĐÃ HOÀN TẤT! <<<")
            break

        if "QUEUED" in status_line:
            # Chờ 15s để xem có chuyển sang RUNNING không trước khi quyết định re-push
            print("Đang ở trạng thái QUEUED. Chờ 15s xem có chuyển sang RUNNING...")
            time.sleep(15)
            status_line = get_status()
            print(f"[{time.strftime('%H:%M:%S')}] Kiểm tra lại: {status_line}")

            if "RUNNING" in status_line:
                print("\n" + "=" * 70)
                print(">>> KERNEL ĐÃ CHUYỂN SANG TRẠNG THÁI RUNNING THÀNH CÔNG! <<<")
                print("=" * 70)
                break

            if "QUEUED" in status_line:
                attempt += 1
                print(f"\nVẫn QUEUED -> Tiến hành re-push lên version mới (Lần {attempt})...")
                push_out = push_kernel()
                print(push_out)
                time.sleep(8)
                continue

        time.sleep(10)


if __name__ == "__main__":
    main()
