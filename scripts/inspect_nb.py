import json
import base64
import io
import tarfile
import yaml

for name, nb_path, cfg_path in [
    ("M1", "kaggle_deploy/tuning_m1_diff_mul/bcb_tuning_m1_diff_mul.ipynb", "configs/tuning/exp_m1_diff_mul.yaml"),
    ("M2", "kaggle_deploy/tuning_m2_diff_mul/bcb_tuning_m2_diff_mul.ipynb", "configs/tuning/exp_m2_diff_mul.yaml"),
]:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for line in nb["cells"][2]["source"]:
        if line.startswith("payload = '"):
            payload = line.split("'")[1]
            break
    tar = tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz")
    cfg = yaml.safe_load(tar.extractfile(cfg_path).read())
    print(f"=== {name} ===")
    print("Epochs:", cfg["training"]["epochs"])
    print("LR:", cfg["training"]["learning_rate"])
    print("Min LR:", cfg["training"].get("min_lr"))
    print("WD:", cfg["training"]["weight_decay"])
    print("Dropout:", cfg["model"]["dropout"])
    print("Label smoothing:", cfg["training"].get("label_smoothing"))
    print("Early stopping patience:", cfg["early_stopping"]["patience"])
