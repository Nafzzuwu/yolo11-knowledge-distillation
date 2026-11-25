"""
diag.py - Diagnosa format keluaran YOLO11 saat training di versi ultralytics yang terpasang.
Tidak melatih apa pun. Jalankan:  python diag.py   lalu kirim seluruh hasilnya.
"""

import inspect

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


def describe(obj, name="out", depth=0):
    pad = "  " * depth
    if isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: Tensor {tuple(obj.shape)}")
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict, keys = {list(obj.keys())}")
        for k, v in obj.items():
            describe(v, f"[{k!r}]", depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} dengan {len(obj)} elemen")
        for i, v in enumerate(obj):
            describe(v, f"[{i}]", depth + 1)
    else:
        print(f"{pad}{name}: {type(obj).__name__}")


print("ultralytics:", ultralytics.__version__, "| torch:", torch.__version__)

model = YOLO("yolo11n.pt").model

print("\n=== Keluaran model mode TRAINING ===")
model.train()
describe(model(torch.zeros(1, 3, 640, 640)))

print("\n=== Keluaran model mode EVAL ===")
model.eval()
with torch.no_grad():
    describe(model(torch.zeros(1, 3, 640, 640)))

print("\n=== Kode v8DetectionLoss.__call__ (maks 60 baris) ===")
print("\n".join(inspect.getsource(v8DetectionLoss.__call__).splitlines()[:60]))

print("\n=== Daftar method v8DetectionLoss ===")
print([n for n in dir(v8DetectionLoss) if not n.startswith("__")])