import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


# ---------------------------------------------------------------------------
# 1. Distillation loss
# ---------------------------------------------------------------------------
class KDState:
    """Menyimpan teacher dan hyperparameter distillation (dipakai oleh patch loss)."""

    teacher = None      # model teacher (nn.Module, mode eval, tanpa gradient)
    temperature = 2.0   # T: melunakkan probabilitas teacher
    w_cls = 1.0         # bobot distillation skor kelas
    w_box = 0.5         # bobot distillation distribusi kotak (DFL)
    logged = False      # penanda: sudah mencetak nilai KD pertama kali atau belum


def distill_term(s_out, t_out, T):
    """Hitung loss distillation dari keluaran head student dan teacher.

    s_out / t_out adalah dict dengan kunci:
      'boxes'  : (B, 4*reg_max, A)  distribusi posisi kotak (DFL), A = jumlah anchor semua skala
      'scores' : (B, nc, A)         logit kelas (sebelum sigmoid)
    """
    s_box, s_cls = s_out["boxes"].float(), s_out["scores"].float()
    t_box, t_cls = t_out["boxes"].float(), t_out["scores"].float()

    if s_cls.shape != t_cls.shape or s_box.shape != t_box.shape:
        raise RuntimeError(
            f"Bentuk keluaran teacher dan student berbeda: student {tuple(s_cls.shape)} vs "
            f"teacher {tuple(t_cls.shape)}. Pastikan jumlah kelas dan imgsz sama."
        )

    b, box_ch, n = s_box.shape
    reg_max = box_ch // 4

    # (a) Kelas: student meniru probabilitas "lunak" milik teacher
    t_soft = torch.sigmoid(t_cls / T)
    bce = F.binary_cross_entropy_with_logits(s_cls / T, t_soft, reduction="none")  # (B, nc, A)
    cls_loss = bce.sum(1).mean() * (T * T)

    # (b) Kotak: KL divergence antar distribusi DFL, difokuskan ke lokasi
    #     yang diyakini teacher ada objeknya (confidence tertinggi per lokasi)
    s_logp = F.log_softmax(s_box.reshape(b, 4, reg_max, n), dim=2)
    t_prob = F.softmax(t_box.reshape(b, 4, reg_max, n), dim=2)
    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(2).mean(1)  # (B, A)
    focus = torch.sigmoid(t_cls).amax(1)                            # (B, A)
    box_loss = (kl * focus).sum() / (focus.sum() + 1e-6)

    return KDState.w_cls * cls_loss + KDState.w_box * box_loss


_original_call = v8DetectionLoss.__call__


def _kd_call(self, preds, batch):
    """Pengganti v8DetectionLoss.__call__: loss asli + loss distillation."""
    loss, loss_items = _original_call(self, preds, batch)

    # Hanya saat training (preds berupa dict) dan teacher sudah dipasang.
    # Saat validasi, preds berupa tuple sehingga distillation otomatis dilewati.
    if KDState.teacher is not None and isinstance(preds, dict):
        with torch.no_grad():
            t_out = KDState.teacher(batch["img"])
        t = t_out[1] if isinstance(t_out, (tuple, list)) else t_out

        # Jaga-jaga: kalau 'scores' teacher ternyata sudah probabilitas (0..1), ubah ke logit
        t_scores = t["scores"]
        if t_scores.min() >= 0 and t_scores.max() <= 1:
            t = {**t, "scores": torch.logit(t_scores.clamp(1e-6, 1 - 1e-6))}

        kd = distill_term(preds, t, KDState.temperature)
        loss = loss + kd * preds["boxes"].shape[0]  # skala sama dengan loss asli (dikali batch size)

        if not KDState.logged:
            print(
                f"[KD] aktif - nilai loss distillation (batch pertama): {kd.item():.4f} "
                f"| rentang skor teacher: {t['scores'].min().item():.2f} s/d {t['scores'].max().item():.2f}"
            )
            KDState.logged = True
    return loss, loss_items


def enable_distillation(teacher_path, device, temperature=2.0, w_cls=1.0, w_box=0.5):
    """Pasang teacher dan aktifkan loss distillation."""
    teacher = YOLO(teacher_path).model.to(device).float().eval()
    for p in teacher.parameters():
        p.requires_grad = False
    KDState.teacher = teacher
    KDState.temperature = temperature
    KDState.w_cls = w_cls
    KDState.w_box = w_box
    v8DetectionLoss.__call__ = _kd_call  # patch di level class (tidak ikut tersimpan ke checkpoint)
    print(f"[KD] Teacher dimuat dari {teacher_path} (T={temperature}, w_cls={w_cls}, w_box={w_box})")


# ---------------------------------------------------------------------------
# 2. Perintah: teacher / baseline / distill / compare
# ---------------------------------------------------------------------------
def train(model_name, run_name, args):
    model = YOLO(model_name)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs_kd",
        name=run_name,
        exist_ok=True,
    )
    print(f"\n[OK] Bobot terbaik {run_name}: {model.trainer.best}\n")


def count_params(weights):
    m = YOLO(weights).model
    return sum(p.numel() for p in m.parameters()) / 1e6


def compare(args):
    print(f"{'Model':<40}{'mAP50':>8}{'mAP50-95':>10}{'Param(M)':>10}{'ms/img':>9}")
    for w in args.weights:
        r = YOLO(w).val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
        print(
            f"{Path(w).parent.parent.name:<40}"
            f"{r.box.map50 * 100:>8.1f}{r.box.map * 100:>10.1f}"
            f"{count_params(w):>10.2f}{r.speed['inference']:>9.1f}"
        )


def main():
    p = argparse.ArgumentParser(description="Knowledge distillation YOLO11 untuk deteksi penyakit daun")
    p.add_argument("mode", choices=["teacher", "baseline", "distill", "compare"])
    p.add_argument("--data", required=True, help="path data.yaml (format YOLO)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="'0' untuk GPU, 'cpu' untuk CPU")
    p.add_argument("--teacher-arch", default="yolo11l.pt")
    p.add_argument("--student-arch", default="yolo11n.pt")
    p.add_argument("--teacher", help="best.pt teacher hasil mode 'teacher' (untuk mode distill)")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--w-box", type=float, default=0.5)
    p.add_argument("--weights", nargs="+", help="daftar .pt yang dibandingkan (mode compare)")
    args = p.parse_args()

    if args.mode == "teacher":
        train(args.teacher_arch, "teacher", args)
    elif args.mode == "baseline":
        train(args.student_arch, "student_baseline", args)
    elif args.mode == "distill":
        if not args.teacher:
            p.error("mode distill membutuhkan --teacher <best.pt>")
        dev = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
        enable_distillation(args.teacher, dev, args.temperature, args.w_cls, args.w_box)
        train(args.student_arch, "student_kd", args)
    elif args.mode == "compare":
        if not args.weights:
            p.error("mode compare membutuhkan --weights <a.pt> <b.pt> ...")
        compare(args)


if __name__ == "__main__":
    main()