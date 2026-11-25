# YOLO11 Knowledge Distillation for Object Detection

Eksperimen **knowledge distillation (KD)** untuk melatih student YOLO11n dengan supervisi tambahan dari teacher YOLO11l. Repo ini berisi skrip training/evaluasi dan catatan demo; dataset serta hasil eksperimen sengaja tidak disertakan.

> **Status:** prototipe/eksperimen. Contoh `coco8` hanya untuk smoke test alur kode (8 gambar objek umum), bukan evaluasi deteksi penyakit daun dan bukan bukti KD meningkatkan akurasi.

## Alur eksperimen

1. **Teacher**: fine-tune YOLO11l pada dataset.
2. **Baseline**: fine-tune YOLO11n tanpa KD.
3. **KD student**: fine-tune YOLO11n dengan detection loss biasa + response-based distillation pada skor kelas dan distribusi box/DFL.
4. **Compare**: validasi bobot dan tampilkan mAP50, mAP50-95, jumlah parameter, dan waktu inference yang dilaporkan Ultralytics.

Skrip memasang loss KD dengan patch pada `v8DetectionLoss`; karena bergantung pada detail internal Ultralytics, versi kompatibilitas perlu dipastikan dan hasil harus diverifikasi sebelum dipakai untuk klaim ilmiah.

## Persiapan

Python 3.10+ disarankan. Pasang PyTorch yang sesuai perangkat (CPU/CUDA) dari instruksi resmi PyTorch, lalu:

```bash
pip install ultralytics
```

Simpan dataset dalam format YOLO dan siapkan file YAML, misalnya `data/leaf-disease/data.yaml`. File YAML dan dataset tidak disertakan. Pastikan `train`, `val`, `names`/`nc` dan path dataset valid.

## Menjalankan

Jalankan dari root repo. Contoh memakai `coco8.yaml` bawaan Ultralytics untuk smoke test; dataset akan diunduh oleh Ultralytics jika belum tersedia.

```bash
# Jalankan satu per satu; atur --device 0 untuk GPU atau --device cpu
python kd_yolo.py teacher --data coco8.yaml --epochs 1 --batch 4 --device cpu
python kd_yolo.py baseline --data coco8.yaml --epochs 1 --batch 4 --device cpu
python kd_yolo.py distill --data coco8.yaml --epochs 1 --batch 4 --device cpu \
  --teacher runs/detect/runs_kd/teacher/weights/best.pt
python kd_yolo.py compare --data coco8.yaml --device cpu --weights \
  runs/detect/runs_kd/teacher/weights/best.pt \
  runs/detect/runs_kd/student_baseline/weights/best.pt \
  runs/detect/runs_kd/student_kd/weights/best.pt
```

Untuk dataset sendiri, ganti `coco8.yaml` dengan path YAML. Secara default arsitektur memakai `yolo11l.pt` untuk teacher dan `yolo11n.pt` untuk student; Ultralytics dapat mengunduh bobot tersebut otomatis. Gunakan `--teacher-arch`/`--student-arch` untuk mengganti arsitektur. KD memerlukan teacher dan student dengan kompatibilitas output (termasuk jumlah kelas, ukuran input dan head); mode KD dirancang untuk satu GPU.

Hasil dibuat di `runs/detect/runs_kd/` (di dalam working directory Ultralytics). `best.pt` teacher menjadi input mode `distill`. Lihat `python kd_yolo.py --help` untuk opsi lengkap.

## Isi repository

- `kd_yolo.py` — training teacher, baseline, KD student, dan perbandingan.
- `diag.py` — diagnostik bentuk output/API Ultralytics; jalankan `python diag.py`.
- `demo.txt` — catatan demonstrasi dan batas interpretasi hasil.

## Data dan hasil

Jangan commit dataset mentah, cache, checkpoint besar (`.pt`), atau seluruh folder hasil training. Letakkan dataset secara lokal di `data/` (di-ignore), simpan hasil di `runs/` atau `runs_kd/` (di-ignore), lalu dokumentasikan sumber, lisensi, preprocessing dan split dataset di README/laporan. Jangan mengunggah dataset pihak ketiga tanpa memeriksa lisensi dan izin redistribusinya. Untuk mereproduksi eksperimen, cantumkan versi dependency, konfigurasi, seed, hardware, dan metrik; bila perlu bagikan checkpoint melalui Git LFS atau release eksternal setelah memeriksa batas dan lisensinya.

## Batasan

Metrik dari dataset kecil/epoch sedikit hanya untuk uji pipeline. Jangan menyimpulkan peningkatan akurasi KD tanpa eksperimen terkontrol pada dataset target, beberapa seed, serta evaluasi yang sesuai. Waktu inference dipengaruhi hardware dan setting.

