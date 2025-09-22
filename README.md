# 🏃 FaceReg – Marathon Photo Finder

FaceReg helps you **find yourself in marathon photos** using [DeepFace](https://github.com/serengil/deepface).
It creates face embeddings from your reference photos and scans through a folder of race images to find matches.

---

## 📂 Project Layout

```
data/
  reference/          # your own face photos
  test_data/          # marathon photos to scan
  output/             # results (embeddings, matches, csv)
src/
  generate_embeddings.py
  scan_img.py
  scraper.py
  crawler.py
```

---

## ⚙️ Setup

```bash
conda create -n FaceReg python=3.10 -y
conda activate FaceReg

# PyTorch (GPU enabled)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core packages
pip install -r requirement.txt
```

---

## 🚀 Usage

1. Put 1–5 clear reference photos in `data/reference/`.
2. Generate embeddings:

   ```bash
   python src/generate_embeddings.py
   ```
3. Place marathon images in `data/test_data/`.
4. Scan:

   ```bash
   python src/scan_img.py
   ```

Matches are copied to `data/output/matches/` and a summary is saved in `data/output/results.csv`.

---

## ⚡ Notes

* ArcFace embeddings (PyTorch) → GPU if available.
* RetinaFace detection (TensorFlow) → GPU only if TF < 2.11 on Windows.
* Threshold for matching is configurable in `scan_img.py`.
