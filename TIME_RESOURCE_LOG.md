# Time & Resource Log
## SAR-EO Building Damage Change Detection

---

## Hardware Used

| Resource        | Details                          |
|:----------------|:---------------------------------|
| Device          | Laptop (CPU only)                |
| GPU             | None (no dedicated graphics card)|
| CPU             | Standard laptop processor        |
| RAM             | System RAM (no VRAM)             |

---

## Training Time

| Phase              | Details                           | Time Taken     |
|:-------------------|:----------------------------------|:---------------|
| Environment setup  | pip install, venv creation        | ~10 minutes    |
| Dataset extraction | Unzipping train/val/test          | ~15 minutes    |
| Training (10 epochs) | ResNet18, 128×128, 2,781 samples | ~4–5 hours     |
| Evaluation (val)   | 334 samples, CPU inference        | ~1.5 minutes   |
| Total              |                                   | **~5–6 hours** |

> **Note**: Full training (50 epochs, ResNet34, 256×256) was estimated at ~4 days on CPU.
> Training was limited to **10 epochs** to keep runtime feasible without a GPU.

---

## Per-Epoch Training Time (Approximate)

| Epoch Range | Approx. Time/Epoch |
|:-----------:|:------------------:|
| 1 – 10      | ~25–30 minutes     |

---

## Storage Used

| Item                        | Size       |
|:----------------------------|:----------:|
| Training dataset (extracted)| ~8.4 GB    |
| Val dataset (extracted)     | ~980 MB    |
| Test dataset (extracted)    | ~186 MB    |
| Model checkpoint (best)     | ~185 MB    |
| Virtual environment (venv)  | ~1.5 GB    |

---

## Software & Libraries

| Library                      | Version   | Purpose                        |
|:-----------------------------|:---------:|:-------------------------------|
| Python                       | 3.10+     | Core language                  |
| PyTorch                      | 2.x       | Deep learning framework        |
| segmentation-models-pytorch  | latest    | Siamese UNet backbone          |
| Albumentations               | latest    | Image augmentations            |
| rasterio                     | latest    | GeoTIFF image loading          |
| NumPy                        | latest    | Array operations               |
| Matplotlib                   | latest    | Visualization                  |
| scikit-learn                 | latest    | Metrics computation            |
| PyYAML                       | latest    | Config file parsing            |
| tqdm                         | latest    | Progress bars                  |
