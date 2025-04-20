# Watermark‑Remover (GAN‐based, Patch‑wise)

> Patch‑based adversarial model that learns to **erase watermarks** from images of any size.  
> Combines a Res‑UNet generator, a spectral‑norm PatchGAN discriminator, and VGG‑19 perceptual loss.

---

## ✨ Key Features
| Feature | Why it matters |
|---------|---------------|
| **Res‑UNet Generator** – 4‑level encoder/decoder with 9 residual blocks (stochastic‑depth)| Preserves spatial detail while boosting capacity. |
| **Single‑scale PatchGAN Discriminator** | Penalises local artefacts without over‑parameterisation. |
| **Perceptual + L1 + GAN losses** | Balances pixel fidelity with perceptual realism. |
| **Patch‑wise training** (`256 × 256` kernels, `64` px stride) | Handles arbitrarily large images; smaller GPU footprint. |
| **Albumentations pipeline** | Real‐world JPEG, noise, motion blur, flips, etc. |
| **AMP + gradient clipping + schedulers** | Stable, mixed‑precision training out of the box. |

---

## 📂 Directory Layout

```text
.
├── dataset-smol/              # default sample dataset (see below)
│   ├── mark/                  # watermarked images  →  {id}_c.jpg
│   └── nomark/                # pristine targets    →  {id}_r.jpg
├── cropper.py                 # optional pre‑processor to align pairs
├── dataset.py                 # WatermarkPatchDataset  (patch extraction)
├── generator_model.py         # Res‑UNet G
├── discriminator_model.py     # PatchGAN D
├── vgg_loss.py                # perceptual loss (VGG‑19)
├── config.py                  # all tunables live here
├── train.py                   # main training script
├── utils.py                   # checkpoint helpers
└── README.md
