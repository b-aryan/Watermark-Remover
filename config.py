import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Core training parameters ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 10
CLIP_GRAD = 1.0
USE_AMP = True

# Directories for WatermarkPatchDataset
MARK_DIR = "dataset-smol/mark"
NOMARK_DIR = "dataset-smol/nomark"

# Patch Extraction Parameters (for WatermarkPatchDataset and run_on_patches.py)
PATCH_KERNEL_SIZE = 256
PATCH_STRIDE = 32

# --- Loss Function Weights ---
L1_LAMBDA = 100
VGG_LAMBDA = 10
GAN_LOSS_LAMBDA = 1

# --- Model Checkpointing ---
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_SAVE_INTERVAL = 5
CHECKPOINT_DISC = 'disc.pth.tar'
CHECKPOINT_GEN = 'gen.pth.tar'

transform_input = A.Compose([
    A.GaussNoise(
        std_range=(0.05, 0.15),  # Standard deviation ~13 to ~38 for 8-bit images
        mean_range=(0.0, 0.0),  # Keep noise centered around zero
        per_channel=True,  # Slightly more realistic for RGB noise
        noise_scale_factor=1.0,  # Full-resolution noise (can reduce if needed)
        p=0.3
    ),
    A.RandomBrightnessContrast(p=0.3),
    A.ImageCompression(
        quality_range=(60, 95),  # Simulate real-world JPEG artifacts
        compression_type='jpeg',  # JPEG is more common in real-world images
        p=0.3
    ),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
    ToTensorV2()
])

# Target (clean image) augmentations — only spatial transforms
transform_target = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
    ToTensorV2()
])