import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache


class WatermarkPatchDataset(Dataset):
    def __init__(self, mark_dir, nomark_dir, kernel_size=256, stride=32,
                 transform_input=None, transform_target=None):
        self.mark_dir = mark_dir
        self.nomark_dir = nomark_dir
        self.kernel_size = kernel_size
        self.stride = stride
        self.transform_input = transform_input  # Static transforms (e.g., normalization)
        self.transform_target = transform_target

        # Store image pairs and precompute patch coordinates
        self.image_pairs = []
        self.patch_coords = []

        mark_files = sorted([f for f in os.listdir(mark_dir) if f.endswith('_c.jpg')])
        for mark_file in mark_files:
            base_name = mark_file.replace('_c.jpg', '')
            mark_path = os.path.join(mark_dir, mark_file)
            nomark_path = os.path.join(nomark_dir, f'{base_name}_r.jpg')

            # Get image dimensions without loading full image
            with Image.open(mark_path) as img:
                width, height = img.size

            self.image_pairs.append((mark_path, nomark_path, width, height))

            # Precompute patch coordinates using unfold logic
            patches_x = (width - kernel_size) // stride + 1
            patches_y = (height - kernel_size) // stride + 1

            for i in range(patches_y):
                for j in range(patches_x):
                    self.patch_coords.append((
                        len(self.image_pairs) - 1,  # Pair index
                        i, j  # Patch grid coordinates (not pixel coordinates)
                    ))

    @lru_cache(maxsize=32)  # Cache most recently used processed images
    def _load_and_preprocess(self, pair_idx):
        """Load images, resize nomark, apply static transforms, and precompute patches"""
        mark_path, nomark_path, width, height = self.image_pairs[pair_idx]

        # Load and resize nomark once
        mark_img = Image.open(mark_path).convert('RGB')
        nomark_img = Image.open(nomark_path).convert('RGB').resize((width, height), Image.LANCZOS)

        # Convert to tensors with static transforms
        mark_tensor = self.transform_input(image=np.array(mark_img))['image'] if self.transform_input \
            else torch.from_numpy(np.array(mark_img)).permute(2, 0, 1).float() / 255.0

        nomark_tensor = self.transform_target(image=np.array(nomark_img))['image'] if self.transform_target \
            else torch.from_numpy(np.array(nomark_img)).permute(2, 0, 1).float() / 255.0

        # Precompute all patches using unfold (vectorized)
        mark_patches = mark_tensor.unfold(1, self.kernel_size, self.stride) \
            .unfold(2, self.kernel_size, self.stride) \
            .permute(1, 2, 0, 3, 4) \
            .reshape(-1, 3, self.kernel_size, self.kernel_size)

        nomark_patches = nomark_tensor.unfold(1, self.kernel_size, self.stride) \
            .unfold(2, self.kernel_size, self.stride) \
            .permute(1, 2, 0, 3, 4) \
            .reshape(-1, 3, self.kernel_size, self.kernel_size)

        return mark_patches, nomark_patches

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        pair_idx, i, j = self.patch_coords[idx]
        mark_patches, nomark_patches = self._load_and_preprocess(pair_idx)

        # Calculate flat index from grid coordinates
        patches_per_row = (self.image_pairs[pair_idx][2] - self.kernel_size) // self.stride + 1
        patch_idx = i * patches_per_row + j

        return mark_patches[patch_idx], nomark_patches[patch_idx]