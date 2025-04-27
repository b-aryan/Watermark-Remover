import os
from PIL import Image
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from tqdm import tqdm
from typing import List
import csv

from run_on_patches_online import process_image_from_data, load_model


def evaluate_models(
        mark_dir: str,
        nomark_dir: str,
        model_paths: List[str],
        output_csv: str,
        device: str,
        kernel_size: int = 256,
        stride: int = 64,
):
    """
    Evaluates multiple watermark removal models and saves results to CSV.
    """
    # Initialize metrics and image pairs
    metric_heads = ['Model', 'PSNR', 'SSIM', 'LPIPS', 'Num_Images']
    all_results = []

    # Find all valid image pairs first
    mark_files = sorted([f for f in os.listdir(mark_dir) if f.endswith('_c.jpg')])
    valid_pairs = []

    for mark_file in mark_files:
        mark_path = os.path.join(mark_dir, mark_file)
        base_name = mark_file.replace('_c.jpg', '')
        nomark_path = os.path.join(nomark_dir, f"{base_name}_r.jpg")

        if os.path.exists(nomark_path):
            valid_pairs.append((mark_path, nomark_path))

    if not valid_pairs:
        raise ValueError(f"No valid image pairs found in {mark_dir} and {nomark_dir}")

    # Normalization parameters for LPIPS
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Evaluate each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating {model_name}...")

        try:
            # Load model
            model = load_model(model_path, device)
            model.eval()

            # Initialize metrics
            psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            lpips_model = lpips.LPIPS(net='vgg', verbose=False).to(device)

            total_psnr = 0.0
            total_ssim = 0.0
            total_lpips = 0.0
            processed_count = 0

            for mark_path, nomark_path in tqdm(valid_pairs, desc="Processing Images"):
                try:
                    # Process image
                    marked_img = Image.open(mark_path).convert('RGB')
                    W, H = marked_img.size
                    generated_img = process_image_from_data(
                        marked_img, model, device, kernel_size, stride, use_tqdm=False
                    )

                    if generated_img is None:
                        continue

                    # Load and resize ground truth
                    nomark_img = Image.open(nomark_path).convert('RGB').resize((W, H), Image.LANCZOS)

                    # Convert to tensors
                    generated_tensor = torch.from_numpy(np.array(generated_img)).permute(2, 0, 1).float() / 255.0
                    nomark_tensor = torch.from_numpy(np.array(nomark_img)).permute(2, 0, 1).float() / 255.0

                    # Move to device
                    generated_tensor = generated_tensor.unsqueeze(0).to(device)
                    nomark_tensor = nomark_tensor.unsqueeze(0).to(device)

                    # Calculate metrics
                    current_psnr = psnr(generated_tensor, nomark_tensor)
                    current_ssim = ssim(generated_tensor, nomark_tensor)

                    # LPIPS calculation
                    generated_normalized = (generated_tensor - mean) / std
                    nomark_normalized = (nomark_tensor - mean) / std
                    current_lpips = lpips_model(generated_normalized, nomark_normalized, normalize=False)

                    total_psnr += current_psnr.item()
                    total_ssim += current_ssim.item()
                    total_lpips += current_lpips.item()
                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {mark_path}: {str(e)}")
                    continue

            # Calculate averages
            if processed_count > 0:
                avg_psnr = total_psnr / processed_count
                avg_ssim = total_ssim / processed_count
                avg_lpips = total_lpips / processed_count

                all_results.append({
                    'Model': model_name,
                    'PSNR': avg_psnr,
                    'SSIM': avg_ssim,
                    'LPIPS': avg_lpips,
                    'Num_Images': processed_count
                })

        except Exception as e:
            print(f"Failed to evaluate {model_name}: {str(e)}")
            continue

    # Save results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metric_heads)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\nSaved results to {output_csv}")
    return all_results


if __name__ == "__main__":
    results = evaluate_models(
        mark_dir="evaluation/mark",
        nomark_dir="evaluation/nomark",
        model_paths=["gen_epoch_13.pth.tar"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        kernel_size=256,
        stride=64,
    )





