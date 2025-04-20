import os
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
from tqdm import tqdm
import requests
import io
from generator_model import Generator

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_GEN = "gen_epoch_86.pth.tar" # Keep your checkpoint name
PATCH_KERNEL_SIZE = 256
PATCH_STRIDE = 64
# DEFAULT_INPUT_DIR = "test/inputs"   # No longer needed for Gradio URL input
# DEFAULT_OUTPUT_DIR = "test/outputs" # Output handled by Gradio
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff') # Still useful for local testing if needed

test_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2()
])

def load_model(checkpoint_path: str, device: str) -> Generator:
    print(f"Loading model from: {checkpoint_path} onto device: {device}")
    model = Generator(in_channels=3, features=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    new_state_dict = OrderedDict()
    has_module_prefix = any(k.startswith("module.") for k in checkpoint["state_dict"])
    for k, v in checkpoint["state_dict"].items():
        name = k.replace("module.", "") if has_module_prefix else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
    return model

def calculate_padding(img_h: int, img_w: int, kernel_size: int, stride: int) -> tuple[int, int]:
    pad_h = kernel_size - img_h if img_h < kernel_size else (stride - (img_h - kernel_size) % stride) % stride
    pad_w = kernel_size - img_w if img_w < kernel_size else (stride - (img_w - kernel_size) % stride) % stride
    return pad_h, pad_w

def download_image(url: str, timeout: int = 15) -> Image.Image | None:
    """Downloads an image from a URL and returns it as a PIL Image object."""
    print(f"Attempting to download image from: {url}")
    try:
        headers = {'User-Agent': 'Gradio-Image-Processor/1.0'} # Be a good net citizen
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        content_type = response.headers.get('Content-Type', '').lower()
        if not content_type.startswith('image/'):
            print(f"Error: URL content type ({content_type}) is not an image.")
            return None

        image_bytes = response.content
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = pil_image.convert('RGB') # Ensure image is in RGB format
        print(f"Image downloaded successfully ({pil_image.width}x{pil_image.height}).")
        return pil_image

    except requests.exceptions.Timeout:
        print(f"Error: Request timed out after {timeout} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except UnidentifiedImageError:
        print("Error: Could not identify image file. The URL might not point to a valid image.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None

def process_image_from_data(
    input_pil_image: Image.Image,
    model: Generator,
    device: str,
    kernel_size: int,
    stride: int,
    use_tqdm: bool = True # Optional: Control progress bar visibility
    ) -> Image.Image | None:
    """
    Processes an input PIL image using the patch-based method and returns the output PIL image.
    Returns None if an error occurs during processing.
    """
    print(f"\nProcessing image data...")
    try:
        image_np = np.array(input_pil_image) # Convert PIL Image to NumPy array
        H, W, _ = image_np.shape
        print(f"  Input dimensions: {W}x{H}")

        # Apply transformations
        transformed = test_transform(image=image_np)
        input_tensor = transformed['image'].to(device)  # Shape: (C, H, W)
        C = input_tensor.shape[0]

        # Calculate and apply padding
        pad_h, pad_w = calculate_padding(H, W, kernel_size, stride)
        print(f"  Calculated padding (H, W): ({pad_h}, {pad_w})")
        padded_tensor = F.pad(input_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        _, H_pad, W_pad = padded_tensor.shape
        print(f"  Padded dimensions: {W_pad}x{H_pad}")

        # Extract patches
        patches = padded_tensor.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        num_patches_h = patches.shape[1]
        num_patches_w = patches.shape[2]
        num_patches_total = num_patches_h * num_patches_w
        print(f"  Extracted {num_patches_total} patches ({num_patches_h} H x {num_patches_w} W)")

        patches = patches.contiguous().view(C, -1, kernel_size, kernel_size)
        # Permute to (num_patches_total, C, kernel_size, kernel_size)
        patches = patches.permute(1, 0, 2, 3).contiguous()

        output_patches = []
        # Set up tqdm iterator if enabled
        patch_iterator = tqdm(patches, total=num_patches_total, desc="  Inferring patches", unit="patch", leave=False, disable=not use_tqdm)

        # --- Inference Loop ---
        with torch.no_grad():
            for patch in patch_iterator:
                # Add batch dimension, run model, remove batch dimension
                output_patch = model(patch.unsqueeze(0)).squeeze(0)
                # Move to CPU immediately to save GPU memory during inference loop
                output_patches.append(output_patch.cpu())

        # Stack output patches back together
        # If GPU memory allows, move back for reconstruction, otherwise keep on CPU
        # Let's try moving back to device for faster reconstruction if possible
        try:
            output_patches = torch.stack(output_patches).to(device)
            print(f"  Output patches moved to {device} for reconstruction.")
        except Exception as e: # Catch potential OOM on device
            print(f"  Warning: Could not move all output patches to {device} ({e}). Reconstruction might be slower on CPU.")
            output_patches = torch.stack(output_patches) # Keep on CPU


        # --- Reconstruction ---
        # Generate 2D Hann window for blending
        window_1d = torch.hann_window(kernel_size, periodic=False, device=device) # periodic=False is common
        window_2d = torch.outer(window_1d, window_1d)
        window_2d = window_2d.unsqueeze(0).to(device) # Add channel dim and ensure on device

        # Initialize output tensor and weight tensor (for weighted averaging)
        output_tensor = torch.zeros((C, H_pad, W_pad), device=device, dtype=output_patches.dtype)
        weight_tensor = torch.zeros((C, H_pad, W_pad), device=device, dtype=window_2d.dtype)

        patch_idx = 0
        reconstruct_iterator = tqdm(total=num_patches_total, desc="  Reconstructing", unit="patch", leave=False, disable=not use_tqdm)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + kernel_size
                w_end = w_start + kernel_size

                # Get current patch (ensure it's on the correct device)
                current_patch = output_patches[patch_idx].to(device)
                weighted_patch = current_patch * window_2d # Apply window

                # Add weighted patch to output tensor
                output_tensor[:, h_start:h_end, w_start:w_end] += weighted_patch
                # Accumulate weights
                weight_tensor[:, h_start:h_end, w_start:w_end] += window_2d

                patch_idx += 1
                reconstruct_iterator.update(1)

        reconstruct_iterator.close() # Close the inner tqdm bar

        # Perform weighted averaging - clamp weights to avoid division by zero
        output_averaged = output_tensor / weight_tensor.clamp(min=1e-6)

        # Crop to original dimensions
        output_cropped = output_averaged[:, :H, :W]
        print(f"  Final output dimensions: {output_cropped.shape[2]}x{output_cropped.shape[1]}")

        # --- Convert to Output Format ---
        # Permute C, H, W -> H, W, C ; Move to CPU ; Convert to NumPy
        output_numpy = output_cropped.permute(1, 2, 0).cpu().numpy()

        # Denormalize: Assuming input was normalized to [-1, 1]
        output_numpy = (output_numpy * 0.5 + 0.5) * 255.0

        # Clip values to [0, 255] and convert to uint8
        output_numpy = output_numpy.clip(0, 255).astype(np.uint8)

        # Convert NumPy array back to PIL Image
        output_image = Image.fromarray(output_numpy)

        print("  Image processing complete.")
        return output_image

    except Exception as e:
        print(f"Error during image processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None

if __name__ == "__main__":
    print("--- Testing Phase 1 Refactoring ---")
    print(f"Using device: {DEVICE}")
    print(f"Using patch kernel size: {PATCH_KERNEL_SIZE}")
    print(f"Using patch stride: {PATCH_STRIDE}")
    print(f"Using model checkpoint: {CHECKPOINT_GEN}")

    # 1. Load the model (as it would be done globally in Gradio app)
    try:
        model = load_model(CHECKPOINT_GEN, DEVICE)
    except Exception as e:
        print(f"Failed to load model. Exiting test. Error: {e}")
        exit()


    # 2. Test URL download
    # Replace with a valid image URL for testing
    # test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    test_url = "https://www.shutterstock.com/shutterstock/photos/2501926843/display_1500/stock-photo-brunette-woman-laying-on-couch-cuddling-light-brown-dog-and-brown-tabby-cat-happy-2501926843.jpg" # A smaller known image
    input_pil = download_image(test_url)

    if input_pil:
        print(f"\nDownloaded image type: {type(input_pil)}, size: {input_pil.size}")

        # 3. Test processing the downloaded image
        output_pil = process_image_from_data(
            input_pil_image=input_pil,
            model=model,
            device=DEVICE,
            kernel_size=PATCH_KERNEL_SIZE,
            stride=PATCH_STRIDE,
            use_tqdm=True # Show progress bars during test
        )

        if output_pil:
            print(f"\nProcessed image type: {type(output_pil)}, size: {output_pil.size}")
            # Save the output locally for verification during testing
            try:
                os.makedirs("test_outputs", exist_ok=True)
                output_filename = "test_output_" + os.path.basename(test_url).split('?')[0] # Basic filename extraction
                if not output_filename.lower().endswith(SUPPORTED_EXTENSIONS):
                     output_filename += ".png" # Ensure it has an extension
                save_path = os.path.join("test_outputs", output_filename)
                output_pil.save(save_path)
                print(f"Saved test output to: {save_path}")
            except Exception as e:
                print(f"Error saving test output: {e}")
        else:
            print("\nImage processing failed.")
    else:
        print("\nImage download failed.")

    print("\n--- Phase 1 Testing Complete ---")