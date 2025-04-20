import os
from PIL import Image
import shutil


def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to hold the suffixes for each x
    pairs = {}

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            base, ext = os.path.splitext(filename)
            if '_r' in base:
                # Split into x and suffix '_r'
                x_part = base.split('_r')[0]
                suffix = '_r'
            else:
                x_part = base
                suffix = ''

            # Update the pairs dictionary
            if x_part not in pairs:
                pairs[x_part] = set()
            pairs[x_part].add(suffix)

    # Process each x that has both '' and '_r' suffixes
    for x in pairs:
        if '' in pairs[x] and '_r' in pairs[x]:
            x_jpg = os.path.join(input_folder, f"{x}.jpg")
            x_r_jpg = os.path.join(input_folder, f"{x}_r.jpg")

            try:
                # Open the images
                with Image.open(x_r_jpg) as img_r, Image.open(x_jpg) as img:
                    h_r, w_r = img_r.height, img_r.width
                    h, w = img.height, img.width

                    aspect_ratio_r = h_r / w_r
                    aspect_ratio = h / w

                    if aspect_ratio_r >= aspect_ratio:
                        print(x)
                    else:
                        # Calculate desired height using integer division
                        desired_height = (h_r * w) // w_r

                        # Crop the image
                        cropped_img = img.crop((0, 0, w, desired_height))

                        # Save the cropped image
                        cropped_filename = os.path.join(output_folder, f"{x}_c.jpg")
                        cropped_img.save(cropped_filename)

                        # Copy the reference image to the output folder
                        shutil.copy(x_r_jpg, os.path.join(output_folder, f"{x}_r.jpg"))
            except Exception as e:
                print(f"Error processing {x}: {e}")

# Example usage:
process_images('dataset_watermark', 'dataset_watermark_cropped')