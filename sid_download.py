import os
import gc
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import io
def download_sid_images(
    output_dir: str = "data/SID_dataset",
    num_per_class: int = 2000,
    split: str = "train"
):
    """
    Download images from SID_set dataset organized by label.
    Args:
        output_dir: Directory to save images
        num_per_class: Number of images to download per class
        split: Dataset split to use (default: "train")
    """
    print(f"Loading SID_set dataset (split: {split})...")
    dataset = load_dataset("saberzl/SID_set", split=split, streaming=True)
    # Create output directories for each label (only authentic and fully_synthetic)
    label_names = {
        0: "authentic",
        1: "fully_synthetic"
    }
    for label, name in label_names.items():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    # Track counts per label
    counts = {0: 0, 1: 0}
    target_count = num_per_class
    print(f"\nDownloading {num_per_class} images per class...")
    print(f"Target: {num_per_class * 2} total images\n")
    # Progress bars for each class
    pbar_authentic = tqdm(total=target_count, desc="Authentic (0)", position=0)
    pbar_synthetic = tqdm(total=target_count, desc="Fully Synthetic (1)", position=1)
    pbars = {0: pbar_authentic, 1: pbar_synthetic}
    dataset_iter = iter(dataset)
    try:
        while True:
            try:
                example = next(dataset_iter)
            except StopIteration:
                break
            label = example["label"]
            # Skip tampered images (label=2)
            if label not in label_names:
                continue
            # Check if we've collected enough for this class
            if counts[label] >= target_count:
                # Check if all classes are complete
                if all(count >= target_count for count in counts.values()):
                    break
                continue
            # Get image and metadata
            img_id = example["img_id"]
            image = example["image"]
            # Save image
            label_name = label_names[label]
            output_path = os.path.join(output_dir, label_name, f"{img_id}.png")
            # Convert and save - properly handle PIL Image to avoid threading issues
            try:
                if isinstance(image, Image.Image):
                    # Copy image to avoid reference issues
                    img_copy = image.copy()
                    img_copy.save(output_path, format='PNG')
                    img_copy.close()
                    del img_copy
                else:
                    # Handle if image is not already PIL Image
                    img = Image.fromarray(image)
                    img.save(output_path, format='PNG')
                    img.close()
                    del img
                # Explicitly close original image if it's a PIL Image
                if isinstance(image, Image.Image):
                    image.close()
            except Exception as img_error:
                print(f"\nWarning: Failed to save {img_id}: {img_error}")
                continue
            counts[label] += 1
            pbars[label].update(1)
            # Clean up example dict
            del example
            del image
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\nError during download: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close progress bars
        for pbar in pbars.values():
            pbar.close()
        # Cleanup
        del dataset_iter
        del dataset
        gc.collect()
    # Print summary
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"Total images downloaded: {sum(counts.values())}")
    print(f"\nBreakdown by class:")
    for label, name in label_names.items():
        print(f"  {name} (label={label}): {counts[label]} images")
    print(f"\nImages saved to: {os.path.abspath(output_dir)}/")
    print("="*60)
    return counts
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download images from SID_set dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="SID_dataset",
        help="Directory to save downloaded images (default: SID_dataset)"
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=1000,
        help="Number of images to download per class (default: 1000)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (default: validation)"
    )
    args = parser.parse_args()
    try:
        # Download images
        counts = download_sid_images(
            output_dir=args.output_dir,
            num_per_class=args.num_per_class,
            split=args.split
        )
        # Force cleanup before exit
        del counts
        gc.collect()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    # Give background threads time to cleanup
    import time
    time.sleep(0.5)
    return 0
if __name__ == "__main__":
    import sys
    sys.exit(main())