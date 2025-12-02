#!/usr/bin/env python3
"""
Download a subset of the SID_set dataset from Hugging Face, organized by label.

This script:
- Streams the SID_set dataset from Hugging Face.
- Saves images as PNG files under <output_dir>/<label_name>/.
- Only keeps labels 0 (authentic) and 1 (fully_synthetic).
- Stops when a fixed number of images per class has been downloaded.
"""

import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import argparse


def download_sid_images(
    output_dir: str = "SID_dataset",
    num_per_class: int = 2000,
    split: str = "train",
) -> dict[int, int]:
    """
    Download images from the SID_set dataset and save them as PNG files
    organized in folders by label.

    Only labels 0 (authentic) and 1 (fully_synthetic) are kept.
    Label 2 (tampered) is skipped.

    Args:
        output_dir (str): Root directory where images will be saved.
            Images are stored under:
                <output_dir>/authentic/
                <output_dir>/fully_synthetic/
        num_per_class (int): Target number of images to download per class
            (for label 0 and label 1).
        split (str): Dataset split to use (e.g., "train", "validation").

    Returns:
        dict[int, int]: Mapping from label ID to number of images actually
            downloaded, e.g. {0: count_authentic, 1: count_fully_synthetic}.
    """
    print(f"Loading SID_set dataset (split: {split})...")
    dataset = load_dataset("saberzl/SID_set", split=split, streaming=True)

    # Map numeric labels to folder names (we only keep 0 and 1).
    label_names: dict[int, str] = {
        0: "authentic",
        1: "fully_synthetic",
    }

    # Ensure output subdirectories exist for each class we keep.
    for name in label_names.values():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    # Track how many images we have saved for each label.
    counts: dict[int, int] = {0: 0, 1: 0}
    target_count: int = num_per_class

    print(f"\nDownloading {num_per_class} images per class...")
    print(f"Target: {num_per_class * 2} total images\n")

    # Progress bars, one per class.
    pbar_authentic = tqdm(total=target_count, desc="Authentic (0)", position=0)
    pbar_synthetic = tqdm(total=target_count, desc="Fully Synthetic (1)", position=1)
    pbars: dict[int, tqdm] = {0: pbar_authentic, 1: pbar_synthetic}

    # Stream over the dataset until both classes are filled or the stream ends.
    for example in dataset:
        # Check if we are done for both classes.
        if all(count >= target_count for count in counts.values()):
            break

        label: int = int(example["label"])

        # Skip tampered images (label=2) or any unexpected label.
        if label not in label_names:
            continue

        # If this class is already full, skip to the next example.
        if counts[label] >= target_count:
            continue

        img_id: str = str(example["img_id"])
        image = example["image"]

        # Build output path:
        #   <output_dir>/<label_name>/<img_id>.png
        label_name: str = label_names[label]
        output_path: str = os.path.join(output_dir, label_name, f"{img_id}.png")

        # Save image as PNG (handle PIL.Image or array-like).
        if isinstance(image, Image.Image):
            image.save(output_path, format="PNG")
        else:
            img = Image.fromarray(image)
            img.save(output_path, format="PNG")

        # Update counters and progress bar.
        counts[label] += 1
        pbars[label].update(1)

    # Close progress bars.
    pbar_authentic.close()
    pbar_synthetic.close()

    # Print a summary of what was actually downloaded.
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Total images downloaded: {sum(counts.values())}")
    print("\nBreakdown by class:")
    for label, name in label_names.items():
        print(f"  {name} (label={label}): {counts[label]} images")
    print(f"\nImages saved to: {os.path.abspath(output_dir)}/")
    print("=" * 60)

    return counts


def main() -> int:
    """
    Parse command-line arguments and download a subset of the SID_set dataset.

    This is the CLI entry point. It:
      - Parses --output-dir, --num-per-class, --split.
      - Calls download_sid_images with those parameters.

    Returns:
        int: Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Download images from the SID_set dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="SID_dataset",
        help="Directory to save downloaded images (default: SID_dataset)",
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=1000,
        help="Number of images to download per class (default: 1000)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help='Dataset split to use (default: "validation")',
    )

    args = parser.parse_args()

    download_sid_images(
        output_dir=args.output_dir,
        num_per_class=args.num_per_class,
        split=args.split,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
