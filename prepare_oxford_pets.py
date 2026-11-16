"""Sort the Oxford-IIIT Pet dataset into one folder per breed.

The original dataset stores every image in a single folder and encodes the
breed in the file name (for example "Abyssinian_100.jpg" or
"american_bulldog_10.jpg"). training.py expects one subfolder per class, so
this script copies each image into a folder named after its breed.

It also:
- Uses the official trainval/test split from the annotations folder when it
  is available, so the test set stays untouched during training
- Converts every image to RGB JPEG (a few originals are grayscale, RGBA, or
  PNG data saved with a .jpg extension) and skips files that can't be opened
- Writes metadata.csv (file, breed, species, split) for data analysis

Download images.tar.gz and annotations.tar.gz from
https://www.robots.ox.ac.uk/~vgg/data/pets/ and extract both, then run:

    python prepare_oxford_pets.py --images-dir images \
        --annotations-dir annotations --output-dir pets

Train on the result with:

    python training.py --data-dir pets/train
"""

import argparse
import csv
import os

from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# In the dataset, cat breed file names start with an uppercase letter and dog
# breed file names start with a lowercase letter. The annotations file has the
# species explicitly, so this is only a fallback when annotations are missing.
SPECIES_BY_ID = {"1": "cat", "2": "dog"}


def breed_from_filename(fileName):
    """Return the breed for a file name like 'american_bulldog_10.jpg'."""
    stem = os.path.splitext(fileName)[0]
    return stem.rsplit("_", 1)[0].lower()


def species_from_filename(fileName):
    return "cat" if fileName[0].isupper() else "dog"


def read_split_file(splitPath):
    """Return {imageStem: speciesName} from trainval.txt or test.txt."""
    entries = {}
    with open(splitPath) as splitFile:
        for line in splitFile:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: <image name> <class id> <species id> <breed id>
            parts = line.split()
            entries[parts[0]] = SPECIES_BY_ID.get(parts[2], "unknown")
    return entries


def load_splits(annotationsDir):
    """Return {imageStem: (split, species)}, or None if annotations are missing."""
    if not annotationsDir:
        return None
    trainvalPath = os.path.join(annotationsDir, "trainval.txt")
    testPath = os.path.join(annotationsDir, "test.txt")
    if not (os.path.exists(trainvalPath) and os.path.exists(testPath)):
        print("Split files not found in the annotations folder, using one 'all' folder.")
        return None

    splits = {}
    for stem, species in read_split_file(trainvalPath).items():
        splits[stem] = ("train", species)
    for stem, species in read_split_file(testPath).items():
        splits[stem] = ("test", species)
    return splits


def save_as_rgb_jpeg(sourcePath, destinationPath):
    """Copy an image as an RGB JPEG. Return False if it can't be opened."""
    try:
        with Image.open(sourcePath) as image:
            image.convert("RGB").save(destinationPath, "JPEG", quality=95)
        return True
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare the Oxford-IIIT Pet dataset.")
    parser.add_argument("--images-dir", required=True, help="Extracted images folder")
    parser.add_argument(
        "--annotations-dir",
        default=None,
        help="Extracted annotations folder (for the official train/test split)",
    )
    parser.add_argument("--output-dir", default="pets", help="Where to write the sorted images")
    args = parser.parse_args()

    splits = load_splits(args.annotations_dir)

    fileNames = sorted(
        name for name in os.listdir(args.images_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )

    metadataRows = []
    skippedFiles = []
    notInSplit = 0

    for fileName in fileNames:
        stem = os.path.splitext(fileName)[0]
        breed = breed_from_filename(fileName)

        if splits is None:
            split = "all"
            species = species_from_filename(fileName)
        elif stem in splits:
            split, species = splits[stem]
        else:
            # A handful of images aren't listed in either split file.
            notInSplit += 1
            continue

        breedDir = os.path.join(args.output_dir, split, breed)
        os.makedirs(breedDir, exist_ok=True)
        outputName = f"{stem}.jpg"
        sourcePath = os.path.join(args.images_dir, fileName)
        destinationPath = os.path.join(breedDir, outputName)

        if not save_as_rgb_jpeg(sourcePath, destinationPath):
            skippedFiles.append(fileName)
            continue

        metadataRows.append(
            {"file": outputName, "breed": breed, "species": species, "split": split}
        )

    os.makedirs(args.output_dir, exist_ok=True)
    metadataPath = os.path.join(args.output_dir, "metadata.csv")
    with open(metadataPath, "w", newline="") as metadataFile:
        writer = csv.DictWriter(metadataFile, fieldnames=["file", "breed", "species", "split"])
        writer.writeheader()
        writer.writerows(metadataRows)

    breedCount = len({row["breed"] for row in metadataRows})
    print(f"Prepared {len(metadataRows)} images across {breedCount} breeds in {args.output_dir}")
    for splitName in sorted({row["split"] for row in metadataRows}):
        splitCount = sum(1 for row in metadataRows if row["split"] == splitName)
        print(f"  {splitName}: {splitCount} images")
    if notInSplit:
        print(f"Left out {notInSplit} images that aren't in either split file")
    if skippedFiles:
        print(f"Skipped {len(skippedFiles)} unreadable files: {', '.join(skippedFiles)}")
    print(f"Wrote metadata to {metadataPath}")


if __name__ == "__main__":
    main()
