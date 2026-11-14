from pathlib import Path
from PIL import Image
import shutil
import hashlib
import xml.etree.ElementTree as ET

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("cleaned_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def get_hash(image_path):
    sha = hashlib.sha256()
    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def clean_dataset():
    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder not found!")
        return
    
    if OUTPUT_DIR.exists():
        print("ERROR: cleaned_dataset already exists.")
        print("Rename the old folder before running again.")
        return

    OUTPUT_DIR.mkdir()

    total_checked = 0
    copied_images = 0
    corrupted_images = 0
    duplicate_images = 0
    missing_xml = 0
    invalid_xml = 0

    for split in ["train", "validation"]:

        images_dir = DATASET_DIR / split / "images"
        annotations_dir = DATASET_DIR / split / "annotations"

        if not images_dir.exists():
            print(f"Missing images folder: {split}")
            continue

        output_images = OUTPUT_DIR / split / "images"
        output_annotations = OUTPUT_DIR / split / "annotations"

        seen_hashes = set()

        # البحث عن ملفات XML داخل مجلد التعليقات التوضيحية
        xml_files = {}
        if annotations_dir.exists():
            for xml_file in annotations_dir.rglob("*.xml"):
                xml_files.setdefault(xml_file.stem, []).append(xml_file)

        print(f"\nProcessing: {split}")

        for img_path in images_dir.rglob("*"):

            if not img_path.is_file():
                continue

            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total_checked += 1

            relative_path = img_path.relative_to(images_dir)
            try:
                with Image.open(img_path) as img:
                    img.verify()

            except Exception:
                corrupted_images += 1
                print(f"Corrupted image: {img_path.name}")
                continue
            class_name = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "unclassified"
            )

            try:
                image_hash = get_hash(img_path)
            except OSError:
                corrupted_images += 1
                continue

            duplicate_key = (class_name, image_hash)

            if duplicate_key in seen_hashes:
                duplicate_images += 1
                print(f"Duplicate image: {img_path.name}")
                continue

            seen_hashes.add(duplicate_key)
            destination = output_images / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, destination)
            copied_images += 1
            matching_xml = xml_files.get(img_path.stem, [])

            if len(matching_xml) != 1:
                missing_xml += 1
                if len(matching_xml) > 1:
                    print(f"Ambiguous XML: {img_path.stem}")
                else:
                    print(f"Missing XML: {img_path.stem}")
                continue

            xml_path = matching_xml[0]

            try:
                ET.parse(xml_path)

                xml_destination = (
                    output_annotations
                    / relative_path.parent
                    / f"{img_path.stem}.xml"
                )

                xml_destination.parent.mkdir(
                    parents=True, exist_ok=True
                )

                shutil.copy2(xml_path, xml_destination)

            except (ET.ParseError, OSError):
                invalid_xml += 1
                print(f"Invalid XML: {xml_path.name}")
    print("\n" + "=" * 50)
    print("DATASET CLEANING REPORT")
    print("=" * 50)

    print(f"Total images checked: {total_checked}")
    print(f"Healthy images copied: {copied_images}")
    print(f"Corrupted images skipped: {corrupted_images}")
    print(f"Duplicate images skipped: {duplicate_images}")
    print(f"Missing or ambiguous XML: {missing_xml}")
    print(f"Invalid XML: {invalid_xml}")

    print("=" * 50)
    print(f"Clean dataset saved in: {OUTPUT_DIR.resolve()}")
    print("CLEANING COMPLETED")


if __name__ == "__main__":
    clean_dataset()