from pathlib import Path
from collections import Counter

# مسار مجموعة البيانا
DATASET_DIR = Path("cleaned_dataset")
# امتدادات الصور
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

print("=" * 55)
print("INDUSTRIAL DEFECT DATASET AUDIT")
print("=" * 55)

# التأكد من وجود المجلد
if not DATASET_DIR.exists():
    print("ERROR: Dataset folder was not found!")
    print("Current working directory:", Path.cwd())
    print("Please check the dataset folder path.")

else:
    print("\nDataset location:", DATASET_DIR.resolve())

    # جمع الملفات والمجلدات
    all_files = [p for p in DATASET_DIR.rglob("*") if p.is_file()]
    all_dirs = [p for p in DATASET_DIR.rglob("*") if p.is_dir()]

    # حساب أنواع الملفات
    extensions = Counter(
        p.suffix.lower() for p in all_files
    )

    # حساب الصور
    images = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("\n1. DATASET SUMMARY")
    print("-" * 40)
    print("Total files:", len(all_files))
    print("Total folders:", len(all_dirs))
    print("Total images:", len(images))

    print("\n2. FILE TYPES")
    print("-" * 40)

    for ext, count in extensions.most_common():
        print(f"{ext or 'No extension'}: {count}")

    print("\n3. FOLDER STRUCTURE")
    print("-" * 40)

    for folder in sorted(all_dirs):
        relative_path = folder.relative_to(DATASET_DIR)
        print(relative_path)

    print("\n4. IMAGE CLASS COUNTS")
    print("-" * 40)

    class_counts = Counter(
        p.parent.name for p in images
    )

    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count} images")

    print("\n5. SAMPLE IMAGE PATHS")
    print("-" * 40)

    for image_path in images[:10]:
        print(image_path.relative_to(DATASET_DIR))

    print("\n6. TABULAR DATA FILES")
    print("-" * 40)

    tabular_extensions = {".csv", ".xlsx", ".xls"}

    tabular_files = [
        p for p in all_files
        if p.suffix.lower() in tabular_extensions
    ]

    if tabular_files:
        for file in tabular_files:
            print(file.relative_to(DATASET_DIR))
    else:
        print("No CSV or Excel files found.")

    print("\n" + "=" * 55)
    print("DATASET AUDIT COMPLETED")
    print("=" * 55)
    