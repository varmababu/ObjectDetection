# =========================
# SLIBR YOLO PIPELINE
# =========================

import os
import glob
import shutil
import random
import importlib
import yaml

# =========================
# PATHS (EDIT THIS)
# =========================
def resolve_raw_root():
    candidates = [
        os.path.join("SLiBR_dataset", "object_detection"),
        os.path.join("dataset", "object_detection"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]

RAW_ROOT = resolve_raw_root()
SPLIT_ROOT = "dataset/split"
OUT_STATS = "runs_stats"

IMAGES_DIR = os.path.join(RAW_ROOT, "images")
LABELS_DIR = os.path.join(RAW_ROOT, "labels")
CLASSES_PATH = os.path.join(RAW_ROOT, "classes.txt")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# UTIL FUNCTIONS
# =========================
def read_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]

def get_image_files(images_dir):
    image_files = []
    for p in glob.glob(os.path.join(images_dir, "*.*")):
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS:
            image_files.append(p)
    return sorted(image_files)

def check_yolo_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    try:
        c, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except:
        return False
    return (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)

# =========================
# 1. SANITY CHECK
# =========================
def sanity_check():
    print("\n🔍 Running Sanity Check...")

    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Missing classes file: {CLASSES_PATH}")
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Missing images directory: {IMAGES_DIR}")
    if not os.path.isdir(LABELS_DIR):
        raise FileNotFoundError(f"Missing labels directory: {LABELS_DIR}")

    class_names = read_classes(CLASSES_PATH)
    images = get_image_files(IMAGES_DIR)
    labels = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))

    invalid_label_lines = 0
    for lp in labels:
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not check_yolo_line(line):
                    invalid_label_lines += 1

    print(f"Images: {len(images)} | Labels: {len(labels)} | Classes: {len(class_names)}")
    if invalid_label_lines:
        print(f"⚠️ Invalid YOLO annotation lines: {invalid_label_lines}")

# =========================
# 2. DATASET STATS
# =========================
def dataset_stats():
    print("\n📊 Generating Dataset Stats...")

    pd = importlib.import_module("pandas")
    tqdm = importlib.import_module("tqdm").tqdm

    os.makedirs(OUT_STATS, exist_ok=True)
    read_classes(CLASSES_PATH)

    rows = []
    label_files = glob.glob(os.path.join(LABELS_DIR, "*.txt"))

    for lp in tqdm(label_files):
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if not check_yolo_line(line):
                    continue
                c, x, y, w, h = line.split()
                rows.append([int(c), float(w), float(h)])

    if not rows:
        print("No valid annotations found. Stats skipped.")
        return

    df = pd.DataFrame(rows, columns=["class_id", "w", "h"])
    df["area"] = df["w"] * df["h"]

    df.to_csv(os.path.join(OUT_STATS, "boxes.csv"), index=False)

    print("Stats saved.")

# =========================
# 3. TRAIN/VAL/TEST SPLIT
# =========================
def create_split():
    print("\n📂 Creating Dataset Split...")

    random.seed(42)

    if os.path.isdir(SPLIT_ROOT):
        shutil.rmtree(SPLIT_ROOT)

    imgs = get_image_files(IMAGES_DIR)
    pairs = []

    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(LABELS_DIR, f"{stem}.txt")
        if os.path.exists(lbl):
            pairs.append((img, lbl))

    random.shuffle(pairs)

    n = len(pairs)
    train = pairs[:int(0.7*n)]
    val   = pairs[int(0.7*n):int(0.9*n)]
    test  = pairs[int(0.9*n):]

    for split, data in zip(["train", "val", "test"], [train, val, test]):
        for img, lbl in data:
            os.makedirs(f"{SPLIT_ROOT}/images/{split}", exist_ok=True)
            os.makedirs(f"{SPLIT_ROOT}/labels/{split}", exist_ok=True)
            shutil.copy(img, f"{SPLIT_ROOT}/images/{split}")
            shutil.copy(lbl, f"{SPLIT_ROOT}/labels/{split}")

    print("Split Done.")

# =========================
# 4. CREATE YAML
# =========================
def create_yaml():
    print("\n🧾 Creating YAML...")

    names = read_classes(CLASSES_PATH)

    data_yaml = {
        "path": SPLIT_ROOT,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)}
    }

    with open(f"{SPLIT_ROOT}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    print("YAML Created.")

def get_training_device():
    try:
        torch = importlib.import_module("torch")
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# =========================
# 5. TRAIN BASELINE
# =========================
def train_baseline():
    print("\n🚀 Training Baseline...")

    YOLO = importlib.import_module("ultralytics").YOLO
    model = YOLO("yolov8m.pt")
    model.train(
        data=f"{SPLIT_ROOT}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=get_training_device()
    )

# =========================
# 6. TRAIN HYBRID
# =========================
def train_hybrid():
    print("\n🚀 Training Hybrid...")

    YOLO = importlib.import_module("ultralytics").YOLO
    model = YOLO("yolov8m.pt")
    model.train(
        data=f"{SPLIT_ROOT}/data.yaml",
        epochs=100,
        imgsz=768,
        batch=8,
        device=get_training_device()
    )

# =========================
# 7. MAIN PIPELINE
# =========================
def main():
    sanity_check()
    dataset_stats()
    create_split()
    create_yaml()

    # Training (uncomment when ready)
    # train_baseline()
    # train_hybrid()

if __name__ == "__main__":
    main()