import argparse
import importlib
import os
from glob import glob

from app import SPLIT_ROOT


def get_device():
    try:
        torch = importlib.import_module("torch")
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_device(device_arg):
    if device_arg == "auto":
        return get_device()
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def resolve_weights(weights_path):
    if os.path.exists(weights_path):
        return weights_path

    candidates = [
        "runs/detect/slibr_train/weights/best.pt",
        "runs/detect/runs/slibr_train/weights/best.pt",
        "runs/detect/runs/detect/slibr_quick_train/weights/best.pt",
    ]

    for c in candidates:
        if os.path.exists(c):
            print(f"Weights not found at '{weights_path}', using '{c}'")
            return c

    best_files = glob(os.path.join("runs", "**", "weights", "best.pt"), recursive=True)
    if best_files:
        best_files.sort(key=os.path.getmtime, reverse=True)
        print(f"Weights not found at '{weights_path}', using latest '{best_files[0]}'")
        return best_files[0]

    raise FileNotFoundError(
        f"Weights file not found: {weights_path}. No fallback best.pt found under runs/."
    )


def evaluate(args):
    YOLO = importlib.import_module("ultralytics").YOLO

    data_yaml = os.path.join(SPLIT_ROOT, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Missing dataset YAML: {data_yaml}. Run app.py once or run train.py first."
        )

    device = parse_device(args.device)

    weights = resolve_weights(args.weights)
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split="test", imgsz=args.imgsz, batch=args.batch, device=device)

    print("Test run completed.")
    print(metrics)


def predict(args):
    YOLO = importlib.import_module("ultralytics").YOLO

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Prediction source not found: {args.source}")

    device = parse_device(args.device)
    weights = resolve_weights(args.weights)
    model = YOLO(weights)
    model.predict(source=args.source, imgsz=args.imgsz, device=device, save=True)
    print("Prediction run completed. Outputs are under runs/detect/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test/evaluate YOLO model")
    parser.add_argument("--weights", default="runs/detect/slibr_train/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or gpu index like 0")
    parser.add_argument("--predict", action="store_true", help="Run prediction instead of test split evaluation")
    parser.add_argument("--source", default=os.path.join(SPLIT_ROOT, "images", "test"), help="Image/file/folder for prediction mode")

    args = parser.parse_args()
    if args.predict:
        predict(args)
    else:
        evaluate(args)
