import argparse
import importlib
import os

from app import SPLIT_ROOT, create_yaml, create_split, sanity_check


def ensure_dataset_yaml():
    data_yaml = os.path.join(SPLIT_ROOT, "data.yaml")
    if os.path.exists(data_yaml):
        return data_yaml

    print("Dataset YAML not found. Preparing split and YAML...")
    sanity_check()
    create_split()
    create_yaml()
    return data_yaml


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


def train(args):
    YOLO = importlib.import_module("ultralytics").YOLO

    data_yaml = ensure_dataset_yaml()
    device = parse_device(args.device)

    print(f"Training with model={args.model}, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={device}")
    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model on SLIBR split dataset")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or gpu index like 0")
    parser.add_argument("--project", default="runs", help="Output project directory")
    parser.add_argument("--name", default="slibr_train", help="Run name")

    args = parser.parse_args()
    train(args)
