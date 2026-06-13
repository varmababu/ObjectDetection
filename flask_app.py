import base64
import importlib
import io
import os
from glob import glob
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def resolve_weights(explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    candidates = [
        "runs/detect/slibr_train/weights/best.pt",
        "runs/detect/runs/slibr_train/weights/best.pt",
        "runs/detect/runs/detect/slibr_quick_train/weights/best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    best_files = glob(os.path.join("runs", "**", "weights", "best.pt"), recursive=True)
    if best_files:
        best_files.sort(key=os.path.getmtime, reverse=True)
        return best_files[0]

    raise FileNotFoundError("No trained weights found. Train a model first.")


def parse_device(device_arg):
    if device_arg == "auto":
        try:
            torch = importlib.import_module("torch")
            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if isinstance(device_arg, str) and device_arg.isdigit():
        return int(device_arg)
    return device_arg


app = Flask(__name__, static_folder="static", template_folder="templates")
YOLO = importlib.import_module("ultralytics").YOLO
MODEL_WEIGHTS = resolve_weights(os.getenv("YOLO_WEIGHTS"))
MODEL = YOLO(MODEL_WEIGHTS)
DEVICE = parse_device(os.getenv("YOLO_DEVICE", "auto"))


@app.route("/")
def index():
    return render_template("index.html", model_path=MODEL_WEIGHTS, device=DEVICE)


def run_inference(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = MODEL.predict(source=image, imgsz=640, device=DEVICE, conf=0.25, verbose=False)
    result = results[0]

    annotated = result.plot()
    out_image = Image.fromarray(annotated[:, :, ::-1])
    buffer = io.BytesIO()
    out_image.save(buffer, format="JPEG", quality=90)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    detections = []
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "class_id": cls_id,
                "label": names.get(cls_id, str(cls_id)),
                "confidence": round(conf, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    return {
        "count": len(detections),
        "detections": detections,
        "image_base64": encoded_image,
    }


@app.route("/predict/upload", methods=["POST"])
def predict_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file name."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        response = run_inference(image_bytes)
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict/camera", methods=["POST"])
def predict_camera():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing camera frame."}), 400

    try:
        header, encoded = image_data.split(",", 1)
        _ = header
        image_bytes = base64.b64decode(encoded)
        response = run_inference(image_bytes)
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "weights": MODEL_WEIGHTS, "device": str(DEVICE)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
