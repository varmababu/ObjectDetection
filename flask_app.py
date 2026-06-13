import base64
import importlib
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image

load_dotenv()

# =========================
# APP SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# =========================
# YOLO CONFIG
# =========================

YOLO = importlib.import_module("ultralytics").YOLO

MODEL_WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    "models/best.pt"
)

MODEL = None


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


DEVICE = parse_device(
    os.getenv("YOLO_DEVICE", "cpu")
)


def get_model():
    global MODEL

    if MODEL is None:
        print(f"Loading model: {MODEL_WEIGHTS}")
        MODEL = YOLO(MODEL_WEIGHTS)

    return MODEL


# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template(
        "index.html",
        model_path=MODEL_WEIGHTS,
        device=DEVICE
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "weights": MODEL_WEIGHTS,
        "device": str(DEVICE)
    })


@app.route("/ping")
def ping():
    return jsonify({
        "status": "alive"
    })


# =========================
# INFERENCE
# =========================

def run_inference(image_bytes):
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    model = get_model()

    results = model.predict(
        source=image,
        imgsz=640,
        device=DEVICE,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    out_image = Image.fromarray(
        annotated[:, :, ::-1]
    )

    buffer = io.BytesIO()

    out_image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    detections = []

    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        x1, y1, x2, y2 = [
            float(v)
            for v in box.xyxy[0].tolist()
        ]

        detections.append(
            {
                "class_id": cls_id,
                "label": names.get(
                    cls_id,
                    str(cls_id)
                ),
                "confidence": round(conf, 4),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            }
        )

    return {
        "count": len(detections),
        "detections": detections,
        "image_base64": encoded_image
    }


# =========================
# IMAGE UPLOAD
# =========================

@app.route("/predict/upload", methods=["POST"])
def predict_upload():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    image_bytes = file.read()

    if not image_bytes:
        return jsonify({
            "error": "Uploaded file is empty"
        }), 400

    try:
        print("Starting upload inference...")

        response = run_inference(
            image_bytes
        )

        print("Upload inference complete")

        return jsonify(response)

    except Exception as exc:
        print("UPLOAD ERROR:", str(exc))

        return jsonify({
            "error": str(exc)
        }), 500


# =========================
# CAMERA INFERENCE
# =========================

@app.route("/predict/camera", methods=["POST"])
def predict_camera():

    payload = request.get_json(
        silent=True
    ) or {}

    image_data = payload.get("image")

    if not image_data:
        return jsonify({
            "error": "Missing camera frame"
        }), 400

    try:
        _, encoded = image_data.split(",", 1)

        image_bytes = base64.b64decode(
            encoded
        )

        print("Starting camera inference...")

        response = run_inference(
            image_bytes
        )

        print("Camera inference complete")

        return jsonify(response)

    except Exception as exc:
        print("CAMERA ERROR:", str(exc))

        return jsonify({
            "error": str(exc)
        }), 500


# =========================
# START SERVER
# =========================

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )