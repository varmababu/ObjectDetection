# SLiBR Object Detection

This project prepares data, trains a YOLO model, evaluates results, and serves inference through a Flask web app.

## Files
- `app.py`: dataset checks, split creation, YAML generation, and optional training pipeline.
- `train.py`: CLI training entrypoint.
- `test.py`: CLI evaluation/prediction entrypoint.
- `flask_app.py`: web app API/UI for upload and camera inference.

## Setup
1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
Prepare dataset split and YAML:

```bash
python app.py
```

Train model:

```bash
python train.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8
```

Evaluate model:

```bash
python test.py
```

Run Flask app:

```bash
python flask_app.py
```

## Git Push Checklist
1. Ensure large dataset folders and model weights are ignored by `.gitignore`.
2. Confirm dependencies are in `requirements.txt`.
3. Initialize git (if not initialized):

```bash
git init
git add .
git commit -m "Initial project commit"
```

4. Add remote and push:

```bash
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
