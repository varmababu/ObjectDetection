const uploadTab = document.getElementById("uploadTab");
const cameraTab = document.getElementById("cameraTab");
const uploadPane = document.getElementById("uploadPane");
const cameraPane = document.getElementById("cameraPane");

const imageFile = document.getElementById("imageFile");
const dropzone = document.querySelector(".dropzone");
const dropText = document.getElementById("dropText");
const uploadBtn = document.getElementById("uploadBtn");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const cameraStream = document.getElementById("cameraStream");
const captureCanvas = document.getElementById("captureCanvas");

const statusEl = document.getElementById("status");
const resultImage = document.getElementById("resultImage");
const countValue = document.getElementById("countValue");
const detectionList = document.getElementById("detectionList");

let mediaStream = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setTab(active) {
  const uploadActive = active === "upload";

  uploadTab.classList.toggle("active", uploadActive);
  cameraTab.classList.toggle("active", !uploadActive);
  uploadPane.classList.toggle("active", uploadActive);
  cameraPane.classList.toggle("active", !uploadActive);
}

function renderResult(data) {
  resultImage.src = `data:image/jpeg;base64,${data.image_base64}`;
  countValue.textContent = String(data.count);

  detectionList.innerHTML = "";
  if (!data.detections.length) {
    const li = document.createElement("li");
    li.textContent = "No objects detected.";
    detectionList.appendChild(li);
    return;
  }

  data.detections.forEach((d) => {
    const li = document.createElement("li");
    li.textContent = `${d.label} | conf=${d.confidence} | bbox=${d.bbox.join(", ")}`;
    detectionList.appendChild(li);
  });
}

async function postUpload() {
  const file = imageFile.files[0];
  if (!file) {
    setStatus("Choose an image first.");
    return;
  }

  if (!file.type.startsWith("image/")) {
    setStatus("Please upload a valid image file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setStatus("Running inference on uploaded image...");
  try {
    const res = await fetch("/predict/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Upload inference failed");
    }
    renderResult(data);
    setStatus("Upload inference complete.");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

function setSelectedFile(file) {
  if (!file) {
    dropText.textContent = "Drop image here or click to browse";
    return;
  }
  dropText.textContent = `Selected: ${file.name}`;
}

function bindDropzone() {
  const prevent = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      prevent(e);
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      prevent(e);
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (!files || !files.length) {
      return;
    }
    imageFile.files = files;
    setSelectedFile(files[0]);
    postUpload();
  });
}

async function startCamera() {
  if (mediaStream) {
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraStream.srcObject = mediaStream;
    setStatus("Camera started.");
  } catch (err) {
    setStatus(`Camera error: ${err.message}`);
  }
}

function stopCamera() {
  if (!mediaStream) {
    return;
  }

  mediaStream.getTracks().forEach((track) => track.stop());
  cameraStream.srcObject = null;
  mediaStream = null;
  setStatus("Camera stopped.");
}

async function captureAndPredict() {
  if (!mediaStream) {
    setStatus("Start camera first.");
    return;
  }

  const w = cameraStream.videoWidth;
  const h = cameraStream.videoHeight;
  if (!w || !h) {
    setStatus("Camera not ready yet.");
    return;
  }

  captureCanvas.width = w;
  captureCanvas.height = h;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(cameraStream, 0, 0, w, h);
  const image = captureCanvas.toDataURL("image/jpeg", 0.9);

  setStatus("Running inference on camera frame...");
  try {
    const res = await fetch("/predict/camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Camera inference failed");
    }
    renderResult(data);
    setStatus("Camera inference complete.");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

uploadTab.addEventListener("click", () => setTab("upload"));
cameraTab.addEventListener("click", () => setTab("camera"));
uploadBtn.addEventListener("click", postUpload);
imageFile.addEventListener("change", () => {
  const file = imageFile.files[0];
  setSelectedFile(file);
  if (file) {
    postUpload();
  }
});
startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);
captureBtn.addEventListener("click", captureAndPredict);

bindDropzone();
window.addEventListener("beforeunload", stopCamera);
