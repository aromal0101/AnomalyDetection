import cv2
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO
from spikingjelly.activation_based import neuron, functional, surrogate

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "snn_ucfcrime.pth"          # trained model
VIDEO_PATH = "test_video.mp4"            # input CCTV video
OUTPUT_PATH = "output_anomaly.mp4"

CLIP_LEN = 16
IMG_SIZE = 64
ANOMALY_THRESHOLD = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# PRINT DEVICE INFO
# =========================================================
print("========== DEVICE INFO ==========")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=================================")

# =========================================================
# SNN MODEL (SAME AS TRAINING)
# =========================================================
class SNNAnomalyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc = nn.Linear(16 * IMG_SIZE * IMG_SIZE, 1)

    def forward(self, clip):
        functional.reset_net(self)
        spike_sum = 0

        for t in range(clip.size(0)):
            x = self.conv(clip[t])
            x = self.lif(x)
            spike_sum += x

        spike_sum = spike_sum / clip.size(0)
        spike_sum = spike_sum.view(1, -1)
        return torch.sigmoid(self.fc(spike_sum))


# =========================================================
# LOAD MODELS
# =========================================================
detector = YOLO("yolov8n.pt")  # human detector

snn = SNNAnomalyNet().to(DEVICE)
snn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
snn.eval()

# =========================================================
# VIDEO SETUP
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

h, w, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20, (w, h))

# Buffer per detected person (simple ID by index)
person_buffers = {}

frame_id = 0

# =========================================================
# INFERENCE LOOP
# =========================================================
while True:
    results = detector(frame, classes=[0], verbose=False)

    if results[0].boxes is not None:
        for pid, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = box.int().tolist()
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crop = torch.tensor(crop).permute(2, 0, 1).float() / 255.0

            if pid not in person_buffers:
                person_buffers[pid] = deque(maxlen=CLIP_LEN)

            person_buffers[pid].append(crop)

            # When temporal clip is ready
            if len(person_buffers[pid]) == CLIP_LEN:
                clip = torch.stack(list(person_buffers[pid])).to(DEVICE)

                with torch.no_grad():
                    score = snn(clip).item()

                if score > ANOMALY_THRESHOLD:
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, f"ANOMALY {score:.2f}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

    out.write(frame)

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

cap.release()
out.release()

print(f"✅ Output saved as {OUTPUT_PATH}")

