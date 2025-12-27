import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from spikingjelly.activation_based import neuron, functional, surrogate

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT = "/home/aromal-project/Downloads/project/UCF-Crime"
CLIP_LEN = 16
IMG_SIZE = 64
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# SNN MODEL
# =========================================================
class SNNAnomalyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc = nn.Linear(16 * IMG_SIZE * IMG_SIZE, 1)

    def forward(self, clip):
        """
        clip shape: [T, 3, H, W]
        """
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
# DATASET
# =========================================================
class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.detector = YOLO("yolov8n.pt")

        # -------- NORMAL --------
        normal_dir = os.path.join(root_dir, "normal")
        if not os.path.exists(normal_dir):
            raise FileNotFoundError(f"Missing folder: {normal_dir}")

        for v in os.listdir(normal_dir):
            if v.endswith(".mp4"):
                self.samples.append((os.path.join(normal_dir, v), 0))

        # -------- ANOMALY (many folders) --------
        anomaly_root = os.path.join(root_dir, "anomaly")
        if not os.path.exists(anomaly_root):
            raise FileNotFoundError(f"Missing folder: {anomaly_root}")

        for atype in os.listdir(anomaly_root):
            atype_path = os.path.join(anomaly_root, atype)
            if not os.path.isdir(atype_path):
                continue
            for v in os.listdir(atype_path):
                if v.endswith(".mp4"):
                    self.samples.append((os.path.join(atype_path, v), 1))

        if len(self.samples) == 0:
            raise RuntimeError("No videos found in dataset")

        print(f"Loaded {len(self.samples)} training videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)

        frames = []

        while len(frames) < CLIP_LEN:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector(frame, classes=[0], verbose=False)

            # --- SAFE CHECK ---
            if results[0].boxes is None:
                continue
            if results[0].boxes.xyxy.shape[0] == 0:
                continue

            x1, y1, x2, y2 = results[0].boxes.xyxy[0].int().tolist()
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crop = torch.tensor(crop).permute(2, 0, 1).float() / 255.0
            frames.append(crop)

        cap.release()

        # --- PAD IF NECESSARY ---
        if len(frames) == 0:
            frames = [torch.zeros(3, IMG_SIZE, IMG_SIZE)] * CLIP_LEN

        while len(frames) < CLIP_LEN:
            frames.append(frames[-1])

        frames = torch.stack(frames)
        label = torch.tensor(label, dtype=torch.float32)

        return frames, label


# =========================================================
# TRAINING
# =========================================================
def train():
    dataset = UCFCrimeDataset(DATASET_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # 🔥 REQUIRED FOR CUDA + YOLO
        drop_last=True
    )

    model = SNNAnomalyNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for clips, labels in loader:
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = []

            for i in range(clips.size(0)):
                outputs.append(model(clips[i]))

            outputs = torch.stack(outputs).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "snn_ucfcrime.pth")
    print("✅ Training complete. Model saved as snn_ucfcrime.pth")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train()

