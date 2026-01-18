import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from spikingjelly.activation_based import functional, layer, surrogate
from spikingjelly.activation_based.neuron import LIFNode
from sklearn.metrics import roc_curve

# --- Configuration ---
SEQUENCE_LENGTH = 10
IMG_SIZE = (112, 112)
BATCH_SIZE = 8
NUM_CLASSES = 2
EPOCHS = 20
LEARNING_RATE = 1e-3

# ==========================================
# 1. DATASET
# ==========================================
class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None, stride=5):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.stride = stride
        self.clips = []
        self.labels = []
        
        for folder in self.root_dir.iterdir():
            if not folder.is_dir(): continue
            images = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png'))
            if len(images) < sequence_length: continue
            label = 0 if 'normal' in folder.name.lower() else 1
            for i in range(0, len(images) - sequence_length + 1, stride):
                self.clips.append(images[i:i + sequence_length])
                self.labels.append(label)
        print(f"Loaded {len(self.clips)} clips")
    
    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        frames = [self.transform(Image.open(p).convert('RGB')) for p in self.clips[idx]]
        return torch.stack(frames), self.labels[idx]

# ==========================================
# 2. CORRECT MULTI-STEP SNN
# ==========================================
class MultiStepSNNClassifier(nn.Module):
    def __init__(self, timesteps=10):
        super().__init__()
        self.timesteps = timesteps
        
        # Encoder layers (ANN -> LIF pattern)
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
        )
        self.lif1 = LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=False)
        
        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 64),
        )
        self.lif2 = LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=False)
        
        # FIXED: Temporal processing as 2D conv over time (no problematic Conv3D)
        self.temp_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=False)
        
        # Classification
        self.classifier = layer.SeqToANNContainer(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
        )
        self.lif4 = LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=False)
        self.fc_out = nn.Linear(256, NUM_CLASSES)
        
        # Enable spike monitoring
        functional.set_monitor_enable(True)
    
    def forward(self, x):
        # Input: (B, T, C, H, W) -> (T, B, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)
        
        # Encoder
        x = functional.seq_to_ann_forward(x, self.conv1)
        x = self.lif1(x)
        
        x = functional.seq_to_ann_forward(x, self.conv2)
        x = self.lif2(x)
        
        # FIXED: Temporal processing - process each timestep with shared conv
        T, B, C, H, W = x.shape
        x_flat = x.reshape(T * B, C, H, W)
        temporal_features = self.temp_conv(x_flat)
        temporal_features = temporal_features.reshape(T, B, -1, H, W)
        x = self.lif3(temporal_features)
        
        # Classification
        features = functional.seq_to_ann_forward(x, self.classifier)
        spike_rates = self.lif4(features)
        
        # Average over time
        firing_rate = spike_rates.mean(dim=0)
        return self.fc_out(firing_rate)
    
    def reset(self):
        functional.reset_net(self)

# ==========================================
# 3. TRAINING WITH SPIKE METRICS
# ==========================================
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    total_spikes = 0
    
    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)
        model.reset()
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(clips)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # FIXED: Correct spike counting
        for module in model.modules():
            if hasattr(module, 'monitor'):
                if 'h' in module.monitor:
                    total_spikes += module.monitor['h'].sum().item()
    
    avg_spikes = total_spikes / (len(train_loader.dataset) * SEQUENCE_LENGTH)
    return total_loss/len(train_loader), avg_spikes

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_scores = []
    all_labels = []
    total_spikes = 0
    
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            model.reset()
            
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Spike counting
            for module in model.modules():
                if hasattr(module, 'monitor'):
                    if 'h' in module.monitor:
                        total_spikes += module.monitor['h'].sum().item()
    
    # FIXED: ROC-based threshold calibration
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
    avg_spikes = total_spikes / (len(val_loader.dataset) * SEQUENCE_LENGTH)
    
    # Calculate optimal threshold (EER-like)
    if len(all_labels) > 0:
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    else:
        eer_threshold = 0.5
    
    return total_loss/len(val_loader), auc, avg_spikes, eer_threshold

# ==========================================
# 4. SPIKE-CAM LOCALIZATION
# ==========================================
class SpikeCAM:
    def __init__(self, model):
        self.model = model
        self.spike_maps = []
        self.hook = model.lif3.register_forward_hook(self.save_spikes)
    
    def save_spikes(self, module, input, output):
        # Save spike activations from temporal layer
        self.spike_maps.append(output.detach())
    
    def __call__(self, x):
        self.spike_maps.clear()
        self.model.reset()
        
        with torch.no_grad():
            _ = self.model(x)
        
        if self.spike_maps:
            # Average spike map over time
            spike_map = torch.stack(self.spike_maps).mean(dim=0)
            return spike_map
        return None
    
    def remove(self):
        self.hook.remove()

def localize_with_spikecam(model, frames_tensor, device):
    """Localize using spike activation maps"""
    spikecam = SpikeCAM(model)
    frames = frames_tensor.unsqueeze(0).to(device)
    
    spike_map = spikecam(frames)
    spikecam.remove()
    
    if spike_map is not None:
        # Average over batch and channels
        heatmap = spike_map[0].mean(dim=0).cpu().numpy()  # (H, W)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            heatmap = np.uint8(255 * heatmap)
            
            # Find regions with high spike activity
            _, binary = cv2.threshold(heatmap, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 50:
                    x, y, w, h = cv2.boundingRect(largest)
                    # Clamp to image
                    H, W = heatmap.shape
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, W-x), min(h, H-y)
                    
                    # Scale to original
                    scale_x = IMG_SIZE[1] / W
                    scale_y = IMG_SIZE[0] / H
                    return (int(x*scale_x), int(y*scale_y), 
                            int(w*scale_x), int(h*scale_y))
    return None

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = VideoDataset(root_dir='Train', 
                               sequence_length=SEQUENCE_LENGTH,
                               transform=transform, 
                               stride=SEQUENCE_LENGTH//2)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = MultiStepSNNClassifier(timesteps=SEQUENCE_LENGTH).to(device)
    
    # Class-balanced loss
    labels = [full_dataset.labels[i] for i in train_dataset.indices]
    normal_count = labels.count(0)
    anomaly_count = labels.count(1)
    class_weights = torch.tensor([1.0, normal_count/max(anomaly_count, 1)], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training
    print("Training SNN...")
    best_auc = 0
    best_threshold = 0.5
    
    for epoch in range(EPOCHS):
        train_loss, train_spikes = train_epoch(model, train_loader, criterion, 
                                              optimizer, device, scaler)
        val_loss, val_auc, val_spikes, val_threshold = validate(model, val_loader, 
                                                               criterion, device)
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.3f}, Val AUC={val_auc:.3f}")
        print(f"  Spikes: {train_spikes:.1f}/timestep, Threshold: {val_threshold:.3f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_threshold = val_threshold
            torch.save({
                'model': model.state_dict(),
                'auc': val_auc,
                'threshold': val_threshold,
                'spikes': val_spikes
            }, "best_snn_model.pth")
    
    # Load and test
    checkpoint = torch.load("best_snn_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"\nBest AUC: {checkpoint['auc']:.3f}, Threshold: {checkpoint['threshold']:.3f}")
    print(f"Energy (spikes/timestep): {checkpoint['spikes']:.1f}")
    
    # Inference
    model.eval()
    test_idx = 0
    test_clip, test_label = val_dataset[test_idx]
    
    with torch.no_grad():
        model.reset()
        outputs = model(test_clip.unsqueeze(0).to(device))
        anomaly_score = F.softmax(outputs, dim=1)[0, 1].item()
    
    print(f"\nTest: {'ANOMALY' if test_label else 'NORMAL'}")
    print(f"Score: {anomaly_score:.3f}, Decision: {'ANOMALY' if anomaly_score > checkpoint['threshold'] else 'NORMAL'}")
    
    # Localization
    if anomaly_score > checkpoint['threshold']:
        bbox = localize_with_spikecam(model, test_clip, device)
        
        # Visualize
        last_frame = test_clip[-1].permute(1, 2, 0).cpu().numpy()
        last_frame = (last_frame * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        last_frame = last_frame.astype(np.uint8)[:, :, ::-1]
        
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(last_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(last_frame, f"Score: {anomaly_score:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(last_frame, f"Spikes: {checkpoint['spikes']:.0f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite("result.jpg", last_frame)
        print("Saved result.jpg")
