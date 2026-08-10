import torch
import numpy as np
import librosa
from torch import nn
from scipy.signal import find_peaks
import os
from config import DEVICE, find_model_path

class ProsodyNet(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lstm = nn.LSTM(input_size=32*16, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_targets)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        B, C, H, W = x.size()
        x_seq = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        out, _ = self.lstm(x_seq)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def extract_prosody_features(audio_data, sr=16000):
    """Extract 5 acoustic prosody features: [Pitch, Intensity, Tempo, Pause Ratio, Stress]"""
    y = audio_data
    duration = len(y) / sr
    if duration <= 0:
        return np.zeros(5, dtype=np.float32)

    # 1. RMS Intensity
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms))

    # 2. Pitch (50-500 Hz human pitch range)
    if intensity > 0.005:
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = []
        for i in range(pitches.shape[1]):
            if mags[:, i].max() > 0.05:
                p = pitches[mags[:, i].argmax(), i]
                if 50 <= p <= 500:
                    pitch_vals.append(p)
        pitch_vals = np.array(pitch_vals)
        pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    else:
        pitch_vals = np.array([])
        pitch = 0.0

    # 3. Speaking Tempo (energy peaks per minute)
    if intensity > 0.005 and len(rms) > 1:
        rms_threshold = float(np.mean(rms) + 0.2 * np.std(rms))
        syllable_peaks, _ = find_peaks(rms, height=rms_threshold, distance=int(0.15 * sr / 512))
        tempo = float((len(syllable_peaks) / duration) * 60.0)
    else:
        tempo = 0.0

    # 4. Pause Silence Ratio
    if intensity > 0.005:
        intervals = librosa.effects.split(y, top_db=25)
        active_samples = sum([end - start for start, end in intervals]) if len(intervals) > 0 else 0
        pause_ratio = float(max(0.0, min(1.0, 1.0 - (active_samples / len(y)))))
    else:
        pause_ratio = 1.0

    # 5. Vocal Stress Pattern
    if len(pitch_vals) > 1 and pitch > 0:
        pitch_std = np.std(pitch_vals) / pitch
    else:
        pitch_std = 0.0

    rms_std = np.std(rms) if len(rms) > 1 else 0.0
    stress = float(np.clip(pitch_std * 2.0 + rms_std * 10.0, 0.0, 1.0))

    return np.array([pitch, intensity, tempo, pause_ratio, stress], dtype=np.float32)

class ProsodyAnalyzer:
    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model = ProsodyNet().to(DEVICE)
        self.mean = np.zeros(5)
        self.std = np.ones(5)
        
        model_path = find_model_path('prosody_net_best.pth')
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.mean = checkpoint.get('mean', np.zeros(5))
                self.std = checkpoint.get('std', np.ones(5))
                print(f"[OK] Loaded ProsodyNet model from {model_path}")
            except Exception as e:
                print(f"[WARNING] Failed loading ProsodyNet model ({e}).")
        else:
            print("[WARNING] prosody_net_best.pth not found. Using initialized model.")
        self.model.eval()

    def analyze(self, audio_data, sr=16000):
        """Analyze audio numpy array and return prosody metrics list [pitch, intensity, tempo, pause_ratio, stress]"""
        features = extract_prosody_features(audio_data, sr=sr)
        return features.tolist()
