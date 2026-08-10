import torch
import numpy as np
import librosa
from torch import nn
import os
from config import DEVICE, SOUND_EMOTIONS, find_model_path

class TwoLayerCNN(nn.Module):
    def __init__(self, input_height=40, input_width=216, num_classes=7):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        fc1_in = 32 * (input_height // 4) * (input_width // 4)
        self.fc1 = nn.Linear(fc1_in, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def extract_mfcc_features(audio_data, sr=16000, max_len=216):
    """Extract MFCC features for sound emotion detection"""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

class SoundEmotionDetector:
    def __init__(self):
        self.emotions = SOUND_EMOTIONS
        # NOTE: This model (TESS-trained) has biased class weights.
        # We use it as-is and rely on energy gates to suppress false positives.
        # For interview use: happy/neutral/angry are the most reliable outputs.
        self.load_model()

    def load_model(self):
        model_path = find_model_path('sound_emotion_model.pth')
        input_height, input_width = 40, 216
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=DEVICE)
                if 'fc1.weight' in checkpoint:
                    fc1_in = checkpoint['fc1.weight'].shape[1]
                    # Calculate input_width from fc1_in: fc1_in = 32 * (10) * (input_width // 4)
                    input_width = (fc1_in // (32 * 10)) * 4
                
                self.model = TwoLayerCNN(input_height, input_width, 7).to(DEVICE)
                self.model.load_state_dict(checkpoint)
                print(f"[OK] Loaded Sound Emotion CNN model from {model_path} ({input_height}x{input_width})")
            except Exception as e:
                print(f"[WARNING] Failed loading Sound Emotion model ({e}). Using initialized model.")
                self.model = TwoLayerCNN(40, 92, 7).to(DEVICE)
        else:
            print("[WARNING] sound_emotion_model.pth not found. Using initialized model.")
            self.model = TwoLayerCNN(40, 92, 7).to(DEVICE)
        self.model.eval()

    def predict_audio(self, audio_data, sr=16000):
        """Predict emotion from float32 audio numpy array — mirrors FinalFuse.py logic exactly"""
        # Need at least 0.5s of audio (8000 samples at 16kHz) — same as FinalFuse
        if len(audio_data) < 8000:
            return 'neutral'

        rms = librosa.feature.rms(y=audio_data)[0]
        intensity = float(np.mean(rms))

        # Hard silence gate — below this is ambient mic noise, not speech (FinalFuse line 704)
        if intensity < 0.015:
            return 'neutral'

        mfcc = extract_mfcc_features(audio_data, sr=sr, max_len=216)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(mfcc_tensor)
            pred = torch.argmax(output, 1).item()
            emotion = self.emotions[pred]

            # Safeguard: fear on borderline energy is almost always mic noise
            if emotion == 'fear' and intensity < 0.025:
                emotion = 'neutral'

            # Safeguard: disgust rarely occurs in interviews — suppress on low energy
            if emotion == 'disgust' and intensity < 0.03:
                emotion = 'neutral'

        return emotion
