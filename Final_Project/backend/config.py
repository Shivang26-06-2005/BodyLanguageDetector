import os
import torch

# Device Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories to search for model weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_SEARCH_PATHS = [
    os.path.join(PROJECT_ROOT, "models_weights"),
    BASE_DIR,
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "..", "Pulse_Mind", "PulseMind-AI-Based-Realtime-Emotion-Detector"),
    os.path.join(PROJECT_ROOT, "..", "Pulse_Mind", "PulseMind-AI-Based-Realtime-Emotion-Detector", "Final_Models"),
]

def find_model_path(filename):
    """Find absolute path for a model weights file"""
    for folder in MODEL_SEARCH_PATHS:
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    return None

# Emotion Labels
VIDEO_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SOUND_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
