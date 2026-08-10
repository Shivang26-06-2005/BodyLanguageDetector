import cv2
import torch
import numpy as np
import librosa
import pyaudio
import threading
import time
import asyncio
import base64
import json
from collections import deque
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

def get_file_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(project_root, filename),
        os.path.join(project_root, "Final_Models", filename),
        os.path.join(script_dir, "Final_Models", filename),
        os.path.abspath(filename),
        os.path.abspath(os.path.join("Final_Models", filename)),
        os.path.abspath(os.path.join("..", filename)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

# ============================
# Device Setup
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================
# Models (same as original)
# ============================
class FER_CNN(nn.Module):
    def __init__(self, num_classes):
        super(FER_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class TwoLayerCNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * (input_height // 4) * (input_width // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class ProsodyNet(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lstm = nn.LSTM(input_size=32 * 16, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_targets)
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        B, C, H, W = x.size()
        x_seq = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        out, _ = self.lstm(x_seq)
        return self.fc(out[:, -1, :])

# ============================
# Feature Extraction
# ============================
def extract_prosody_features(audio_data, sr=16000):
    y = audio_data
    duration = len(y) / sr
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = np.array([pitches[mags[:, i].argmax(), i] for i in range(pitches.shape[1])])
    pitch_vals = pitch_vals[pitch_vals > 0]
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms))
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except:
        tempo = 0.0
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    prev_end = 0
    for start, end in intervals:
        pauses.append((prev_end, start))
        prev_end = end
    pause_ratio = float(sum([e - s for s, e in pauses]) / duration) if duration > 0 else 0.0
    pitch_stress = 0.0
    if len(pitch_vals) > 1:
        peaks, _ = find_peaks(pitch_vals, prominence=5)
        pitch_stress = float(np.mean(pitch_vals[peaks])) if len(peaks) > 0 else 0.0
    rms_stress = 0.0
    if len(rms) > 1:
        rms_peaks, _ = find_peaks(rms, prominence=0.01)
        rms_stress = float(np.mean(rms[rms_peaks])) if len(rms_peaks) > 0 else 0.0
    stress = pitch_stress + rms_stress
    return np.array([pitch, intensity, tempo, pause_ratio, stress], dtype=np.float32)


def extract_mfcc_features(audio_data, sr=16000, max_len=92):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# ============================
# Alert System
# ============================
THREAT_EMOTIONS = {'Angry', 'Fear', 'Disgust'}
ALERT_THRESHOLD = 0.3  # nervousness or low confidence triggers alert

class AlertLog:
    def __init__(self, max_size=50):
        self.logs = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, alert_type, message, severity="medium"):
        with self.lock:
            self.logs.appendleft({
                "id": int(time.time() * 1000),
                "timestamp": time.strftime("%H:%M:%S"),
                "type": alert_type,
                "message": message,
                "severity": severity
            })

    def get_all(self):
        with self.lock:
            return list(self.logs)

alert_log = AlertLog()

# ============================
# Detector (adapted for API)
# ============================
class CampusSecurityDetector:
    def __init__(self):
        self.video_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.sound_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.load_models()
        self.init_bart_feedback()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.setup_audio()
        self.video_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.audio_buffer = deque(maxlen=int(16000 * 2))
        self.audio_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.current_results = {
            'video_emotion': 'Neutral',
            'sound_emotion': 'neutral',
            'prosody_features': [0.0] * 5,
            'feedback_scores': {'engagement': 0.5, 'confidence': 0.5, 'nervousness': 0.5, 'positivity': 0.5},
            'face_count': 0,
            'face_locations': [],
            'threat_level': 'low',
            'timestamp': time.strftime("%H:%M:%S")
        }
        self.processing_audio = False
        self.processing_video = False
        self.processing_feedback = False
        self.last_video_detection = 0
        self.last_audio_detection = 0
        self.last_feedback_analysis = 0
        self.video_detection_interval = 0.3
        self.audio_detection_interval = 1.0
        self.feedback_analysis_interval = 2.0
        self.frame_count = 0
        self.websocket_clients = set()
        self.running = False
        self.stats = {
            'total_faces_detected': 0,
            'alerts_triggered': 0,
            'session_start': time.strftime("%H:%M:%S"),
            'emotion_history': deque(maxlen=100)
        }

    def load_models(self):
        # Video Emotion Model
        try:
            path = get_file_path('fer_model_finetuned_v2.pth') or get_file_path('fer_model_finetuned.pth')
            if not path:
                raise FileNotFoundError("fer_model_finetuned_v2.pth not found in workspace")
            self.video_model = FER_CNN(7).to(DEVICE)
            self.video_model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.video_model.eval()
            print(f"[OK] Video model loaded successfully from {path}")
        except Exception as e:
            self.video_model = FER_CNN(7).to(DEVICE)
            self.video_model.eval()
            print(f"[WARNING] Using dummy video model (Reason: {e})")

        # Sound Emotion Model
        try:
            path = get_file_path('sound_emotion_model.pth')
            if not path:
                raise FileNotFoundError("sound_emotion_model.pth not found in workspace")
            checkpoint = torch.load(path, map_location=DEVICE)
            self.sound_model = TwoLayerCNN(40, 92, 7).to(DEVICE)
            self.sound_model.load_state_dict(checkpoint)
            self.sound_model.eval()
            print(f"[OK] Sound model loaded successfully from {path}")
        except Exception as e:
            self.sound_model = TwoLayerCNN(40, 92, 7).to(DEVICE)
            self.sound_model.eval()
            print(f"[WARNING] Using dummy sound model (Reason: {e})")

        # Prosody Model
        try:
            path = get_file_path('prosody_net_best.pth')
            if not path:
                raise FileNotFoundError("prosody_net_best.pth not found in workspace")
            checkpoint = torch.load(path, map_location=DEVICE)
            self.prosody_model = ProsodyNet().to(DEVICE)
            self.prosody_model.load_state_dict(checkpoint['model_state_dict'])
            self.prosody_mean = checkpoint['mean']
            self.prosody_std = checkpoint['std']
            self.prosody_model.eval()
            print(f"[OK] Prosody model loaded successfully from {path}")
        except Exception as e:
            self.prosody_model = ProsodyNet().to(DEVICE)
            self.prosody_mean = np.zeros(5)
            self.prosody_std = np.ones(5)
            self.prosody_model.eval()
            print(f"[WARNING] Using dummy prosody model (Reason: {e})")

    def init_bart_feedback(self):
        try:
            self.bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            self.bart_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
            print("[OK] BART model loaded")
        except Exception as e:
            self.bart_tokenizer = None
            self.bart_model = None
            print(f"[WARNING] BART model unavailable: {e}")

    def setup_audio(self):
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32, channels=1, rate=16000, input=True,
                frames_per_buffer=1024, stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("[OK] Audio stream started")
        except Exception as e:
            self.stream = None
            print(f"[WARNING] Audio setup failed: {e}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        with self.audio_lock:
            self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)

    def compute_threat_level(self, video_emotion, sound_emotion, feedback_scores):
        threat_score = 0
        if video_emotion in THREAT_EMOTIONS:
            threat_score += 2
        if sound_emotion in ['angry', 'fear', 'disgust']:
            threat_score += 2
        if feedback_scores.get('nervousness', 0.5) > 0.7:
            threat_score += 1
        if feedback_scores.get('confidence', 0.5) < 0.3:
            threat_score += 1
        if threat_score >= 4:
            return 'critical'
        elif threat_score >= 2:
            return 'high'
        elif threat_score >= 1:
            return 'medium'
        return 'low'

    def analyze_with_bart(self, emotion_text):
        if not self.bart_tokenizer:
            return {'engagement': 0.5, 'confidence': 0.5, 'nervousness': 0.5, 'positivity': 0.5}
        scores = {}
        for aspect in ['engagement', 'confidence', 'nervousness', 'positivity']:
            try:
                text = f"The person shows {emotion_text} emotion, indicating {aspect} level"
                inputs = self.bart_tokenizer(text, f"This shows high {aspect}", return_tensors='pt', truncation=True)
                with torch.no_grad():
                    outputs = self.bart_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    scores[aspect] = probs[0][2].item()
            except:
                scores[aspect] = 0.5
        return scores

    def process_frame(self, frame):
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_video_detection >= self.video_detection_interval and not self.processing_video:
            self.processing_video = True
            threading.Thread(target=self._detect_video_emotion, args=(frame.copy(),), daemon=True).start()
            self.last_video_detection = current_time
        if current_time - self.last_audio_detection >= self.audio_detection_interval and not self.processing_audio:
            self.processing_audio = True
            threading.Thread(target=self._detect_sound_emotion, daemon=True).start()
            self.last_audio_detection = current_time
        if current_time - self.last_feedback_analysis >= self.feedback_analysis_interval and not self.processing_feedback:
            self.processing_feedback = True
            threading.Thread(target=self._analyze_feedback, daemon=True).start()
            self.last_feedback_analysis = current_time

    def _detect_video_emotion(self, frame):
        try:
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3, minSize=(30, 30))
            face_count = len(faces)
            self.stats['total_faces_detected'] += face_count
            emotion = 'Neutral'
            face_locations_list = []
            if face_count > 0:
                sx, sy = frame.shape[1] / 320, frame.shape[0] / 240
                scaled = [(int(x*sx), int(y*sy), int(w*sx), int(h*sy)) for (x,y,w,h) in faces]
                face_locations_list = [list(f) for f in scaled]
                x, y, w, h = scaled[0]
                roi = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w], (48, 48))
                tensor = self.video_transform(Image.fromarray(roi)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = torch.argmax(self.video_model(tensor), 1).item()
                    emotion = self.video_emotions[pred]
            with self.results_lock:
                self.current_results['video_emotion'] = emotion
                self.current_results['face_count'] = face_count
                self.current_results['face_locations'] = face_locations_list
                self.current_results['timestamp'] = time.strftime("%H:%M:%S")
            # Auto-alert for threat emotions
            if emotion in THREAT_EMOTIONS and face_count > 0:
                alert_log.add("EMOTION", f"Threat emotion detected: {emotion} on {face_count} subject(s)", "high")
                self.stats['alerts_triggered'] += 1
            self.stats['emotion_history'].append({'time': time.strftime("%H:%M:%S"), 'emotion': emotion})
        except Exception as e:
            print(f"Video error: {e}")
        finally:
            self.processing_video = False

    def _detect_sound_emotion(self):
        try:
            with self.audio_lock:
                if len(self.audio_buffer) < 8000:
                    return
                audio_data = np.array(list(self.audio_buffer))
            mfcc = extract_mfcc_features(audio_data)
            tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = torch.argmax(self.sound_model(tensor), 1).item()
                emotion = self.sound_emotions[pred]
            prosody = extract_prosody_features(audio_data)
            with self.results_lock:
                self.current_results['sound_emotion'] = emotion
                self.current_results['prosody_features'] = prosody.tolist()
        except Exception as e:
            print(f"Sound error: {e}")
        finally:
            self.processing_audio = False

    def _analyze_feedback(self):
        try:
            with self.results_lock:
                ve = self.current_results['video_emotion']
                se = self.current_results['sound_emotion']
            scores = self.analyze_with_bart(f"{ve} and {se}")
            threat = self.compute_threat_level(ve, se, scores)
            with self.results_lock:
                self.current_results['feedback_scores'] = scores
                self.current_results['threat_level'] = threat
            if threat in ('high', 'critical'):
                alert_log.add("THREAT", f"Threat level: {threat.upper()} — Video: {ve}, Audio: {se}", threat)
                self.stats['alerts_triggered'] += 1
        except Exception as e:
            print(f"Feedback error: {e}")
        finally:
            self.processing_feedback = False

    def get_current_state(self):
        with self.results_lock:
            return dict(self.current_results)

    def get_stats(self):
        return {
            'total_faces': self.stats['total_faces_detected'],
            'alerts': self.stats['alerts_triggered'],
            'session_start': self.stats['session_start'],
            'emotion_history': list(self.stats['emotion_history'])[-20:]
        }

    def cleanup(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()

# ============================
# FastAPI App
# ============================
app = FastAPI(title="Campus Security AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

detector = CampusSecurityDetector()
camera_active = False
cap = None
cap_lock = threading.Lock()
active_connections: list[WebSocket] = []

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = get_file_path("static/index.html") or get_file_path("index.html")
    if index_path and os.path.exists(index_path):
        return FileResponse(index_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_index = os.path.abspath(os.path.join(script_dir, "..", "static", "index.html"))
    if os.path.exists(parent_index):
        return FileResponse(parent_index)
    raise HTTPException(status_code=404, detail="static/index.html not found")

@app.get("/api/status")
async def get_status():
    state = detector.get_current_state()
    return {"status": "online", "camera_active": camera_active, "data": state}

@app.get("/api/stats")
async def get_stats():
    return detector.get_stats()

@app.get("/api/alerts")
async def get_alerts():
    return {"alerts": alert_log.get_all()}

@app.post("/api/alerts/clear")
async def clear_alerts():
    alert_log.logs.clear()
    return {"message": "Alerts cleared"}

@app.post("/api/camera/start")
async def start_camera():
    global camera_active, cap
    with cap_lock:
        if camera_active:
            return {"message": "Camera already running"}
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open camera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_active = True
    alert_log.add("SYSTEM", "Camera feed started", "low")
    return {"message": "Camera started"}

@app.post("/api/camera/stop")
async def stop_camera():
    global camera_active, cap
    with cap_lock:
        camera_active = False
        if cap:
            cap.release()
            cap = None
    alert_log.add("SYSTEM", "Camera feed stopped", "low")
    return {"message": "Camera stopped"}

@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            if camera_active and cap and cap.isOpened():
                with cap_lock:
                    ret, frame = cap.read()
                if ret:
                    detector.process_frame(frame)
                    # Encode frame as JPEG → base64
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    state = detector.get_current_state()
                    await websocket.send_json({
                        "type": "frame",
                        "image": frame_b64,
                        "data": state,
                        "alerts": alert_log.get_all()[:5]
                    })
            else:
                state = detector.get_current_state()
                await websocket.send_json({
                    "type": "data",
                    "data": state,
                    "alerts": alert_log.get_all()[:5]
                })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WS error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# Mount static files
def get_static_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = [
        os.path.join(project_root, "static"),
        os.path.join(script_dir, "static"),
        os.path.abspath("static"),
        os.path.abspath(os.path.join("..", "static"))
    ]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "index.html")):
            return cand
    return os.path.join(project_root, "static")

static_dir = get_static_dir()
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)