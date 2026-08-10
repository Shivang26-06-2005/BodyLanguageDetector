import cv2
import torch
import numpy as np
import librosa
import pyaudio
import threading
import time
import math
from collections import deque
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mediapipe as mp
import warnings
import os

warnings.filterwarnings('ignore')

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

def distance(a, b):
    """Calculate 3D Euclidean distance between two landmarks"""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def get_precise_direction(dx, dy, dz, thresh=0.01):
    """Returns a combined direction string for 3D movement"""
    directions = []
    if abs(dx) > thresh:
        directions.append("Right" if dx > 0 else "Left")
    if abs(dy) > thresh:
        directions.append("Down" if dy > 0 else "Up")
    if abs(dz) > thresh:
        directions.append("Back" if dz > 0 else "Forward")
    if not directions:
        directions.append("No Movement")
    return "-".join(directions)

# ============================
# 1. Device Setup
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================
# 2. Video Emotion Model (FER_CNN)
# ============================
class FER_CNN(nn.Module):
    def __init__(self, num_classes):
        super(FER_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================
# 3. Sound Emotion Model (CNN)
# ============================
class TwoLayerCNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(32*(input_height//4)*(input_width//4), 128)
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

# ============================
# 4. Prosody Model (CNN+LSTM)
# ============================
class ProsodyNet(nn.Module):
    def __init__(self, num_targets=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lstm = nn.LSTM(input_size=32*16, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_targets)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        B,C,H,W = x.size()
        x_seq = x.permute(0,3,1,2).contiguous().view(B,W,C*H)
        out,_ = self.lstm(x_seq)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ============================
# 5. Feature Extraction Functions
# ============================
def extract_prosody_features(audio_data, sr=16000):
    """Extract prosody features from audio data"""
    y = audio_data
    duration = len(y) / sr
    if duration <= 0:
        return np.zeros(5, dtype=np.float32)
    
    # Intensity (RMS Volume)
    rms = librosa.feature.rms(y=y)[0]
    intensity = float(np.mean(rms))
    
    # Pitch (filtered for voice energy and human pitch range 50-500 Hz)
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
    
    # Speaking Tempo (Syllable rate / energy peaks per minute)
    if intensity > 0.005 and len(rms) > 1:
        rms_threshold = float(np.mean(rms) + 0.2 * np.std(rms))
        syllable_peaks, _ = find_peaks(rms, height=rms_threshold, distance=int(0.15 * sr / 512))
        tempo = float((len(syllable_peaks) / duration) * 60.0)
    else:
        tempo = 0.0
    
    # Pause Ratio (Fraction of non-speech / silence over total duration)
    if intensity > 0.005:
        intervals = librosa.effects.split(y, top_db=25)
        active_samples = sum([end - start for start, end in intervals]) if len(intervals) > 0 else 0
        pause_ratio = float(max(0.0, min(1.0, 1.0 - (active_samples / len(y)))))
    else:
        pause_ratio = 1.0
    
    # Stress Pattern (Normalized variation in pitch dynamics & acoustic energy)
    if len(pitch_vals) > 1 and pitch > 0:
        pitch_std = np.std(pitch_vals) / pitch
    else:
        pitch_std = 0.0
        
    rms_std = np.std(rms) if len(rms) > 1 else 0.0
    stress = float(np.clip(pitch_std * 2.0 + rms_std * 10.0, 0.0, 1.0))
    
    return np.array([pitch, intensity, tempo, pause_ratio, stress], dtype=np.float32)

def extract_mfcc_features(audio_data, sr=16000, max_len=92):
    """Extract MFCC features for sound emotion detection"""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def extract_mel_spectrogram(audio_data, sr=16000, max_len=128):
    """Extract mel-spectrogram for prosody model"""
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0),(0,max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db

# ============================
# 6. Multi-Modal Emotion Detector Class
# ============================
class MultiModalEmotionDetector:
    def __init__(self):
        # Emotion labels
        self.video_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.sound_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Load models
        self.load_models()
        
        # Initialize BART for feedback & final report
        self.init_bart_feedback()
        
        # Video capture and face detection
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Pose tracking initialization
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.baseline_z = None
        
        # Gesture counters & tracking history
        self.gesture_counters = {
            "Leaning Forward": 0,
            "Leaning Back": 0,
            "Leaning Left": 0,
            "Leaning Right": 0,
            "Upright Posture": 0,
            "Arms Crossed": 0,
            "Hand Raised": 0,
            "Hand Lowered": 0,
            "Pointing": 0,
            "Waving": 0,
            "Object in Hand": 0
        }
        
        self.wrist_history = {"left": deque(maxlen=5), "right": deque(maxlen=5)}
        self.shoulder_history = deque(maxlen=2)
        self.head_history = deque(maxlen=2)
        self.left_wrist_history = deque(maxlen=2)
        self.right_wrist_history = deque(maxlen=2)
        
        # Audio setup
        self.setup_audio()
        
        # Transformation for video
        self.video_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Detection intervals and buffers
        self.face_locations = []
        self.face_labels = []
        self.last_video_detection = time.time()
        self.last_audio_detection = time.time()
        
        # Intervals
        self.video_detection_interval = 0.3  # Video emotion every 300ms
        self.audio_detection_interval = 1.0  # Audio emotion every 1s
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(16000 * 2))  # 2 seconds buffer
        self.audio_lock = threading.Lock()
        
        # Processing flags
        self.processing_audio = False
        self.processing_video = False
        
        # Results storage with thread locks
        self.results_lock = threading.Lock()
        self.current_results = {
            'video_emotion': 'Neutral',
            'sound_emotion': 'neutral',
            'prosody_features': [0.0] * 5,
            'active_posture': 'Upright Posture',
            'active_gesture': 'Hand Lowered',
            'movement_info': []
        }
        
        # Cumulative Session History for Final BART Report
        self.session_history = {
            'video_emotions': [],
            'sound_emotions': [],
            'postures': [],
            'gestures_counter': self.gesture_counters.copy(),
            'prosody_records': []
        }
        
        # Frame processing optimization
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        
    def load_models(self):
        """Load all pre-trained models"""
        try:
            path = get_file_path('fer_model_finetuned_v2.pth')
            if not path:
                raise FileNotFoundError("fer_model_finetuned_v2.pth not found")
            self.video_model = FER_CNN(7).to(DEVICE)
            self.video_model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.video_model.eval()
            print(f"[OK] Video emotion model loaded from {path}")
        except Exception as e:
            print(f"[WARNING] Video model not loaded ({e}), creating dummy model")
            self.video_model = FER_CNN(7).to(DEVICE)
            self.video_model.eval()
        
        try:
            path = get_file_path('sound_emotion_model.pth')
            if not path:
                raise FileNotFoundError("sound_emotion_model.pth not found")
            checkpoint = torch.load(path, map_location=DEVICE)
            
            fc1_weight_shape = checkpoint['fc1.weight'].shape
            input_height, input_width = 40, 92
            
            self.sound_model = TwoLayerCNN(input_height, input_width, 7).to(DEVICE)
            self.sound_model.load_state_dict(checkpoint)
            self.sound_model.eval()
            print(f"[OK] Sound emotion model loaded from {path} (dimensions {input_height}x{input_width})")
        except Exception as e:
            print(f"[WARNING] Sound model loading failed ({e}), creating dummy model")
            self.sound_model = TwoLayerCNN(40, 92, 7).to(DEVICE)
            self.sound_model.eval()
            
        try:
            path = get_file_path('prosody_net_best.pth')
            if not path:
                raise FileNotFoundError("prosody_net_best.pth not found")
            checkpoint = torch.load(path, map_location=DEVICE)
            self.prosody_model = ProsodyNet().to(DEVICE)
            self.prosody_model.load_state_dict(checkpoint['model_state_dict'])
            self.prosody_mean = checkpoint['mean']
            self.prosody_std = checkpoint['std']
            self.prosody_model.eval()
            print(f"[OK] Prosody model loaded from {path}")
        except Exception as e:
            print(f"[WARNING] Prosody model not loaded ({e}), creating dummy model")
            self.prosody_model = ProsodyNet().to(DEVICE)
            self.prosody_mean = np.zeros(5)
            self.prosody_std = np.ones(5)
            self.prosody_model.eval()
    
    def init_bart_feedback(self):
        """Initialize BART model for candidate analysis & final report"""
        try:
            self.bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            self.bart_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
            print("[OK] BART candidate evaluation model loaded")
        except Exception as e:
            print(f"[WARNING] BART model failed to load: {e}")
            self.bart_tokenizer = None
            self.bart_model = None
    
    def setup_audio(self):
        """Setup audio recording"""
        self.audio = pyaudio.PyAudio()
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        
        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("[OK] Audio stream started")
        except Exception as e:
            print(f"[WARNING] Audio setup failed: {e}")
            self.stream = None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        with self.audio_lock:
            self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)
    
    def process_body_pose(self, frame):
        """Track 3D body posture and hand/arm gestures using MediaPipe"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        
        posture = "Upright Posture"
        active_gesture = "Hand Lowered"
        movement_info = []
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw MediaPipe pose skeleton on frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # --- Torso midpoints for leaning posture ---
            l_sh, r_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip, r_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            mid_sh_x = (l_sh.x + r_sh.x)/2
            mid_sh_y = (l_sh.y + r_sh.y)/2
            mid_sh_z = (l_sh.z + r_sh.z)/2
            mid_hip_x = (l_hip.x + r_hip.x)/2
            mid_hip_y = (l_hip.y + r_hip.y)/2
            mid_hip_z = (l_hip.z + r_hip.z)/2
            
            if self.baseline_z is None:
                self.baseline_z = (mid_sh_z + mid_hip_z)/2
                
            dx = mid_hip_x - mid_sh_x
            dz = ((mid_hip_z + mid_sh_z)/2) - self.baseline_z
            lean_thresh_z = 0.03
            lean_thresh_x = 0.03
            
            if dz < -lean_thresh_z:
                if dx > lean_thresh_x:
                    posture = "Leaning Forward-Right"
                    self.gesture_counters["Leaning Forward"] += 1
                    self.gesture_counters["Leaning Right"] += 1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Forward-Left"
                    self.gesture_counters["Leaning Forward"] += 1
                    self.gesture_counters["Leaning Left"] += 1
                else:
                    posture = "Leaning Forward"
                    self.gesture_counters["Leaning Forward"] += 1
            elif dz > lean_thresh_z:
                if dx > lean_thresh_x:
                    posture = "Leaning Back-Right"
                    self.gesture_counters["Leaning Back"] += 1
                    self.gesture_counters["Leaning Right"] += 1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Back-Left"
                    self.gesture_counters["Leaning Back"] += 1
                    self.gesture_counters["Leaning Left"] += 1
                else:
                    posture = "Leaning Back"
                    self.gesture_counters["Leaning Back"] += 1
            else:
                if dx > lean_thresh_x:
                    posture = "Leaning Right"
                    self.gesture_counters["Leaning Right"] += 1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Left"
                    self.gesture_counters["Leaning Left"] += 1
                else:
                    posture = "Upright Posture"
                    self.gesture_counters["Upright Posture"] += 1
            
            # --- Arm & Hand Gestures ---
            l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Arms Crossed
            if distance(l_wrist, r_elbow) < 0.12 and distance(r_wrist, l_elbow) < 0.12:
                active_gesture = "Arms Crossed"
                self.gesture_counters["Arms Crossed"] += 1
                
            # Hand Raised / Lowered
            if l_wrist.y < l_sh.y or r_wrist.y < r_sh.y:
                active_gesture = "Hand Raised"
                self.gesture_counters["Hand Raised"] += 1
            else:
                self.gesture_counters["Hand Lowered"] += 1
                
            # Pointing
            l_angle = math.degrees(math.atan2(l_wrist.y - l_elbow.y, l_wrist.x - l_elbow.x))
            r_angle = math.degrees(math.atan2(r_wrist.y - r_elbow.y, r_wrist.x - r_elbow.x))
            if abs(l_angle) < 30 or abs(r_angle) < 30:
                active_gesture = "Pointing"
                self.gesture_counters["Pointing"] += 1
                
            # Waving
            self.wrist_history["left"].append(l_wrist.x)
            self.wrist_history["right"].append(r_wrist.x)
            if len(self.wrist_history["left"]) == 5 and max(self.wrist_history["left"]) - min(self.wrist_history["left"]) > 0.05:
                active_gesture = "Waving"
                self.gesture_counters["Waving"] += 1
            if len(self.wrist_history["right"]) == 5 and max(self.wrist_history["right"]) - min(self.wrist_history["right"]) > 0.05:
                active_gesture = "Waving"
                self.gesture_counters["Waving"] += 1
                
            # Object in Hand
            if distance(l_wrist, l_elbow) < 0.05 or distance(r_wrist, r_elbow) < 0.05:
                self.gesture_counters["Object in Hand"] += 1
                
            # --- 3D Movement Tracking ---
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            
            head_mid = ((nose.x + left_eye.x + right_eye.x)/3, (nose.y + left_eye.y + right_eye.y)/3, (nose.z + left_eye.z + right_eye.z)/3)
            self.head_history.append(head_mid)
            
            shoulder_mid = (mid_sh_x, mid_sh_y, mid_sh_z)
            self.shoulder_history.append(shoulder_mid)
            
            self.left_wrist_history.append((l_wrist.x, l_wrist.y, l_wrist.z))
            self.right_wrist_history.append((r_wrist.x, r_wrist.y, r_wrist.z))
            
            if len(self.shoulder_history) == 2:
                dx_s, dy_s, dz_s = [self.shoulder_history[1][i]-self.shoulder_history[0][i] for i in range(3)]
                movement_info.append(f"Shoulders: {get_precise_direction(dx_s, dy_s, dz_s)}")
                
            if len(self.head_history) == 2:
                dx_h, dy_h, dz_h = [self.head_history[1][i]-self.head_history[0][i] for i in range(3)]
                movement_info.append(f"Head: {get_precise_direction(dx_h, dy_h, dz_h)}")
        
        with self.results_lock:
            self.current_results['active_posture'] = posture
            self.current_results['active_gesture'] = active_gesture
            self.current_results['movement_info'] = movement_info
            
            # Record for final session history
            self.session_history['postures'].append(posture)
            self.session_history['gestures_counter'] = self.gesture_counters.copy()
            
    def detect_video_emotion_threaded(self, frame):
        """Detect face emotion using OpenCV Haar Cascade with proper square padded bounding box"""
        if self.processing_video:
            return
        
        self.processing_video = True
        frame_copy = frame.copy()
        
        def process():
            try:
                h_frame, w_frame = frame_copy.shape[:2]
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.mp_face_mesh.process(rgb)
                rgb.flags.writeable = True
                
                now = time.time()
                box = None
                
                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    landmarks = results.multi_face_landmarks[0].landmark
                    xs = [lm.x * w_frame for lm in landmarks]
                    ys = [lm.y * h_frame for lm in landmarks]
                    
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    fw = max_x - min_x
                    fh = max_y - min_y
                    cx, cy = int(min_x + fw / 2), int(min_y + fh / 2)
                    
                    # MediaPipe tight square face crop with 15% margin (eyebrows to chin, cheek to cheek)
                    side = int(max(fw, fh) * 1.15)
                    sx = max(0, cx - side // 2)
                    sy = max(0, cy - side // 2)
                    sw = min(w_frame - sx, side)
                    sh = min(h_frame - sy, side)
                    actual_side = min(sw, sh)
                    box = (sx, sy, actual_side, actual_side)
                else:
                    # Fallback to Haar Cascade
                    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50)
                    )
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        fx, fy, fw, fh = faces[0]
                        cx, cy = fx + fw // 2, fy + fh // 2
                        side = int(max(fw, fh) * 1.20)
                        sx = max(0, cx - side // 2)
                        sy = max(0, cy - side // 2)
                        sw = min(w_frame - sx, side)
                        sh = min(h_frame - sy, side)
                        actual_side = min(sw, sh)
                        box = (sx, sy, actual_side, actual_side)
                
                if box is not None:
                    x, y, w, h = box
                    with self.results_lock:
                        # Exponential smoothing for stable bounding box visualization
                        if hasattr(self, 'last_face_box') and self.last_face_box is not None:
                            lx, ly, lw, lh = self.last_face_box
                            sx = int(0.7 * x + 0.3 * lx)
                            sy = int(0.7 * y + 0.3 * ly)
                            sw = int(0.7 * w + 0.3 * lw)
                            sh = int(0.7 * h + 0.3 * lh)
                        else:
                            sx, sy, sw, sh = x, y, w, h
                            
                        self.last_face_box = (sx, sy, sw, sh)
                        self.last_face_time = now
                        self.face_locations = [(sx, sy, sw, sh)]
                    
                    sx, sy = max(0, sx), max(0, sy)
                    sw, sh = min(w_frame - sx, sw), min(h_frame - sy, sh)
                    
                    if sw > 20 and sh > 20:
                        face_roi = gray[sy:sy+sh, sx:sx+sw]
                        face_roi = cv2.resize(face_roi, (48, 48))
                        face_pil = Image.fromarray(face_roi)
                        face_tensor = self.video_transform(face_pil).unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            output = self.video_model(face_tensor)
                            probs = torch.softmax(output, dim=-1)[0].cpu().numpy()
                            
                            # Temporal Probability Smoothing (EMA decay 0.35)
                            if not hasattr(self, 'smoothed_probs') or self.smoothed_probs is None:
                                self.smoothed_probs = probs
                            else:
                                self.smoothed_probs = 0.35 * probs + 0.65 * self.smoothed_probs
                                
                            pred = int(np.argmax(self.smoothed_probs))
                            emotion = self.video_emotions[pred]
                            
                            with self.results_lock:
                                self.current_results['video_emotion'] = emotion
                                self.face_labels = [emotion]
                                self.session_history['video_emotions'].append(emotion)
                else:
                    # Keep last valid face box active for 1.5s to eliminate box flickering
                    with self.results_lock:
                        if hasattr(self, 'last_face_time') and (now - self.last_face_time < 1.5):
                            pass
                        else:
                            self.face_locations = []
                            self.face_labels = []
                    
            except Exception as e:
                print(f"Video processing error: {e}")
            finally:
                self.processing_video = False
        
        threading.Thread(target=process, daemon=True).start()
    
    def detect_sound_emotion_threaded(self):
        """Detect emotion from audio buffer in separate thread"""
        if self.processing_audio:
            return
        
        self.processing_audio = True
        
        def process():
            try:
                with self.audio_lock:
                    if len(self.audio_buffer) < 8000:
                        return
                    audio_data = np.array(list(self.audio_buffer))
                
                # Check RMS volume intensity (energy)
                rms = librosa.feature.rms(y=audio_data)[0]
                intensity = float(np.mean(rms))
                
                # High energy threshold to suppress ambient mic noise / silence 'fear' output
                if intensity < 0.015:
                    emotion = 'neutral'
                else:
                    mfcc = extract_mfcc_features(audio_data)
                    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = self.sound_model(mfcc_tensor)
                        pred = torch.argmax(output, 1).item()
                        emotion = self.sound_emotions[pred]
                        
                        # Extra safeguard: if low speech volume yields 'fear', override to 'neutral'
                        if emotion == 'fear' and intensity < 0.025:
                            emotion = 'neutral'
                    
                with self.results_lock:
                    self.current_results['sound_emotion'] = emotion
                    self.session_history['sound_emotions'].append(emotion)
                        
            except Exception as e:
                print(f"Sound processing error: {e}")
            finally:
                self.processing_audio = False
        
        threading.Thread(target=process, daemon=True).start()
    
    def detect_prosody_features_threaded(self):
        """Detect prosody features using accurate acoustic analysis"""
        def process():
            try:
                with self.audio_lock:
                    if len(self.audio_buffer) < 8000:
                        return
                    audio_data = np.array(list(self.audio_buffer))
                
                prosody_feats = extract_prosody_features(audio_data)
                
                with self.results_lock:
                    self.current_results['prosody_features'] = prosody_feats.tolist()
                    self.session_history['prosody_records'].append(prosody_feats.tolist())
                    
            except Exception as e:
                print(f"Prosody processing error: {e}")
        
        threading.Thread(target=process, daemon=True).start()
    
    def draw_results(self, frame):
        """Draw visual overlays & multi-panel HUD on the frame"""
        height, width = frame.shape[:2]
        
        # Draw face bounding boxes (OpenCV detection)
        for i, (x, y, w, h) in enumerate(self.face_locations):
            label = self.current_results['video_emotion']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Video: {label}", (max(0, x), max(20, y-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Create bottom HUD panel
        panel_height = 180
        panel_width = width
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        with self.results_lock:
            video_emotion = self.current_results['video_emotion']
            sound_emotion = self.current_results['sound_emotion']
            prosody_feats = self.current_results['prosody_features']
            active_posture = self.current_results['active_posture']
            active_gesture = self.current_results['active_gesture']
            movement_info = self.current_results['movement_info']
        
        # Left Column (Emotions & Posture)
        cv2.putText(panel, f"Video Emotion: {video_emotion}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(panel, f"Sound Emotion: {sound_emotion}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(panel, f"Body Posture: {active_posture}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(panel, f"Active Gesture: {active_gesture}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Middle Column (Prosody Metrics)
        prosody_p = f"Pitch: {prosody_feats[0]:.1f} Hz | Int: {prosody_feats[1]:.3f} RMS"
        prosody_t = f"Tempo: {prosody_feats[2]:.1f} syll/min | Pause: {prosody_feats[3]*100:.0f}%"
        prosody_s = f"Vocal Stress: {prosody_feats[4]:.2f}"
        
        col2_x = int(width * 0.45)
        cv2.putText(panel, "Prosody Audio Metrics:", (col2_x, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(panel, prosody_p, (col2_x, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(panel, prosody_t, (col2_x, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(panel, prosody_s, (col2_x, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Bottom Row (3D Movement Status)
        mov_str = " | ".join(movement_info) if movement_info else "No Active Motion"
        cv2.putText(panel, f"3D Motion: {mov_str[:70]}", (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)
        
        combined = np.vstack([frame, panel])
        return combined
    
    def generate_final_bart_report(self):
        """Synthesize accumulated session history and generate BART Interview Analysis Report"""
        print("\n" + "="*70)
        print("     FINAL AI CANDIDATE INTERVIEW ASSESSMENT REPORT (BART NLI)    ")
        print("="*70)
        
        # Compile session counts
        v_emotions = self.session_history['video_emotions']
        s_emotions = self.session_history['sound_emotions']
        postures = self.session_history['postures']
        gestures = self.session_history['gestures_counter']
        prosody = np.array(self.session_history['prosody_records']) if len(self.session_history['prosody_records']) > 0 else np.zeros((1, 5))
        
        # Most frequent Video & Sound Emotions
        dom_v = max(set(v_emotions), key=v_emotions.count) if v_emotions else "Neutral"
        dom_s = max(set(s_emotions), key=s_emotions.count) if s_emotions else "neutral"
        dom_posture = max(set(postures), key=postures.count) if postures else "Upright Posture"
        
        # Average Prosody Metrics
        avg_pitch = float(np.mean(prosody[:, 0]))
        avg_int = float(np.mean(prosody[:, 1]))
        avg_tempo = float(np.mean(prosody[:, 2]))
        avg_pause = float(np.mean(prosody[:, 3]))
        avg_stress = float(np.mean(prosody[:, 4]))
        
        # Construct Multi-Modal Session Summary Text Prompt
        session_summary_text = (
            f"Candidate Interview Performance Summary:\n"
            f"- Facial Expression: Primarily {dom_v}.\n"
            f"- Vocal Emotion: Primarily {dom_s}.\n"
            f"- Body Posture & Language: Maintained {dom_posture} with Hand Raised ({gestures.get('Hand Raised', 0)} times), Arms Crossed ({gestures.get('Arms Crossed', 0)} times), Waving ({gestures.get('Waving', 0)} times).\n"
            f"- Vocal Dynamics: Average pitch {avg_pitch:.1f} Hz, volume {avg_int:.3f} RMS, speaking tempo {avg_tempo:.1f} syllables per min, pause silence ratio {avg_pause*100:.1f}%, and vocal stress {avg_stress:.2f}."
        )
        
        print("\n[SESSION SUMMARY NARRATIVE]")
        print(session_summary_text)
        
        # Evaluate 5 Candidate Performance Hypotheses using BART Zero-Shot Classification
        hypotheses = {
            'Confidence & Poise': "This candidate demonstrates high confidence, poise, and self-assurance during the interview",
            'Interview Engagement': "This candidate demonstrates high engagement, active listening, and attentiveness",
            'Nervousness & Vocal Anxiety': "This candidate displays signs of nervousness, tension, or vocal anxiety",
            'Professional Posture & Demeanor': "This candidate maintains professional posture and effective body language",
            'Overall Candidate Positivity': "This candidate maintains a positive and encouraging attitude"
        }
        
        scores = {}
        if self.bart_tokenizer is not None and self.bart_model is not None:
            print("\n[BART NLI EVALUATION RUNNING...]")
            for aspect, hypothesis in hypotheses.items():
                try:
                    inputs = self.bart_tokenizer(session_summary_text, hypothesis, return_tensors='pt', truncation=True)
                    with torch.no_grad():
                        outputs = self.bart_model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        # Index 2 = Entailment probability
                        score = probs[0][2].item() * 100
                        scores[aspect] = score
                except Exception as e:
                    scores[aspect] = 50.0
        else:
            scores = {k: 50.0 for k in hypotheses.keys()}
        
        # Display Final Scorecard
        print("\n[CANDIDATE PERFORMANCE SCORECARD]")
        print("-" * 55)
        for aspect, score in scores.items():
            print(f" * {aspect:<35}: {score:>5.1f}%")
        print("-" * 55)
        
        # Save Report to file
        report_filename = "interview_candidate_report.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write("=========================================================\n")
            f.write("     FINAL AI CANDIDATE INTERVIEW ASSESSMENT REPORT      \n")
            f.write("=========================================================\n\n")
            f.write(session_summary_text + "\n\n")
            f.write("CANDIDATE PERFORMANCE SCORECARD:\n")
            f.write("-" * 55 + "\n")
            for aspect, score in scores.items():
                f.write(f" * {aspect:<35}: {score:>5.1f}%\n")
            f.write("-" * 55 + "\n")
        
        print(f"\n[OK] Report saved successfully to {os.path.abspath(report_filename)}")
        print("="*70 + "\n")
    
    def run(self):
        """Main detection loop (optimized)"""
        print("Starting Multi-Modal Emotion & Candidate Assessment...")
        print("Press 'q' to quit and generate Final BART Candidate Report")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time()
            self.frame_count += 1
            
            # Process MediaPipe Pose
            self.process_body_pose(frame)
            
            # Check thread triggers on every frame
            if current_time - self.last_video_detection >= self.video_detection_interval:
                self.detect_video_emotion_threaded(frame)
                self.last_video_detection = current_time
            
            if current_time - self.last_audio_detection >= self.audio_detection_interval:
                self.detect_sound_emotion_threaded()
                self.detect_prosody_features_threaded()
                self.last_audio_detection = current_time
            
            display_frame = self.draw_results(frame)
            
            # FPS tracking & periodic logging
            fps_counter += 1
            if fps_counter >= 30:
                fps_elapsed = time.time() - fps_start_time
                fps = fps_counter / fps_elapsed
                fps_counter = 0
                fps_start_time = time.time()
                
                with self.results_lock:
                    res = self.current_results.copy()
                
                print(f"\n=== FPS: {fps:.1f} ===")
                print(f"Video: {res['video_emotion']}, Sound: {res['sound_emotion']}, Posture: {res['active_posture']}")
                print(f"Prosody Pitch: {res['prosody_features'][0]:.1f} Hz, Volume: {res['prosody_features'][1]:.3f} RMS")
            
            cv2.imshow('Multi-Modal Candidate Assessment System', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Generate Final BART Assessment Report on Exit
        self.generate_final_bart_report()
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        cv2.destroyAllWindows()
        print("Cleanup completed")

# ============================
# 7. Main Execution
# ============================
if __name__ == "__main__":
    try:
        detector = MultiModalEmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()