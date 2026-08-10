import cv2
import torch
import numpy as np
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import mediapipe as mp
import os
import time
import threading
from config import DEVICE, VIDEO_EMOTIONS, find_model_path


class FER_CNN(nn.Module):
    def __init__(self, num_classes=7):
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


class FacialEmotionDetector:
    def __init__(self):
        self.emotions = VIDEO_EMOTIONS  # ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

        # ── Transform identical to FinalFuse.py ──────────────────────────────
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.lock = threading.Lock()

        # ── Face detectors — mirrored exactly from FinalFuse.py ──────────────
        # CRITICAL FIX: static_image_mode=False (tracking mode) with 0.5 thresholds
        # FinalFuse uses False/0.5/0.5; our old code used True/0.3/0.3 which caused bad crops
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,          # TRACKING mode — same as FinalFuse
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,     # Same as FinalFuse (was 0.3)
            min_tracking_confidence=0.5       # Same as FinalFuse (was 0.3)
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.smoothed_probs = None
        self.last_face_box = None
        self.last_face_time = 0.0
        self.recent_preds = []  # 3-frame majority vote window

        # Logit-level bias correction applied BEFORE softmax.
        # This model systematically over-predicts Sad/Fear on neutral faces.
        # Penalise Sad/Fear at logit level; boost Happy/Neutral so they can win.
        # Order matches VIDEO_EMOTIONS: ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
        self.LOGIT_BIAS = torch.tensor(
            [0.0, 0.2, -0.5, 0.6, -1.0, 0.1, 0.4],
            dtype=torch.float32
        ).to(DEVICE)

        self.load_model()

    def load_model(self):
        self.model = FER_CNN(num_classes=7).to(DEVICE)
        model_path = find_model_path('fer_model_finetuned_v2.pth')
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                print(f"[OK] Loaded FER_CNN video emotion model weights from {model_path}")
            except Exception as e:
                print(f"[WARNING] Failed loading FER_CNN weights ({e}). Using initialized model.")
        else:
            print(f"[WARNING] fer_model_finetuned_v2.pth not found. Using initialized model.")
        self.model.eval()

    def reset(self):
        self.smoothed_probs = None
        self.last_face_box = None
        self.last_face_time = 0.0
        self.recent_preds = []

    def predict_frame(self, frame):
        """
        Processes a single BGR frame — logic mirrors FinalFuse.py exactly:
          • MediaPipe FaceMesh in tracking mode (static_image_mode=False, conf 0.5)
          • 1.15x crop margin (FinalFuse uses 1.15 for MP, 1.20 for Haar fallback)
          • Minimum face size sw > 20
          • EMA: 0.35 * new_probs + 0.65 * old_probs  (same as FinalFuse line 660)
          • Keep last valid box for 1.5 s on detection loss (same as FinalFuse)
        """
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        now = time.time()
        box = None

        with self.lock:
            results = self.mp_face_mesh.process(rgb)

        # ── Face localisation (mirrors FinalFuse lines 585-624) ───────────────
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            landmarks = results.multi_face_landmarks[0].landmark
            xs = [lm.x * w_frame for lm in landmarks]
            ys = [lm.y * h_frame for lm in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            fw = max_x - min_x
            fh = max_y - min_y
            cx = int(min_x + fw / 2)
            cy = int(min_y + fh / 2)
            # 1.15x margin — exactly as FinalFuse line 598
            side = int(max(fw, fh) * 1.15)
            sx   = max(0, cx - side // 2)
            sy   = max(0, cy - side // 2)
            sw   = min(w_frame - sx, side)
            sh   = min(h_frame - sy, side)
            actual_side = min(sw, sh)
            box  = (sx, sy, actual_side, actual_side)
        else:
            # Haar fallback — 1.20x margin same as FinalFuse line 618
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                cx = fx + fw // 2
                cy = fy + fh // 2
                side = int(max(fw, fh) * 1.20)
                sx   = max(0, cx - side // 2)
                sy   = max(0, cy - side // 2)
                sw   = min(w_frame - sx, side)
                sh   = min(h_frame - sy, side)
                actual_side = min(sw, sh)
                box  = (sx, sy, actual_side, actual_side)

        # ── Box smoothing (mirrors FinalFuse lines 630-641) ──────────────────
        if box is not None:
            x, y, w, h = box
            if self.last_face_box is not None:
                lx, ly, lw, lh = self.last_face_box
                sx = int(0.7 * x + 0.3 * lx)
                sy = int(0.7 * y + 0.3 * ly)
                sw = int(0.7 * w + 0.3 * lw)
                sh = int(0.7 * h + 0.3 * lh)
            else:
                sx, sy, sw, sh = x, y, w, h
            self.last_face_box = (sx, sy, sw, sh)
            self.last_face_time = now
        else:
            # Keep last box for 1.5 s (mirrors FinalFuse lines 670-676)
            if self.last_face_box is not None and (now - self.last_face_time < 1.5):
                sx, sy, sw, sh = self.last_face_box
            else:
                # Truly no face
                self.last_face_box = None
                neutral_dict = {e: (1.0 if e == 'Neutral' else 0.0) for e in self.emotions}
                return {'emotion': 'Neutral', 'raw_emotions': neutral_dict, 'box': None}

        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(w_frame - sx, sw), min(h_frame - sy, sh)

        # ── Minimum face size guard: sw > 20 (FinalFuse line 646) ────────────
        if sw <= 20 or sh <= 20:
            neutral_dict = {e: (1.0 if e == 'Neutral' else 0.0) for e in self.emotions}
            return {'emotion': 'Neutral', 'raw_emotions': neutral_dict, 'box': None}

        # ── FER inference ─────────────────────────────────────────────────────
        face_roi    = gray[sy:sy+sh, sx:sx+sw]
        face_roi    = cv2.resize(face_roi, (48, 48))
        face_pil    = Image.fromarray(face_roi)
        face_tensor = self.transform(face_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(face_tensor)

            # ── Logit-level bias correction (applied BEFORE softmax) ──────────
            # FinalFuse works at 30fps so EMA recovers fast; we run at 3.3fps
            # so we must correct the model's Sad/Fear bias at the logit level.
            # Subtract bias from logits: negative bias = penalise, positive = boost.
            output_biased = output + self.LOGIT_BIAS
            probs = torch.softmax(output_biased, dim=-1)[0].cpu().numpy()

            # ── EMA tuned for 300ms API interval (0.6 new / 0.4 old) ─────────
            # FinalFuse uses 0.35/0.65 at 30fps ≈ same real-time decay as
            # 0.6/0.4 at 3.3fps (300ms). Prevents Sad from staying sticky.
            if self.smoothed_probs is None:
                self.smoothed_probs = probs.copy()
            else:
                self.smoothed_probs = 0.6 * probs + 0.4 * self.smoothed_probs

            # ── 3-frame majority vote for stability without stickiness ────────
            frame_pred = int(np.argmax(self.smoothed_probs))
            self.recent_preds.append(frame_pred)
            if len(self.recent_preds) > 3:
                self.recent_preds.pop(0)
            from collections import Counter
            pred_idx = Counter(self.recent_preds).most_common(1)[0][0]
            predicted_emotion = self.emotions[pred_idx]
            raw_dict = {self.emotions[i]: float(self.smoothed_probs[i])
                        for i in range(len(self.emotions))}

        # ── Draw face box + label on frame ───────────────────────────────────
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        cv2.putText(
            frame, f"Video: {predicted_emotion}",
            (max(0, sx), max(25, sy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return {
            'emotion'    : predicted_emotion,
            'raw_emotions': raw_dict,
            'box'        : [sx, sy, sw, sh]
        }
