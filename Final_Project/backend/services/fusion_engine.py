import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import base64
import threading
import time
from models.fer_model import FacialEmotionDetector
from models.pose_model import BodyPoseTracker
from models.sound_model import SoundEmotionDetector
from models.prosody_model import ProsodyAnalyzer
from models.bart_reporter import BARTCandidateReporter

class MultiModalFusionEngine:
    def __init__(self):
        print("[INFO] Initializing PulseMind MultiModal Fusion Engine...")
        self.lock = threading.Lock()
        
        # Initialize modular AI sub-systems
        self.fer_detector = FacialEmotionDetector()
        self.pose_tracker = BodyPoseTracker()
        self.sound_detector = SoundEmotionDetector()
        self.prosody_analyzer = ProsodyAnalyzer()
        self.report_generator = BARTCandidateReporter()

        # Shared Live Candidate State
        self.current_results = {
            'video_emotion': 'Neutral',
            'sound_emotion': 'neutral',
            'raw_emotions': {e: (1.0 if e == 'Neutral' else 0.0) for e in self.fer_detector.emotions},
            'emotion_summary': {'Happiness': 0.0, 'Nervous': 0.0, 'Neutral': 1.0},
            'active_posture': 'Upright Posture',
            'active_gesture': 'Hand Lowered',
            'movement_info': ['No Movement'],
            'movement_details': {
                'head_eyes': 'No Movement',
                'shoulders': 'No Movement',
                'left_hand': 'No Movement',
                'right_hand': 'No Movement'
            },
            'prosody_features': [0.0, 0.0, 0.0, 0.0, 0.0],  # [F0, RMS, Tempo, Pause, Stress]
            'eye_contact': 85,
            'voice_tone': 'Neutral Speech',
            'confidence_score': 85,
            'feedback': 'PulseMind AI active. Analyzing live video, pose, & speech prosody...',
            'annotated_image': None,
            'gestures': {'fidgeting': False, 'handMovement': False}
        }

        # Session history for BART report synthesis
        self.session_history = {
            'video_emotions': [],
            'sound_emotions': [],
            'postures': [],
            'prosody_samples': [],
            'gestures_counter': {}
        }

    def decode_base64_image(self, base64_str):
        """Decodes base64 JPEG from React HTML canvas. cv2.imdecode returns BGR directly."""
        try:
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            img_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Already returns BGR — no extra conversion needed
            return img
        except Exception as e:
            print(f"[ERROR] Failed base64 image decoding: {e}")
            return None

    def process_frame(self, frame_or_base64):
        """Processes video frame base64 string or BGR numpy matrix and returns live prediction state"""
        if isinstance(frame_or_base64, str):
            frame = self.decode_base64_image(frame_or_base64)
        else:
            frame = frame_or_base64

        if frame is None or frame.size == 0:
            return self.get_live_state()

        with self.lock:
            # 1. Body Pose & Gestures Tracking (draws MediaPipe skeleton & active movement stats on frame)
            pose_res = self.pose_tracker.process_frame(frame)

            # 2. Video FER Emotion Detection (draws face box & label on frame)
            fer_res = self.fer_detector.predict_frame(frame)

            # 3. Draw On-Screen HUD Overlay Badges on top left of frame
            cv2.putText(frame, f"AI Emotion: {fer_res['emotion']}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Posture: {pose_res['active_posture']}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {pose_res['active_gesture']}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # 4. Encode annotated frame to JPEG Base64 image string
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                annotated_b64 = None

            self.current_results['video_emotion'] = fer_res['emotion']
            self.current_results['raw_emotions'] = fer_res['raw_emotions']
            if annotated_b64:
                self.current_results['annotated_image'] = annotated_b64

            raw_em = fer_res['raw_emotions']
            happy_score = float(raw_em.get('Happy', 0.0))
            nervous_score = float(raw_em.get('Fear', 0.0) * 0.4 + raw_em.get('Sad', 0.0) * 0.3 + raw_em.get('Disgust', 0.0) * 0.15 + raw_em.get('Angry', 0.0) * 0.15)
            neutral_score = float(raw_em.get('Neutral', 0.0) + raw_em.get('Surprise', 0.0) * 0.5)

            tot = happy_score + nervous_score + neutral_score
            if tot > 0:
                self.current_results['emotion_summary'] = {
                    'Happiness': round(happy_score / tot, 2),
                    'Nervous': round(nervous_score / tot, 2),
                    'Neutral': round(neutral_score / tot, 2)
                }

            # Update Pose, Gestures & Active 3D Movement Details
            self.current_results['active_posture'] = pose_res['active_posture']
            self.current_results['active_gesture'] = pose_res['active_gesture']
            self.current_results['movement_info'] = pose_res['movement_info']
            self.current_results['movement_details'] = pose_res.get('movement_details', {})

            is_fidgeting = "Arms Crossed" in pose_res['active_gesture'] or len(pose_res['movement_info']) > 1
            has_hands = "Hand Raised" in pose_res['active_gesture'] or "Waving" in pose_res['active_gesture'] or "Pointing" in pose_res['active_gesture']
            self.current_results['gestures'] = {
                'fidgeting': is_fidgeting,
                'handMovement': has_hands
            }

            # Eye Contact Estimation
            if fer_res['box'] is not None:
                self.current_results['eye_contact'] = min(98, max(65, 88 + (5 if fer_res['emotion'] in ['Happy', 'Neutral'] else -10)))
            else:
                self.current_results['eye_contact'] = 50

            # Dynamic Confidence Score calculation
            conf = 75
            if fer_res['emotion'] == 'Happy':
                conf += 12
            elif fer_res['emotion'] in ['Fear', 'Sad']:
                conf -= 10

            if pose_res['active_posture'] == 'Upright Posture':
                conf += 8
            elif 'Leaning Forward' in pose_res['active_posture']:
                conf += 5
            elif 'Leaning Back' in pose_res['active_posture']:
                conf -= 8

            if is_fidgeting:
                conf -= 5

            self.current_results['confidence_score'] = max(30, min(99, conf))

            # Dynamic AI Feedback
            feedback_parts = []
            if fer_res['emotion'] == 'Happy':
                feedback_parts.append("Positive facial demeanor detected.")
            elif fer_res['emotion'] in ['Fear', 'Sad']:
                feedback_parts.append("Maintain relaxed facial posture.")
            else:
                feedback_parts.append("Calm neutral facial expression.")

            if pose_res['active_posture'] == 'Upright Posture':
                feedback_parts.append("Excellent upright body posture.")
            elif 'Leaning Forward' in pose_res['active_posture']:
                feedback_parts.append("High candidate engagement level.")
            else:
                feedback_parts.append("Maintain centered upright posture.")

            self.current_results['feedback'] = " ".join(feedback_parts)

            # Record session history
            self.session_history['video_emotions'].append(fer_res['emotion'])
            self.session_history['postures'].append(pose_res['active_posture'])
            self.session_history['gestures_counter'] = pose_res['gesture_counters']

        return self.get_live_state()

    def process_audio(self, pcm_audio_data, sample_rate=16000):
        """Processes live microphone PCM float32/int16 buffer array"""
        if pcm_audio_data is None or len(pcm_audio_data) == 0:
            return self.get_live_state()

        # 1. Sound Emotion Detection — returns plain string e.g. 'happy'
        sound_emotion = self.sound_detector.predict_audio(pcm_audio_data, sample_rate)

        # 2. Speech Prosody Feature Extraction — returns list [pitch, intensity, tempo, pause_ratio, stress]
        prosody_res = self.prosody_analyzer.analyze(pcm_audio_data, sample_rate)

        with self.lock:
            self.current_results['sound_emotion'] = sound_emotion
            self.current_results['prosody_features'] = prosody_res

            # Voice tone from prosody features [pitch, intensity, tempo, pause_ratio, stress]
            stress = prosody_res[4] if len(prosody_res) > 4 else 0.0
            pitch  = prosody_res[0] if len(prosody_res) > 0 else 0.0
            intensity = prosody_res[1] if len(prosody_res) > 1 else 0.0
            if stress > 0.6:
                voice_tone = "Hesitant / Anxious"
            elif pitch > 180 and stress < 0.4:
                voice_tone = "Confident & Energetic"
            elif intensity > 0.02:
                voice_tone = "Clear & Articulate"
            else:
                voice_tone = "Neutral Speech"
            self.current_results['voice_tone'] = voice_tone

            self.session_history['sound_emotions'].append(sound_emotion)
            self.session_history['prosody_samples'].append(prosody_res)

        return self.get_live_state()

    def reset_session(self):
        """Resets candidate session history"""
        with self.lock:
            self.fer_detector.reset()
            self.pose_tracker.reset()
            self.session_history = {
                'video_emotions': [],
                'sound_emotions': [],
                'postures': [],
                'prosody_samples': [],
                'gestures_counter': {}
            }
        return {'status': 'reset'}

    def get_live_state(self):
        """Returns JSON-serializable dictionary for API response"""
        with self.lock:
            return {
                'status': 'success',
                'videoEmotion': self.current_results['video_emotion'],
                'soundEmotion': self.current_results['sound_emotion'],
                'raw_emotions': self.current_results['raw_emotions'],
                'emotionSummary': self.current_results['emotion_summary'],
                'activePosture': self.current_results['active_posture'],
                'activeGesture': self.current_results['active_gesture'],
                'movementInfo': self.current_results['movement_info'],
                'movementDetails': self.current_results['movement_details'],
                'prosodyFeatures': self.current_results['prosody_features'],
                'eyeContact': self.current_results['eye_contact'],
                'voiceTone': self.current_results['voice_tone'],
                'confidenceScore': self.current_results['confidence_score'],
                'feedback': self.current_results['feedback'],
                'annotatedImage': self.current_results['annotated_image'],
                'gestures': self.current_results['gestures']
            }

    def generate_final_report(self):
        """Triggers BART NLI Zero-Shot Candidate Evaluation Report"""
        with self.lock:
            report = self.report_generator.generate_report(self.session_history)
        return {'status': 'success', 'report': report}
