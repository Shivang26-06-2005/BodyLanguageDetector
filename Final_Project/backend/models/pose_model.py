import cv2
import mediapipe as mp
import math
from collections import deque

import threading

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def distance(a, b):
    """Calculate 3D Euclidean distance between two landmarks (from FinalFuse.py)"""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def get_precise_direction(dx, dy, dz, thresh=0.01):
    """Returns a combined direction string for 3D movement (from FinalFuse.py)"""
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

class BodyPoseTracker:
    def __init__(self):
        print("[INFO] Initializing MediaPipe 3D Pose Tracker (from FinalFuse.py)...")
        self.lock = threading.Lock()
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.counters = {
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

        self.baseline_z = None
        self.wrist_history = {"left": deque(maxlen=5), "right": deque(maxlen=5)}
        
        # 3D History tracking for Head, Shoulders, Left Hand, Right Hand (from FinalFuse.py)
        self.shoulder_history = deque(maxlen=2)
        self.head_history = deque(maxlen=2)
        self.left_wrist_history = deque(maxlen=2)
        self.right_wrist_history = deque(maxlen=2)

    def reset(self):
        """Reset session state — called by fusion_engine.reset_session() on interview start."""
        self.baseline_z = None
        self.wrist_history = {"left": deque(maxlen=5), "right": deque(maxlen=5)}
        self.shoulder_history = deque(maxlen=2)
        self.head_history = deque(maxlen=2)
        self.left_wrist_history = deque(maxlen=2)
        self.right_wrist_history = deque(maxlen=2)
        for key in self.counters:
            self.counters[key] = 0

    def process_frame(self, frame):
        """
        Exact process_body_pose implementation from FinalFuse.py:
        Tracks 3D body posture, leaning, gestures, and active 3D movement (Head, Shoulders, Hands).
        """
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        with self.lock:
            results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True

        posture = "Upright Posture"
        active_gesture = "Hand Lowered"
        movement_info = []
        movement_details = {
            "head_eyes": "No Movement",
            "shoulders": "No Movement",
            "left_hand": "No Movement",
            "right_hand": "No Movement"
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 1. Draw MediaPipe 3D Pose Skeleton on frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 2. Torso Midpoints for 3D Posture Leaning (from FinalFuse.py)
            l_sh, r_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip, r_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            mid_sh_x = (l_sh.x + r_sh.x) / 2
            mid_sh_y = (l_sh.y + r_sh.y) / 2
            mid_sh_z = (l_sh.z + r_sh.z) / 2

            mid_hip_x = (l_hip.x + r_hip.x) / 2
            mid_hip_y = (l_hip.y + r_hip.y) / 2
            mid_hip_z = (l_hip.z + r_hip.z) / 2

            if self.baseline_z is None:
                self.baseline_z = (mid_sh_z + mid_hip_z) / 2

            dx = mid_hip_x - mid_sh_x
            dz = ((mid_hip_z + mid_sh_z) / 2) - self.baseline_z
            lean_thresh_z = 0.02
            lean_thresh_x = 0.02

            if dz < -lean_thresh_z:
                if dx > lean_thresh_x:
                    posture = "Leaning Forward-Right"
                    self.counters["Leaning Forward"] += 1
                    self.counters["Leaning Right"] += 1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Forward-Left"
                    self.counters["Leaning Forward"] += 1
                    self.counters["Leaning Left"] += 1
                else:
                    posture = "Leaning Forward"
                    self.counters["Leaning Forward"] += 1
            elif dz > lean_thresh_z:
                if dx > lean_thresh_x:
                    posture = "Leaning Back-Right"
                    self.counters["Leaning Back"] += 1
                    self.counters["Leaning Right"] += 1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Back-Left"
                    self.counters["Leaning Back"] += 1
                    self.counters["Leaning Left"] += 1
                else:
                    posture = "Leaning Back"
                    self.counters["Leaning Back"] += 1
            else:
                if dx > lean_thresh_x:
                    posture = "Leaning Right"
                    self.counters["Leaning Right"] += 1
                elif dx < -lean_thresh_x:
                    posture = "Leaning Left"
                    self.counters["Leaning Left"] += 1
                else:
                    posture = "Upright Posture"
                    self.counters["Upright Posture"] += 1

            # 3. Gestures Detection (from FinalFuse.py)
            l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            if distance(l_wrist, r_elbow) < 0.12 and distance(r_wrist, l_elbow) < 0.12:
                active_gesture = "Arms Crossed"
                self.counters["Arms Crossed"] += 1
            elif l_wrist.y < l_sh.y or r_wrist.y < r_sh.y:
                active_gesture = "Hand Raised"
                self.counters["Hand Raised"] += 1
            else:
                active_gesture = "Hand Lowered"
                self.counters["Hand Lowered"] += 1

            # Pointing
            l_angle = math.degrees(math.atan2(l_wrist.y - l_elbow.y, l_wrist.x - l_elbow.x))
            r_angle = math.degrees(math.atan2(r_wrist.y - r_elbow.y, r_wrist.x - r_elbow.x))
            if abs(l_angle) < 30 or abs(r_angle) < 30:
                active_gesture = "Pointing"
                self.counters["Pointing"] += 1

            # Waving
            self.wrist_history["left"].append(l_wrist.x)
            self.wrist_history["right"].append(r_wrist.x)
            if len(self.wrist_history["left"]) == 5 and max(self.wrist_history["left"]) - min(self.wrist_history["left"]) > 0.05:
                active_gesture = "Waving"
                self.counters["Waving"] += 1
            if len(self.wrist_history["right"]) == 5 and max(self.wrist_history["right"]) - min(self.wrist_history["right"]) > 0.05:
                active_gesture = "Waving"
                self.counters["Waving"] += 1

            # Object in Hand
            if distance(l_wrist, l_elbow) < 0.05 or distance(r_wrist, r_elbow) < 0.05:
                self.counters["Object in Hand"] += 1

            # 4. Active 3D Movement Tracking (from FinalFuse.py)
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

            head_mid = ((nose.x + left_eye.x + right_eye.x) / 3,
                        (nose.y + left_eye.y + right_eye.y) / 3,
                        (nose.z + left_eye.z + right_eye.z) / 3)
            self.head_history.append(head_mid)

            shoulder_mid = (mid_sh_x, mid_sh_y, mid_sh_z)
            self.shoulder_history.append(shoulder_mid)

            self.left_wrist_history.append((l_wrist.x, l_wrist.y, l_wrist.z))
            self.right_wrist_history.append((r_wrist.x, r_wrist.y, r_wrist.z))

            if len(self.shoulder_history) == 2:
                dx_s, dy_s, dz_s = [self.shoulder_history[1][i] - self.shoulder_history[0][i] for i in range(3)]
                s_dir = get_precise_direction(dx_s, dy_s, dz_s)
                movement_details["shoulders"] = s_dir
                movement_info.append(f"Shoulders: {s_dir}")

            if len(self.head_history) == 2:
                dx_h, dy_h, dz_h = [self.head_history[1][i] - self.head_history[0][i] for i in range(3)]
                h_dir = get_precise_direction(dx_h, dy_h, dz_h)
                movement_details["head_eyes"] = h_dir
                movement_info.append(f"Head: {h_dir}")

            if len(self.left_wrist_history) == 2:
                dx_l, dy_l, dz_l = [self.left_wrist_history[1][i] - self.left_wrist_history[0][i] for i in range(3)]
                l_dir = get_precise_direction(dx_l, dy_l, dz_l)
                movement_details["left_hand"] = l_dir
                movement_info.append(f"Left Hand: {l_dir}")

            if len(self.right_wrist_history) == 2:
                dx_r, dy_r, dz_r = [self.right_wrist_history[1][i] - self.right_wrist_history[0][i] for i in range(3)]
                r_dir = get_precise_direction(dx_r, dy_r, dz_r)
                movement_details["right_hand"] = r_dir
                movement_info.append(f"Right Hand: {r_dir}")

            # Draw movement text on top-right of frame (from FinalFuse.py)
            start_y = 30
            for i, info in enumerate(movement_info):
                cv2.putText(frame, info, (max(10, w - 260), start_y + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        return {
            'active_posture': posture,
            'active_gesture': active_gesture,
            'movement_info': movement_info,
            'movement_details': movement_details,
            'gesture_counters': dict(self.counters)
        }
