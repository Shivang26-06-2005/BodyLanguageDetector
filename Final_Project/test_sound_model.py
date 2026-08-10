"""
Quick sound emotion model test — runs inference on synthetic audio signals
to verify all 7 labels are reachable and the model responds to different inputs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import numpy as np
import torch
from models.sound_model import SoundEmotionDetector, extract_mfcc_features
from config import SOUND_EMOTIONS, DEVICE

detector = SoundEmotionDetector()
model    = detector.model
emotions = SOUND_EMOTIONS
sr       = 16000

print(f"\n{'='*55}")
print(f"  Sound Emotion Model — Sample Inference Test")
print(f"  Device: {DEVICE}   Labels: {emotions}")
print(f"{'='*55}\n")

# ── 1. Raw logits over all 7 classes on silence ──────────────
silence = np.zeros(16000, dtype=np.float32)
mfcc_s  = extract_mfcc_features(silence, sr=sr, max_len=216)
t_s     = torch.tensor(mfcc_s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    logits_s = model(t_s)[0].cpu().numpy()
print("Silence logits:")
for i, (e, v) in enumerate(zip(emotions, logits_s)):
    bar = "#" * max(0, int((v - logits_s.min()) / (logits_s.max() - logits_s.min() + 1e-9) * 20))
    print(f"  [{i}] {e:10s}  {v:+.4f}  {bar}")

# ── 2. Predict on 5 different synthesised audio signals ──────
def make_tone(freq, duration=1.0, amp=0.3):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def make_noise(amp=0.3, duration=1.0):
    return (np.random.randn(int(sr * duration)) * amp).astype(np.float32)

def make_pulse(rate=4, amp=0.4, duration=1.0):
    """Simulates choppy / stressed speech bursts"""
    sig = np.zeros(int(sr * duration), dtype=np.float32)
    period = int(sr / rate)
    for i in range(0, len(sig) - period, period):
        sig[i:i + period//2] = amp
    return sig

samples = [
    ("Low hum 120Hz (male voice range)",  make_tone(120, amp=0.4)),
    ("Mid tone 250Hz",                    make_tone(250, amp=0.3)),
    ("High pitch 400Hz (excited/happy)",  make_tone(400, amp=0.5)),
    ("White noise (ambient mic)",         make_noise(amp=0.05)),
    ("Loud noise (strong signal)",        make_noise(amp=0.4)),
    ("Choppy pulses (stressed speech)",   make_pulse(rate=6, amp=0.4)),
    ("Quiet pulse (soft speech)",         make_pulse(rate=3, amp=0.1)),
]

print(f"\n{'-'*55}")
print(f"  Predicted emotion for each synthetic sample:")
print(f"{'-'*55}")
for name, audio in samples:
    result = detector.predict_audio(audio, sr=sr)
    import librosa
    rms = float(np.mean(librosa.feature.rms(y=audio)[0]))
    print(f"  {name}")
    print(f"    RMS={rms:.4f}  =>  Predicted: [{result}]")

# ── 3. Force all 7 classes — raw argmax without any guard ────
print(f"\n{'-'*55}")
print(f"  Raw argmax (no guards) across 20 random noise inputs:")
print(f"{'-'*55}")
pred_counts = {e: 0 for e in emotions}
for _ in range(20):
    audio = (np.random.randn(16000) * np.random.uniform(0.05, 0.5)).astype(np.float32)
    mfcc  = extract_mfcc_features(audio, sr=sr, max_len=216)
    t     = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = int(torch.argmax(model(t), dim=1).item())
    pred_counts[emotions[pred]] += 1

for e, count in pred_counts.items():
    bar = "#" * count
    print(f"  {e:12s} {bar} ({count}/20)")

print(f"\n{'='*55}")
print("  Done. If multiple classes appear above, model is healthy.")
print(f"{'='*55}\n")
