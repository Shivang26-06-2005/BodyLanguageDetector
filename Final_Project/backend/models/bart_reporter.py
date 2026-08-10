import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BARTCandidateReporter:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            print("[INFO] Initializing BART NLI model (facebook/bart-large-mnli)...")
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            self.model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
            print("[OK] Loaded BART NLI candidate evaluation model successfully")
        except Exception as e:
            print(f"[WARNING] BART model failed to load ({e}). Rule-based fallback active.")
            self.tokenizer = None
            self.model = None

    def generate_report(self, session_history):
        """Synthesize accumulated session history and generate BART Interview Assessment Report"""
        v_emotions = session_history.get('video_emotions', [])
        s_emotions = session_history.get('sound_emotions', [])
        postures = session_history.get('postures', [])
        gestures = session_history.get('gestures_counter', {})
        prosody_records = session_history.get('prosody_records', [])

        prosody = np.array(prosody_records) if len(prosody_records) > 0 else np.zeros((1, 5))

        dom_v = max(set(v_emotions), key=v_emotions.count) if v_emotions else "Neutral"
        dom_s = max(set(s_emotions), key=s_emotions.count) if s_emotions else "neutral"
        dom_posture = max(set(postures), key=postures.count) if postures else "Upright Posture"

        avg_pitch = float(np.mean(prosody[:, 0]))
        avg_int = float(np.mean(prosody[:, 1]))
        avg_tempo = float(np.mean(prosody[:, 2]))
        avg_pause = float(np.mean(prosody[:, 3]))
        avg_stress = float(np.mean(prosody[:, 4]))

        session_summary_text = (
            f"Candidate Interview Performance Summary:\n"
            f"- Facial Expression: Primarily {dom_v}.\n"
            f"- Vocal Emotion: Primarily {dom_s}.\n"
            f"- Body Posture & Language: Maintained {dom_posture} with Hand Raised ({gestures.get('Hand Raised', 0)} times), "
            f"Arms Crossed ({gestures.get('Arms Crossed', 0)} times), Waving ({gestures.get('Waving', 0)} times).\n"
            f"- Vocal Dynamics: Average pitch {avg_pitch:.1f} Hz, volume {avg_int:.3f} RMS, "
            f"speaking tempo {avg_tempo:.1f} syllables per min, pause silence ratio {avg_pause*100:.1f}%, and vocal stress {avg_stress:.2f}."
        )

        hypotheses = {
            'Confidence & Poise': "This candidate demonstrates high confidence, poise, and self-assurance during the interview",
            'Interview Engagement': "This candidate demonstrates high engagement, active listening, and attentiveness",
            'Nervousness & Vocal Anxiety': "This candidate displays signs of nervousness, tension, or vocal anxiety",
            'Professional Posture & Demeanor': "This candidate maintains professional posture and effective body language",
            'Overall Candidate Positivity': "This candidate maintains a positive and encouraging attitude"
        }

        scores = {}
        if self.tokenizer is not None and self.model is not None:
            for aspect, hypothesis in hypotheses.items():
                try:
                    inputs = self.tokenizer(session_summary_text, hypothesis, return_tensors='pt', truncation=True)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        # Index 2 = Entailment probability
                        score = float(probs[0][2].item() * 100)
                        scores[aspect] = round(score, 1)
                except Exception as e:
                    scores[aspect] = 50.0
        else:
            # Rule-based fallback metrics derived from session stats
            positivity = 80.0 if dom_v in ['Happy', 'Neutral'] else 45.0
            anxiety = round(float(avg_stress * 100.0), 1)
            confidence = round(max(0.0, min(100.0, 100.0 - anxiety*0.5 + (20 if dom_posture == 'Upright Posture' else 0))), 1)
            engagement = 85.0 if gestures.get('Hand Raised', 0) > 0 else 70.0
            posture_score = 90.0 if dom_posture == 'Upright Posture' else 65.0

            scores = {
                'Confidence & Poise': confidence,
                'Interview Engagement': engagement,
                'Nervousness & Vocal Anxiety': anxiety,
                'Professional Posture & Demeanor': posture_score,
                'Overall Candidate Positivity': positivity
            }

        return {
            'narrative': session_summary_text,
            'scores': scores,
            'dominant_video_emotion': dom_v,
            'dominant_sound_emotion': dom_s,
            'dominant_posture': dom_posture,
            'vocal_metrics': {
                'avg_pitch_hz': round(avg_pitch, 1),
                'avg_intensity_rms': round(avg_int, 4),
                'avg_tempo_syll_min': round(avg_tempo, 1),
                'avg_pause_ratio_pct': round(avg_pause * 100, 1),
                'avg_stress': round(avg_stress, 2)
            }
        }
