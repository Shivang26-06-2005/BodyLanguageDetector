import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure backend root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE
from services.fusion_engine import MultiModalFusionEngine

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Initialize Fusion Engine singleton
engine = MultiModalFusionEngine()

@app.route('/api/status', methods=['GET'])
def status():
    """Server health check and model status"""
    return jsonify({
        'status': 'online',
        'engine': 'PulseMind AI Multimodal Fusion Engine',
        'device': str(DEVICE),
        'version': '2.0'
    })

@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    """Predict emotion, 3D pose, gestures & confidence from video frame"""
    try:
        data = request.get_json(force=True)
        image_data = data.get('image', None)
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

        result = engine.process_frame(image_data)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] /api/predict_frame exception: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict_audio', methods=['POST'])
def predict_audio():
    """Predict sound emotion & prosody metrics from audio array"""
    try:
        data = request.get_json(force=True)
        audio_list = data.get('audio', None)
        sr = data.get('sample_rate', 16000)

        if not audio_list or not isinstance(audio_list, list):
            return jsonify({'status': 'error', 'message': 'No audio array provided'}), 400

        audio_np = np.array(audio_list, dtype=np.float32)
        result = engine.process_audio(audio_np, sample_rate=sr)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] /api/predict_audio exception: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict_multimodal', methods=['POST'])
def predict_multimodal():
    """Unified endpoint accepting video frame + optional audio chunk"""
    try:
        data = request.get_json(force=True)
        image_data = data.get('image', None)
        audio_list = data.get('audio', None)

        if image_data:
            engine.process_frame(image_data)

        if audio_list and isinstance(audio_list, list):
            audio_np = np.array(audio_list, dtype=np.float32)
            engine.process_audio(audio_np)

        return jsonify(engine.get_live_state())
    except Exception as e:
        print(f"[ERROR] /api/predict_multimodal exception: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate_report', methods=['GET', 'POST'])
def generate_report():
    """Generate final BART candidate assessment report and scorecard"""
    try:
        report_data = engine.generate_final_report()
        return jsonify(report_data)
    except Exception as e:
        print(f"[ERROR] /api/generate_report exception: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    """Reset session history"""
    try:
        return jsonify(engine.reset_session())
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n========================================================")
    print(f"  PulseMind AI Flask Backend Server Running on Port {port}  ")
    print(f"========================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
