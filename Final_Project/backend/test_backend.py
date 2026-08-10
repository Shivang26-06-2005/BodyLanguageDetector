import os
import sys
import numpy as np
import cv2
import base64

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

def run_tests():
    print("Testing Flask endpoints...")
    client = app.test_client()

    # 1. Test /api/status
    res = client.get('/api/status')
    assert res.status_code == 200
    print("[PASS] /api/status ->", res.get_json())

    # 2. Test /api/predict_frame with dummy black frame
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', dummy_frame)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    res_frame = client.post('/api/predict_frame', json={'image': b64_str})
    assert res_frame.status_code == 200
    print("[PASS] /api/predict_frame ->", res_frame.get_json()['status'], "| emotion:", res_frame.get_json()['videoEmotion'])

    # 3. Test /api/predict_audio with dummy audio array
    dummy_audio = [0.0] * 8000
    res_audio = client.post('/api/predict_audio', json={'audio': dummy_audio})
    assert res_audio.status_code == 200
    print("[PASS] /api/predict_audio ->", res_audio.get_json()['status'], "| tone:", res_audio.get_json()['voiceTone'])

    # 4. Test /api/generate_report
    res_report = client.get('/api/generate_report')
    assert res_report.status_code == 200
    report_json = res_report.get_json()
    print("[PASS] /api/generate_report ->", report_json['status'])
    print("       Scorecard Scores:", report_json['report']['scores'])

    print("\n[SUCCESS] All backend routes tested and functioning perfectly!")

if __name__ == '__main__':
    run_tests()
