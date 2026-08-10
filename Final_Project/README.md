# HireVisor — AI Candidate Assessment System (Final Working Project)

HireVisor is an AI-powered human interview candidate assessment system. It combines multi-modal deep learning models (FER_CNN facial emotion recognition, MediaPipe 3D Pose & Body Language tracking, Sound Emotion CNN, ProsodyNet voice dynamics, and BART NLI candidate scorecard generation) connected via a Flask REST API backend to a React frontend.

---

##  Directory Structure (`Final_Project/`)

```text
Final_Project/
├── backend/
│   ├── app.py                 # Main Flask REST API server (Port 5000)
│   ├── config.py              # Configuration & model path resolver
│   ├── requirements.txt       # Backend dependencies
│   ├── test_backend.py        # Automated test verification script
│   ├── models/
│   │   ├── fer_model.py       # FER_CNN video facial emotion model
│   │   ├── sound_model.py     # Sound emotion CNN classifier
│   │   ├── prosody_model.py   # ProsodyNet audio metrics model
│   │   ├── pose_model.py      # MediaPipe 3D pose & gesture tracker
│   │   └── bart_reporter.py   # BART NLI zero-shot candidate reporter
│   └── services/
│       └── fusion_engine.py   # MultiModal fusion manager
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── StartInterview.js  # Live video AI analyzer & camera component
│   │   │   ├── Home.js           # Landing page
│   │   │   └── Navbar.js         # Navigation menu
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
│
└── models_weights/
    ├── fer_model_finetuned_v2.pth
    ├── sound_emotion_model.pth
    └── prosody_net_best.pth
```

---

##  Quick Start Guide

### Step 1: Launch Flask AI Backend
Open a terminal in the `backend` folder:
```bash
cd backend
python app.py
```
*(The Flask backend will start on `http://127.0.0.1:5000`)*

### Step 2: Launch React Frontend
Open another terminal in the `frontend` folder:
```bash
cd frontend
npm start
```
*(The React frontend app will launch on `http://localhost:3000`)*

---

##  Testing Backend Server
To test all REST API endpoints and model loading:
```bash
cd backend
python test_backend.py
```
