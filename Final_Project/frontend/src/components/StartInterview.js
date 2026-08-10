import React, { useState, useRef, useEffect, useCallback } from "react";

/**
 * StartInterview.js
 * HierVisor — Connected to PulseMind AI Modular Flask Backend (Port 5000)
 */

export default function StartInterview() {
  const API_BASE_URL =
    typeof window !== "undefined"
      ? `http://${window.location.hostname}:5000`
      : "http://127.0.0.1:5000";

  // media & audio refs
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const audioContextRef = useRef(null);
  const scriptProcessorRef = useRef(null);

  // UI & responsive state
  const [isRunning, setIsRunning] = useState(false);
  const [backendConnected, setBackendConnected] = useState(false);
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined" ? window.innerWidth <= 900 : false
  );

  // Button hover states
  const [hoverStart, setHoverStart] = useState(false);
  const [hoverStop, setHoverStop] = useState(false);
  const [hoverReport, setHoverReport] = useState(false);

  // Live inference state from PulseMind AI Flask Backend
  const [rawEmotions, setRawEmotions] = useState({
    Happy: 0.0, Fear: 0.0, Angry: 0.0, Surprise: 0.0, Sad: 0.0, Disgust: 0.0, Neutral: 1.0
  });
  const [videoEmotion, setVideoEmotion] = useState("Neutral");
  const [soundEmotion, setSoundEmotion] = useState("neutral");
  const [activePosture, setActivePosture] = useState("Upright Posture");
  const [activeGesture, setActiveGesture] = useState("Hand Lowered");
  const [movementDetails, setMovementDetails] = useState({
    head_eyes: "No Movement",
    shoulders: "No Movement",
    left_hand: "No Movement",
    right_hand: "No Movement"
  });
  const [prosodyFeatures, setProsodyFeatures] = useState([0.0, 0.0, 0.0, 0.0, 0.0]); // Pitch, Int, Tempo, Pause, Stress
  const [eyeContact, setEyeContact] = useState(85);
  const [voiceTone, setVoiceTone] = useState("Neutral Speech");
  const [confidenceScore, setConfidenceScore] = useState(85);
  const [feedback, setFeedback] = useState("Click 'Start Interview' to begin live AI candidate assessment.");
  const [annotatedImage, setAnnotatedImage] = useState(null);

  // BART Report modal state
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportData, setReportData] = useState(null);

  // Check backend server status on mount
  const checkBackendStatus = useCallback(() => {
    fetch(`${API_BASE_URL}/api/status`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "online") {
          setBackendConnected(true);
        } else {
          setBackendConnected(false);
        }
      })
      .catch(() => {
        setBackendConnected(false);
      });
  }, [API_BASE_URL]);

  useEffect(() => {
    checkBackendStatus();
    const statusTimer = setInterval(checkBackendStatus, 3000);
    return () => clearInterval(statusTimer);
  }, [checkBackendStatus]);

  // Responsive handler
  useEffect(() => {
    function onResize() {
      setIsMobile(window.innerWidth <= 900);
    }
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Original Frame capture and backend inference sender
  const captureAndSendFrame = () => {
    if (!videoRef.current || !streamRef.current) return;
    const video = videoRef.current;
    
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
      canvasRef.current.width = 640;
      canvasRef.current.height = 480;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameBase64 = canvas.toDataURL("image/jpeg", 0.75);

    fetch(`${API_BASE_URL}/api/predict_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: frameBase64 }),
    })
      .then((res) => res.json())
      .then((data) => handleInferenceResult(data))
      .catch((err) => {
        console.error("API inference error:", err);
        setBackendConnected(false);
      });
  };

  // Handle Flask backend inference result
  const handleInferenceResult = (data) => {
    if (!data || data.status !== "success") return;
    setBackendConnected(true);
    if (data.raw_emotions) setRawEmotions(data.raw_emotions);
    if (data.videoEmotion) setVideoEmotion(data.videoEmotion);
    if (data.soundEmotion) setSoundEmotion(data.soundEmotion);
    if (data.activePosture) setActivePosture(data.activePosture);
    if (data.activeGesture) setActiveGesture(data.activeGesture);
    if (data.movementDetails) setMovementDetails(data.movementDetails);
    if (data.prosodyFeatures) setProsodyFeatures(data.prosodyFeatures);
    if (typeof data.eyeContact === "number") setEyeContact(data.eyeContact);
    if (data.voiceTone) setVoiceTone(data.voiceTone);
    if (typeof data.confidenceScore === "number") setConfidenceScore(data.confidenceScore);
    if (data.feedback) setFeedback(data.feedback);
    if (data.annotatedImage) setAnnotatedImage(data.annotatedImage);
  };

  // Start webcam, mic audio recorder, and frame loop
  const startInterview = async () => {
    if (isRunning) return;
    try {
      // Clear old state & reset backend session
      setAnnotatedImage(null);
      fetch(`${API_BASE_URL}/api/reset_session`, { method: "POST" }).catch(() => {});

      const constraints = { video: { width: 1280, height: 720 }, audio: true };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setIsRunning(true);
      setFeedback("PulseMind AI active. Analyzing live video, pose, & speech prosody...");

      // Initialize Web Audio API for Live Speech Prosody & Audio Emotion Inference
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        const audioCtx = new AudioContextClass({ sampleRate: 16000 });
        audioContextRef.current = audioCtx;

        const source = audioCtx.createMediaStreamSource(stream);
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        scriptProcessorRef.current = processor;

        let audioBuffer = [];
        processor.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0);
          for (let i = 0; i < inputData.length; i++) {
            audioBuffer.push(inputData[i]);
          }
          // Process 1-second chunks (16000 audio samples at 16kHz)
          if (audioBuffer.length >= 16000) {
            const chunk = audioBuffer.splice(0, 16000);
            fetch(`${API_BASE_URL}/api/predict_audio`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ audio: chunk, sample_rate: 16000 }),
            })
              .then((res) => res.json())
              .then((data) => handleInferenceResult(data))
              .catch((err) => console.error("Audio API error:", err));
          }
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);
      } catch (audioErr) {
        console.warn("Web Audio API stream initialization note:", audioErr);
      }

      // Start frame capture loop (~3.3 FPS / 300ms interval for real-time video inference)
      sendIntervalRef.current = setInterval(() => {
        captureAndSendFrame();
      }, 300);
    } catch (err) {
      console.error("Error accessing camera/mic:", err);
      alert("Unable to access camera or microphone. Please grant permission and try again.");
    }
  };

  // Stop webcam, audio stream & cleanup
  const stopInterview = () => {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
    if (scriptProcessorRef.current) {
      try { scriptProcessorRef.current.disconnect(); } catch (e) {}
      scriptProcessorRef.current = null;
    }
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (e) {}
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      try {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      } catch (e) {}
    }
    setIsRunning(false);
    setAnnotatedImage(null);
    setFeedback("Interview stopped. Click 'Generate Final BART Report' for assessment.");
  };

  // Fetch and display Final BART Candidate Assessment Report
  const generateBARTReport = () => {
    setReportLoading(true);
    setShowReportModal(true);

    fetch(`${API_BASE_URL}/api/generate_report`)
      .then((res) => res.json())
      .then((data) => {
        setReportLoading(false);
        if (data && data.status === "success") {
          setReportData(data.report);
        } else {
          alert("Failed to generate BART report. Please ensure backend is running.");
        }
      })
      .catch((err) => {
        setReportLoading(false);
        console.error("Report generation error:", err);
        alert("Error connecting to backend for report generation.");
      });
  };

  // Download report JSON file
  const downloadReport = () => {
    if (!reportData) return;
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hiervisor_candidate_report_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopInterview();
    };
    // eslint-disable-next-line
  }, []);

  // Styles
  const pageStyle = {
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    background: "#f8fafc",
    minHeight: "100vh",
    padding: isMobile ? "80px 12px 40px" : "90px 32px 40px",
    boxSizing: "border-box",
  };

  const headerStyle = {
    maxWidth: 1200,
    margin: "0 auto 20px",
    textAlign: "center",
  };

  const titleStyle = {
    fontSize: isMobile ? "1.6rem" : "2.2rem",
    fontWeight: 800,
    margin: 0,
    color: "#0f1724",
    letterSpacing: "-0.5px",
  };

  const subtitleStyle = {
    marginTop: 8,
    color: "#475569",
    fontSize: isMobile ? "0.9rem" : "1.02rem",
  };

  const statusBadgeStyle = {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 14px",
    borderRadius: 20,
    fontSize: "0.85rem",
    fontWeight: 600,
    marginTop: 10,
    backgroundColor: backendConnected ? "#ecfdf5" : "#fff1f2",
    color: backendConnected ? "#047857" : "#be123c",
    border: `1px solid ${backendConnected ? "#a7f3d0" : "#fecdd3"}`,
  };

  const layoutStyle = {
    maxWidth: 1200,
    margin: "16px auto 0",
    display: "flex",
    gap: 24,
    flexDirection: isMobile ? "column" : "row",
    alignItems: "stretch",
    justifyContent: "center",
  };

  const leftStyle = {
    flex: isMobile ? "unset" : "1 1 50%",
    background: "#fff",
    borderRadius: 16,
    padding: 20,
    boxShadow: "0 10px 25px rgba(15,23,42,0.06)",
    display: "flex",
    flexDirection: "column",
    gap: 14,
  };

  const videoWrapperStyle = {
    width: "100%",
    height: 380,
    borderRadius: 12,
    overflow: "hidden",
    background: "#0f172a",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  };

  const controlRowStyle = {
    display: "flex",
    gap: 12,
    alignItems: "center",
    marginTop: 6,
    flexWrap: "wrap",
  };

  const buttonBase = {
    padding: "12px 20px",
    borderRadius: 10,
    border: "none",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: "0.95rem",
    transition: "all 0.2s ease",
    boxShadow: "0 4px 12px rgba(2,6,23,0.08)",
  };

  const startButtonStyle = {
    ...buttonBase,
    background: hoverStart ? "#0f766e" : "#0d9488",
    color: "#fff",
  };

  const stopButtonStyle = {
    ...buttonBase,
    background: hoverStop ? "#be123c" : "#e11d48",
    color: "#fff",
  };

  const reportButtonStyle = {
    ...buttonBase,
    background: hoverReport ? "#1e1b4b" : "#312e81",
    color: "#fff",
  };

  const rightStyle = {
    flex: isMobile ? "unset" : "1 1 50%",
    background: "#fff",
    borderRadius: 16,
    padding: 20,
    boxShadow: "0 10px 25px rgba(15,23,42,0.06)",
    display: "flex",
    flexDirection: "column",
    gap: 14,
  };

  const statRowStyle = {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 10,
    background: "#f8fafc",
    padding: 12,
    borderRadius: 10,
  };

  const smallLabelStyle = { color: "#64748b", fontSize: "0.8rem", fontWeight: 500 };

  const barOuter = {
    background: "#e2e8f0",
    borderRadius: 6,
    height: 8,
    overflow: "hidden",
    marginTop: 4,
  };

  const barInner = (pct, color) => ({
    width: `${Math.round((pct || 0) * 100)}%`,
    height: "100%",
    background: color,
    transition: "width 0.4s ease",
  });

  const confidenceOuter = { background: "#eef2ff", borderRadius: 8, height: 14, overflow: "hidden" };
  const confidenceInner = (v) => ({
    width: `${Math.round(v)}%`,
    height: "100%",
    background: `linear-gradient(90deg, #10b981, #06b6d4)`,
    transition: "width 0.4s ease",
  });

  return (
    <div id="start-interview-section" style={pageStyle}>
      <header style={headerStyle}>
        <h1 style={titleStyle}>HierVisor — AI Candidate Assessment Studio</h1>
        <p style={subtitleStyle}>
          PulseMind Multi-Modal AI (FER_CNN, MediaPipe 3D Pose, Speech Prosody & BART NLI)
        </p>
        <div style={statusBadgeStyle}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: backendConnected ? "#10b981" : "#ef4444",
            }}
          ></span>
          {backendConnected
            ? "PulseMind AI Flask Engine: Online (Port 5000)"
            : "PulseMind AI Flask Engine: Disconnected (Run python backend/app.py)"}
        </div>
      </header>

      <main style={layoutStyle}>
        {/* LEFT: Live Webcam Feed & MediaPipe AI Overlay */}
        <section style={leftStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ margin: 0, fontSize: "1.1rem", color: "#0f1724" }}>Live AI Stream & MediaPipe Overlay</h3>
            <div
              style={{
                color: isRunning ? "#059669" : "#64748b",
                fontSize: "0.85rem",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              {isRunning && (
                <span style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: "#10b981" }} />
              )}
              {isRunning ? "PulseMind AI Active" : "Idle"}
            </div>
          </div>

          <div style={videoWrapperStyle}>
            {!isRunning && (
              <div style={{ color: "#94a3b8", textAlign: "center", padding: 20 }}>
                <div style={{ fontSize: 44, marginBottom: 8 }}>📷</div>
                <div style={{ fontWeight: 600, fontSize: "1.05rem" }}>Camera & microphone standby</div>
                <div style={{ fontSize: "0.85rem", marginTop: 6, color: "#64748b" }}>
                  Click "Start Interview" to launch MediaPipe pose tracking, facial emotion & audio analysis
                </div>
              </div>
            )}

            {/* Live webcam video element - ALWAYS visible when running */}
            <video
              ref={videoRef}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                display: isRunning ? "block" : "none",
                position: "relative",
                zIndex: 1
              }}
              playsInline
              muted
              autoPlay
            />

            {/* Live Annotated AI Image Stream Overlay (MediaPipe Skeleton + Bounding Box + Emotion Tag) */}
            {isRunning && annotatedImage && (
              <img
                src={annotatedImage}
                alt="Live AI Inference Stream Overlay"
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  zIndex: 2,
                  pointerEvents: "none"
                }}
              />
            )}
          </div>

          <div style={controlRowStyle}>
            <button
              style={startButtonStyle}
              onClick={startInterview}
              onMouseOver={() => setHoverStart(true)}
              onMouseOut={() => setHoverStart(false)}
              disabled={isRunning}
            >
              Start Interview
            </button>

            <button
              style={stopButtonStyle}
              onClick={stopInterview}
              onMouseOver={() => setHoverStop(true)}
              onMouseOut={() => setHoverStop(false)}
              disabled={!isRunning}
            >
              Stop Interview
            </button>

            <button
              style={reportButtonStyle}
              onClick={generateBARTReport}
              onMouseOver={() => setHoverReport(true)}
              onMouseOut={() => setHoverReport(false)}
            >
              Generate Final BART Report
            </button>
          </div>
        </section>

        {/* RIGHT: Full Backend Multi-Modal Analytics Dashboard */}
        <aside style={rightStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ margin: 0, fontWeight: 700, color: "#0f1724", fontSize: "1.1rem" }}>
              PulseMind Multi-Modal Analytics
            </h3>
            <div style={{ display: "flex", gap: 8 }}>
              <span style={{ background: "#ecfdf5", color: "#047857", padding: "4px 10px", borderRadius: 12, fontSize: "0.8rem", fontWeight: 700 }}>
                Face: {videoEmotion}
              </span>
              <span style={{ background: "#eff6ff", color: "#1d4ed8", padding: "4px 10px", borderRadius: 12, fontSize: "0.8rem", fontWeight: 700 }}>
                Audio: {soundEmotion}
              </span>
            </div>
          </div>

          {/* MediaPipe Active 3D Movement Analysis (Head, Eyes, Shoulders, Left/Right Hands) */}
          <div style={{ background: "#f8fafc", padding: 12, borderRadius: 10, border: "1px solid #e2e8f0" }}>
            <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "#1e293b", marginBottom: 8 }}>
              🏃 MediaPipe 3D Active Movement Tracking
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <div style={{ background: "#fff", padding: 8, borderRadius: 8, border: "1px solid #f1f5f9" }}>
                <div style={{ color: "#64748b", fontSize: "0.75rem" }}>👤 Head / Neck / Eyes</div>
                <div style={{ fontWeight: 700, color: movementDetails.head_eyes !== "No Movement" ? "#059669" : "#64748b", fontSize: "0.85rem", marginTop: 2 }}>
                  {movementDetails.head_eyes || "No Movement"}
                </div>
              </div>

              <div style={{ background: "#fff", padding: 8, borderRadius: 8, border: "1px solid #f1f5f9" }}>
                <div style={{ color: "#64748b", fontSize: "0.75rem" }}>🦾 Shoulders</div>
                <div style={{ fontWeight: 700, color: movementDetails.shoulders !== "No Movement" ? "#059669" : "#64748b", fontSize: "0.85rem", marginTop: 2 }}>
                  {movementDetails.shoulders || "No Movement"}
                </div>
              </div>

              <div style={{ background: "#fff", padding: 8, borderRadius: 8, border: "1px solid #f1f5f9" }}>
                <div style={{ color: "#64748b", fontSize: "0.75rem" }}>✋ Left Hand</div>
                <div style={{ fontWeight: 700, color: movementDetails.left_hand !== "No Movement" ? "#2563eb" : "#64748b", fontSize: "0.85rem", marginTop: 2 }}>
                  {movementDetails.left_hand || "No Movement"}
                </div>
              </div>

              <div style={{ background: "#fff", padding: 8, borderRadius: 8, border: "1px solid #f1f5f9" }}>
                <div style={{ color: "#64748b", fontSize: "0.75rem" }}>🤚 Right Hand</div>
                <div style={{ fontWeight: 700, color: movementDetails.right_hand !== "No Movement" ? "#2563eb" : "#64748b", fontSize: "0.85rem", marginTop: 2 }}>
                  {movementDetails.right_hand || "No Movement"}
                </div>
              </div>
            </div>
          </div>

          {/* 7-Facial Emotion Probability Distribution */}
          <div style={{ background: "#f8fafc", padding: 12, borderRadius: 12, border: "1px solid #e2e8f0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: "0.85rem", color: "#1e293b" }}>FER_CNN 7-Emotion Breakdown</span>
              <span style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: 600 }}>Deep CNN</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 14px" }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 600, color: "#334155" }}>
                  <span>😊 Happy</span>
                  <span>{Math.round((rawEmotions.Happy || 0) * 100)}%</span>
                </div>
                <div style={barOuter}><div style={barInner(rawEmotions.Happy || 0, "#10b981")} /></div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 600, color: "#334155" }}>
                  <span>😐 Neutral</span>
                  <span>{Math.round((rawEmotions.Neutral || 0) * 100)}%</span>
                </div>
                <div style={barOuter}><div style={barInner(rawEmotions.Neutral || 0, "#64748b")} /></div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 600, color: "#334155" }}>
                  <span>😰 Fear / Nervous</span>
                  <span>{Math.round((rawEmotions.Fear || 0) * 100)}%</span>
                </div>
                <div style={barOuter}><div style={barInner(rawEmotions.Fear || 0, "#f97316")} /></div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 600, color: "#334155" }}>
                  <span>😲 Surprise</span>
                  <span>{Math.round((rawEmotions.Surprise || 0) * 100)}%</span>
                </div>
                <div style={barOuter}><div style={barInner(rawEmotions.Surprise || 0, "#8b5cf6")} /></div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 600, color: "#334155" }}>
                  <span>😢 Sad</span>
                  <span>{Math.round((rawEmotions.Sad || 0) * 100)}%</span>
                </div>
                <div style={barOuter}><div style={barInner(rawEmotions.Sad || 0, "#3b82f6")} /></div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 600, color: "#334155" }}>
                  <span>😡 Angry</span>
                  <span>{Math.round((rawEmotions.Angry || 0) * 100)}%</span>
                </div>
                <div style={barOuter}><div style={barInner(rawEmotions.Angry || 0, "#ef4444")} /></div>
              </div>
            </div>
          </div>

          {/* MediaPipe 3D Body Language & Posture */}
          <div style={statRowStyle}>
            <div>
              <div style={smallLabelStyle}>Body Posture</div>
              <div style={{ fontWeight: 700, color: "#0f1724", fontSize: "0.85rem", marginTop: 2 }}>{activePosture}</div>
            </div>
            <div>
              <div style={smallLabelStyle}>Active Gesture</div>
              <div style={{ fontWeight: 700, color: "#0f1724", fontSize: "0.85rem", marginTop: 2 }}>{activeGesture}</div>
            </div>
            <div>
              <div style={smallLabelStyle}>Eye Contact Score</div>
              <div style={{ fontWeight: 700, color: "#0f1724", fontSize: "0.85rem", marginTop: 2 }}>{eyeContact}%</div>
            </div>
          </div>

          {/* Speech Prosody & Vocal Dynamics */}
          <div style={{ background: "#f8fafc", padding: 12, borderRadius: 10, border: "1px solid #e2e8f0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: "0.85rem", color: "#1e293b" }}>🎙️ Speech Prosody Features</span>
              <span style={{ fontWeight: 700, fontSize: "0.85rem", color: "#0d9488" }}>{voiceTone}</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, textAlign: "center" }}>
              <div style={{ background: "#fff", padding: 8, borderRadius: 8, border: "1px solid #f1f5f9" }}>
                <div style={{ color: "#64748b", fontSize: "0.75rem" }}>Pitch (F0)</div>
                <div style={{ fontWeight: 700, color: "#0f1724", fontSize: "0.85rem", marginTop: 2 }}>
                  {prosodyFeatures[0] > 0 ? `${prosodyFeatures[0].toFixed(1)} Hz` : "Listening..."}
                </div>
              </div>

              <div style={{ background: "#fff", padding: 8, borderRadius: 8, border: "1px solid #f1f5f9" }}>
                <div style={{ color: "#64748b", fontSize: "0.75rem" }}>Volume (RMS)</div>
                <div style={{ fontWeight: 700, color: "#0f1724", fontSize: "0.85rem", marginTop: 2 }}>
                  {prosodyFeatures[1] > 0 ? prosodyFeatures[1].toFixed(3) : "0.000"}
                </div>
              </div>

              <div style={{ background: "#fff", padding: 8, borderRadius: 8, border: "1px solid #f1f5f9" }}>
                <div style={{ color: "#64748b", fontSize: "0.75rem" }}>Vocal Stress</div>
                <div style={{ fontWeight: 700, color: "#0f1724", fontSize: "0.85rem", marginTop: 2 }}>
                  {prosodyFeatures[4].toFixed(2)}
                </div>
              </div>
            </div>
          </div>

          {/* Multimodal Confidence Meter */}
          <div style={{ background: "#f8fafc", padding: 10, borderRadius: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontWeight: 700, color: "#0f1724", fontSize: "0.85rem" }}>Multimodal Confidence Score</div>
              <div style={{ color: "#0f1724", fontWeight: 800, fontSize: "1.05rem" }}>{confidenceScore}%</div>
            </div>
            <div style={{ marginTop: 6, ...confidenceOuter }}>
              <div style={confidenceInner(confidenceScore)} />
            </div>
          </div>

          {/* Real-time AI Guidance Feedback */}
          <div style={{ background: "#eff6ff", padding: 10, borderRadius: 10, borderLeft: "4px solid #3b82f6" }}>
            <div style={{ fontWeight: 700, color: "#1e40af", fontSize: "0.78rem" }}>PULSEMIND AI REAL-TIME FEEDBACK</div>
            <div style={{ marginTop: 4, color: "#1e3a8a", fontSize: "0.85rem", lineHeight: "1.4" }}>{feedback}</div>
          </div>
        </aside>
      </main>

      {/* BART REPORT MODAL */}
      {showReportModal && (
        <div style={{
          position: "fixed",
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: "rgba(15, 23, 42, 0.75)",
          backdropFilter: "blur(4px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 2000,
          padding: 20
        }}>
          <div style={{
            background: "#fff",
            borderRadius: 16,
            maxWidth: 700,
            width: "100%",
            maxHeight: "90vh",
            overflowY: "auto",
            padding: 28,
            boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
            position: "relative"
          }}>
            <button
              onClick={() => setShowReportModal(false)}
              style={{
                position: "absolute",
                top: 20, right: 20,
                border: "none",
                background: "#f1f5f9",
                borderRadius: "50%",
                width: 32, height: 32,
                cursor: "pointer",
                fontWeight: "bold",
                color: "#64748b"
              }}
            >
              ✕
            </button>

            <h2 style={{ margin: "0 0 6px", fontSize: "1.4rem", color: "#0f1724" }}>
              BART Candidate Assessment Report (NLI)
            </h2>
            <p style={{ color: "#64748b", margin: "0 0 20px", fontSize: "0.9rem" }}>
              Synthesized Multi-Modal Performance Evaluation Scorecard
            </p>

            {reportLoading && (
              <div style={{ textAlign: "center", padding: "40px 0", color: "#3b82f6" }}>
                <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>Running BART NLI Zero-Shot Inference...</div>
                <div style={{ fontSize: "0.85rem", color: "#64748b", marginTop: 6 }}>Evaluating 5 candidate performance metrics</div>
              </div>
            )}

            {!reportLoading && reportData && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {/* Scorecard Cards */}
                <div style={{ background: "#f8fafc", borderRadius: 12, padding: 16 }}>
                  <h4 style={{ margin: "0 0 12px", color: "#0f1724" }}>Candidate Performance Scorecard</h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {Object.entries(reportData.scores || {}).map(([aspect, score]) => (
                      <div key={aspect}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.9rem", fontWeight: 600, color: "#334155" }}>
                          <span>{aspect}</span>
                          <span style={{ color: "#0d9488" }}>{score}%</span>
                        </div>
                        <div style={{ ...barOuter, height: 10, marginTop: 4 }}>
                          <div style={barInner(score / 100.0, "#0d9488")} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Narrative Summary */}
                <div style={{ background: "#eff6ff", borderRadius: 12, padding: 16, borderLeft: "4px solid #3b82f6" }}>
                  <h4 style={{ margin: "0 0 8px", color: "#1e40af" }}>Session Performance Narrative</h4>
                  <pre style={{
                    whiteSpace: "pre-wrap",
                    fontFamily: "inherit",
                    margin: 0,
                    color: "#1e3a8a",
                    fontSize: "0.9rem",
                    lineHeight: "1.5"
                  }}>
                    {reportData.narrative}
                  </pre>
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: 12, justifyContent: "flex-end", marginTop: 8 }}>
                  <button
                    onClick={downloadReport}
                    style={{
                      padding: "10px 18px",
                      borderRadius: 8,
                      border: "none",
                      background: "#0d9488",
                      color: "#fff",
                      fontWeight: 600,
                      cursor: "pointer"
                    }}
                  >
                    Download JSON Report
                  </button>

                  <button
                    onClick={() => setShowReportModal(false)}
                    style={{
                      padding: "10px 18px",
                      borderRadius: 8,
                      border: "1px solid #cbd5e1",
                      background: "#fff",
                      color: "#475569",
                      fontWeight: 600,
                      cursor: "pointer"
                    }}
                  >
                    Close
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <footer style={{ maxWidth: 1200, margin: "32px auto 12px", textAlign: "center", color: "#64748b" }}>
        <div>HierVisor • Powered by PulseMind AI Modular Flask Engine</div>
        <div style={{ marginTop: 6, fontSize: "0.85rem" }}>© {new Date().getFullYear()} HierVisor</div>
      </footer>
    </div>
  );
}
