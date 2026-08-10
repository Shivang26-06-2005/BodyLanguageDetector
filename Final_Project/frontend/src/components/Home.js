import React, { useState } from "react";

const HomePage = () => {
  // Button hover state
  const [btnHover, setBtnHover] = useState(false);

  const scrollToAnalyzer = () => {
    const el = document.getElementById("start-interview-section");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  // Reusable section styles
  const sectionStyle = (bgColor) => ({
    padding: "60px 20px",
    backgroundColor: bgColor,
    textAlign: "center",
  });

  const headingStyle = {
    fontSize: "2.5rem",
    fontWeight: "bold",
    marginBottom: "20px",
  };

  const paragraphStyle = {
    fontSize: "1.1rem",
    maxWidth: "800px",
    margin: "0 auto 20px",
    lineHeight: "1.6",
  };

  const buttonStyle = {
    padding: "12px 30px",
    fontSize: "1rem",
    border: "none",
    borderRadius: "5px",
    backgroundColor: btnHover ? "#0056b3" : "#007BFF",
    color: "#fff",
    cursor: "pointer",
    transition: "background-color 0.3s ease",
  };

  const threeColGrid = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
    gap: "20px",
    maxWidth: "1000px",
    margin: "0 auto",
  };

  const cardStyle = {
    backgroundColor: "#fff",
    padding: "20px",
    borderRadius: "8px",
    boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
    textAlign: "left",
  };

  return (
    <div>
      {/* Hero Section */}
      <section style={sectionStyle("#f8f9fa")}>
        <h1 style={headingStyle}>AI-Based Human Interview Analyzer</h1>
        <p style={paragraphStyle}>
          Real-time analysis of micro-expressions, eye contact, and gestures powered by PulseMind AI neural networks to give candidates instant feedback.
        </p>
        <button
          style={buttonStyle}
          onClick={scrollToAnalyzer}
          onMouseOver={() => setBtnHover(true)}
          onMouseOut={() => setBtnHover(false)}
        >
          Start Demo
        </button>
      </section>

      {/* Features Section */}
      <section id="features" style={sectionStyle("#e9ecef")}>
        <h2 style={headingStyle}>How It Works</h2>
        <div style={threeColGrid}>
          <div style={cardStyle}>
            <h3>Step 1: Facial Micro-expressions</h3>
            <p>FER_CNN deep neural network detects 7 emotion categories in real-time.</p>
          </div>
          <div style={cardStyle}>
            <h3>Step 2: Speech & Prosody</h3>
            <p>
              ProsodyNet LSTM & Sound CNN analyze pitch, volume, BPM tempo, and voice tone.
            </p>
          </div>
          <div style={cardStyle}>
            <h3>Step 3: Vision & Tracking</h3>
            <p>
              OpenCV gaze tracking monitors eye contact and hand/fidgeting movements.
            </p>
          </div>
          <div style={cardStyle}>
            <h3>Step 4: Multimodal Fusion</h3>
            <p>
              Integrates speech, tone, and body language to calculate real-time confidence scores.
            </p>
          </div>
        </div>
      </section>

      {/* Why This Matters */}
      <section style={sectionStyle("#f8f9fa")}>
        <h2 style={headingStyle}>Why This Matters</h2>
        <div style={threeColGrid}>
          <div style={cardStyle}>
            <h3>What We Observe</h3>
            <p>Words, voice tone, facial expressions, and gestures.</p>
          </div>
          <div style={cardStyle}>
            <h3>What the Model Does</h3>
            <p>
              Reads, listens, and watches to detect emotions in real-time.
            </p>
          </div>
          <div style={cardStyle}>
            <h3>Why It Helps</h3>
            <p>
              Provides consistent, unbiased insights for better hiring decisions.
            </p>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" style={sectionStyle("#e9ecef")}>
        <h2 style={headingStyle}>FAQ</h2>
        <div style={{ maxWidth: "800px", margin: "0 auto", textAlign: "left" }}>
          <h4>1. How do you ensure privacy?</h4>
          <p>
            All video frames are processed locally in real-time and never saved without consent.
          </p>
          <h4>2. How accurate is the analysis?</h4>
          <p>
            The AI model suite is trained on multimodal FER, audio emotion, and prosody datasets for high accuracy.
          </p>
          <h4>3. Can bias affect results?</h4>
          <p>
            The model is designed to focus strictly on objective non-verbal Cues.
          </p>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" style={sectionStyle("#f8f9fa")}>
        <h2 style={headingStyle}>What People Say</h2>
        <div style={threeColGrid}>
          <div style={cardStyle}>
            <p>
              "This tool gave me confidence in my interviews and helped me improve my delivery."
            </p>
            <strong>- Candidate</strong>
          </div>
          <div style={cardStyle}>
            <p>
              "It’s like having a body language coach for every candidate — unbiased and precise."
            </p>
            <strong>- Recruiter</strong>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer
        id="contact"
        style={{
          backgroundColor: "#343a40",
          color: "#fff",
          padding: "20px",
          textAlign: "center",
        }}
      >
        <p>© 2025 AI Body Language Analyzer | All Rights Reserved</p>
        <p>
          <button
            onClick={scrollToAnalyzer}
            style={{ background: "none", border: "none", color: "#00d1ff", cursor: "pointer", textDecoration: "underline", margin: "0 10px" }}
          >
            Start Analyzer
          </button>
          |
          <span style={{ margin: "0 10px" }}>HierVisor Powered by PulseMind AI</span>
        </p>
      </footer>
    </div>
  );
};

export default HomePage;
