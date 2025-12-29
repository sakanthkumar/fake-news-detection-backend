import React, { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./home.css";
import { isAuthenticated } from "./auth";

function Home() {
  const navigate = useNavigate();

  useEffect(() => {
    // If user already logged in, redirect to dashboard
    if (isAuthenticated()) {
      navigate("/dashboard", { replace: true });
    }
  }, [navigate]);

  return (
    <div className="home-wrapper">
      <div className="overlay" />

      <div className="home-content">
        <h1 className="home-title">Truth Lens</h1>
        <p className="home-subtitle">AI-powered Fake News Detection System</p>

        <p className="home-description">
          Analyze news headlines or URLs instantly with an AI model trained to identify misinformation.
          Get credibility scores, explanations, and real-time analysis.
        </p>

        <div className="home-buttons">
          <Link to="/register" className="btn home-btn-primary">Register</Link>
          <Link to="/login" className="btn home-btn-secondary">Login</Link>
        </div>
      </div>
    </div>
  );
}

export default Home;
