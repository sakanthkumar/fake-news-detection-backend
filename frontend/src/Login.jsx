// src/Login.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { loginUser, fetchMe } from "./api"; // fetchMe verifies cookie-session
import "./register.css"; // reuse styles from Register

export default function Login() {
  const [form, setForm] = useState({ identifier: "", password: "", remember: false });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const onChange = (e) => {
    const { name, type, value, checked } = e.target;
    setForm(prev => ({ ...prev, [name]: type === "checkbox" ? checked : value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    const identifier = form.identifier?.trim();
    const password = form.password;

    if (!identifier || !password) {
      setError("Please provide your username/email and password.");
      return;
    }

    setLoading(true);
    try {
      // Call backend login endpoint. For cookie-flow, backend should set httpOnly cookie.
      const result = await loginUser({ identifier, password });

      // Accept token variants (token, accessToken) if backend returns it in JSON
      const token = result?.token || result?.accessToken || result?.data?.token;

      // If backend returned success (cookie-only or token), proceed. We accept either:
      //  - result.success && token  -> token-based
      //  - result.success && no token -> cookie-based (server sets cookie)
      if (result?.success) {
        // If token returned, persist according to Remember me
        if (token) {
          if (form.remember) {
            localStorage.setItem("auth_token", token);
          } else {
            sessionStorage.setItem("auth_token", token);
          }
        }

        // Try to verify session by calling /api/me (recommended for cookie flow)
        try {
          const me = await fetchMe();
          if (me?.success) {
            if (me.user?.username) localStorage.setItem("username", me.user.username);
            navigate("/dashboard", { replace: true });
            return;
          } else {
            // If /api/me failed but token exists, still proceed
            if (token) {
              if (result.user?.username) localStorage.setItem("username", result.user.username);
              navigate("/dashboard", { replace: true });
              return;
            }
            // otherwise, treat as error
            setError("Login succeeded but session verification failed.");
            return;
          }
        } catch (meErr) {
          console.error("Session verification error:", meErr);
          // If token available, let user proceed; otherwise show error.
          if (token) {
            if (result.user?.username) localStorage.setItem("username", result.user.username);
            navigate("/dashboard", { replace: true });
            return;
          }
          setError("Login succeeded but session verification failed.");
          return;
        }
      } else {
        // login not successful
        const msg = result?.message || result?.error || "Login failed. Check credentials.";
        setError(msg);
      }
    } catch (err) {
      console.error("Login error:", err);
      const msg = err?.response?.data?.error || err?.message || "Network error. Try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // Redirect to backend Google OAuth start (Flask: /auth/google)
  const handleGoogleSignIn = () => {
    // Dynamically determine backend URL using same logic as api.js
    const API_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === "production" ? "https://news-authenticity-api.onrender.com" : "http://localhost:5000");
    window.location.href = `${API_URL}/auth/google`;
  };

  // Placeholder for GitHub (replace with your backend route /auth/github)
  const handleGithubSignIn = () => {
    const API_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === "production" ? "https://news-authenticity-api.onrender.com" : "http://localhost:5000");
    window.location.href = `${API_URL}/auth/github`;
  };

  return (
    <div className="register-page">
      <div className="register-card" style={{ maxWidth: 520 }}>
        <h2 className="register-title">Welcome back</h2>
        <p className="register-subtitle">Sign in to access your dashboard and analyze headlines.</p>

        {error && (
          <div
            className="register-message"
            style={{ background: "rgba(255,80,80,0.12)", color: "#ffdede" }}
            role="alert"
            aria-live="polite"
          >
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="register-form" noValidate>
          <label className="form-label-custom">
            Username or Email
            <input
              name="identifier"
              value={form.identifier}
              onChange={onChange}
              className="input-field"
              placeholder="username or email"
              autoComplete="username"
              required
              disabled={loading}
            />
          </label>

          <label className="form-label-custom">
            Password
            <div className="password-wrapper">
              <input
                name="password"
                value={form.password}
                onChange={onChange}
                className="input-field"
                placeholder="Your password"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                required
                disabled={loading}
              />
              <button
                type="button"
                className="toggle-btn"
                onClick={() => setShowPassword(s => !s)}
                aria-label="Toggle password visibility"
                disabled={loading}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </label>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--muted)" }}>
              <input
                type="checkbox"
                name="remember"
                checked={form.remember}
                onChange={onChange}
                style={{ width: 16, height: 16 }}
                disabled={loading}
              />
              Remember me
            </label>

            <Link to="/forgot" className="forgot-link" style={{ color: "var(--muted)" }}>Forgot?</Link>
          </div>

          <button type="submit" className="submit-btn" disabled={loading} style={{ marginTop: 12 }}>
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <div style={{ marginTop: 12, textAlign: "center", color: "var(--muted)" }}>Or sign in with</div>

        <div className="social-row" style={{ marginTop: 10 }}>
          <button
            type="button"
            className="btn social google"
            onClick={handleGoogleSignIn}
            disabled={loading}
            aria-disabled={loading}
          >
            Google
          </button>

          <button
            type="button"
            className="btn social github"
            onClick={handleGithubSignIn}
            disabled={loading}
            aria-disabled={loading}
          >
            GitHub
          </button>
        </div>

        <p className="login-link" style={{ marginTop: 14 }}>
          New to Truth Lens? <Link to="/register">Create an account</Link>
        </p>
      </div>
    </div>
  );
}
