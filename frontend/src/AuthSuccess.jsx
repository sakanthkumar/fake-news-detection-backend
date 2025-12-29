// src/AuthSuccess.jsx
import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

/**
 * Simple landing page after OAuth callback.
 * Backend sets the httpOnly cookie and redirects the browser to /auth/success.
 * This component simply confirms and routes the user to the dashboard.
 */
export default function AuthSuccess() {
  const navigate = useNavigate();

  useEffect(() => {
    // Optionally you could call /api/me here to verify the session before redirecting.
    const t = setTimeout(() => {
      navigate("/dashboard", { replace: true });
    }, 400);

    return () => clearTimeout(t);
  }, [navigate]);

  return (
    <div style={{ minHeight: "60vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div style={{ textAlign: "center" }}>
        <h3>Signing you in…</h3>
        <p style={{ color: "#666" }}>Completing authentication — redirecting to your dashboard.</p>
      </div>
    </div>
  );
}
