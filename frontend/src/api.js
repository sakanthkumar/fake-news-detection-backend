// src/api.js
import axios from "axios";

const API_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === "production" ? "https://news-authenticity-api.onrender.com" : "http://localhost:5000");

const api = axios.create({
  baseURL: API_URL,
  headers: { "Content-Type": "application/json" },
  withCredentials: true,
  timeout: 60000, // 60 seconds (for Render cold starts)
});

/* Request interceptor: attach auth token if present in localStorage OR sessionStorage
   This is a fallback for token-based flows; cookie-based flows do not require this. */
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("auth_token") || sessionStorage.getItem("auth_token");
    if (token) config.headers["Authorization"] = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

/* ---------- Demo API (only if backend supports it) ---------- */
export async function analyzeHeadlineDemo(headline) {
  try {
    const res = await api.post("/api/predict/demo", { headline });
    return res.data;
  } catch (err) {
    const message = err.response?.data?.error || err.message || "Demo API error";
    throw new Error(message);
  }
}

/* ---------- Authentication ---------- */

export const registerUser = async (userData) => {
  try {
    const res = await api.post("/register", userData);
    return res.data;
  } catch (err) {
    console.error("Registration Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Registration failed" };
  }
};

/**
 * loginUser
 * Accepts { identifier, password } or { username/email, password }.
 * Backend for cookie-flow should set httpOnly cookie on success and return { success: true, user? }.
 * If backend also returns a token, this function returns it (caller may persist).
 */
export const loginUser = async (credentials) => {
  try {
    let payload;
    if (credentials.email || credentials.username) {
      payload = {};
      if (credentials.email) payload.email = credentials.email;
      if (credentials.username) payload.username = credentials.username;
      payload.password = credentials.password;
    } else if (credentials.identifier) {
      const id = credentials.identifier.trim();
      const looksLikeEmail = /\S+@\S+\.\S+/.test(id);
      payload = looksLikeEmail ? { email: id, password: credentials.password } : { username: id, password: credentials.password };
    } else {
      payload = credentials;
    }

    const res = await api.post("/login", payload);
    return res.data; // expected: { success: true, user?, token? } or { success: false, error }
  } catch (err) {
    console.error("Login Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Login failed" };
  }
};

/**
 * logoutUser
 * Calls backend to clear httpOnly cookie (server must implement /auth/logout or similar).
 * Also clears any client-side storage.
 */
export const logoutUser = async () => {
  try {
    await api.post("/auth/logout");
  } catch (err) {
    console.error("Logout error:", err);
  } finally {
    localStorage.removeItem("auth_token");
    sessionStorage.removeItem("auth_token");
    localStorage.removeItem("username");
  }
};


/* ---------- Session / utility ---------- */

/**
 * fetchMe - verify current session (reads cookie server-side)
 * Returns { success: true, user } or { success: false, error }
 */
export const fetchMe = async () => {
  try {
    const res = await api.get("/api/me");
    return res.data;
  } catch (err) {
    console.error("/api/me error:", err.response || err.message);
    return { error: err.response?.data?.error || "Not authenticated" };
  }
};

/* ---------- Core prediction APIs ---------- */

export const predictHeadline = async (headline) => {
  try {
    const res = await api.post("/api/predict", { headline });
    return res.data;
  } catch (err) {
    console.error("Prediction Error:", err.response || err.message);
    return { error: err.response?.data?.error || err.message || "Prediction failed" };
  }
};

export const explainHeadline = async (headline, num_features = 6, num_samples = 500) => {
  try {
    const res = await api.post("/api/explain", { headline, num_features, num_samples });
    return res.data;
  } catch (err) {
    console.error("Explanation Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Failed to get explanation" };
  }
};

export const reportPrediction = async (headline, prediction, feedback) => {
  try {
    const res = await api.post("/api/report", { headline, prediction, feedback });
    return res.data;
  } catch (err) {
    console.error("Report Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Failed to send report" };
  }
};

export const scrapeHeadline = async (url) => {
  try {
    const res = await api.post("/api/scrape", { url });
    return res.data;
  } catch (err) {
    console.error("Scrape Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Scraping failed" };
  }
};

/* ---------- Bulk / Web APIs ---------- */

export const fetchAndPredict = async () => {
  try {
    const res = await api.get("/api/web-predict");
    return res.data;
  } catch (err) {
    console.error("Fetch & Predict Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Web fetch & predict failed" };
  }
};

export const fetchAndSave = async () => {
  try {
    const res = await api.post("/api/fetch-and-save", {});
    return res.data;
  } catch (err) {
    console.error("Fetch & Save Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Web fetch & save failed" };
  }
};

export const fetchHistory = async () => {
  try {
    const res = await api.get("/api/history");
    return res.data;
  } catch (err) {
    console.error("Fetch History Error:", err.response || err.message);
    return { error: err.response?.data?.error || "Failed to fetch history" };
  }
};

/* ---------- Health Check ---------- */

export const checkApiHealth = async () => {
  try {
    const res = await api.get("/api/health");
    return res.data;
  } catch (err) {
    console.error("API Health Check Error:", err.response || err.message);
    return { error: "API is unreachable" };
  }
};

export default api;
