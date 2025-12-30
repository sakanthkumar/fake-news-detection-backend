import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Sun, Moon, LogOut, History, Trash2, Loader2, Copy } from "lucide-react";

import {
  predictHeadline,
  scrapeHeadline,
  fetchAndPredict,
  fetchAndSave,
  explainHeadline,
  checkApiHealth,
  logoutUser,
  fetchMe,
  reportPrediction
} from "./api";

import "./dashboard.css";

export default function Dashboard() {
  // --- STATE MANAGEMENT ---
  const [headline, setHeadline] = useState("");
  const [url, setUrl] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [explainingIndex, setExplainingIndex] = useState(null);
  const [darkMode, setDarkMode] = useState(true);
  const [modelStatus, setModelStatus] = useState(null);

  // New: username state to display who is logged in
  const [username, setUsername] = useState(null);



  const navigate = useNavigate();

  // --- EFFECTS ---
  // Check API health once
  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const health = await checkApiHealth();
        setModelStatus(health);
      } catch (error) {
        console.error("Health check failed:", error);
        setModelStatus({ status: "error" });
      }
    };
    checkModelStatus();
  }, []);

  // Dark mode body class sync
  useEffect(() => {
    document.body.className = darkMode ? "dark-mode" : "";
  }, [darkMode]);

  // Fetch authenticated user (works with cookie or bearer token)
  useEffect(() => {
    let mounted = true;
    const loadMe = async () => {
      try {
        const res = await fetchMe();
        if (!mounted) return;
        if (res?.success && res?.user) setUsername(res.user.username || res.user.email);
        else setUsername(null);
      } catch (err) {
        console.error("fetchMe failed:", err);
        setUsername(null);
      }
    };
    loadMe();
    return () => { mounted = false; };
  }, []);




  // --- RENDER HELPERS ---
  const renderCredibility = (p) => {
    if (!p || p.credibility === undefined) return null;
    return (
      <div className="mt-3">
        <strong>Credibility Score:</strong>
        <div className="progress mt-1" style={{ height: "18px" }}>
          <div
            className={`progress-bar ${p.credibility >= 60 ? "bg-success" : p.credibility >= 30 ? "bg-warning" : "bg-danger"
              }`}
            role="progressbar"
            style={{ width: `${p.credibility}%` }}
          >
            {p.credibility}%
          </div>
        </div>
      </div>
    );
  };

  const renderExplanation = (p) => {
    if (!p?.explanation) return null;
    return (
      <div className="mt-3 p-3 border rounded">
        <strong>Why this prediction?</strong>
        <ul className="list-group list-group-flush">
          {p.explanation.map((item, idx) => (
            <li key={idx} className="list-group-item d-flex justify-content-between align-items-center">
              <span>"{item.word}"</span>
              <span className={`badge ${item.impact === "supports" ? "bg-success" : "bg-danger"}`}>
                {item.impact}
              </span>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // --- CORE HANDLERS ---
  const processAndStoreResults = (results) => {
    const resultsArray = Array.isArray(results) ? results : [results];
    setPredictions(resultsArray);
    setHistory((prev) => [...resultsArray.filter((r) => !r.error), ...prev]);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!headline.trim()) return;
    setLoading(true);
    setLoadingMessage("Analyzing Headline...");
    try {
      const result = await predictHeadline(headline);
      processAndStoreResults(result);
    } catch (error) {
      setPredictions([{ headline, error: "Prediction failed. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleScrape = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;
    setLoading(true);
    setLoadingMessage("Scraping & Analyzing URL...");
    try {
      const result = await scrapeHeadline(url);
      processAndStoreResults(result);
    } catch (error) {
      setPredictions([{ headline: url, error: "Scraping failed. Try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleWebPredict = async () => {
    setLoading(true);
    setLoadingMessage("Fetching latest news...");
    try {
      const data = await fetchAndPredict();
      if (data.error) setPredictions([{ headline: "Web Fetch", error: data.error }]);
      else processAndStoreResults(data.predictions);
    } catch (error) {
      setPredictions([{ headline: "Web Fetch", error: "Fetching failed. See console." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleWebSave = async () => {
    setLoading(true);
    setLoadingMessage("Saving to database...");
    try {
      const data = await fetchAndSave();
      alert(data.error ? `Failed: ${data.error}` : `✅ Successfully saved ${data.total_headlines_saved} headlines.`);
    } catch (error) {
      alert("Web fetching & saving failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async (headlineToExplain, index) => {
    setExplainingIndex(index);
    try {
      const result = await explainHeadline(headlineToExplain);
      if (result.error) {
        alert(`Could not get explanation: ${result.error}`);
      } else {
        setPredictions((prev) => prev.map((p, i) => (i === index ? { ...p, explanation: result.explanation } : p)));
      }
    } catch (error) {
      console.error("Explain failed:", error);
      alert("Failed to fetch explanation. See console for details.");
    } finally {
      setExplainingIndex(null);
    }
  };

  const handleReport = async (headline, prediction) => {
    const feedback = prompt("Is this prediction Incorrect? Type 'Fake' or 'Real' to correct us:");
    if (!feedback) return;

    const res = await reportPrediction(headline, prediction, feedback);
    if (res.success) {
      alert("Thanks! Your feedback has been recorded.");
    } else {
      alert("Failed to send report.");
    }
  };

  const openGoogleImages = (headline) => {
    const query = encodeURIComponent(headline);
    window.open(`https://www.google.com/search?tbm=isch&q=${query}`, "_blank");
  };

  const handleLogout = async () => {
    try {
      await logoutUser();
    } catch (err) {
      console.error("logoutUser error:", err);
    } finally {
      // client-side cleanup
      navigate("/login");
    }
  };

  // --- MAIN RENDER ---
  return (
    <div className="dashboard-wrapper">
      <main className="dashboard-main">
        <header className="d-flex justify-content-between align-items-center mb-5">
          <div>
            <h1 className="mb-0">Fake News Detection</h1>
            {modelStatus ? (
              <small className={modelStatus.status === "error" ? "text-danger" : (modelStatus.model_loaded ? "text-success" : "text-warning")}>
                Model Status: {modelStatus.status === "error" ? "Offline" : (modelStatus.model_loaded ? "Ready" : "Loading...")}
              </small>
            ) : (
              <small className="text-secondary">Model Status: Checking...</small>
            )}
            {/* Welcome message */}
            <div className="text-muted mt-1">Signed in as: {username ?? "Guest"}</div>
          </div>

          <div className="d-flex align-items-center">
            <button className="btn btn-outline-secondary me-2" onClick={() => setDarkMode(!darkMode)}>
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button className="btn btn-danger d-flex align-items-center gap-2" onClick={handleLogout}>
              <LogOut size={16} /> Logout
            </button>
          </div>
        </header>

        <div className="row g-4">
          <div className="col-lg-6">
            <div className="custom-card p-4">
              <h5 className="card-title mb-3">Check Headline</h5>
              <form onSubmit={handlePredict}>
                <input
                  className="form-control mb-2"
                  placeholder="Enter news headline..."
                  value={headline}
                  onChange={(e) => setHeadline(e.target.value)}
                  required
                />
                <button className="btn btn-primary w-100" disabled={loading || !headline.trim()}>
                  Analyze Headline
                </button>
              </form>
            </div>
          </div>

          <div className="col-lg-6">
            <div className="custom-card p-4">
              <h5 className="card-title mb-3">Check News URL</h5>
              <form onSubmit={handleScrape}>
                <input
                  className="form-control mb-2"
                  placeholder="Enter news URL..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  required
                />
                <button className="btn btn-warning w-100" disabled={loading || !url.trim()}>
                  Analyze URL
                </button>
              </form>
            </div>
          </div>
        </div>

        <div className="row mt-4">
          <div className="col-12">
            <div className="custom-card p-3">
              <div className="d-flex gap-3">
                <button onClick={handleWebPredict} disabled={loading} className="btn btn-success flex-grow-1">
                  Fetch & Analyze Latest News
                </button>
                <button onClick={handleWebSave} disabled={loading} className="btn btn-secondary flex-grow-1">
                  Save to Database
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-5">
          <div className="d-flex justify-content-between align-items-center mb-3">
            <h4>Results</h4>
            {predictions.length > 0 && (
              <button
                onClick={() => setPredictions([])}
                className="btn btn-sm btn-outline-danger d-flex align-items-center gap-1"
              >
                <Trash2 size={14} /> Clear Results
              </button>
            )}
          </div>

          {loading ? (
            <div className="text-center my-5">
              <Loader2 className="spinner-border" style={{ width: "3rem", height: "3rem" }} />
              <p className="mt-2 text-muted">{loadingMessage}</p>
            </div>
          ) : predictions.length > 0 ? (
            <div className="results-container">
              {predictions.map((p, idx) => (
                <div key={idx} className={`custom-card p-4 mb-3 ${p.error ? "border-danger" : ""}`}>
                  {p.headline && (
                    <div className="d-flex justify-content-between">
                      <h5 className="card-title mb-3">{p.headline}</h5>
                      <button
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => navigator.clipboard.writeText(p.headline)}
                      >
                        <Copy size={14} />
                      </button>
                    </div>
                  )}

                  {p.error ? (
                    <div className="alert alert-danger mb-0">{p.error}</div>
                  ) : (
                    <>
                      <div className="d-flex align-items-center gap-2 mb-3">
                        <strong>Verdict:</strong>
                        {p.prediction === "REAL" && p.credibility < 20 ? (
                          <span className="badge bg-warning text-dark">Likely Real (Unverified)</span>
                        ) : (
                          <span className={`badge ${p.prediction === "REAL" ? "bg-success" : "bg-danger"}`}>
                            {p.prediction}
                          </span>
                        )}
                        {typeof p.confidence === "number" && (
                          <span className="text-muted">({p.confidence.toFixed(1)}% confidence)</span>
                        )}
                      </div>

                      {p.prediction === "REAL" && p.credibility < 20 && (
                        <div className="alert alert-warning py-2 small">
                          <i className="bi bi-exclamation-triangle me-2"></i>
                          This follows news patterns, but we couldn't find it in trusted sources. Verify locally.
                        </div>
                      )}

                      {renderCredibility(p)}

                      {p.prediction && (
                        <div className="mt-3 d-flex gap-2">
                          <button
                            className="btn btn-outline-info btn-sm"
                            onClick={() => handleExplain(p.headline, idx)}
                            disabled={explainingIndex === idx || p.explanation}
                          >
                            {explainingIndex === idx ? "Analyzing..." : p.explanation ? "View Analysis" : (
                              <>
                                <i className="bi bi-lightbulb me-1"></i> Explain Why
                              </>
                            )}
                          </button>

                          <button
                            className="btn btn-outline-secondary btn-sm"
                            onClick={() => openGoogleImages(p.headline)}
                          >
                            <i className="bi bi-images me-1"></i> Verify Image
                          </button>

                          <button
                            className="btn btn-outline-danger btn-sm ms-auto"
                            onClick={() => handleReport(p.headline, p.prediction)}
                          >
                            <i className="bi bi-flag me-1"></i> Report Incorrect
                          </button>
                        </div>
                      )}

                      {renderExplanation(p)}
                    </>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div
              className="text-center py-5 rounded-3"
              style={{ backgroundColor: "var(--card-bg)", border: "2px dashed var(--border-color)" }}
            >
              <p className="text-muted">Analysis results will appear here.</p>
            </div>
          )}
        </div>



      </main>

      <aside className="dashboard-history">
        <h4 className="d-flex align-items-center gap-2 mb-4">
          <History size={20} /> Prediction History
        </h4>
        <div className="d-flex flex-column gap-2">
          {history.length > 0 ? (
            history.map((h, idx) => (
              <div key={idx} className="history-item" onClick={() => setPredictions([h])}>
                <p className="fw-bold">{h.headline}</p>
                <span className={`badge ${h.prediction === "Real" ? "bg-success" : "bg-danger"}`}>{h.prediction}</span>
              </div>
            ))
          ) : (
            <p className="text-muted small">Your session history will appear here.</p>
          )}
        </div>
      </aside>
    </div>
  );
}
