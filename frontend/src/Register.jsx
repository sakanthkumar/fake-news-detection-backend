import React, { useState } from "react";
import axios from "axios";
import { registerUser } from "./api";
import { useNavigate, Link } from "react-router-dom";
import "./register.css"; // NEW CSS FILE

function Register() {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: ""
  });

  const [message, setMessage] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      const response = await registerUser(formData);

      if (response.success) {
        setMessage(response.message || "Registration successful!");
        setTimeout(() => {
          navigate("/login");
        }, 1500);
      } else {
        setMessage(response.error || "Registration failed. Try again.");
      }

    } catch (error) {
      console.error("Error in registration:", error);
      setMessage("Registration failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="register-page">
      <div className="register-card">

        <h2 className="register-title">Create Your Account</h2>
        <p className="register-subtitle">Join Truth Lens to detect fake news with AI</p>

        {message && <div className="register-message">{message}</div>}

        <form onSubmit={handleSubmit} className="register-form">

          {/* Username */}
          <label className="form-label-custom">
            Username
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="input-field"
              required
            />
          </label>

          {/* Email */}
          <label className="form-label-custom">
            Email
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="input-field"
              required
            />
          </label>

          {/* Password */}
          <label className="form-label-custom">
            Password
            <div className="password-wrapper">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                className="input-field"
                required
              />
              <button
                type="button"
                className="toggle-btn"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </label>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? "Creating Account..." : "Register"}
          </button>
        </form>

        <p className="login-link">
          Already have an account? <Link to="/login">Login</Link>
        </p>

      </div>
    </div>
  );
}

export default Register;
