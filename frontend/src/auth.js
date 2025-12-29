// src/utils/auth.js
export function isAuthenticated() {
  return !!(localStorage.getItem("auth_token") || sessionStorage.getItem("auth_token"));
}
export function logoutUser() {
  localStorage.removeItem("auth_token");
  sessionStorage.removeItem("auth_token");
  localStorage.removeItem("username");
}
