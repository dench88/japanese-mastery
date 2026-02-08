/**
 * Shared auth helpers for all pages.
 * Include via <script src="/static/auth.js"></script> before page scripts.
 */

function getToken() {
  return localStorage.getItem("token");
}

function requireAuth() {
  const token = getToken();
  if (!token) {
    window.location.href = "/static/login.html";
    return null;
  }
  return token;
}

function authHeaders(extra = {}) {
  const token = getToken();
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

function logout() {
  localStorage.removeItem("token");
  window.location.href = "/static/login.html";
}

/**
 * Wrapper around fetch that adds auth headers and redirects on 401.
 */
async function authFetch(url, options = {}) {
  const token = getToken();
  if (!token) {
    window.location.href = "/static/login.html";
    throw new Error("Not authenticated");
  }
  const headers = options.headers || {};
  // Don't set Content-Type for FormData (browser sets multipart boundary)
  if (options.body instanceof FormData) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    headers["Authorization"] = `Bearer ${token}`;
    if (!headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
  }
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    localStorage.removeItem("token");
    window.location.href = "/static/login.html";
    throw new Error("Session expired");
  }
  return res;
}
