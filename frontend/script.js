// frontend/script.js
const BASE_URL = "http://127.0.0.1:5000/api";

function showLoading(el, msg = "Working...") {
  el.innerHTML = `<div style="color:#cfc">${msg} <progress max="100" value="50" style="width:140px"></progress></div>`;
}
function showError(el, msg) { el.innerHTML = `<div style="color:#ff8080">Error: ${msg}</div>`; }
function showOutput(el, msg) { el.innerHTML = `<pre style="color:#ddd; white-space:pre-wrap;">${msg}</pre>`; }
async function safeJSON(res) { try { return await res.json(); } catch(e) { return { error: "Invalid JSON response" }; } }

// HARSHAD: first non-harshad factorial
async function firstNonHarshad() {
  const out = document.getElementById("harshad-first-out");
  showLoading(out, "Searching for first non-Harshad factorial...");
  try {
    const res = await fetch(`${BASE_URL}/harshad/factorial`);
    const data = await safeJSON(res);
    if (data.error) return showError(out, data.error);
    showOutput(out, JSON.stringify(data, null, 2));
  } catch (err) {
    showError(out, err.message || "Failed to fetch");
  }
}

// HARSHAD: range scan
async function scanHarshadRange() {
  const s = document.getElementById("harshad-range-start").value;
  const e = document.getElementById("harshad-range-end").value;
  const out = document.getElementById("harshad-range-out");
  showLoading(out, `Scanning factorials ${s}..${e}...`);
  try {
    const res = await fetch(`${BASE_URL}/harshad/range?start_k=${s}&end_k=${e}`);
    const data = await safeJSON(res);
    if (data.error) return showError(out, data.error);
    showOutput(out, JSON.stringify(data, null, 2));
  } catch (err) {
    showError(out, err.message || "Failed to fetch");
  }
}

// HARSHAD: consecutive
async function findConsecutive() {
  const k = document.getElementById("harshad-k").value;
  const start = document.getElementById("harshad-start").value;
  const out = document.getElementById("harshad-consec-out");
  showLoading(out, `Searching for ${k} consecutive Harshads starting at ${start}...`);
  try {
    const res = await fetch(`${BASE_URL}/harshad/consecutive?k=${k}&start_hint=${start}`);
    const data = await safeJSON(res);
    if (data.error) return showError(out, data.error);
    showOutput(out, JSON.stringify(data, null, 2));
  } catch (err) {
    showError(out, err.message || "Failed to fetch");
  }
}

// POLYNOMIAL
async function computePolynomial() {
  const n = document.getElementById("poly-n").value;
  const out = document.getElementById("poly-out");
  showLoading(out, `Computing modified Legendre polynomial (n=${n})...`);
  try {
    const res = await fetch(`${BASE_URL}/polynomial?order=${n}`);
    const data = await safeJSON(res);
    if (data.error) return showError(out, data.error);
    showOutput(out, JSON.stringify(data, null, 2));
  } catch (err) {
    showError(out, err.message || "Failed to fetch");
  }
}

// QUADRATURE
async function computeQuadrature() {
  const n = document.getElementById("quad-n").value;
  const out = document.getElementById("quad-out");
  showLoading(out, `Computing quadrature (n=${n})...`);
  try {
    const res = await fetch(`${BASE_URL}/quadrature?n=${n}`);
    const data = await safeJSON(res);
    if (data.error) return showError(out, data.error);
    // display table and plot
    let html = `<pre style="color:#ddd">${data.table}</pre>`;
    if (data.plot) {
      html += `<img src="data:image/png;base64,${data.plot}" alt="quad-plot" style="width:100%;max-width:700px;margin-top:10px;border-radius:8px;">`;
    }
    out.innerHTML = html;
  } catch (err) {
    showError(out, err.message || "Failed to fetch");
  }
}

// HEAT
async function solveHeat() {
  const n = document.getElementById("heat-n").value;
  const eta = document.getElementById("heat-eta").value;
  const out = document.getElementById("heat-out");
  showLoading(out, `Solving heat equation (n=${n}, eta_max=${eta})...`);
  try {
    const res = await fetch(`${BASE_URL}/heat?n=${n}&eta_max=${eta}`);
    const data = await safeJSON(res);
    if (data.error) return showError(out, data.error);
    let html = `<pre style="color:#ddd">${data.message}\nMax error: ${data.max_error}\nMean error: ${data.mean_error}</pre>`;
    if (data.plot) {
      html += `<img src="data:image/png;base64,${data.plot}" alt="heat-plot" style="width:100%;max-width:700px;margin-top:10px;border-radius:8px;">`;
    }
    out.innerHTML = html;
  } catch (err) {
    showError(out, err.message || "Failed to fetch");
  }
}

// bind buttons
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn-harshad-first").onclick = firstNonHarshad;
  document.getElementById("btn-harshad-range").onclick = scanHarshadRange;
  document.getElementById("btn-harshad-consec").onclick = findConsecutive;
  document.getElementById("btn-poly").onclick = computePolynomial;
  document.getElementById("btn-quad").onclick = computeQuadrature;
  document.getElementById("btn-heat").onclick = solveHeat;
});
