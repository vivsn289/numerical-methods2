const API_BASE = "http://127.0.0.1:5000/api";

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupHarshad();
  setupPolynomial();
  setupQuadrature();
  setupHeat();
});

function el(id){return document.getElementById(id)}
function showEl(id){ const e=el(id); if(e) e.classList.remove("hidden") }
function hideEl(id){ const e=el(id); if(e) e.classList.add("hidden") }
function setProgress(barId, pct, msgId, msg){
  const b = el(barId); if(b) b.value = pct;
  const m = el(msgId); if(m) m.textContent = msg || "";
}

/* Tabs */
function setupTabs(){
  const tabs = Array.from(document.querySelectorAll(".tab"));
  tabs.forEach(t => t.addEventListener("click", () => {
    tabs.forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    const target = t.dataset.target;
    document.querySelectorAll(".pane").forEach(p => p.classList.remove("active"));
    document.getElementById(target).classList.add("active");
  }));
}

/* Job lifecycle: start --> poll --> fetch result */
async function startJobAndPoll(task, params, progressBarId, progressMsgId, resultContainerId){
  const resp = await fetch(`${API_BASE}/job/start`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({task, params})
  });
  if(!resp.ok) throw new Error("Failed to start job");
  const data = await resp.json();
  const jobId = data.job_id;

  // poll until done
  while(true){
    await new Promise(r => setTimeout(r, 350));
    const p = await fetch(`${API_BASE}/job/progress/${jobId}`);
    if(!p.ok) throw new Error("progress fetch failed");
    const payload = await p.json();
    setProgress(progressBarId, payload.progress || 0, progressMsgId, payload.message || payload.status);
    if(payload.status === "done") break;
    if(payload.status === "error") throw new Error("Job error: " + (payload.message || "unknown"));
  }

  const r = await fetch(`${API_BASE}/job/result/${jobId}`);
  if(!r.ok) throw new Error("fetch result failed");
  const rr = await r.json();
  renderResult(resultContainerId, rr.result);
  setProgress(progressBarId, 100, progressMsgId, "Completed");
}

/* -------------------- Rendering Section -------------------- */

function renderResult(id, payload){
  const elRes = el(id);
  if(!elRes) return;
  elRes.classList.remove("hidden");

  // --- HARSHAD factorial result ---
  if(payload && payload.k && payload.factorial_value){
    const k = payload.k;
    const ds = payload.digit_sum;
    const isHarshad = payload.is_harshad;
    const next = payload.next_factorial || {};
    const factorialValue = payload.factorial_value;
    const truncated = factorialValue.length > 120 ? factorialValue.slice(0, 120) + "..." : factorialValue;

    elRes.innerHTML = `
      <h3>🧮 First Non-Harshad Factorial Found</h3>
      <table class="result-table">
        <tr><td><b>k</b></td><td>${k}</td></tr>
        <tr><td><b>Digit Sum</b></td><td>${ds}</td></tr>
        <tr><td><b>Harshad?</b></td><td>${isHarshad ? "✅ Yes" : "❌ No"}</td></tr>
        <tr><td><b>Factorial (${k}!)</b></td><td class="mono small">${truncated}</td></tr>
      </table>
      <h4>Next Factorial (${next.k || "?"}!)</h4>
      <table class="result-table">
        <tr><td><b>Digit Sum</b></td><td>${next.digit_sum || "—"}</td></tr>
        <tr><td><b>Harshad?</b></td><td>${next.is_harshad ? "✅ Yes" : "❌ No"}</td></tr>
      </table>
    `;
    return;
  }

  // --- HARSHAD consecutive ---
  if(payload && payload.numbers && payload.numbers.length){
    const nums = payload.numbers;
    const seq = nums.join(", ");
    elRes.innerHTML = `
      <h3>🔢 Consecutive Harshad Numbers Found</h3>
      <table class="result-table">
        <tr><td><b>Length</b></td><td>${nums.length}</td></tr>
        <tr><td><b>Start</b></td><td>${payload.start}</td></tr>
        <tr><td><b>End</b></td><td>${payload.end}</td></tr>
      </table>
      <p><b>Sequence:</b></p>
      <div class="mono small">${seq}</div>
    `;
    return;
  }

  // --- POLYNOMIAL ---
  if(payload && payload.order && payload.coefficients){
    const coeffs = payload.coefficients.map(c => c.toFixed(6)).join(", ");
    const roots = payload.roots.map(r => r.toFixed(6)).join(", ");
    elRes.innerHTML = `
      <h3>📈 Modified Legendre Polynomial</h3>
      <table class="result-table">
        <tr><td><b>Order</b></td><td>${payload.order}</td></tr>
        <tr><td><b>LU Method</b></td><td>${payload.lu_info.method}</td></tr>
        <tr><td><b>LU Shape</b></td><td>${payload.lu_info.lu_shape}</td></tr>
      </table>
      <h4>Coefficients</h4>
      <div class="mono small">${coeffs}</div>
      <h4>Roots</h4>
      <div class="mono small">${roots}</div>
    `;
    return;
  }

  // --- QUADRATURE ---
  if(payload && payload.nodes && payload.weights){
    elRes.innerHTML = `
      <h3>📊 Gaussian Quadrature Results</h3>
      <table class="result-table">
        <tr><td><b>n</b></td><td>${payload.n}</td></tr>
        <tr><td><b>Σ(weights)</b></td><td>${payload.weights.reduce((a,b)=>a+b,0).toFixed(6)}</td></tr>
      </table>
      <h4>First few nodes</h4>
      <div class="mono small">${payload.nodes.slice(0,5).map(x=>x.toFixed(6)).join(", ")} ...</div>
      <h4>First few weights</h4>
      <div class="mono small">${payload.weights.slice(0,5).map(x=>x.toFixed(6)).join(", ")} ...</div>
      <p>Matrix plot generated ✅</p>
    `;
    return;
  }

  // --- HEAT EQUATION ---
  if(payload && payload.etas && payload.plot){
    elRes.innerHTML = `
      <h3>🔥 Heat Equation Solution</h3>
      <table class="result-table">
        <tr><td><b>Nodes (n)</b></td><td>${payload.etas.length}</td></tr>
        <tr><td><b>η max</b></td><td>${payload.etas[payload.etas.length-1].toFixed(2)}</td></tr>
      </table>
      <p>Below: Numeric (dots) vs Analytic (line)</p>
      <img src="data:image/png;base64,${payload.plot}" style="max-width:100%;border-radius:8px;margin-top:10px;">
    `;
    return;
  }

  // --- fallback raw ---
  elRes.innerHTML = `<pre class="mono small">${JSON.stringify(payload, null, 2)}</pre>`;
}

/* -------------------- Task Forms -------------------- */

function setupHarshad(){
  const f1 = el("form-harshad");
  f1.addEventListener("submit", async ev => {
    ev.preventDefault();
    hideEl("harshad-factorial-result");
    setProgress("harshad-factorial-progress", 0, "harshad-factorial-msg", "starting...");
    try {
      await startJobAndPoll(
        "harshad_factorial",
        {max_k: Number(el("harshad-max").value) || 500},
        "harshad-factorial-progress",
        "harshad-factorial-msg",
        "harshad-factorial-result"
      );
    } catch(e){ renderResult("harshad-factorial-result",{error:e.message}); }
  });

  const f2 = el("form-harshad-consec");
  f2.addEventListener("submit", async ev => {
    ev.preventDefault();
    hideEl("harshad-consec-result");
    setProgress("harshad-consec-progress", 0, "harshad-consec-msg", "starting...");
    try {
      await startJobAndPoll(
        "harshad_consec",
        {length:Number(el("harshad-k").value)||10,start_hint:Number(el("harshad-hint").value)||2,max_iter:2000000},
        "harshad-consec-progress",
        "harshad-consec-msg",
        "harshad-consec-result"
      );
    } catch(e){ renderResult("harshad-consec-result",{error:e.message}); }
  });
}

function setupPolynomial(){
  el("form-poly").addEventListener("submit", async ev => {
    ev.preventDefault();
    hideEl("poly-result");
    setProgress("poly-progress",0,"poly-progress-msg","starting...");
    try{
      await startJobAndPoll("polynomial",{order:Number(el("poly-order").value)||10},
        "poly-progress","poly-progress-msg","poly-result");
    }catch(e){renderResult("poly-result",{error:e.message});}
  });
}

function setupQuadrature(){
  el("form-quad").addEventListener("submit", async ev => {
    ev.preventDefault();
    hideEl("quad-result");
    setProgress("quad-progress",0,"quad-progress-msg","starting...");
    try{
      await startJobAndPoll("quadrature",{n:Number(el("quad-n").value)||32},
        "quad-progress","quad-progress-msg","quad-result");
    }catch(e){renderResult("quad-result",{error:e.message});}
  });
}

function setupHeat(){
  el("form-heat").addEventListener("submit", async ev => {
    ev.preventDefault();
    hideEl("heat-result");
    setProgress("heat-progress",0,"heat-progress-msg","starting...");
    try{
      await startJobAndPoll("heat",{n:Number(el("heat-n").value)||32,eta_max:Number(el("heat-eta").value)||3.0},
        "heat-progress","heat-progress-msg","heat-result");
    }catch(e){renderResult("heat-result",{error:e.message});}
  });
}
