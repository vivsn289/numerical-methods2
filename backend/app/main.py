"""
Flask API with a simple in-memory job manager for iterative numerical tasks.

Provides:
- POST /api/job/start      start a job (returns job_id)
- GET  /api/job/progress/<job_id>  poll job progress (status/progress/message)
- GET  /api/job/result/<job_id>    fetch result once done

Also synchronous endpoints kept:
- GET /api/harshad/factorial?max_n=...
- GET /api/harshad/consecutive?k=...&start_hint=...
- GET /api/polynomial?order=...
- GET /api/quadrature?n=...
- GET /api/heat?n=...&eta_max=...
"""

import uuid
import threading
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from . import harshad, polynomial, gaussian_quadrature, heat_equation

app = Flask(__name__)
CORS(app)

_jobs = {}
_jobs_lock = threading.Lock()


def _new_job(task_name: str):
    job_id = uuid.uuid4().hex
    rec = {"id": job_id, "task": task_name, "status": "pending", "progress": 0, "message": "", "result": None}
    with _jobs_lock:
        _jobs[job_id] = rec
    return rec


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job(job_id: str):
    with _jobs_lock:
        return _jobs.get(job_id)


@app.route("/")
def home():
    return "<h3>Numerical Methods API — Job manager available at /api/job/*</h3>"


@app.route("/api/job/start", methods=["POST"])
def start_job():
    """
    Start a background job.

    JSON body:
      { "task": "harshad_factorial" | "harshad_consec" | "polynomial" | "quadrature" | "heat",
        "params": {...} }
    """
    try:
        payload = request.get_json(force=True)
        task = payload.get("task")
        params = payload.get("params", {})

        allowed = {"harshad_factorial", "harshad_consec", "polynomial", "quadrature", "heat"}
        if task not in allowed:
            return jsonify({"error": "unknown task"}), 400

        job = _new_job(task)

        def runner():
            try:
                _update_job(job["id"], status="running", progress=0, message="starting")
                if task == "harshad_factorial":
                    max_k = int(params.get("max_k", 500))
                    res = harshad.first_nonharshad_factorial(max_k=max_k,
                                                            progress_callback=lambda p, m=None: _update_job(job["id"], progress=p, message=(m or "")))
                    _update_job(job["id"], result=res, progress=100, status="done", message="completed")
                elif task == "harshad_consec":
                    length = int(params.get("length", 10))
                    start_hint = int(params.get("start_hint", 2))
                    max_iter = int(params.get("max_iter", 2_000_000))
                    res = harshad.find_consecutive_harshads(length=length, start_hint=start_hint, max_iter=max_iter,
                                                            progress_callback=lambda p, m=None: _update_job(job["id"], progress=p, message=(m or "")))
                    _update_job(job["id"], result=res, progress=100, status="done", message="completed")
                elif task == "polynomial":
                    order = int(params.get("order", 10))
                    res = polynomial.compute_polynomial_pipeline(order=order,
                                                                 progress_callback=lambda p, m=None: _update_job(job["id"], progress=p, message=(m or "")))
                    _update_job(job["id"], result=res, progress=100, status="done", message="completed")
                elif task == "quadrature":
                    n = int(params.get("n", 32))
                    res = gaussian_quadrature.compute_quadrature_pipeline(n=n,
                                                                          progress_callback=lambda p, m=None: _update_job(job["id"], progress=p, message=(m or "")))
                    _update_job(job["id"], result=res, progress=100, status="done", message="completed")
                elif task == "heat":
                    n = int(params.get("n", 32))
                    eta_max = float(params.get("eta_max", 3.0))
                    res = heat_equation.compute_heat_pipeline(n=n, eta_max=eta_max,
                                                              progress_callback=lambda p, m=None: _update_job(job["id"], progress=p, message=(m or "")))
                    _update_job(job["id"], result=res, progress=100, status="done", message="completed")
                else:
                    _update_job(job["id"], status="error", message="unsupported task")
            except Exception as e:
                traceback.print_exc()
                _update_job(job["id"], status="error", message=str(e))

        t = threading.Thread(target=runner, daemon=True)
        t.start()

        return jsonify({"job_id": job["id"]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/job/progress/<job_id>")
def job_progress(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"status": job["status"], "progress": job["progress"], "message": job.get("message", "")})


@app.route("/api/job/result/<job_id>")
def job_result(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "done":
        return jsonify({"error": "job not finished", "status": job["status"], "progress": job["progress"]}), 400
    return jsonify({"result": job["result"]})


# ----------------------------
# Synchronous compatibility endpoints
# ----------------------------
@app.route("/api/harshad/factorial")
def harshad_factorial_sync():
    try:
        n = int(request.args.get("max_n", 500))
        res = harshad.first_nonharshad_factorial(max_k=n)
        return jsonify({"result": res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/harshad/consecutive")
def harshad_consec_sync():
    try:
        k = int(request.args.get("k", 10))
        start_hint = int(request.args.get("start_hint", 2))
        max_iter = int(request.args.get("max_iter", 2_000_000))
        res = harshad.find_consecutive_harshads(length=k, start_hint=start_hint, max_iter=max_iter)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/polynomial")
def polynomial_sync():
    try:
        order = int(request.args.get("order", 10))
        res = polynomial.compute_polynomial_pipeline(order=order)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/quadrature")
def quadrature_sync():
    try:
        n = int(request.args.get("n", 32))
        res = gaussian_quadrature.compute_quadrature_pipeline(n=n)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/heat")
def heat_sync():
    try:
        n = int(request.args.get("n", 32))
        eta_max = float(request.args.get("eta_max", 3.0))
        res = heat_equation.compute_heat_pipeline(n=n, eta_max=eta_max)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
