"""
FastAPI main application
Provides REST API endpoints for Harshad numbers and polynomial computations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import time
import traceback

from app.harshad import (
    first_nonharshad_factorial,
    find_consecutive_harshads,
    explain_max_consecutive,
    is_harshad
)
from app.polynomial import (
    generate_legendre_coefficients,
    build_companion_matrix,
    compute_roots_eigenvalue,
    solve_lu_system,
    newton_raphson_root
)

app = FastAPI(
    title="Numerical Methods API",
    description="Harshad numbers and polynomial root-finding",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Request/Response Models =============

class ConsecutiveHarshadRequest(BaseModel):
    length: int = Field(..., ge=1, le=50)
    start_hint: Optional[int] = Field(default=1, ge=1)


class PolynomialGenerateRequest(BaseModel):
    n: int = Field(..., ge=1, le=100)
    normalized: bool = Field(default=False)


class PolynomialRootsRequest(BaseModel):
    n: int = Field(..., ge=1, le=100)
    method: str = Field(default="companion_eig")
    precision: int = Field(default=64, ge=53, le=512)
    normalized: bool = Field(default=False)


class LUSolveRequest(BaseModel):
    n: int = Field(..., ge=1, le=100)
    method: str = Field(default="doolittle")
    normalized: bool = Field(default=False)


class NewtonRootsRequest(BaseModel):
    n: int = Field(..., ge=1, le=100)
    which: str = Field(default="both")  # "min", "max", or "both"
    precision: int = Field(default=64)
    normalized: bool = Field(default=False)
    initial_guesses: Optional[List[float]] = None


# ============= Harshad Endpoints =============

@app.post("/harshad/first_nonharshad")
async def find_first_nonharshad():
    """
    Find the first factorial that is not a Harshad number.
    Returns k, factorial value, and explanation.
    """
    try:
        start_time = time.time()
        result = first_nonharshad_factorial()
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "k": result["k"],
            "factorial_value": result["factorial_value"],
            "is_harshad": result["is_harshad"],
            "explanation": result["explanation"],
            "next_factorial": result.get("next_factorial"),
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{traceback.format_exc()}")


@app.post("/harshad/consecutive")
async def find_consecutive(request: ConsecutiveHarshadRequest):
    """
    Find N consecutive Harshad numbers.
    """
    try:
        start_time = time.time()
        
        if request.length >= 20:
            return {
                "success": False,
                "error": "Cannot find 20 or more consecutive Harshad numbers",
                "explanation": explain_max_consecutive()
            }
        
        result = find_consecutive_harshads(request.length, request.start_hint)
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "length": request.length,
            "consecutive_numbers": result["numbers"],
            "start": result["start"],
            "end": result["end"],
            "verification": result["verification"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/harshad/explain_max")
async def explain_maximum_consecutive():
    """
    Explain why there cannot be 20 or more consecutive Harshad numbers.
    """
    return {
        "explanation": explain_max_consecutive()
    }


# ============= Polynomial Endpoints =============

@app.post("/polynomial/generate")
async def generate_polynomial(request: PolynomialGenerateRequest):
    """
    Generate Legendre polynomial coefficients for degree n.
    """
    try:
        start_time = time.time()
        result = generate_legendre_coefficients(request.n, request.normalized)
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "degree": request.n,
            "normalized": request.normalized,
            "coefficients": result["coefficients"],
            "coefficient_summary": result["summary"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/polynomial/companion")
async def get_companion_matrix(request: PolynomialGenerateRequest):
    """
    Build companion matrix for the polynomial.
    """
    try:
        start_time = time.time()
        
        # Generate coefficients first
        poly_result = generate_legendre_coefficients(request.n, request.normalized)
        coeffs = poly_result["coefficients"]
        
        # Build companion matrix
        result = build_companion_matrix(coeffs)
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "degree": request.n,
            "shape": result["shape"],
            "matrix_preview": result["preview"],
            "properties": result["properties"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/polynomial/roots")
async def compute_roots(request: PolynomialRootsRequest):
    """
    Compute roots of the polynomial using eigenvalue method.
    """
    try:
        start_time = time.time()
        
        result = compute_roots_eigenvalue(
            request.n,
            request.method,
            request.precision,
            request.normalized
        )
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "degree": request.n,
            "method": request.method,
            "precision": request.precision,
            "roots": result["roots"],
            "smallest_root": result["smallest_root"],
            "largest_root": result["largest_root"],
            "error_estimates": result["error_estimates"],
            "elapsed_seconds": round(elapsed, 4),
            "warnings": result.get("warnings", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/polynomial/lu_solve")
async def lu_solve(request: LUSolveRequest):
    """
    Solve Ax = b using LU decomposition where A is companion matrix.
    b = [1, 2, 3, ..., n]
    """
    try:
        start_time = time.time()
        
        result = solve_lu_system(request.n, request.method, request.normalized)
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "degree": request.n,
            "method": request.method,
            "solution_x": result["x"],
            "residual_norm": result["residual_norm"],
            "condition_number": result.get("condition_number"),
            "elapsed_seconds": round(elapsed, 4),
            "verification": result.get("verification")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/polynomial/newton_roots")
async def newton_refine_roots(request: NewtonRootsRequest):
    """
    Refine smallest and/or largest roots using Newton-Raphson method.
    """
    try:
        start_time = time.time()
        
        result = newton_raphson_root(
            request.n,
            request.which,
            request.precision,
            request.normalized,
            request.initial_guesses
        )
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "degree": request.n,
            "which": request.which,
            "precision": request.precision,
            "results": result["results"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
