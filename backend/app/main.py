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

from app.gaussian_quadrature import (
    compute_gauss_legendre_golub_welsch,
    lagrange_weights,
    orthogonal_collocation_matrices,
    compute_up_to_n,
    test_quadrature
)
from app.heat_equation import (
    solve_heat_equation_collocation,
    analytical_solution,
    plot_comparison_data,
    solve_multiple_times,
    convergence_study
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

class GaussQuadratureRequest(BaseModel):
    n: int = Field(..., ge=1, le=64)
    compute_lagrange: bool = Field(default=True)


class CollocationMatricesRequest(BaseModel):
    n: int = Field(..., ge=2, le=64)


class HeatEquationRequest(BaseModel):
    n: int = Field(default=32, ge=4, le=64)
    t_final: float = Field(default=0.1, ge=0.0, le=10.0)
    num_time_points: int = Field(default=50, ge=2, le=200)
    T0: float = Field(default=0.0)
    Ts: float = Field(default=1.0)


class ConvergenceStudyRequest(BaseModel):
    n_values: List[int] = Field(default=[8, 16, 32])
    t_final: float = Field(default=0.1)


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

@app.post("/gaussian/compute")
async def compute_gaussian_quadrature(request: GaussQuadratureRequest):
    """
    Compute Gauss-Legendre nodes and weights using Golub-Welsch algorithm.
    Optionally compute weights using Lagrangian interpolation.
    """
    try:
        start_time = time.time()
        
        # Compute using Golub-Welsch
        result = compute_gauss_legendre_golub_welsch(request.n)
        
        # Compute using Lagrange if requested
        if request.compute_lagrange:
            nodes_array = np.array(result["nodes"])
            lagrange_w = lagrange_weights(nodes_array)
            result["weights_lagrange"] = lagrange_w.tolist()
            
            # Compare both methods
            weights_gw = np.array(result["weights"])
            diff = np.abs(weights_gw - lagrange_w)
            result["comparison"] = {
                "max_difference": float(np.max(diff)),
                "rms_difference": float(np.sqrt(np.mean(diff**2)))
            }
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "n": request.n,
            "nodes": result["nodes"],
            "weights": result["weights"],
            "weights_lagrange": result.get("weights_lagrange"),
            "comparison": result.get("comparison"),
            "sum_weights": sum(result["weights"]),
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/gaussian/compute_range")
async def compute_gaussian_range(max_n: int = 64):
    """
    Compute Gauss-Legendre quadrature for n=1 to max_n.
    """
    try:
        start_time = time.time()
        
        result = compute_up_to_n(max_n)
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "max_n": max_n,
            "results": result["results"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/gaussian/collocation_matrices")
async def compute_collocation_matrices(request: CollocationMatricesRequest):
    """
    Compute A and B matrices for orthogonal collocation at n points.
    """
    try:
        start_time = time.time()
        
        # First compute nodes and weights
        quad_result = compute_gauss_legendre_golub_welsch(request.n)
        nodes = np.array(quad_result["nodes"])
        weights = np.array(quad_result["weights"])
        
        # Compute collocation matrices
        result = orthogonal_collocation_matrices(nodes, weights)
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "n": request.n,
            "A_matrix": result["A_matrix"],
            "B_matrix": result["B_matrix"],
            "A_shape": result["A_shape"],
            "B_shape": result["B_shape"],
            "A_norm": result["A_norm"],
            "B_norm": result["B_norm"],
            "nodes": quad_result["nodes"],
            "weights": quad_result["weights"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/gaussian/test_quadrature")
async def test_gauss_quadrature(n: int = 32):
    """
    Test Gauss-Legendre quadrature accuracy on known integrals.
    """
    try:
        start_time = time.time()
        
        # Compute quadrature
        result = compute_gauss_legendre_golub_welsch(n)
        nodes = result["nodes"]
        weights = result["weights"]
        
        # Test on x^2 (exact for sufficient n)
        test_result = test_quadrature(nodes, weights)
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "n": n,
            "test_function": "x^2",
            "exact_value": test_result["exact"],
            "approximate_value": test_result["approximate"],
            "absolute_error": test_result["absolute_error"],
            "relative_error": test_result["relative_error"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
@app.post("/heat_equation/solve")
async def solve_heat_eq(request: HeatEquationRequest):
    """
    Solve the 1D heat equation using Gauss-Legendre collocation method.
    Compare with analytical solution.
    """
    try:
        start_time = time.time()
        
        # Compute nodes and matrices
        quad_result = compute_gauss_legendre_golub_welsch(request.n)
        nodes = np.array(quad_result["nodes"])
        weights = np.array(quad_result["weights"])
        
        colloc_matrices = orthogonal_collocation_matrices(nodes, weights)
        B_matrix = np.array(colloc_matrices["B_matrix"])
        
        # Solve heat equation
        t_eval = np.linspace(0, request.t_final, request.num_time_points)
        result = solve_heat_equation_collocation(
            request.n, nodes, B_matrix,
            t_span=(0, request.t_final),
            t_eval=t_eval,
            T0=request.T0,
            Ts=request.Ts
        )
        
        elapsed = time.time() - start_time
        
        if result["success"]:
            return {
                "success": True,
                "n": request.n,
                "nodes": result["nodes"],
                "t_eval": result["t_eval"],
                "solution": result["solution"],
                "final_solution": {
                    "numerical": result["theta_numerical"],
                    "analytical": result["theta_analytical"],
                    "error": result["error"]
                },
                "max_error": result["max_error"],
                "rms_error": result["rms_error"],
                "elapsed_seconds": round(elapsed, 4)
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{traceback.format_exc()}")


@app.post("/heat_equation/compare")
async def compare_solutions(n: int = 32, t_final: float = 0.1):
    """
    Get comparison data between numerical and analytical solution at final time.
    """
    try:
        start_time = time.time()
        
        # Compute nodes and matrices
        quad_result = compute_gauss_legendre_golub_welsch(n)
        nodes = np.array(quad_result["nodes"])
        weights = np.array(quad_result["weights"])
        
        colloc_matrices = orthogonal_collocation_matrices(nodes, weights)
        B_matrix = np.array(colloc_matrices["B_matrix"])
        
        # Solve
        result = solve_heat_equation_collocation(
            n, nodes, B_matrix,
            t_span=(0, t_final),
            t_eval=np.array([t_final])
        )
        
        if result["success"]:
            plot_data = plot_comparison_data(result)
            elapsed = time.time() - start_time
            
            return {
                "success": True,
                "n": n,
                "t_final": t_final,
                "comparison": plot_data,
                "elapsed_seconds": round(elapsed, 4)
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/heat_equation/convergence")
async def study_convergence(request: ConvergenceStudyRequest):
    """
    Study convergence of collocation method with different n values.
    """
    try:
        start_time = time.time()
        
        result = convergence_study(request.n_values, request.t_final)
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "convergence_data": result["convergence_data"],
            "t_final": result["t_final"],
            "n_values": result["n_values"],
            "elapsed_seconds": round(elapsed, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/assignment3/info")
async def assignment3_info():
    """
    Get information about Assignment 3 implementation.
    """
    return {
        "assignment": "Assignment 3",
        "description": "Gaussian Quadrature and Heat Equation Solver",
        "methods": {
            "gaussian_quadrature": {
                "algorithm": "Golub-Welsch (1969)",
                "description": "Compute nodes and weights from Jacobi matrix eigenvalues",
                "max_n": 64
            },
            "lagrangian_weights": {
                "description": "Alternative weight computation using Lagrangian interpolation"
            },
            "orthogonal_collocation": {
                "description": "Compute A and B matrices for derivative approximation",
                "use_case": "Solving differential equations"
            },
            "heat_equation": {
                "problem": "1D heat diffusion in a beam",
                "method": "Gauss-Legendre orthogonal collocation",
                "comparison": "Analytical vs numerical solution"
            }
        },
        "endpoints": {
            "/gaussian/compute": "Compute nodes and weights for given n",
            "/gaussian/compute_range": "Compute for n=1 to max_n",
            "/gaussian/collocation_matrices": "Get A and B matrices",
            "/gaussian/test_quadrature": "Test accuracy on known integral",
            "/heat_equation/solve": "Solve heat equation with collocation",
            "/heat_equation/compare": "Compare numerical vs analytical",
            "/heat_equation/convergence": "Study convergence with different n"
        }
    }
