"""
Heat Equation Solver using Gauss-Legendre Orthogonal Collocation

Solves the 1D heat equation with orthogonal collocation method.
Implements the transformation to ODE form and comparison with analytical solution.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Callable
import warnings


def analytical_solution(x: np.ndarray, t: float, T0: float = 0.0, Ts: float = 1.0) -> np.ndarray:
    """
    Analytical solution for the dimensionless heat equation.
    
    θ(ξ,τ) = T0 + sum_{n=0}^∞ [2(Ts-T0)/(2n+1)π] * sin[(2n+1)πξ] * exp[-(2n+1)²π²τ]
    
    For the specific problem with T0=0 at x=0 and initial condition θ=1:
    θ(ξ,τ) = 1 - ξ - (2/π) * sum_{n=1}^∞ [sin(nπξ)/n * exp(-n²π²τ)]
    
    Args:
        x: Spatial coordinates (dimensionless, 0 to 1)
        t: Time (dimensionless)
        T0: Temperature at x=0 (dimensionless)
        Ts: Initial temperature (dimensionless)
        
    Returns:
        Temperature distribution θ(x,t)
    """
    # Series approximation
    n_terms = 100
    theta = np.ones_like(x) * (1 - x)  # Steady state part
    
    for n in range(1, n_terms + 1):
        term = (2.0 / (np.pi * n)) * np.sin(n * np.pi * x) * np.exp(-n**2 * np.pi**2 * t)
        theta += term
    
    return theta


def setup_collocation_ode(nodes: np.ndarray, B_matrix: np.ndarray, 
                          boundary_indices: tuple = (0, -1)) -> Dict:
    """
    Set up the ODE system for orthogonal collocation.
    
    The heat equation: ∂θ/∂τ = ∂²θ/∂ξ²
    
    Using collocation: dθ_i/dτ = sum_j B_ij * θ_j
    
    With boundary conditions: θ(0,τ) = T0, θ(1,τ) = boundary condition
    
    Args:
        nodes: Collocation nodes (transformed to [0,1])
        B_matrix: Second derivative matrix
        boundary_indices: Indices of boundary nodes
        
    Returns:
        Dictionary with ODE setup information
    """
    n = len(nodes)
    
    # Transform nodes from [-1,1] to [0,1]
    nodes_01 = (nodes + 1) / 2
    
    # Adjust B matrix for coordinate transformation
    # d²/dξ² on [0,1] = 4 * d²/dx² on [-1,1]
    B_scaled = 4.0 * B_matrix
    
    # Interior nodes (excluding boundaries)
    interior_mask = np.ones(n, dtype=bool)
    interior_mask[boundary_indices[0]] = False
    interior_mask[boundary_indices[1]] = False
    interior_indices = np.where(interior_mask)[0]
    
    return {
        "nodes_01": nodes_01.tolist(),
        "B_scaled": B_scaled.tolist(),
        "interior_indices": interior_indices.tolist(),
        "num_interior": len(interior_indices),
        "num_total": n
    }


def solve_heat_equation_collocation(
    n: int,
    nodes: np.ndarray,
    B_matrix: np.ndarray,
    t_span: tuple = (0, 1.0),
    t_eval: np.ndarray = None,
    T0: float = 0.0,
    Ts: float = 1.0
) -> Dict:
    """
    Solve the heat equation using Gauss-Legendre collocation.
    
    Problem:
        ∂θ/∂τ = ∂²θ/∂ξ²
        BC: θ(0,τ) = T0, θ(1,τ) = boundary_condition
        IC: θ(ξ,0) = Ts
    
    Args:
        n: Number of collocation points
        nodes: Gauss-Legendre nodes on [-1,1]
        B_matrix: Second derivative matrix
        t_span: Time span (start, end)
        t_eval: Time points for evaluation
        T0: Temperature at left boundary
        Ts: Initial temperature
        
    Returns:
        Dictionary with solution and comparison to analytical
    """
    # Transform nodes to [0,1]
    nodes_01 = (nodes + 1) / 2
    
    # Scale B matrix for [0,1] domain
    B_scaled = 4.0 * B_matrix
    
    # Initial condition
    y0 = np.ones(n) * Ts
    y0[0] = T0  # Left boundary
    
    # Define ODE system
    def ode_system(t, y):
        dydt = np.zeros(n)
        
        # Boundary conditions (fixed)
        dydt[0] = 0  # Left boundary is constant
        dydt[-1] = 0  # Right boundary (can be modified for different BC)
        
        # Interior points: use collocation
        for i in range(1, n - 1):
            dydt[i] = np.dot(B_scaled[i, :], y)
        
        return dydt
    
    # Solve ODE
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 50)
    
    try:
        solution = solve_ivp(
            ode_system,
            t_span,
            y0,
            method='BDF',  # Good for stiff problems
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10
        )
        
        if not solution.success:
            warnings.warn(f"ODE solver warning: {solution.message}")
        
        # Compare with analytical solution at final time
        t_final = t_eval[-1]
        theta_numerical = solution.y[:, -1]
        theta_analytical = analytical_solution(nodes_01, t_final, T0, Ts)
        
        # Compute error
        error = np.abs(theta_numerical - theta_analytical)
        max_error = np.max(error)
        rms_error = np.sqrt(np.mean(error**2))
        
        return {
            "success": True,
            "t_eval": t_eval.tolist(),
            "nodes": nodes_01.tolist(),
            "solution": solution.y.tolist(),
            "final_time": t_final,
            "theta_numerical": theta_numerical.tolist(),
            "theta_analytical": theta_analytical.tolist(),
            "error": error.tolist(),
            "max_error": float(max_error),
            "rms_error": float(rms_error),
            "num_points": n,
            "solver_message": solution.message
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "num_points": n
        }


def plot_comparison_data(result: Dict) -> Dict:
    """
    Prepare data for plotting numerical vs analytical solution.
    
    Args:
        result: Result from solve_heat_equation_collocation
        
    Returns:
        Dictionary with data ready for plotting
    """
    if not result["success"]:
        return {"error": "Solution failed"}
    
    return {
        "spatial_points": result["nodes"],
        "numerical_solution": result["theta_numerical"],
        "analytical_solution": result["theta_analytical"],
        "error": result["error"],
        "max_error": result["max_error"],
        "rms_error": result["rms_error"],
        "time": result["final_time"]
    }


def solve_multiple_times(
    n: int,
    nodes: np.ndarray,
    B_matrix: np.ndarray,
    time_points: List[float]
) -> Dict:
    """
    Solve heat equation and save solution at multiple time points.
    
    Args:
        n: Number of collocation points
        nodes: Gauss-Legendre nodes
        B_matrix: Second derivative matrix
        time_points: List of time points to save solution
        
    Returns:
        Dictionary with solutions at each time point
    """
    results = {}
    
    for t in time_points:
        result = solve_heat_equation_collocation(
            n, nodes, B_matrix,
            t_span=(0, t),
            t_eval=np.array([t])
        )
        
        if result["success"]:
            results[f"t={t}"] = {
                "nodes": result["nodes"],
                "theta_numerical": result["theta_numerical"],
                "theta_analytical": result["theta_analytical"],
                "error": result["error"]
            }
    
    return {
        "time_points": time_points,
        "solutions": results,
        "num_points": n
    }


def convergence_study(
    n_values: List[int],
    t_final: float = 0.1
) -> Dict:
    """
    Study convergence of collocation method with different n.
    
    Args:
        n_values: List of n values to test
        t_final: Final time for comparison
        
    Returns:
        Dictionary with convergence data
    """
    from gaussian_quadrature import compute_gauss_legendre_golub_welsch, orthogonal_collocation_matrices
    
    results = []
    
    for n in n_values:
        try:
            # Compute nodes and matrices
            quad_result = compute_gauss_legendre_golub_welsch(n)
            nodes = np.array(quad_result["nodes"])
            weights = np.array(quad_result["weights"])
            
            colloc_matrices = orthogonal_collocation_matrices(nodes, weights)
            B_matrix = np.array(colloc_matrices["B_matrix"])
            
            # Solve
            solution = solve_heat_equation_collocation(
                n, nodes, B_matrix,
                t_span=(0, t_final),
                t_eval=np.array([t_final])
            )
            
            if solution["success"]:
                results.append({
                    "n": n,
                    "max_error": solution["max_error"],
                    "rms_error": solution["rms_error"]
                })
        except Exception as e:
            results.append({
                "n": n,
                "error": str(e)
            })
    
    return {
        "convergence_data": results,
        "t_final": t_final,
        "n_values": n_values
    }