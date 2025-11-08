"""
Iterative, large-input safe Harshad module with progress callback.
"""

from typing import Any, Dict, List, Optional, Callable

ProgressCB = Optional[Callable[[int, Optional[str]], None]]


def digit_sum(n: int) -> int:
    """Digit sum using string conversion. Handles negatives safely."""
    return sum(int(ch) for ch in str(abs(int(n))))


def is_harshad(n: int) -> bool:
    """True if n is divisible by its digit sum."""
    n = int(n)
    if n <= 0:
        return False
    s = digit_sum(n)
    return s != 0 and (n % s == 0)


def factorial_iterative(max_k: int, progress_callback: ProgressCB = None) -> Dict[int, int]:
    """
    Generator-like factorial computation up to max_k.
    Returns a dict {k: factorial(k)} for 1..max_k.
    """
    results = {}
    f = 1
    for k in range(1, max_k + 1):
        f *= k
        results[k] = f
        if progress_callback and (k % max(1, max_k // 100) == 0):
            progress_callback(int(100 * k / max_k), f"computed {k}!")
    return results


def first_nonharshad_factorial(max_k: int = 5000, progress_callback: ProgressCB = None) -> Dict[str, Any]:
    """Find first factorial that is NOT Harshad."""
    fact = 1
    for k in range(1, max_k + 1):
        fact *= k
        ds = digit_sum(fact)
        if progress_callback and k % max(1, max_k // 100) == 0:
            progress_callback(int(100 * k / max_k), f"checked {k}!")
        if ds != 0 and fact % ds != 0:
            next_fact = fact * (k + 1)
            return {
                "k": k,
                "factorial_value": str(fact),
                "digit_sum": ds,
                "is_harshad": False,
                "next_factorial": {
                    "k": k + 1,
                    "digit_sum": digit_sum(next_fact),
                    "is_harshad": (next_fact % digit_sum(next_fact) == 0)
                }
            }
    return {"error": f"No non-Harshad factorial found up to {max_k}"}


def find_consecutive_harshads(length: int, start_hint: int = 1, max_iter: int = 10_000_000,
                              progress_callback: ProgressCB = None) -> Dict[str, Any]:
    """Find length consecutive Harshad numbers iteratively."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if length >= 25:
        raise ValueError("Finding >25 consecutive Harshads is computationally infeasible.")

    count, start_seq = 0, None
    for i in range(start_hint, start_hint + max_iter):
        if progress_callback and i % max(1, max_iter // 200) == 0:
            pct = int(100 * (i - start_hint) / max_iter)
            progress_callback(min(pct, 99), f"scanning {i}")
        if is_harshad(i):
            count += 1
            if count == 1:
                start_seq = i
            if count == length:
                nums = list(range(start_seq, start_seq + length))
                return {
                    "start": start_seq,
                    "end": start_seq + length - 1,
                    "numbers": nums,
                    "verification": [
                        {"n": n, "digit_sum": digit_sum(n), "is_harshad": is_harshad(n)} for n in nums
                    ]
                }
        else:
            count, start_seq = 0, None
    return {"error": f"Run not found within {max_iter} iterations"}
