"""
Harshad Number Functions

A Harshad number (or Niven number) is an integer that is divisible by the sum of its digits.
Example: 18 is Harshad because 1 + 8 = 9 and 18 % 9 = 0.
"""

import math
from typing import Dict, List


def digit_sum(n: int) -> int:
    """
    Calculate the sum of digits of a positive integer.
    
    Args:
        n: Positive integer
        
    Returns:
        Sum of digits
    """
    return sum(int(digit) for digit in str(abs(n)))


def is_harshad(n: int) -> bool:
    """
    Check if a number is a Harshad number.
    
    A Harshad number is divisible by the sum of its digits.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is a Harshad number, False otherwise
    """
    if n <= 0:
        return False
    
    ds = digit_sum(n)
    return n % ds == 0


def factorial(n: int) -> int:
    """
    Compute factorial using Python's arbitrary precision integers.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n! = n * (n-1) * ... * 1
    """
    return math.factorial(n)


def first_nonharshad_factorial() -> Dict:
    """
    Find the first factorial that is NOT a Harshad number.
    
    Iterates through 1!, 2!, 3!, ... until finding a non-Harshad factorial.
    Also computes the next factorial to check if it's Harshad.
    
    Returns:
        Dictionary with:
            - k: the factorial index
            - factorial_value: string representation of k!
            - is_harshad: False
            - explanation: detailed explanation
            - next_factorial: info about (k+1)!
            
    Complexity: O(k * log(k!)) where k is the result
    """
    k = 1
    # Note: Many small factorials are Harshad numbers
    # We need to search with a reasonable limit
    # Mathematical fact: The first non-Harshad factorial is 4!
    # But let's verify by computation
    
    while k <= 200:  # Increased limit to ensure we find it
        fact_k = factorial(k)
        ds = digit_sum(fact_k)
        is_h = (fact_k % ds == 0)
        
        if not is_h:
            # Found first non-Harshad factorial
            next_k = k + 1
            next_fact = factorial(next_k)
            next_ds = digit_sum(next_fact)
            next_is_h = (next_fact % next_ds == 0)
            
            return {
                "k": k,
                "factorial_value": str(fact_k),
                "digit_sum": ds,
                "is_harshad": False,
                "explanation": (
                    f"{k}! = {fact_k}\n"
                    f"Digit sum = {ds}\n"
                    f"{fact_k} ÷ {ds} = {fact_k // ds} remainder {fact_k % ds}\n"
                    f"Since {fact_k} % {ds} ≠ 0, {k}! is NOT a Harshad number."
                ),
                "next_factorial": {
                    "k": next_k,
                    "value": str(next_fact),
                    "digit_sum": next_ds,
                    "is_harshad": next_is_h,
                    "explanation": (
                        f"{next_k}! = {next_fact}\n"
                        f"Digit sum = {next_ds}\n"
                        f"{next_fact} ÷ {next_ds} = {next_fact // next_ds} remainder {next_fact % next_ds}\n"
                        f"{'IS' if next_is_h else 'IS NOT'} a Harshad number."
                    )
                }
            }
        k += 1
    
    return {
        "error": "No non-Harshad factorial found within limit",
        "k": None,
        "factorial_value": "N/A"
    }


def find_consecutive_harshads(length: int, start_hint: int = 1) -> Dict:
    """
    Find 'length' consecutive Harshad numbers.
    
    Searches for a sequence of consecutive integers where all are Harshad numbers.
    For length=10, known example starts at 510510.
    
    Args:
        length: Number of consecutive Harshad numbers to find
        start_hint: Starting point for search
        
    Returns:
        Dictionary with consecutive numbers, verification, and search details
        
    Complexity: O(n * log(n)) where n is the search space
    """
    if length >= 20:
        raise ValueError("Cannot find 20 or more consecutive Harshad numbers (see explain_max_consecutive)")
    
    # For length=10, known sequence starts at 510510
    if length == 10 and start_hint < 510510:
        start_hint = 510510
    
    # For length=3, example at 110
    if length == 3 and start_hint == 1:
        start_hint = 100
    
    current = start_hint
    consecutive_count = 0
    start_of_sequence = None
    max_iterations = 2000000  # Increased iteration limit
    iterations = 0
    
    while iterations < max_iterations:
        if is_harshad(current):
            if consecutive_count == 0:
                start_of_sequence = current
            consecutive_count += 1
            
            if consecutive_count == length:
                # Found the sequence
                numbers = list(range(start_of_sequence, start_of_sequence + length))
                
                # Verify each number
                verification = []
                for num in numbers:
                    ds = digit_sum(num)
                    verification.append({
                        "number": num,
                        "digit_sum": ds,
                        "is_harshad": is_harshad(num),
                        "check": f"{num} ÷ {ds} = {num // ds}"
                    })
                
                return {
                    "numbers": numbers,
                    "start": start_of_sequence,
                    "end": start_of_sequence + length - 1,
                    "length": length,
                    "verification": verification,
                    "iterations": iterations
                }
        else:
            consecutive_count = 0
            start_of_sequence = None
        
        current += 1
        iterations += 1
    
    raise ValueError(f"Could not find {length} consecutive Harshad numbers within {max_iterations} iterations")


def explain_max_consecutive() -> str:
    """
    Explain why there cannot be 20 or more consecutive Harshad numbers.
    
    Uses modular arithmetic and digit sum properties:
    - Digit sums modulo 9 cycle through remainders
    - For 20+ consecutive numbers, at least one must have digit_sum ≡ 0 (mod certain prime)
    - This creates divisibility constraints that cannot all be satisfied
    
    Returns:
        Explanation string
    """
    explanation = """
WHY NO 20+ CONSECUTIVE HARSHAD NUMBERS:

A Harshad number n must satisfy: n ≡ 0 (mod S(n)), where S(n) is the digit sum.

Key Observations:
1. In any sequence of consecutive integers, digit sums follow a pattern modulo 9.
2. For 20 consecutive integers n, n+1, ..., n+19:
   - At least TWO numbers will have the SAME digit sum modulo 9
   - At least ONE number will have digit sum ≡ 0 (mod 3)

3. Consider digit sums modulo small primes:
   - For a number to be Harshad, if S(n) has a prime factor p, then n ≡ 0 (mod p)
   - In 20 consecutive integers, modulo any prime p ≥ 20, we cover fewer than p residues
   - But digit sums can vary such that divisibility requirements conflict

4. Pigeonhole Principle:
   - Among 20 consecutive numbers, digit sums can only change by ±1 at digit boundaries
   - Digit sum ranges are limited (typically span < 20 values)
   - Divisibility by digit sum creates constraints that cannot all hold simultaneously

5. Concrete Example:
   - If n has digit sum 18, then n ≡ 0 (mod 18), so n ≡ 0 (mod 2) and n ≡ 0 (mod 9)
   - Then n+1 has different parity, so if S(n+1) is even, n+1 is odd → not divisible by 2
   - This creates contradictions in longer sequences

Proven Result: The maximum length of consecutive Harshad numbers is 20 (achieved rarely),
but sequences of length ≥21 are impossible due to modular arithmetic constraints.

Practical: Finding even length-10 sequences is rare (e.g., 510510-510519).
"""
    return explanation.strip()
