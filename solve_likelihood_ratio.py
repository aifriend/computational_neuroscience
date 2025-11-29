#!/usr/bin/env python3
"""
Solve likelihood ratio test with asymmetric costs.

Given:
- s1: N(μ=5, σ=0.5)
- s2: N(μ=7, σ=1)
- P(s1) = P(s2) = 0.5
- C(decide s2 | s1) = 2
- C(decide s1 | s2) = 1
"""

import numpy as np
from scipy.stats import norm

# Parameters
mu1, sigma1 = 5, 0.5
mu2, sigma2 = 7, 1.0
P_s1 = 0.5
P_s2 = 0.5
C_s2_given_s1 = 2
C_s1_given_s2 = 1

# Calculate threshold
threshold = (P_s1 * C_s2_given_s1) / (P_s2 * C_s1_given_s2)
print(f"Likelihood ratio threshold: {threshold}")

# For Gaussian distributions, the likelihood ratio is:
# LR(r) = P(r|s2) / P(r|s1) = (σ1/σ2) * exp(-0.5*((r-μ2)/σ2)^2 + 0.5*((r-μ1)/σ1)^2)
# We want LR(r) = threshold

# Taking log:
# log(LR) = log(σ1/σ2) - 0.5*((r-μ2)/σ2)^2 + 0.5*((r-μ1)/σ1)^2 = log(threshold)

# Expanding:
# log(σ1/σ2) - 0.5*(r-μ2)^2/σ2^2 + 0.5*(r-μ1)^2/σ1^2 = log(threshold)
# Multiply by 2:
# 2*log(σ1/σ2) - (r-μ2)^2/σ2^2 + (r-μ1)^2/σ1^2 = 2*log(threshold)

# Expand (r-μ)^2 terms:
# (r-μ1)^2/σ1^2 - (r-μ2)^2/σ2^2 = 2*log(threshold) - 2*log(σ1/σ2)
# (r^2 - 2*r*μ1 + μ1^2)/σ1^2 - (r^2 - 2*r*μ2 + μ2^2)/σ2^2 = 2*log(threshold) - 2*log(σ1/σ2)

# Rearranging:
# r^2*(1/σ1^2 - 1/σ2^2) - 2*r*(μ1/σ1^2 - μ2/σ2^2) + (μ1^2/σ1^2 - μ2^2/σ2^2) = 2*log(threshold) - 2*log(σ1/σ2)

# Coefficients for quadratic equation: a*r^2 + b*r + c = 0
a = (1/sigma1**2 - 1/sigma2**2)
b = -2 * (mu1/sigma1**2 - mu2/sigma2**2)
c = (mu1**2/sigma1**2 - mu2**2/sigma2**2) - 2*np.log(threshold) + 2*np.log(sigma1/sigma2)

print(f"\nQuadratic coefficients:")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")

# Solve quadratic equation
# Check for division by zero
if abs(a) < 1e-10:
    print("Error: Coefficient 'a' is too close to zero. Cannot solve quadratic equation.")
    threshold_r = None
else:
    discriminant = b**2 - 4*a*c
    print(f"\nDiscriminant: {discriminant}")

    if discriminant >= 0:
        r1 = (-b + np.sqrt(discriminant)) / (2*a)
        r2 = (-b - np.sqrt(discriminant)) / (2*a)
        print(f"\nSolutions:")
        print(f"r1 = {r1}")
        print(f"r2 = {r2}")
        
        # Check which solution is between the means (reasonable threshold)
        solutions = [r1, r2]
        valid_solutions = [r for r in solutions if mu1 <= r <= mu2]
        
        if valid_solutions:
            threshold_r = valid_solutions[0]
            print(f"\nValid threshold (between μ1={mu1} and μ2={mu2}): {threshold_r}")
        else:
            # If neither is between, choose the one closer to the midpoint
            midpoint = (mu1 + mu2) / 2
            threshold_r = min(solutions, key=lambda x: abs(x - midpoint))
            print(f"\nThreshold (closest to midpoint): {threshold_r}")
    else:
        print("No real solutions!")
        threshold_r = None

# Verify the solution
if threshold_r is not None:
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")
    
    # Calculate likelihood ratio at threshold
    pdf_s1 = norm.pdf(threshold_r, mu1, sigma1)
    pdf_s2 = norm.pdf(threshold_r, mu2, sigma2)
    LR = pdf_s2 / pdf_s1
    
    print(f"At r = {threshold_r}:")
    print(f"  P(r|s1) = {pdf_s1:.6f}")
    print(f"  P(r|s2) = {pdf_s2:.6f}")
    print(f"  LR = P(r|s2)/P(r|s1) = {LR:.6f}")
    print(f"  Required threshold = {threshold:.6f}")
    print(f"  Difference = {abs(LR - threshold):.6f}")
    
    # Check decision rule: decide s2 if LR > threshold
    print(f"\nDecision rule: Decide s2 if LR > {threshold}")
    print(f"At threshold r = {threshold_r}, LR = {LR:.6f}")

# Test all given options
print(f"\n{'='*60}")
print("Testing given options:")
print(f"{'='*60}")

options = [5.978, 5.830, 2.69, 5.667]

for r_test in options:
    pdf_s1_test = norm.pdf(r_test, mu1, sigma1)
    pdf_s2_test = norm.pdf(r_test, mu2, sigma2)
    LR_test = pdf_s2_test / pdf_s1_test
    
    print(f"\nr = {r_test}:")
    print(f"  P(r|s1) = {pdf_s1_test:.6f}")
    print(f"  P(r|s2) = {pdf_s2_test:.6f}")
    print(f"  LR = {LR_test:.6f}")
    print(f"  Required threshold = {threshold:.6f}")
    print(f"  |LR - threshold| = {abs(LR_test - threshold):.6f}")

# Find which option is closest to the threshold
if threshold_r is not None:
    print(f"\n{'='*60}")
    print("Best match:")
    print(f"{'='*60}")
    
    best_option = min(options, key=lambda r: abs(norm.pdf(r, mu2, sigma2) / norm.pdf(r, mu1, sigma1) - threshold))
    print(f"\nThe option closest to the optimal threshold is: {best_option}")

