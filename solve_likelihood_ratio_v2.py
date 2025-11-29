#!/usr/bin/env python3
"""
Solve likelihood ratio test with asymmetric costs - testing both interpretations.

Given:
- s1: N(μ=5, σ=0.5)
- s2: N(μ=7, σ=1)
- P(s1) = P(s2) = 0.5
- C(decide s2 | s1) = 2
- C(decide s1 | s2) = 1
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

# Parameters
mu1, sigma1 = 5, 0.5
mu2, sigma2 = 7, 1.0
P_s1 = 0.5
P_s2 = 0.5
C_s2_given_s1 = 2
C_s1_given_s2 = 1

print("="*70)
print("INTERPRETATION 1: Standard Bayesian Decision Theory")
print("="*70)
print("\nDecision rule: Decide s2 if P(r|s2)/P(r|s1) > threshold")
print("where threshold = [P(s1) × C(s2|s1)] / [P(s2) × C(s1|s2)]")

threshold1 = (P_s1 * C_s2_given_s1) / (P_s2 * C_s1_given_s2)
print(f"\nThreshold = (0.5 × 2) / (0.5 × 1) = {threshold1}")

# Solve: P(r|s2) / P(r|s1) = threshold1
# Taking log and rearranging:
# log(P(r|s2)) - log(P(r|s1)) = log(threshold1)
# -0.5*((r-μ2)/σ2)^2 - log(σ2√(2π)) + 0.5*((r-μ1)/σ1)^2 + log(σ1√(2π)) = log(threshold1)
# Simplifying: (r-μ1)^2/σ1^2 - (r-μ2)^2/σ2^2 = 2*log(threshold1) - 2*log(σ1/σ2)

a1 = (1/sigma1**2 - 1/sigma2**2)
b1 = -2 * (mu1/sigma1**2 - mu2/sigma2**2)
c1 = (mu1**2/sigma1**2 - mu2**2/sigma2**2) - 2*np.log(threshold1) + 2*np.log(sigma1/sigma2)

discriminant1 = b1**2 - 4*a1*c1
r1_1 = (-b1 + np.sqrt(discriminant1)) / (2*a1)
r1_2 = (-b1 - np.sqrt(discriminant1)) / (2*a1)

valid_r1 = [r for r in [r1_1, r1_2] if mu1 <= r <= mu2]
if valid_r1:
    threshold_r1 = valid_r1[0]
    print(f"\nSolution: r = {threshold_r1:.6f}")
    
    pdf_s1_1 = norm.pdf(threshold_r1, mu1, sigma1)
    pdf_s2_1 = norm.pdf(threshold_r1, mu2, sigma2)
    LR1 = pdf_s2_1 / pdf_s1_1
    print(f"Verification: P(r|s2)/P(r|s1) = {LR1:.6f} (target: {threshold1:.6f})")

print("\n" + "="*70)
print("INTERPRETATION 2: Alternative Asymmetric Cost Formulation")
print("="*70)
print("\nDecision rule based on: P(r|s1) / P(r|s2) = C(s2|s1) / C(s1|s2)")
print("This means: P(r|s1) / P(r|s2) = 2/1 = 2")

threshold2 = C_s2_given_s1 / C_s1_given_s2
print(f"\nThreshold = 2 / 1 = {threshold2}")

# Solve: P(r|s1) / P(r|s2) = threshold2
# Taking log:
# log(P(r|s1)) - log(P(r|s2)) = log(threshold2)
# -0.5*((r-μ1)/σ1)^2 - log(σ1√(2π)) + 0.5*((r-μ2)/σ2)^2 + log(σ2√(2π)) = log(threshold2)
# Simplifying: (r-μ2)^2/σ2^2 - (r-μ1)^2/σ1^2 = 2*log(threshold2) - 2*log(σ2/σ1)

a2 = (1/sigma2**2 - 1/sigma1**2)
b2 = -2 * (mu2/sigma2**2 - mu1/sigma1**2)
c2 = (mu2**2/sigma2**2 - mu1**2/sigma1**2) - 2*np.log(threshold2) + 2*np.log(sigma2/sigma1)

discriminant2 = b2**2 - 4*a2*c2
r2_1 = (-b2 + np.sqrt(discriminant2)) / (2*a2)
r2_2 = (-b2 - np.sqrt(discriminant2)) / (2*a2)

valid_r2 = [r for r in [r2_1, r2_2] if mu1 <= r <= mu2]
if valid_r2:
    threshold_r2 = valid_r2[0]
    print(f"\nSolution: r = {threshold_r2:.6f}")
    
    pdf_s1_2 = norm.pdf(threshold_r2, mu1, sigma1)
    pdf_s2_2 = norm.pdf(threshold_r2, mu2, sigma2)
    LR2 = pdf_s1_2 / pdf_s2_2
    print(f"Verification: P(r|s1)/P(r|s2) = {LR2:.6f} (target: {threshold2:.6f})")

print("\n" + "="*70)
print("SYMMETRIC CASE (for comparison)")
print("="*70)
print("\nDecision rule: P(r|s1) = P(r|s2)")

# Solve: P(r|s1) = P(r|s2)
# Taking log:
# -0.5*((r-μ1)/σ1)^2 - log(σ1√(2π)) = -0.5*((r-μ2)/σ2)^2 - log(σ2√(2π))
# Simplifying: (r-μ2)^2/σ2^2 - (r-μ1)^2/σ1^2 = 2*log(σ2/σ1)

a_sym = (1/sigma2**2 - 1/sigma1**2)
b_sym = -2 * (mu2/sigma2**2 - mu1/sigma1**2)
c_sym = (mu2**2/sigma2**2 - mu1**2/sigma1**2) - 2*np.log(sigma2/sigma1)

discriminant_sym = b_sym**2 - 4*a_sym*c_sym
r_sym_1 = (-b_sym + np.sqrt(discriminant_sym)) / (2*a_sym)
r_sym_2 = (-b_sym - np.sqrt(discriminant_sym)) / (2*a_sym)

valid_r_sym = [r for r in [r_sym_1, r_sym_2] if mu1 <= r <= mu2]
if valid_r_sym:
    threshold_r_sym = valid_r_sym[0]
    print(f"\nSolution: r = {threshold_r_sym:.6f}")
    
    pdf_s1_sym = norm.pdf(threshold_r_sym, mu1, sigma1)
    pdf_s2_sym = norm.pdf(threshold_r_sym, mu2, sigma2)
    print(f"Verification: P(r|s1) = {pdf_s1_sym:.6f}, P(r|s2) = {pdf_s2_sym:.6f}")
    print(f"Difference: {abs(pdf_s1_sym - pdf_s2_sym):.6f}")

print("\n" + "="*70)
print("TESTING GIVEN OPTIONS")
print("="*70)

options = [5.978, 5.830, 2.69, 5.667]

for r_test in options:
    pdf_s1_test = norm.pdf(r_test, mu1, sigma1)
    pdf_s2_test = norm.pdf(r_test, mu2, sigma2)
    
    LR_forward = pdf_s2_test / pdf_s1_test  # P(r|s2)/P(r|s1)
    LR_reverse = pdf_s1_test / pdf_s2_test  # P(r|s1)/P(r|s2)
    
    print(f"\nr = {r_test}:")
    print(f"  P(r|s1) = {pdf_s1_test:.6f}")
    print(f"  P(r|s2) = {pdf_s2_test:.6f}")
    print(f"  P(r|s2)/P(r|s1) = {LR_forward:.6f} (target for Interp1: {threshold1:.6f})")
    print(f"  P(r|s1)/P(r|s2) = {LR_reverse:.6f} (target for Interp2: {threshold2:.6f})")
    print(f"  |P(r|s1) - P(r|s2)| = {abs(pdf_s1_test - pdf_s2_test):.6f} (symmetric case)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if valid_r1:
    print(f"\nInterpretation 1 (Bayesian): r = {threshold_r1:.6f}")
if valid_r2:
    print(f"Interpretation 2 (Alternative): r = {threshold_r2:.6f}")
if valid_r_sym:
    print(f"Symmetric case: r = {threshold_r_sym:.6f}")

print(f"\nGiven options: {options}")
print("\nWhich option matches which interpretation?")
for r_test in options:
    pdf_s1_test = norm.pdf(r_test, mu1, sigma1)
    pdf_s2_test = norm.pdf(r_test, mu2, sigma2)
    LR_reverse = pdf_s1_test / pdf_s2_test
    
    if abs(LR_reverse - 2.0) < 0.01:
        print(f"  r = {r_test}: P(r|s1)/P(r|s2) ≈ 2.0 → Matches Interpretation 2!")

