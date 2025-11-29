#!/usr/bin/env python3
"""
Solve Maximum Likelihood (ML) criterion for disease diagnosis.

Given:
- Prevalence: P(Disease) = 1/100,000,000
- Test Sensitivity: P(Test Positive | Disease) = 0.99
- Test False Positive Rate: P(Test Positive | No Disease) = 0.02
- Patient tests positive
- Question: Using ML criterion, diagnose positive?
"""

print("="*70)
print("Maximum Likelihood (ML) Criterion for Disease Diagnosis")
print("="*70)

# Parameters
P_disease = 1 / 100_000_000
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.02

print("\nGiven Information:")
print(f"  Prevalence: P(Disease) = {P_disease:.2e} (1 in 100 million)")
print(f"  Test Sensitivity: P(Test Positive | Disease) = {P_positive_given_disease}")
print(f"  False Positive Rate: P(Test Positive | No Disease) = {P_positive_given_no_disease}")
print(f"  Observation: Patient tests positive")

print("\n" + "-"*70)
print("Maximum Likelihood (ML) Criterion:")
print("-"*70)
print("ML criterion chooses the hypothesis that maximizes the likelihood")
print("of the observed data. It does NOT consider prior probabilities.")
print("\nWe compare:")
print("  Hypothesis 1 (Has Disease):")
print(f"    Likelihood = P(Test Positive | Disease) = {P_positive_given_disease}")
print("\n  Hypothesis 2 (No Disease):")
print(f"    Likelihood = P(Test Positive | No Disease) = {P_positive_given_no_disease}")

print("\n" + "="*70)
print("Decision:")
print("="*70)

if P_positive_given_disease > P_positive_given_no_disease:
    print(f"\n✓ {P_positive_given_disease} > {P_positive_given_no_disease}")
    print("  The likelihood of observing a positive test is HIGHER")
    print("  under the hypothesis that the patient HAS the disease.")
    print("\n  ANSWER: YES - Diagnose as positive")
else:
    print(f"\n✗ {P_positive_given_disease} < {P_positive_given_no_disease}")
    print("  The likelihood of observing a positive test is HIGHER")
    print("  under the hypothesis that the patient does NOT have the disease.")
    print("\n  ANSWER: NO - Do not diagnose as positive")

print("\n" + "-"*70)
print("Important Note:")
print("-"*70)
print("ML criterion ignores the prior probability (prevalence).")
print("Even though the disease is extremely rare (1 in 100 million),")
print("the ML criterion only considers which hypothesis makes the")
print("observed positive test result more likely.")
print("\nThis is different from Bayesian inference, which would consider")
print("the prior probability and might lead to a different conclusion")
print("due to the extreme rarity of the disease.")

