#!/usr/bin/env python3
"""
Solve Maximum A Posteriori (MAP) criterion for disease diagnosis.

Given:
- Prevalence: P(Disease) = 1/100,000,000
- Test Sensitivity: P(Test Positive | Disease) = 0.99
- Test False Positive Rate: P(Test Positive | No Disease) = 0.02
- Patient tests positive
- Question: Using MAP criterion, diagnose positive?
"""

print("="*70)
print("Maximum A Posteriori (MAP) Criterion for Disease Diagnosis")
print("="*70)

# Parameters
P_disease = 1 / 100_000_000
P_no_disease = 1 - P_disease
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.02

print("\nGiven Information:")
print(f"  Prevalence: P(Disease) = {P_disease:.2e} (1 in 100 million)")
print(f"  P(No Disease) = {P_no_disease:.10f}")
print(f"  Test Sensitivity: P(Test Positive | Disease) = {P_positive_given_disease}")
print(f"  False Positive Rate: P(Test Positive | No Disease) = {P_positive_given_no_disease}")
print(f"  Observation: Patient tests positive")

print("\n" + "-"*70)
print("Maximum A Posteriori (MAP) Criterion:")
print("-"*70)
print("MAP criterion chooses the hypothesis with the HIGHEST POSTERIOR")
print("probability. It incorporates prior probabilities using Bayes' theorem.")
print("\nWe need to calculate:")
print("  P(Disease | Test Positive)")
print("  P(No Disease | Test Positive)")
print("\nAnd choose whichever is higher.")

print("\n" + "-"*70)
print("Bayes' Theorem Calculation:")
print("-"*70)

# Calculate P(Test Positive) using law of total probability
P_positive = (P_positive_given_disease * P_disease + 
              P_positive_given_no_disease * P_no_disease)

print(f"\nStep 1: Calculate P(Test Positive)")
print(f"  P(Test Positive) = P(Test Positive | Disease) × P(Disease) +")
print(f"                     P(Test Positive | No Disease) × P(No Disease)")
print(f"  P(Test Positive) = {P_positive_given_disease} × {P_disease:.2e} +")
print(f"                     {P_positive_given_no_disease} × {P_no_disease:.10f}")
print(f"  P(Test Positive) = {P_positive:.10f}")

# Calculate posterior probabilities using Bayes' theorem
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive
P_no_disease_given_positive = (P_positive_given_no_disease * P_no_disease) / P_positive

print(f"\nStep 2: Calculate P(Disease | Test Positive)")
print(f"  P(Disease | Test Positive) = P(Test Positive | Disease) × P(Disease) / P(Test Positive)")
print(f"  P(Disease | Test Positive) = {P_positive_given_disease} × {P_disease:.2e} / {P_positive:.10f}")
print(f"  P(Disease | Test Positive) = {P_disease_given_positive:.10f}")

print(f"\nStep 3: Calculate P(No Disease | Test Positive)")
print(f"  P(No Disease | Test Positive) = P(Test Positive | No Disease) × P(No Disease) / P(Test Positive)")
print(f"  P(No Disease | Test Positive) = {P_positive_given_no_disease} × {P_no_disease:.10f} / {P_positive:.10f}")
print(f"  P(No Disease | Test Positive) = {P_no_disease_given_positive:.10f}")

print("\n" + "="*70)
print("Decision:")
print("="*70)

print(f"\nPosterior Probabilities:")
print(f"  P(Disease | Test Positive) = {P_disease_given_positive:.10f}")
print(f"  P(No Disease | Test Positive) = {P_no_disease_given_positive:.10f}")

if P_disease_given_positive > P_no_disease_given_positive:
    print(f"\n✓ P(Disease | Test Positive) > P(No Disease | Test Positive)")
    print("  The posterior probability is HIGHER for the disease hypothesis.")
    print("\n  ANSWER: YES - Diagnose as positive")
else:
    print(f"\n✗ P(Disease | Test Positive) < P(No Disease | Test Positive)")
    print("  The posterior probability is HIGHER for the no-disease hypothesis.")
    print("\n  ANSWER: NO - Do not diagnose as positive")

print("\n" + "-"*70)
print("Key Insight:")
print("-"*70)
print("Even though the test is quite accurate (99% sensitivity, 2% false positive),")
print("the extreme rarity of the disease (1 in 100 million) means that a positive")
print("test is still much more likely to be a false positive than a true positive.")
print(f"\nThe posterior probability of disease is only {P_disease_given_positive*100:.6f}%,")
print(f"while the probability of no disease is {P_no_disease_given_positive*100:.6f}%.")

# Show the ratio
ratio = P_no_disease_given_positive / P_disease_given_positive
print(f"\nThe patient is approximately {ratio:.0f} times more likely to NOT have")
print("the disease than to have it, despite testing positive!")

