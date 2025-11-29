#!/usr/bin/env python3
"""
Explain the difference between ML and MAP criteria.

From the previous problems:
- ML criterion: YES (diagnose as positive)
- MAP criterion: NO (do not diagnose as positive)

Why do they differ?
"""

print("="*70)
print("Why ML and MAP Criteria Differ")
print("="*70)

print("\n" + "-"*70)
print("Key Difference:")
print("-"*70)
print("\nMaximum Likelihood (ML) Criterion:")
print("  • Chooses hypothesis that maximizes P(Data | Hypothesis)")
print("  • IGNORES prior probabilities")
print("  • Only considers: P(Test Positive | Disease) vs P(Test Positive | No Disease)")
print("  • Result: 0.99 > 0.02 → Diagnose as positive (YES)")

print("\nMaximum A Posteriori (MAP) Criterion:")
print("  • Chooses hypothesis that maximizes P(Hypothesis | Data)")
print("  • INCORPORATES prior probabilities using Bayes' theorem")
print("  • Considers: P(Disease | Test Positive) vs P(No Disease | Test Positive)")
print("  • Result: 0.0000004950 < 0.9999995050 → Do not diagnose (NO)")

print("\n" + "="*70)
print("Evaluating Answer Options:")
print("="*70)

options = [
    "Unlike MAP, ML assumes the same model for all people.",
    "Since ML assumes a Gaussian distribution, unlike MAP, it oversimplifies the world.",
    "There is no difference between the two, because in this case they are equivalent.",
    "The role of prior probability is different between the two."
]

print("\nOption 1: 'Unlike MAP, ML assumes the same model for all people.'")
print("  ✗ INCORRECT - Both ML and MAP use the same probabilistic model.")
print("    The difference is not about model assumptions, but about how")
print("    prior information is incorporated.")

print("\nOption 2: 'Since ML assumes a Gaussian distribution, unlike MAP, it oversimplifies the world.'")
print("  ✗ INCORRECT - Neither ML nor MAP assumes Gaussian distributions here.")
print("    This problem uses discrete probabilities (disease/no disease).")
print("    The Gaussian assumption is irrelevant to this problem.")

print("\nOption 3: 'There is no difference between the two, because in this case they are equivalent.'")
print("  ✗ INCORRECT - We just demonstrated they give DIFFERENT answers:")
print("    • ML: YES (diagnose as positive)")
print("    • MAP: NO (do not diagnose as positive)")
print("    They are clearly NOT equivalent in this case.")

print("\nOption 4: 'The role of prior probability is different between the two.'")
print("  ✓ CORRECT!")
print("    • ML IGNORES prior probabilities (prevalence of disease)")
print("    • MAP INCORPORATES prior probabilities using Bayes' theorem")
print("    • This is exactly why they differ:")
print("      - ML only looks at likelihoods: P(Test Positive | Disease) = 0.99")
print("        vs P(Test Positive | No Disease) = 0.02")
print("      - MAP considers posteriors: P(Disease | Test Positive) ≈ 0.0000005")
print("        vs P(No Disease | Test Positive) ≈ 0.9999995")
print("    • The extreme rarity of the disease (1 in 100 million) dramatically")
print("      affects the MAP decision but is completely ignored by ML.")

print("\n" + "="*70)
print("ANSWER: Option 4")
print("="*70)
print("\n'The role of prior probability is different between the two.'")
print("\nThis is the fundamental difference:")
print("  • ML: Prior probabilities play NO role")
print("  • MAP: Prior probabilities are ESSENTIAL (via Bayes' theorem)")

print("\n" + "-"*70)
print("Mathematical Formulation:")
print("-"*70)
print("\nML: argmax_h P(Data | h)")
print("    = argmax_h P(Test Positive | h)")
print("    where h ∈ {Disease, No Disease}")
print("\nMAP: argmax_h P(h | Data)")
print("     = argmax_h P(h) × P(Data | h) / P(Data)")
print("     = argmax_h P(h) × P(Data | h)")
print("     where P(h) is the PRIOR probability")
print("\nThe prior P(Disease) = 1/100,000,000 is what makes MAP differ from ML!")

