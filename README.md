# Computational Neuroscience - Encoding Problems

This repository contains solutions to computational neuroscience problems focusing on statistical inference, decision theory, and neural encoding.

## Overview

This project implements solutions for various computational neuroscience problems including:

- **Maximum Likelihood (ML) and Maximum A Posteriori (MAP) Criteria**: Disease diagnosis using different statistical decision criteria
- **Likelihood Ratio Tests**: Decision making with asymmetric costs
- **Spike-Triggered Average (STA)**: Computing neural receptive fields from stimulus-response data

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── compute_sta.py                     # Spike-triggered average computation
├── solve_ml_diagnosis.py             # ML criterion for disease diagnosis
├── solve_map_diagnosis.py            # MAP criterion for disease diagnosis
├── explain_ml_map_difference.py      # Explanation of ML vs MAP differences
├── solve_likelihood_ratio.py         # Likelihood ratio test (version 1)
├── solve_likelihood_ratio_v2.py      # Likelihood ratio test (version 2)
├── solve_likelihood_ratio_final.py   # Final likelihood ratio solution
├── quiz2.py                          # Quiz 2 problem solution
└── c1p8.pickle                       # Data file for quiz2.py
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

## Installation

1. Clone this repository:
```bash
git clone https://github.com/aifriend/computational_neuroscience_washington.git
cd computational_neuroscience_washington
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Maximum Likelihood (ML) Diagnosis

Solve disease diagnosis using the ML criterion:

```bash
python solve_ml_diagnosis.py
```

This script demonstrates how the ML criterion chooses the hypothesis that maximizes the likelihood of observed data, ignoring prior probabilities.

### Maximum A Posteriori (MAP) Diagnosis

Solve disease diagnosis using the MAP criterion:

```bash
python solve_map_diagnosis.py
```

This script demonstrates how the MAP criterion incorporates prior probabilities using Bayes' theorem.

### ML vs MAP Comparison

Understand the difference between ML and MAP criteria:

```bash
python explain_ml_map_difference.py
```

### Likelihood Ratio Tests

Solve likelihood ratio tests with asymmetric costs:

```bash
python solve_likelihood_ratio_final.py
```

This script finds the optimal decision threshold when different types of errors have different costs.

### Spike-Triggered Average

Compute the spike-triggered average from neural data:

```bash
python quiz2.py
```

This script loads neural data from `c1p8.pickle` and computes the STA, visualizing the neural receptive field.

## Problem Descriptions

### Problem 1: Disease Diagnosis with ML and MAP

Given:
- Disease prevalence: 1 in 100 million
- Test sensitivity: 99%
- False positive rate: 2%
- Patient tests positive

**Question**: Should the patient be diagnosed as positive?

- **ML Criterion**: Chooses the hypothesis that maximizes P(Test Positive | Hypothesis), ignoring prior probabilities.
- **MAP Criterion**: Chooses the hypothesis that maximizes P(Hypothesis | Test Positive), incorporating prior probabilities via Bayes' theorem.

**Key Insight**: ML and MAP can give different answers when prior probabilities are extreme, as demonstrated in this problem.

### Problem 2: Likelihood Ratio Test with Asymmetric Costs

Given:
- Two stimuli: s1 ~ N(μ=5, σ=0.5) and s2 ~ N(μ=7, σ=1)
- Equal priors: P(s1) = P(s2) = 0.5
- Asymmetric costs: C(decide s2 | s1) = 2, C(decide s1 | s2) = 1

**Question**: Find the optimal decision threshold r.

The solution uses the likelihood ratio test formulation:
P(r|s1) / P(r|s2) = C(s2|s1) / C(s1|s2) = 2

### Problem 3: Spike-Triggered Average

Compute the average stimulus preceding each spike to estimate the neural receptive field. The STA is computed as:

STA(τ) = E[stimulus(t-τ) | spike at time t]

This provides an estimate of the stimulus feature that the neuron is most sensitive to.

## Mathematical Background

### Maximum Likelihood (ML)

ML criterion chooses the hypothesis h that maximizes the likelihood:

```
h_ML = argmax_h P(Data | h)
```

### Maximum A Posteriori (MAP)

MAP criterion chooses the hypothesis h that maximizes the posterior probability:

```
h_MAP = argmax_h P(h | Data) = argmax_h P(h) × P(Data | h)
```

### Likelihood Ratio Test

For decision making with costs, the optimal threshold satisfies:

```
P(r|s1) / P(r|s2) = C(s2|s1) / C(s1|s2)
```

where C(s2|s1) is the cost of deciding s2 when the true state is s1.

## License

This project is provided for educational purposes.

## Author

Created as part of computational neuroscience coursework.

## References

- Statistical decision theory
- Bayesian inference
- Neural encoding and spike-triggered analysis

