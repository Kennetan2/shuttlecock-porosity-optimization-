# shuttlecock-porosity-optimization-
Machine learning-assisted optimisation of shuttlecock porosity using Gaussian Process Regression and Bayesian optimisation

# Optimization of Shuttlecock Porosity Through Data-Driven Predictions of Aerodynamic Roughness Length

## Overview

This repository contains the datasets, source code, and supporting figures used in the study:

"Optimising Feather Porosity for Drag Dominated Flight using Gaussian Process Regression and Bayesian Optimisation"

The objective of this study was to determine the feather porosity that minimizes deviation from an analytically derived ideal shuttlecock trajectory. Experimental kinematic data were extracted through computer vision-assisted motion tracking and subsequently analyzed using Gaussian Process Regression (GPR) and Bayesian Optimization (BO).

---

## Research Question

Optimising Feather Porosity for Drag Dominated Flight using Gaussian Process Regression and Bayesian Optimisation. This is being submitted to the JEI research. 

---

## Methodology

The workflow consisted of:

1. Construction of shuttlecocks with varying feather porosities.
2. Controlled free-fall drop tests.
3. Extraction of displacement-time data using computer vision-assisted motion tracking.
4. Computation of horizontal velocity, vertical velocity, horizontal acceleration, and vertical acceleration.
5. Calculation of aerodynamic roughness length using a drag-based analytical model.
6. Development of a Gaussian Process Regression surrogate model.
7. Bayesian Optimization to identify the optimal porosity.

---

## Repository Structure

data/

* raw_shuttlecock_data.md
* porosity_measurements.csv
* optimization_data.csv

code/

* gpr_bayesian_optimization.py

figures/

* figure5_kinematic_profiles.png
* figure6_gpr_response_surface.png
* figure7_objective_function.png

---

## Experimental Variables

Input Variable:

* Feather porosity (ϕ)

Derived Variables:

* Horizontal velocity (vx)
* Vertical velocity (vy)
* Horizontal acceleration (ax)
* Vertical acceleration (ay)
* Aerodynamic roughness length (L)

Optimization Objective:

* J(ϕ)=|Lideal−μ(ϕ)|

---

## Machine Learning Framework

Gaussian Process Regression was implemented using Scikit-learn.

Optimized Kernel:

1.01² × RBF(length_scale = 0.0232) + WhiteKernel(noise_level = 1×10⁻⁸)

Bayesian Optimization was subsequently used to identify the porosity value minimizing aerodynamic deviation.

---

## Key Results

Optimal Porosity:
ϕ = 0.175525

Reported Porosity:
ϕ ≈ 0.176

Maximum Aerodynamic Roughness Length:
L = 3.75 m at ϕ = 0.15

Minimum Objective Value:
2.2 × 10⁻⁵

---

## Software and Libraries

Python Version:
Python 3.14.5

Libraries:

* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* SciPy

---

## Reproducibility

All datasets and scripts required to reproduce the Gaussian Process Regression model, Bayesian Optimization routine, and associated figures are provided in this repository.

---

## Author

Kenne Tan Wen Tong

2026
