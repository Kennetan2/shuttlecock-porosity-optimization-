import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize_scalar

# Experimental dataset
porosity = np.array([0.10, 0.12, 0.15, 0.18, 0.22, 0.31], dtype=float)
roughness_length = np.array([2.91, 3.08, 3.75, 3.42, 3.18, 2.67], dtype=float)

X = porosity.reshape(-1, 1)
y = roughness_length

# Gaussian Process Regression model
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=0.05, length_scale_bounds=(1e-3, 1.0))
    + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=20,
    random_state=42
)

gpr.fit(X, y)

# Ideal aerodynamic roughness length
L_ideal = 3.50

# Objective function
def objective(phi):
    phi_arr = np.array([[phi]])
    predicted_L = gpr.predict(phi_arr)[0]
    return abs(L_ideal - predicted_L)

# Bayesian-style surrogate optimization
result = minimize_scalar(
    objective,
    bounds=(porosity.min(), porosity.max()),
    method="bounded"
)

optimal_porosity = result.x
minimum_objective = result.fun

print("Optimized kernel:", gpr.kernel_)
print(f"Optimal porosity: {optimal_porosity:.6f}")
print(f"Minimum objective value: {minimum_objective:.6f}")

# GPR prediction curve
x_grid = np.linspace(porosity.min(), porosity.max(), 1000).reshape(-1, 1)
y_mean, y_std = gpr.predict(x_grid, return_std=True)

# Objective-function values
objective_vals = np.abs(y_mean - L_ideal)

# Figure 6: Gaussian Process Regression
plt.figure(figsize=(6.4, 4.8), dpi=300)

plt.scatter(X, y)
plt.plot(x_grid, y_mean)

plt.fill_between(
    x_grid.ravel(),
    y_mean - 1.96 * y_std,
    y_mean + 1.96 * y_std,
    alpha=0.2
)

plt.axhline(L_ideal, linestyle="--")
plt.axvline(optimal_porosity, linestyle=":")

plt.xlabel("Porosity, φ")
plt.ylabel("Aerodynamic roughness length, L (m)")
plt.title("Gaussian Process Regression Model")

plt.tight_layout()
plt.savefig("figure_gpr.jpg", dpi=600, bbox_inches="tight")
plt.show()

# Figure 7: Objective function
plt.figure(figsize=(6.4, 4.8), dpi=300)

plt.plot(x_grid, objective_vals)
plt.axvline(optimal_porosity, linestyle=":")
plt.scatter(optimal_porosity, minimum_objective, color="black", zorder=5)

plt.xlabel("Porosity, φ")
plt.ylabel(r"Objective function, $J(\phi)=|L_{\mathrm{ideal}}-\mu(\phi)|$")
plt.title("Optimization of Shuttlecock Porosity")

plt.tight_layout()
plt.savefig("figure_optimization.jpg", dpi=600, bbox_inches="tight")
plt.show()
