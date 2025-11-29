"""Optuna 多目的最適化のサンプルスクリプト

使い方例:
  python run_optuna_moo.py --trials 200 --output pareto.png

このスクリプトは 2 目的を同時に最小化し、パレート最適解集合をプロットしてファイルに保存します。
入力変数と出力変数の相関関係も可視化します。
"""
import argparse
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


def objective(trial: optuna.trial.Trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    # 例：2つの目的関数（両方とも最小化）
    f1 = (x - 2.0) ** 2 + (y - 1.0) ** 2 + np.random.normal(0, 0.5)
    f2 = (x + 2.0) ** 2 + (y + 1.0) ** 2 + np.random.normal(0, 0.5)
    return f1, f2


def non_dominated_indices(values_list):
    # values_list: list of tuples/lists (obj1, obj2, ...)
    n = len(values_list)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            vi = values_list[i]
            vj = values_list[j]
            # vj dominates vi if all vj_k <= vi_k and at least one < (minimization)
            if all(vj_k <= vi_k for vj_k, vi_k in zip(vj, vi)) and any(vj_k < vi_k for vj_k, vi_k in zip(vj, vi)):
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]


def plot_pareto(study: optuna.study.Study, output: str = "pareto.png"):
    trials = [t for t in study.trials if t.values is not None]
    if not trials:
        print("No completed trials with objective values.")
        return

    objs = [t.values for t in trials]
    xs = [v[0] for v in objs]
    ys = [v[1] for v in objs]

    pareto_idx = non_dominated_indices(objs)
    pareto_points = [objs[i] for i in pareto_idx]
    px = [p[0] for p in pareto_points]
    py = [p[1] for p in pareto_points]

    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Multi-objective Optimization Analysis", fontsize=14, fontweight="bold")

    # Plot 1: Pareto Front
    ax = axes[0, 0]
    ax.scatter(xs, ys, c="C0", alpha=0.5, s=40, label="All trials")
    ax.scatter(px, py, c="C1", s=100, marker="*", label="Pareto front", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Objective 1 (minimize)", fontsize=10)
    ax.set_ylabel("Objective 2 (minimize)", fontsize=10)
    ax.set_title("Pareto Front", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 2: Objective 1 Distribution
    ax = axes[0, 1]
    ax.hist(xs, bins=20, alpha=0.7, color="C0", edgecolor="black", label="All trials")
    if px:
        ax.axvline(min(px), color="C1", linestyle="--", linewidth=2, label="Pareto range")
        ax.axvline(max(px), color="C1", linestyle="--", linewidth=2)
    ax.set_xlabel("Objective 1", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Objective 1 Distribution", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Objective 2 Distribution
    ax = axes[1, 0]
    ax.hist(ys, bins=20, alpha=0.7, color="C0", edgecolor="black", label="All trials")
    if py:
        ax.axvline(min(py), color="C1", linestyle="--", linewidth=2, label="Pareto range")
        ax.axvline(max(py), color="C1", linestyle="--", linewidth=2)
    ax.set_xlabel("Objective 2", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Objective 2 Distribution", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = f"""
    Total Trials: {len(trials)}
    Pareto-optimal Trials: {len(pareto_idx)}
    
    Objective 1:
      Min: {min(xs):.4f}
      Max: {max(xs):.4f}
      Mean: {sum(xs)/len(xs):.4f}
      Pareto Min: {min(px):.4f}
      Pareto Max: {max(px):.4f}
    
    Objective 2:
      Min: {min(ys):.4f}
      Max: {max(ys):.4f}
      Mean: {sum(ys)/len(ys):.4f}
      Pareto Min: {min(py):.4f}
      Pareto Max: {max(py):.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center", 
            family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Plot 5: Input x vs Objectives
    params_x = [t.params["x"] for t in trials]
    params_y = [t.params["y"] for t in trials]
    pareto_x = [params_x[i] for i in pareto_idx]
    pareto_y = [params_y[i] for i in pareto_idx]
    
    ax = axes[2, 0]
    ax.scatter(params_x, xs, c="C0", alpha=0.5, s=40, label="All trials (Obj1)")
    ax.scatter(params_x, ys, c="C2", alpha=0.5, s=40, label="All trials (Obj2)")
    ax.scatter(pareto_x, [objs[i][0] for i in pareto_idx], c="C1", s=100, marker="*", label="Pareto (Obj1)")
    ax.scatter(pareto_x, [objs[i][1] for i in pareto_idx], c="C3", s=100, marker="*", label="Pareto (Obj2)")
    ax.set_xlabel("Input Parameter x", fontsize=10)
    ax.set_ylabel("Objective Values", fontsize=10)
    ax.set_title("Input x vs Objectives (Correlation)", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: Input y vs Objectives
    ax = axes[2, 1]
    ax.scatter(params_y, xs, c="C0", alpha=0.5, s=40, label="All trials (Obj1)")
    ax.scatter(params_y, ys, c="C2", alpha=0.5, s=40, label="All trials (Obj2)")
    ax.scatter(pareto_y, [objs[i][0] for i in pareto_idx], c="C1", s=100, marker="*", label="Pareto (Obj1)")
    ax.scatter(pareto_y, [objs[i][1] for i in pareto_idx], c="C3", s=100, marker="*", label="Pareto (Obj2)")
    ax.set_xlabel("Input Parameter y", fontsize=10)
    ax.set_ylabel("Objective Values", fontsize=10)
    ax.set_title("Input y vs Objectives (Correlation)", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=100, bbox_inches="tight")
    print(f"Saved comprehensive analysis figure to: {output}")


def plot_gpr_slices(study: optuna.study.Study, output: str = "slices.png"):
    """Plot GPR mean predictions and std bands along each input dimension.
    
    For each input variable, fixes other variables at values that minimize
    each objective, then plots the predicted mean and ±1σ bands.
    """
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials for GPR slice plots. Skipping.")
        return

    X = np.array([[t.params["x"], t.params["y"]] for t in trials])
    y1 = np.array([t.values[0] for t in trials])
    y2 = np.array([t.values[1] for t in trials])

    # Find the index of best (minimum) values for each objective
    best_idx_obj1 = np.argmin(y1)
    best_idx_obj2 = np.argmin(y2)

    # Best parameter values for each objective
    best_x_obj1 = X[best_idx_obj1, 0]
    best_y_obj1 = X[best_idx_obj1, 1]
    best_x_obj2 = X[best_idx_obj2, 0]
    best_y_obj2 = X[best_idx_obj2, 1]

    # Fit GPRs
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr1.fit(X, y1)
    gpr2.fit(X, y2)

    # Create slice plots for each input variable
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("GPR Mean Predictions & Uncertainty Bands (±3σ)", fontsize=14, fontweight="bold")

    param_names = ["x", "y"]
    param_bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    # Plot slices along x (fixing y to best values for each objective)
    grid_x = np.linspace(param_bounds[0][0], param_bounds[0][1], 200)
    
    # Objective 1: slice along x, y fixed to best_y_obj1
    X_slice_obj1 = np.column_stack([grid_x, np.full_like(grid_x, best_y_obj1)])
    mu1_slice, std1_slice = gpr1.predict(X_slice_obj1, return_std=True)
    
    ax = axes[0, 0]
    ax.plot(grid_x, mu1_slice, "b-", linewidth=2, label="Mean")
    ax.fill_between(grid_x, mu1_slice - 3*std1_slice, mu1_slice + 3*std1_slice, alpha=0.3, color="blue", label="±3σ")
    ax.scatter(X[:, 0], y1, alpha=0.5, s=30, color="gray", label="Samples")
    ax.set_xlabel("x (with y fixed to best for Obj1)", fontsize=10)
    ax.set_ylabel("Objective 1", fontsize=10)
    ax.set_title(f"Obj1: x vs Output (y={best_y_obj1:.3f})", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Objective 2: slice along x, y fixed to best_y_obj2
    X_slice_obj2_x = np.column_stack([grid_x, np.full_like(grid_x, best_y_obj2)])
    mu2_slice_x, std2_slice_x = gpr2.predict(X_slice_obj2_x, return_std=True)
    
    ax = axes[0, 1]
    ax.plot(grid_x, mu2_slice_x, "r-", linewidth=2, label="Mean")
    ax.fill_between(grid_x, mu2_slice_x - 3*std2_slice_x, mu2_slice_x + 3*std2_slice_x, alpha=0.3, color="red", label="±3σ")
    ax.scatter(X[:, 0], y2, alpha=0.5, s=30, color="gray", label="Samples")
    ax.set_xlabel("x (with y fixed to best for Obj2)", fontsize=10)
    ax.set_ylabel("Objective 2", fontsize=10)
    ax.set_title(f"Obj2: x vs Output (y={best_y_obj2:.3f})", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot slices along y
    grid_y = np.linspace(param_bounds[1][0], param_bounds[1][1], 200)

    # Objective 1: slice along y, x fixed to best_x_obj1
    X_slice_obj1_y = np.column_stack([np.full_like(grid_y, best_x_obj1), grid_y])
    mu1_slice_y, std1_slice_y = gpr1.predict(X_slice_obj1_y, return_std=True)
    
    ax = axes[1, 0]
    ax.plot(grid_y, mu1_slice_y, "b-", linewidth=2, label="Mean")
    ax.fill_between(grid_y, mu1_slice_y - 3*std1_slice_y, mu1_slice_y + 3*std1_slice_y, alpha=0.3, color="blue", label="±3σ")
    ax.scatter(X[:, 1], y1, alpha=0.5, s=30, color="gray", label="Samples")
    ax.set_xlabel("y (with x fixed to best for Obj1)", fontsize=10)
    ax.set_ylabel("Objective 1", fontsize=10)
    ax.set_title(f"Obj1: y vs Output (x={best_x_obj1:.3f})", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Objective 2: slice along y, x fixed to best_x_obj2
    X_slice_obj2_y = np.column_stack([np.full_like(grid_y, best_x_obj2), grid_y])
    mu2_slice_y, std2_slice_y = gpr2.predict(X_slice_obj2_y, return_std=True)
    
    ax = axes[1, 1]
    ax.plot(grid_y, mu2_slice_y, "r-", linewidth=2, label="Mean")
    ax.fill_between(grid_y, mu2_slice_y - 3*std2_slice_y, mu2_slice_y + 3*std2_slice_y, alpha=0.3, color="red", label="±3σ")
    ax.scatter(X[:, 1], y2, alpha=0.5, s=30, color="gray", label="Samples")
    ax.set_xlabel("y (with x fixed to best for Obj2)", fontsize=10)
    ax.set_ylabel("Objective 2", fontsize=10)
    ax.set_title(f"Obj2: y vs Output (x={best_x_obj2:.3f})", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved GPR slice plots to: {output}")
    return gpr1, gpr2


def plot_gpr_2d_surfaces(study: optuna.study.Study, output: str = "gpr_surfaces.png"):
    """Plot 2D contour maps of GPR predictions: mean and variance for each objective.
    
    Creates a 2x2 grid:
    - Top row: Predictive mean for Obj1 and Obj2
    - Bottom row: Predictive variance for Obj1 and Obj2
    """
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials for 2D surface plots. Skipping.")
        return

    X = np.array([[t.params["x"], t.params["y"]] for t in trials])
    y1 = np.array([t.values[0] for t in trials])
    y2 = np.array([t.values[1] for t in trials])

    # Fit GPRs
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr1.fit(X, y1)
    gpr2.fit(X, y2)

    # Create 2D grid for predictions
    grid_size = 100
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0
    
    xx = np.linspace(x_min, x_max, grid_size)
    yy = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xx, yy)
    Xgrid = np.column_stack([XX.ravel(), YY.ravel()])

    # Predictions
    mu1, std1 = gpr1.predict(Xgrid, return_std=True)
    mu2, std2 = gpr2.predict(Xgrid, return_std=True)
    
    # Reshape to 2D
    MU1 = mu1.reshape(XX.shape)
    MU2 = mu2.reshape(XX.shape)
    VAR1 = (std1 ** 2).reshape(XX.shape)
    VAR2 = (std2 ** 2).reshape(XX.shape)

    # Pareto indices to highlight
    objs = [t.values for t in trials]
    pareto_idx = non_dominated_indices(objs)
    pareto_xy = np.array([[trials[i].params["x"], trials[i].params["y"]] for i in pareto_idx]) if pareto_idx else np.empty((0, 2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("GPR 2D Surfaces: Predictive Mean and Variance", fontsize=14, fontweight="bold")

    # ===== Row 1: Predictive Mean =====
    ax = axes[0, 0]
    contour1_mean = ax.contourf(XX, YY, MU1, levels=40, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c="white", s=30, alpha=0.6, edgecolor="black", linewidth=0.5, label="Samples")
    if pareto_xy.size:
        ax.scatter(pareto_xy[:, 0], pareto_xy[:, 1], c="red", s=100, marker="*", label="Pareto", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Input Variable 1 (x)", fontsize=11)
    ax.set_ylabel("Input Variable 2 (y)", fontsize=11)
    ax.set_title("Objective 1: Predictive Mean μ(x, y)", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    cbar1_mean = fig.colorbar(contour1_mean, ax=ax, label="Mean μ")

    ax = axes[0, 1]
    contour2_mean = ax.contourf(XX, YY, MU2, levels=40, cmap="plasma")
    ax.scatter(X[:, 0], X[:, 1], c="white", s=30, alpha=0.6, edgecolor="black", linewidth=0.5, label="Samples")
    if pareto_xy.size:
        ax.scatter(pareto_xy[:, 0], pareto_xy[:, 1], c="red", s=100, marker="*", label="Pareto", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Input Variable 1 (x)", fontsize=11)
    ax.set_ylabel("Input Variable 2 (y)", fontsize=11)
    ax.set_title("Objective 2: Predictive Mean μ(x, y)", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    cbar2_mean = fig.colorbar(contour2_mean, ax=ax, label="Mean μ")

    # ===== Row 2: Predictive Variance =====
    ax = axes[1, 0]
    contour1_var = ax.contourf(XX, YY, VAR1, levels=40, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c="white", s=30, alpha=0.6, edgecolor="black", linewidth=0.5, label="Samples")
    if pareto_xy.size:
        ax.scatter(pareto_xy[:, 0], pareto_xy[:, 1], c="red", s=100, marker="*", label="Pareto", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Input Variable 1 (x)", fontsize=11)
    ax.set_ylabel("Input Variable 2 (y)", fontsize=11)
    ax.set_title("Objective 1: Predictive Variance σ²(x, y)", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    cbar1_var = fig.colorbar(contour1_var, ax=ax, label="Variance σ²")

    ax = axes[1, 1]
    contour2_var = ax.contourf(XX, YY, VAR2, levels=40, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c="white", s=30, alpha=0.6, edgecolor="black", linewidth=0.5, label="Samples")
    if pareto_xy.size:
        ax.scatter(pareto_xy[:, 0], pareto_xy[:, 1], c="red", s=100, marker="*", label="Pareto", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Input Variable 1 (x)", fontsize=11)
    ax.set_ylabel("Input Variable 2 (y)", fontsize=11)
    ax.set_title("Objective 2: Predictive Variance σ²(x, y)", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    cbar2_var = fig.colorbar(contour2_var, ax=ax, label="Variance σ²")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved GPR 2D surface plots to: {output}")


def plot_sensitivity_analysis(study: optuna.study.Study, output: str = "sensitivity_analysis.png"):
    """Plot three sensitivity analysis methods: Partial Correlation, Local Sensitivity, and PDP.
    
    - Partial Correlation: Linear correlation after removing effects of other variables
    - Local Sensitivity: Derivative of GPR model w.r.t. input parameters
    - Partial Dependence Plot: Expected output when varying one input while marginalizing others
    """
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials for sensitivity analysis. Skipping.")
        return

    X = np.array([[t.params["x"], t.params["y"]] for t in trials])
    y1 = np.array([t.values[0] for t in trials])
    y2 = np.array([t.values[1] for t in trials])

    # Fit GPRs
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr1.fit(X, y1)
    gpr2.fit(X, y2)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Sensitivity Analysis: Input-Output Response", fontsize=14, fontweight="bold")

    # ===== 1. Partial Correlation =====
    # Remove effect of one variable and compute correlation
    from scipy.stats import linregress

    # For x variable (remove effect of y)
    _, residuals_x_obj1 = linregress(X[:, 1], y1)[:2], linregress(X[:, 1], y1)[4:]
    slope_y_on_y1, _, _, _, residuals_y_obj1 = linregress(X[:, 1], y1)
    residuals_y_obj1 = y1 - (slope_y_on_y1 * X[:, 1] + linregress(X[:, 1], y1)[1])
    
    slope_y_on_x, intercept_y_on_x, _, _, _ = linregress(X[:, 1], X[:, 0])
    residuals_x_on_x = X[:, 0] - (slope_y_on_x * X[:, 1] + intercept_y_on_x)
    
    partial_corr_x_obj1 = np.corrcoef(residuals_x_on_x, residuals_y_obj1)[0, 1]
    
    # For y variable (remove effect of x)
    slope_x_on_y1, intercept_x_on_y1, _, _, _ = linregress(X[:, 0], y1)
    residuals_x_on_y1 = y1 - (slope_x_on_y1 * X[:, 0] + intercept_x_on_y1)
    
    slope_x_on_x, intercept_x_on_x, _, _, _ = linregress(X[:, 0], X[:, 1])
    residuals_y_on_x = X[:, 1] - (slope_x_on_x * X[:, 0] + intercept_x_on_x)
    
    partial_corr_y_obj1 = np.corrcoef(residuals_y_on_x, residuals_x_on_y1)[0, 1]
    
    # Similar for objective 2
    slope_y_on_y2, intercept_y_on_y2, _, _, _ = linregress(X[:, 1], y2)
    residuals_y_obj2 = y2 - (slope_y_on_y2 * X[:, 1] + intercept_y_on_y2)
    partial_corr_x_obj2 = np.corrcoef(residuals_x_on_x, residuals_y_obj2)[0, 1]
    
    slope_x_on_y2, intercept_x_on_y2, _, _, _ = linregress(X[:, 0], y2)
    residuals_x_on_y2 = y2 - (slope_x_on_y2 * X[:, 0] + intercept_x_on_y2)
    partial_corr_y_obj2 = np.corrcoef(residuals_y_on_x, residuals_x_on_y2)[0, 1]

    ax = axes[0, 0]
    var_names = ["x", "y"]
    partial_corr_obj1 = [partial_corr_x_obj1, partial_corr_y_obj1]
    colors_obj1 = ["blue" if pc > 0 else "red" for pc in partial_corr_obj1]
    bars1 = ax.bar(var_names, partial_corr_obj1, color=colors_obj1, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Partial Correlation", fontsize=10)
    ax.set_title("Obj1: Partial Correlation (after removing other variable effects)", fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars1, partial_corr_obj1)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05 if val > 0 else val - 0.1, f"{val:.3f}", ha="center", fontsize=9)

    ax = axes[0, 1]
    partial_corr_obj2 = [partial_corr_x_obj2, partial_corr_y_obj2]
    colors_obj2 = ["blue" if pc > 0 else "red" for pc in partial_corr_obj2]
    bars2 = ax.bar(var_names, partial_corr_obj2, color=colors_obj2, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Partial Correlation", fontsize=10)
    ax.set_title("Obj2: Partial Correlation (after removing other variable effects)", fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, val) in enumerate(zip(bars2, partial_corr_obj2)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05 if val > 0 else val - 0.1, f"{val:.3f}", ha="center", fontsize=9)

    # ===== 2. Local Sensitivity (derivatives at mean point) =====
    # Compute numerical gradient of GPR predictions at mean input
    X_mean = X.mean(axis=0)
    eps = 1e-4
    
    # Derivative w.r.t. x
    X_plus_x = X_mean + np.array([eps, 0])
    X_minus_x = X_mean - np.array([eps, 0])
    mu1_plus_x = gpr1.predict(X_plus_x.reshape(1, -1))[0]
    mu1_minus_x = gpr1.predict(X_minus_x.reshape(1, -1))[0]
    grad_x_obj1 = (mu1_plus_x - mu1_minus_x) / (2 * eps)
    
    mu2_plus_x = gpr2.predict(X_plus_x.reshape(1, -1))[0]
    mu2_minus_x = gpr2.predict(X_minus_x.reshape(1, -1))[0]
    grad_x_obj2 = (mu2_plus_x - mu2_minus_x) / (2 * eps)
    
    # Derivative w.r.t. y
    X_plus_y = X_mean + np.array([0, eps])
    X_minus_y = X_mean - np.array([0, eps])
    mu1_plus_y = gpr1.predict(X_plus_y.reshape(1, -1))[0]
    mu1_minus_y = gpr1.predict(X_minus_y.reshape(1, -1))[0]
    grad_y_obj1 = (mu1_plus_y - mu1_minus_y) / (2 * eps)
    
    mu2_plus_y = gpr2.predict(X_plus_y.reshape(1, -1))[0]
    mu2_minus_y = gpr2.predict(X_minus_y.reshape(1, -1))[0]
    grad_y_obj2 = (mu2_plus_y - mu2_minus_y) / (2 * eps)

    ax = axes[1, 0]
    local_sens_obj1 = [abs(grad_x_obj1), abs(grad_y_obj1)]
    colors_sens = ["green", "orange"]
    bars3 = ax.bar(var_names, local_sens_obj1, color=colors_sens, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Local Sensitivity (|dμ/dx|)", fontsize=10)
    ax.set_title(f"Obj1: Local Sensitivity at mean point (x={X_mean[0]:.2f}, y={X_mean[1]:.2f})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars3, local_sens_obj1):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01*max(local_sens_obj1), f"{val:.3f}", ha="center", fontsize=9)

    ax = axes[1, 1]
    local_sens_obj2 = [abs(grad_x_obj2), abs(grad_y_obj2)]
    bars4 = ax.bar(var_names, local_sens_obj2, color=colors_sens, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Local Sensitivity (|dμ/dx|)", fontsize=10)
    ax.set_title(f"Obj2: Local Sensitivity at mean point (x={X_mean[0]:.2f}, y={X_mean[1]:.2f})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars4, local_sens_obj2):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01*max(local_sens_obj2), f"{val:.3f}", ha="center", fontsize=9)

    # ===== 3. Partial Dependence Plot =====
    # Compute PDP by averaging over other variables
    grid_x = np.linspace(-5, 5, 100)
    grid_y = np.linspace(-5, 5, 100)
    
    # For x: average over y (use all y values from data)
    pdp_x_obj1 = []
    pdp_x_obj2 = []
    for x_val in grid_x:
        X_slice = np.column_stack([np.full(len(X), x_val), X[:, 1]])
        mu1_vals = gpr1.predict(X_slice)
        mu2_vals = gpr2.predict(X_slice)
        pdp_x_obj1.append(mu1_vals.mean())
        pdp_x_obj2.append(mu2_vals.mean())
    
    # For y: average over x
    pdp_y_obj1 = []
    pdp_y_obj2 = []
    for y_val in grid_y:
        X_slice = np.column_stack([X[:, 0], np.full(len(X), y_val)])
        mu1_vals = gpr1.predict(X_slice)
        mu2_vals = gpr2.predict(X_slice)
        pdp_y_obj1.append(mu1_vals.mean())
        pdp_y_obj2.append(mu2_vals.mean())

    ax = axes[2, 0]
    ax.plot(grid_x, pdp_x_obj1, "b-", linewidth=2, label="x effect")
    ax.plot(grid_y, pdp_y_obj1, "orange", linewidth=2, label="y effect")
    ax.scatter(X[:, 0], y1, alpha=0.3, s=20, color="gray", label="Samples")
    ax.set_xlabel("Input Value", fontsize=10)
    ax.set_ylabel("Obj1 (Partial Dependence)", fontsize=10)
    ax.set_title("Obj1: Partial Dependence Plot", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(grid_x, pdp_x_obj2, "b-", linewidth=2, label="x effect")
    ax.plot(grid_y, pdp_y_obj2, "orange", linewidth=2, label="y effect")
    ax.scatter(X[:, 0], y2, alpha=0.3, s=20, color="gray", label="Samples")
    ax.set_xlabel("Input Value", fontsize=10)
    ax.set_ylabel("Obj2 (Partial Dependence)", fontsize=10)
    ax.set_title("Obj2: Partial Dependence Plot", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved sensitivity analysis figure to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Optuna multi-objective example")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--output", type=str, default="pareto.png", help="Output image file for Pareto front")
    parser.add_argument("--slices-output", type=str, default="slices.png", help="Output image file for GPR slice plots with confidence bands")
    parser.add_argument("--analysis-output", type=str, default="sensitivity_analysis.png", help="Output image file for sensitivity analysis (Partial Correlation, Local Sensitivity, PDP)")
    parser.add_argument("--surfaces-output", type=str, default="gpr_surfaces.png", help="Output image file for 2D GPR surfaces (mean and variance)")
    args = parser.parse_args()

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=args.trials)

    # 結果のサマリ
    print("Total trials:", len(study.trials))
    # パレート最適なトライアルを手動で抽出して表示
    trials = [t for t in study.trials if t.values is not None]
    objs = [t.values for t in trials]
    pareto_idx = non_dominated_indices(objs)

    print("Pareto-optimal trials:")
    for idx in pareto_idx:
        t = trials[idx]
        print(f"  Trial#{t.number}: values={t.values}, params={t.params}")

    plot_pareto(study, output=args.output)
    # Compute and plot predictive uncertainty (std) from GPRs for each objective
    # Plot GPR slices with confidence bands
    try:
        plot_gpr_slices(study, output=args.slices_output)
    except Exception as e:
        print("GPR slice plot failed:", e)
    # Plot sensitivity analysis (Partial Correlation, Local Sensitivity, PDP)
    try:
        plot_sensitivity_analysis(study, output=args.analysis_output)
    except Exception as e:
        print("Sensitivity analysis plot failed:", e)
    # Plot 2D GPR surfaces (mean and variance)
    try:
        plot_gpr_2d_surfaces(study, output=args.surfaces_output)
    except Exception as e:
        print("GPR 2D surface plot failed:", e)


if __name__ == "__main__":
    main()
