"""Optuna 多目的最適化のサンプルスクリプト

使い方例:
  python run_optuna_moo.py --trials 200 --output pareto.png

このスクリプトは 2 目的を同時に最小化し、パレート最適解集合をプロットしてファイルに保存します。
入力変数と出力変数の相関関係も可視化します。
"""
import argparse
import os
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


def objective(trial: optuna.trial.Trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    # 例：2つの目的関数（両方とも最小化）
    f1 = (x - 2.0) ** 2 + (y - 1.0) ** 2 + np.random.normal(0, 0.5) - x - y
    f2 = (x + 2.0) ** 2 + (y + 1.0) ** 2 + np.random.normal(0, 0.5) + x + y
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
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

    # Plot 4: Summary Statistics
    ax = axes[0, 1]
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
            family="monospace", bbox=dict(boxstyle="round,pad=1.2", facecolor="wheat", alpha=0.5))

    # Plot 5: Input x vs Objectives
    params_x = [t.params["x"] for t in trials]
    params_y = [t.params["y"] for t in trials]
    pareto_x = [params_x[i] for i in pareto_idx]
    pareto_y = [params_y[i] for i in pareto_idx]
    
    ax = axes[1, 0]
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
    ax = axes[1, 1]
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

    # Fit GPRs
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr1.fit(X, y1)
    gpr2.fit(X, y2)

    # Find predicted optima (argmin of GPR mean) for each objective via grid search
    grid_size = 120
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0
    xx = np.linspace(x_min, x_max, grid_size)
    yy = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    mu1_grid = gpr1.predict(grid)
    mu2_grid = gpr2.predict(grid)
    opt1 = grid[np.argmin(mu1_grid)]  # (x*, y*) for objective 1
    opt2 = grid[np.argmin(mu2_grid)]  # (x*, y*) for objective 2
    
    # Predicted optimal parameter values for each objective
    best_x_obj1, best_y_obj1 = opt1[0], opt1[1]
    best_x_obj2, best_y_obj2 = opt2[0], opt2[1]

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
    ax.set_xlabel("x (with y fixed to predicted optimum for Obj1)", fontsize=10)
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
    ax.set_xlabel("x (with y fixed to predicted optimum for Obj2)", fontsize=10)
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
    ax.set_xlabel("y (with x fixed to predicted optimum for Obj1)", fontsize=10)
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
    ax.set_xlabel("y (with x fixed to predicted optimum for Obj2)", fontsize=10)
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


def plot_partial_correlation(study: optuna.study.Study, output: str = "partial_correlation.png"):
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials for partial correlation. Skipping.")
        return

    X = np.array([[t.params["x"], t.params["y"]] for t in trials])
    y1 = np.array([t.values[0] for t in trials])
    y2 = np.array([t.values[1] for t in trials])

    from scipy.stats import linregress

    slope_y_on_y1, intercept_y_on_y1, _, _, _ = linregress(X[:, 1], y1)
    residuals_y_obj1 = y1 - (slope_y_on_y1 * X[:, 1] + intercept_y_on_y1)
    slope_y_on_x, intercept_y_on_x, _, _, _ = linregress(X[:, 1], X[:, 0])
    residuals_x_on_x = X[:, 0] - (slope_y_on_x * X[:, 1] + intercept_y_on_x)
    partial_corr_x_obj1 = np.corrcoef(residuals_x_on_x, residuals_y_obj1)[0, 1]

    slope_x_on_y1, intercept_x_on_y1, _, _, _ = linregress(X[:, 0], y1)
    residuals_x_on_y1 = y1 - (slope_x_on_y1 * X[:, 0] + intercept_x_on_y1)
    slope_x_on_x, intercept_x_on_x, _, _, _ = linregress(X[:, 0], X[:, 1])
    residuals_y_on_x = X[:, 1] - (slope_x_on_x * X[:, 0] + intercept_x_on_x)
    partial_corr_y_obj1 = np.corrcoef(residuals_y_on_x, residuals_x_on_y1)[0, 1]

    slope_y_on_y2, intercept_y_on_y2, _, _, _ = linregress(X[:, 1], y2)
    residuals_y_obj2 = y2 - (slope_y_on_y2 * X[:, 1] + intercept_y_on_y2)
    partial_corr_x_obj2 = np.corrcoef(residuals_x_on_x, residuals_y_obj2)[0, 1]

    slope_x_on_y2, intercept_x_on_y2, _, _, _ = linregress(X[:, 0], y2)
    residuals_x_on_y2 = y2 - (slope_x_on_y2 * X[:, 0] + intercept_x_on_y2)
    partial_corr_y_obj2 = np.corrcoef(residuals_y_on_x, residuals_x_on_y2)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    var_names = ["x", "y"]

    ax = axes[0]
    vals1 = [partial_corr_x_obj1, partial_corr_y_obj1]
    colors1 = ["blue" if v > 0 else "red" for v in vals1]
    bars1 = ax.bar(var_names, vals1, color=colors1, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Partial Correlation")
    ax.set_title("Obj1 Partial Correlation")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, vals1):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val > 0 else -0.1), f"{val:.3f}", ha="center", fontsize=9)

    ax = axes[1]
    vals2 = [partial_corr_x_obj2, partial_corr_y_obj2]
    colors2 = ["blue" if v > 0 else "red" for v in vals2]
    bars2 = ax.bar(var_names, vals2, color=colors2, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Partial Correlation")
    ax.set_title("Obj2 Partial Correlation")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val > 0 else -0.1), f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved partial correlation figure to: {output}")


def plot_local_sensitivity(study: optuna.study.Study, output: str = "local_sensitivity.png"):
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials for local sensitivity. Skipping.")
        return

    X = np.array([[t.params["x"], t.params["y"]] for t in trials])
    y1 = np.array([t.values[0] for t in trials])
    y2 = np.array([t.values[1] for t in trials])

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr1.fit(X, y1)
    gpr2.fit(X, y2)
    # Find predicted optima (argmin of GPR mean) for each objective via grid search
    grid_size = 120
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0
    xx = np.linspace(x_min, x_max, grid_size)
    yy = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    mu1_grid = gpr1.predict(grid)
    mu2_grid = gpr2.predict(grid)
    opt1 = grid[np.argmin(mu1_grid)]  # (x*, y*) for objective 1
    opt2 = grid[np.argmin(mu2_grid)]  # (x*, y*) for objective 2

    # Finite-difference gradients at each predicted optimum
    eps = 1e-4
    def finite_diff(gpr, point):
        px = point.copy(); px[0] += eps
        mx = point.copy(); mx[0] -= eps
        py = point.copy(); py[1] += eps
        my = point.copy(); my[1] -= eps
        f_px = gpr.predict(px.reshape(1, -1))[0]
        f_mx = gpr.predict(mx.reshape(1, -1))[0]
        f_py = gpr.predict(py.reshape(1, -1))[0]
        f_my = gpr.predict(my.reshape(1, -1))[0]
        dfdx = (f_px - f_mx) / (2 * eps)
        dfdy = (f_py - f_my) / (2 * eps)
        return dfdx, dfdy

    grad_x_obj1, grad_y_obj1 = finite_diff(gpr1, opt1)
    grad_x_obj2, grad_y_obj2 = finite_diff(gpr2, opt2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    var_names = ["x", "y"]

    ax = axes[0]
    vals1 = [abs(grad_x_obj1), abs(grad_y_obj1)]
    colors = ["green", "orange"]
    bars1 = ax.bar(var_names, vals1, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Local Sensitivity (|∂μ/∂var|)")
    ax.set_title(f"Obj1 Local Sensitivity at predicted optimum (x={opt1[0]:.2f}, y={opt1[1]:.2f})")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, vals1):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01*max(vals1), f"{val:.3f}", ha="center", fontsize=9)
    ax.annotate(f"Opt1=(x={opt1[0]:.2f}, y={opt1[1]:.2f})", xy=(0.5, 0.9), xycoords="axes fraction", ha="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    ax = axes[1]
    vals2 = [abs(grad_x_obj2), abs(grad_y_obj2)]
    bars2 = ax.bar(var_names, vals2, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Local Sensitivity (|∂μ/∂var|)")
    ax.set_title(f"Obj2 Local Sensitivity at predicted optimum (x={opt2[0]:.2f}, y={opt2[1]:.2f})")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01*max(vals2), f"{val:.3f}", ha="center", fontsize=9)
    ax.annotate(f"Opt2=(x={opt2[0]:.2f}, y={opt2[1]:.2f})", xy=(0.5, 0.9), xycoords="axes fraction", ha="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved local sensitivity figure to: {output}")


def plot_partial_dependence(study: optuna.study.Study, output: str = "partial_dependence.png"):
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials for partial dependence. Skipping.")
        return

    X = np.array([[t.params["x"], t.params["y"]] for t in trials])
    y1 = np.array([t.values[0] for t in trials])
    y2 = np.array([t.values[1] for t in trials])

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr1.fit(X, y1)
    gpr2.fit(X, y2)

    # Calculate mean values for annotation
    mean_x = X[:, 0].mean()
    mean_y = X[:, 1].mean()

    grid_x = np.linspace(-5, 5, 100)
    grid_y = np.linspace(-5, 5, 100)

    pdp_x_obj1 = []
    pdp_x_obj2 = []
    for x_val in grid_x:
        X_slice = np.column_stack([np.full(len(X), x_val), X[:, 1]])
        mu1_vals = gpr1.predict(X_slice)
        mu2_vals = gpr2.predict(X_slice)
        pdp_x_obj1.append(mu1_vals.mean())
        pdp_x_obj2.append(mu2_vals.mean())

    pdp_y_obj1 = []
    pdp_y_obj2 = []
    for y_val in grid_y:
        X_slice = np.column_stack([X[:, 0], np.full(len(X), y_val)])
        mu1_vals = gpr1.predict(X_slice)
        mu2_vals = gpr2.predict(X_slice)
        pdp_y_obj1.append(mu1_vals.mean())
        pdp_y_obj2.append(mu2_vals.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(grid_x, pdp_x_obj1, "b-", linewidth=2, label=f"x effect (y averaged over samples)")
    ax.plot(grid_y, pdp_y_obj1, "orange", linewidth=2, label=f"y effect (x averaged over samples)")
    ax.scatter(X[:, 0], y1, alpha=0.3, s=20, color="gray", label="Samples")
    ax.set_xlabel("Input Value")
    ax.set_ylabel("Obj1 (PDP)")
    ax.set_title(f"Obj1 Partial Dependence Plot\n(x: y fixed at mean={mean_y:.2f}, y: x fixed at mean={mean_x:.2f})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(grid_x, pdp_x_obj2, "b-", linewidth=2, label=f"x effect (y averaged over samples)")
    ax.plot(grid_y, pdp_y_obj2, "orange", linewidth=2, label=f"y effect (x averaged over samples)")
    ax.scatter(X[:, 0], y2, alpha=0.3, s=20, color="gray", label="Samples")
    ax.set_xlabel("Input Value")
    ax.set_ylabel("Obj2 (PDP)")
    ax.set_title(f"Obj2 Partial Dependence Plot\n(x: y fixed at mean={mean_y:.2f}, y: x fixed at mean={mean_x:.2f})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved partial dependence figure to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Optuna multi-objective example")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--output", type=str, default="pareto.png", help="Output image file for Pareto front")
    parser.add_argument("--slices-output", type=str, default="slices.png", help="Output image file for GPR slice plots with confidence bands")
    parser.add_argument("--partial-dependence", type=str, default="partial_dependence.png", help="Output image file for partial dependence plot (PDP)")
    parser.add_argument("--surfaces-output", type=str, default="gpr_surfaces.png", help="Output image file for 2D GPR surfaces (mean and variance)")
    parser.add_argument("--partial-correlation", type=str, default="partial_correlation.png", help="Output image file for partial correlation plot")
    parser.add_argument("--local-sensitivity", type=str, default="local_sensitivity.png", help="Output image file for local sensitivity plot")
    args = parser.parse_args()

    # Ensure results are saved under the 'results' directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    def to_results_path(filename: str) -> str:
        base = os.path.basename(filename)
        return os.path.join(results_dir, base)

    args.output = to_results_path(args.output)
    args.slices_output = to_results_path(args.slices_output)
    args.partial_dependence = to_results_path(args.partial_dependence)
    args.surfaces_output = to_results_path(args.surfaces_output)
    args.partial_correlation = to_results_path(args.partial_correlation)
    args.local_sensitivity = to_results_path(args.local_sensitivity)

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
    # Plot sensitivity analysis (split into three figures)
    try:
        plot_partial_correlation(study, output=args.partial_correlation)
        plot_local_sensitivity(study, output=args.local_sensitivity)
        plot_partial_dependence(study, output=args.partial_dependence)
    except Exception as e:
        print("Sensitivity analysis plots failed:", e)
    # Plot 2D GPR surfaces (mean and variance)
    try:
        plot_gpr_2d_surfaces(study, output=args.surfaces_output)
    except Exception as e:
        print("GPR 2D surface plot failed:", e)


if __name__ == "__main__":
    main()
