"""Optuna 多目的最適化のサンプルスクリプト (4入力、3出力)

使い方例:
  python run_optuna_moo_4d.py --trials 300 --output pareto_4d.png

このスクリプトは 3 目的を同時に最小化し、パレート最適解集合をプロットしてファイルに保存します。
入力変数と出力変数の相関関係も可視化します。
"""
import argparse
import os
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D


def objective(trial: optuna.trial.Trial):
    x1 = trial.suggest_float("x1", -5.0, 5.0)
    x2 = trial.suggest_float("x2", -5.0, 5.0)
    x3 = trial.suggest_float("x3", -5.0, 5.0)
    x4 = trial.suggest_float("x4", -5.0, 5.0)
    
    # 例：3つの目的関数（すべて最小化）
    f1 = (x1 - 2.0) ** 2 + (x2 - 1.0) ** 2 + x3 ** 2 + np.random.normal(0, 0.5) - x1 - x2
    f2 = (x1 + 2.0) ** 2 + (x2 + 1.0) ** 2 + (x4 - 1.0) ** 2 + np.random.normal(0, 0.5) + x1 + x2
    f3 = x1 ** 2 + x2 ** 2 + (x3 + 2.0) ** 2 + (x4 + 1.0) ** 2 + np.random.normal(0, 0.5)
    
    return f1, f2, f3


def non_dominated_indices(values_list):
    # values_list: list of tuples/lists (obj1, obj2, obj3, ...)
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


def plot_pareto(study: optuna.study.Study, output: str = "pareto_4d.png"):
    trials = [t for t in study.trials if t.values is not None]
    if not trials:
        print("No completed trials with objective values.")
        return

    objs = [t.values for t in trials]
    obj1 = [v[0] for v in objs]
    obj2 = [v[1] for v in objs]
    obj3 = [v[2] for v in objs]

    pareto_idx = non_dominated_indices(objs)
    pareto_points = [objs[i] for i in pareto_idx]
    p1 = [p[0] for p in pareto_points]
    p2 = [p[1] for p in pareto_points]
    p3 = [p[2] for p in pareto_points]

    # Create figure with 3D plot and 2D projections
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Multi-objective Optimization Analysis (4 inputs, 3 objectives)", fontsize=14, fontweight="bold")

    # 3D Pareto front
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.scatter(obj1, obj2, obj3, c="C0", alpha=0.3, s=30, label="All trials")
    ax.scatter(p1, p2, p3, c="C1", s=100, marker="*", label="Pareto front", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Objective 1", fontsize=9)
    ax.set_ylabel("Objective 2", fontsize=9)
    ax.set_zlabel("Objective 3", fontsize=9)
    ax.set_title("3D Pareto Front", fontweight="bold")
    ax.legend(loc="best", fontsize=8)

    # 2D projections
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(obj1, obj2, c="C0", alpha=0.4, s=30, label="All trials")
    ax.scatter(p1, p2, c="C1", s=80, marker="*", label="Pareto front", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Objective 1", fontsize=9)
    ax.set_ylabel("Objective 2", fontsize=9)
    ax.set_title("Obj1 vs Obj2", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(obj1, obj3, c="C0", alpha=0.4, s=30, label="All trials")
    ax.scatter(p1, p3, c="C1", s=80, marker="*", label="Pareto front", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Objective 1", fontsize=9)
    ax.set_ylabel("Objective 3", fontsize=9)
    ax.set_title("Obj1 vs Obj3", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(obj2, obj3, c="C0", alpha=0.4, s=30, label="All trials")
    ax.scatter(p2, p3, c="C1", s=80, marker="*", label="Pareto front", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Objective 2", fontsize=9)
    ax.set_ylabel("Objective 3", fontsize=9)
    ax.set_title("Obj2 vs Obj3", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary Statistics
    ax = fig.add_subplot(2, 3, 5)
    ax.axis("off")
    stats_text = f"""
Total Trials: {len(trials)}
Pareto-optimal: {len(pareto_idx)}

Objective 1:
  Min: {min(obj1):.4f}
  Max: {max(obj1):.4f}
  Mean: {sum(obj1)/len(obj1):.4f}
  Pareto Range: [{min(p1):.4f}, {max(p1):.4f}]

Objective 2:
  Min: {min(obj2):.4f}
  Max: {max(obj2):.4f}
  Mean: {sum(obj2)/len(obj2):.4f}
  Pareto Range: [{min(p2):.4f}, {max(p2):.4f}]

Objective 3:
  Min: {min(obj3):.4f}
  Max: {max(obj3):.4f}
  Mean: {sum(obj3)/len(obj3):.4f}
  Pareto Range: [{min(p3):.4f}, {max(p3):.4f}]
    """
    ax.text(0.05, 0.5, stats_text, fontsize=9, verticalalignment="center",
            family="monospace", bbox=dict(boxstyle="round,pad=1.0", facecolor="wheat", alpha=0.5))

    # Input distribution for Pareto solutions
    ax = fig.add_subplot(2, 3, 6)
    if pareto_idx:
        params_dict = {f"x{i+1}": [trials[idx].params[f"x{i+1}"] for idx in pareto_idx] for i in range(4)}
        positions = range(1, 5)
        bp = ax.boxplot([params_dict[f"x{i}"] for i in range(1, 5)], positions=positions, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xticklabels([f"x{i}" for i in range(1, 5)])
        ax.set_ylabel("Parameter Value", fontsize=9)
        ax.set_title("Pareto Input Distribution", fontweight="bold")
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "No Pareto solutions", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved comprehensive analysis figure to: {output}")


def plot_input_correlation(study: optuna.study.Study, output: str = "input_correlation_4d.png"):
    """Plot correlation between input parameters and objectives"""
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials for correlation plots. Skipping.")
        return

    params = {f"x{i}": [t.params[f"x{i}"] for t in trials] for i in range(1, 5)}
    objs = [[t.values[i] for t in trials] for i in range(3)]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Input-Output Correlation Matrix (4 inputs, 3 objectives)", fontsize=14, fontweight="bold")

    pareto_idx = non_dominated_indices([t.values for t in trials])
    
    colors = ['C0', 'C2', 'C3']
    obj_names = ['Obj1', 'Obj2', 'Obj3']
    
    for obj_idx in range(3):
        for param_idx in range(4):
            ax = axes[obj_idx, param_idx]
            param_name = f"x{param_idx+1}"
            
            # All trials
            ax.scatter(params[param_name], objs[obj_idx], c=colors[obj_idx], alpha=0.4, s=25, label="All trials")
            
            # Pareto trials
            if pareto_idx:
                pareto_x = [params[param_name][i] for i in pareto_idx]
                pareto_y = [objs[obj_idx][i] for i in pareto_idx]
                ax.scatter(pareto_x, pareto_y, c='red', s=80, marker="*", label="Pareto", edgecolors="black", linewidth=0.5)
            
            # Correlation coefficient
            corr = np.corrcoef(params[param_name], objs[obj_idx])[0, 1]
            ax.text(0.05, 0.95, f"r={corr:.3f}", transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), 
                   verticalalignment='top', fontsize=8)
            
            ax.set_xlabel(param_name, fontsize=9)
            ax.set_ylabel(obj_names[obj_idx], fontsize=9)
            ax.grid(True, alpha=0.3)
            if param_idx == 0 and obj_idx == 0:
                ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved input correlation plots to: {output}")


def plot_gpr_slices(study: optuna.study.Study, output: str = "slices_4d.png"):
    """Plot GPR predictions along each input dimension for each objective"""
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 10:
        print("Not enough trials for GPR slice plots. Skipping.")
        return

    X = np.array([[t.params[f"x{i}"] for i in range(1, 5)] for t in trials])
    Y = [np.array([t.values[i] for t in trials]) for i in range(3)]

    # Fit GPRs for each objective
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gprs = [GaussianProcessRegressor(kernel=kernel, normalize_y=True) for _ in range(3)]
    for i in range(3):
        gprs[i].fit(X, Y[i])

    # Find optimal points for each objective
    grid_size = 50
    ranges = [np.linspace(-5.0, 5.0, grid_size) for _ in range(4)]
    grid_points = np.array(np.meshgrid(*ranges)).reshape(4, -1).T
    
    optima = []
    for gpr in gprs:
        mu_grid = gpr.predict(grid_points)
        opt_point = grid_points[np.argmin(mu_grid)]
        optima.append(opt_point)

    # Create slice plots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("GPR Predictions with ±2σ Uncertainty (fixing other variables at predicted optima)", 
                 fontsize=14, fontweight="bold")

    colors_obj = ['blue', 'green', 'purple']
    obj_names = ['Obj1', 'Obj2', 'Obj3']

    for obj_idx in range(3):
        opt_point = optima[obj_idx]
        
        for param_idx in range(4):
            ax = axes[obj_idx, param_idx]
            
            # Create slice along this parameter
            grid_1d = np.linspace(-5.0, 5.0, 200)
            X_slice = np.tile(opt_point, (200, 1))
            X_slice[:, param_idx] = grid_1d
            
            mu, std = gprs[obj_idx].predict(X_slice, return_std=True)
            
            # Plot
            ax.plot(grid_1d, mu, color=colors_obj[obj_idx], linewidth=2, label="Mean")
            ax.fill_between(grid_1d, mu - 2*std, mu + 2*std, alpha=0.3, color=colors_obj[obj_idx], label="±2σ")
            
            # Sample points
            param_vals = X[:, param_idx]
            ax.scatter(param_vals, Y[obj_idx], alpha=0.3, s=15, color="gray")
            
            ax.set_xlabel(f"x{param_idx+1}", fontsize=9)
            if param_idx == 0:
                ax.set_ylabel(obj_names[obj_idx], fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Show fixed values
            other_params = [f"x{i}={opt_point[i-1]:.2f}" for i in range(1, 5) if i != param_idx+1]
            ax.set_title(f"{obj_names[obj_idx]}: {', '.join(other_params)}", fontsize=8)
            
            if param_idx == 0 and obj_idx == 0:
                ax.legend(loc="best", fontsize=7)

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved GPR slice plots to: {output}")


def plot_partial_correlation(study: optuna.study.Study, output: str = "partial_correlation_4d.png"):
    """Plot partial correlation for each input-output pair"""
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 10:
        print("Not enough trials for partial correlation. Skipping.")
        return

    X = np.array([[t.params[f"x{i}"] for i in range(1, 5)] for t in trials])
    Y = [np.array([t.values[i] for t in trials]) for i in range(3)]

    from scipy.stats import linregress

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Partial Correlation Analysis", fontsize=14, fontweight="bold")
    
    param_names = [f"x{i}" for i in range(1, 5)]
    obj_names = ['Obj1', 'Obj2', 'Obj3']
    colors = ['blue', 'green', 'purple']

    for obj_idx in range(3):
        partial_corrs = []
        
        for target_idx in range(4):
            # Compute residuals
            y_target = Y[obj_idx].copy()
            x_target = X[:, target_idx].copy()
            
            # Regress out other variables
            other_indices = [i for i in range(4) if i != target_idx]
            if len(other_indices) > 0:
                X_other = X[:, other_indices]
                
                # Multivariate regression for y on other x's
                from sklearn.linear_model import LinearRegression
                reg_y = LinearRegression()
                reg_y.fit(X_other, y_target)
                y_residual = y_target - reg_y.predict(X_other)
                
                # Multivariate regression for target x on other x's
                reg_x = LinearRegression()
                reg_x.fit(X_other, x_target)
                x_residual = x_target - reg_x.predict(X_other)
                
                partial_corr = np.corrcoef(x_residual, y_residual)[0, 1]
            else:
                partial_corr = np.corrcoef(x_target, y_target)[0, 1]
            
            partial_corrs.append(partial_corr)
        
        ax = axes[obj_idx]
        bar_colors = [colors[obj_idx] if v > 0 else 'red' for v in partial_corrs]
        bars = ax.bar(param_names, partial_corrs, color=bar_colors, alpha=0.7, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Partial Correlation")
        ax.set_title(f"{obj_names[obj_idx]} Partial Correlation")
        ax.grid(True, alpha=0.3, axis="y")
        
        for bar, val in zip(bars, partial_corrs):
            ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val > 0 else -0.1), 
                   f"{val:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved partial correlation plot to: {output}")


def plot_local_sensitivity(study: optuna.study.Study, output: str = "local_sensitivity_4d.png"):
    """Plot local sensitivity at predicted optima"""
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 10:
        print("Not enough trials for local sensitivity. Skipping.")
        return

    X = np.array([[t.params[f"x{i}"] for i in range(1, 5)] for t in trials])
    Y = [np.array([t.values[i] for t in trials]) for i in range(3)]

    # Fit GPRs
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    gprs = [GaussianProcessRegressor(kernel=kernel, normalize_y=True) for _ in range(3)]
    for i in range(3):
        gprs[i].fit(X, Y[i])

    # Find optima
    grid_size = 50
    ranges = [np.linspace(-5.0, 5.0, grid_size) for _ in range(4)]
    grid_points = np.array(np.meshgrid(*ranges)).reshape(4, -1).T
    
    optima = []
    for gpr in gprs:
        mu_grid = gpr.predict(grid_points)
        opt_point = grid_points[np.argmin(mu_grid)]
        optima.append(opt_point)

    # Compute finite difference gradients
    eps = 0.5
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Local Sensitivity at Predicted Optima", fontsize=14, fontweight="bold")
    
    param_names = [f"x{i}" for i in range(1, 5)]
    obj_names = ['Obj1', 'Obj2', 'Obj3']
    colors_list = [['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                   ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
                   ['#bcbd22', '#17becf', '#aec7e8', '#ffbb78']]

    for obj_idx in range(3):
        opt_point = optima[obj_idx]
        sensitivities = []
        
        for param_idx in range(4):
            point_plus = opt_point.copy()
            point_minus = opt_point.copy()
            point_plus[param_idx] += eps
            point_minus[param_idx] -= eps
            
            f_plus = gprs[obj_idx].predict(point_plus.reshape(1, -1))[0]
            f_minus = gprs[obj_idx].predict(point_minus.reshape(1, -1))[0]
            
            sensitivity = abs((f_plus - f_minus) / (2 * eps))
            sensitivities.append(sensitivity)
        
        ax = axes[obj_idx]
        bars = ax.bar(param_names, sensitivities, color=colors_list[obj_idx], alpha=0.7, edgecolor="black")
        ax.set_ylabel("Local Sensitivity (|∂μ/∂x|)")
        opt_str = ", ".join([f"x{i+1}={opt_point[i]:.2f}" for i in range(4)])
        ax.set_title(f"{obj_names[obj_idx]}\n({opt_str})", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        
        for bar, val in zip(bars, sensitivities):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01*max(sensitivities), 
                   f"{val:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved local sensitivity plot to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Optuna multi-objective example (4 inputs, 3 objectives)")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--output", type=str, default="pareto_4d.png", help="Output image file for Pareto front")
    parser.add_argument("--input-correlation", type=str, default="input_correlation_4d.png", help="Output for input-output correlation")
    parser.add_argument("--slices-output", type=str, default="slices_4d.png", help="Output image file for GPR slice plots")
    parser.add_argument("--partial-correlation", type=str, default="partial_correlation_4d.png", help="Output for partial correlation")
    parser.add_argument("--local-sensitivity", type=str, default="local_sensitivity_4d.png", help="Output for local sensitivity")
    args = parser.parse_args()

    # Ensure results are saved under the 'results_4d' directory
    results_dir = os.path.join(os.path.dirname(__file__), "results_4d")
    os.makedirs(results_dir, exist_ok=True)
    
    def to_results_path(filename: str) -> str:
        base = os.path.basename(filename)
        return os.path.join(results_dir, base)

    args.output = to_results_path(args.output)
    args.input_correlation = to_results_path(args.input_correlation)
    args.slices_output = to_results_path(args.slices_output)
    args.partial_correlation = to_results_path(args.partial_correlation)
    args.local_sensitivity = to_results_path(args.local_sensitivity)

    # Create study with 3 objectives (all minimize)
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    study.optimize(objective, n_trials=args.trials)

    # 結果のサマリ
    print("Total trials:", len(study.trials))
    
    # パレート最適なトライアルを手動で抽出して表示
    trials = [t for t in study.trials if t.values is not None]
    objs = [t.values for t in trials]
    pareto_idx = non_dominated_indices(objs)

    print(f"Pareto-optimal trials: {len(pareto_idx)}")
    for idx in pareto_idx[:10]:  # Show first 10
        t = trials[idx]
        print(f"  Trial#{t.number}: values={[f'{v:.4f}' for v in t.values]}, params={t.params}")
    
    if len(pareto_idx) > 10:
        print(f"  ... and {len(pareto_idx) - 10} more Pareto-optimal trials")

    # Generate all plots
    plot_pareto(study, output=args.output)
    
    try:
        plot_input_correlation(study, output=args.input_correlation)
    except Exception as e:
        print("Input correlation plot failed:", e)
    
    try:
        plot_gpr_slices(study, output=args.slices_output)
    except Exception as e:
        print("GPR slice plot failed:", e)
    
    try:
        plot_partial_correlation(study, output=args.partial_correlation)
    except Exception as e:
        print("Partial correlation plot failed:", e)
    
    try:
        plot_local_sensitivity(study, output=args.local_sensitivity)
    except Exception as e:
        print("Local sensitivity plot failed:", e)


if __name__ == "__main__":
    main()
