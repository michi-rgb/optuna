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
    f1 = (x - 2.0) ** 2 + (y - 1.0) ** 2
    f2 = (x + 2.0) ** 2 + (y + 1.0) ** 2
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


def plot_gpr_uncertainty(study: optuna.study.Study, output: str = "sensitivity.png", grid_size: int = 80):
    """Fit Gaussian Process for each objective and plot predictive std over input space.

    Produces a side-by-side figure with predictive standard deviation (uncertainty)
    for objective1 and objective2 on a 2D grid of (x, y).
    """
    trials = [t for t in study.trials if t.values is not None]
    if len(trials) < 5:
        print("Not enough trials to fit GPRs (need at least 5). Skipping GPR uncertainty plot.")
        return

    X = np.array([[t.params["x"], t.params["y"]] for t in trials])
    y1 = np.array([t.values[0] for t in trials])
    y2 = np.array([t.values[1] for t in trials])

    # grid bounds with small margin
    margin_x = 0.1 * (X[:, 0].max() - X[:, 0].min() if np.ptp(X[:, 0]) > 0 else 1.0)
    margin_y = 0.1 * (X[:, 1].max() - X[:, 1].min() if np.ptp(X[:, 1]) > 0 else 1.0)
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y

    xx = np.linspace(x_min, x_max, grid_size)
    yy = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xx, yy)
    Xgrid = np.column_stack([XX.ravel(), YY.ravel()])

    # kernel: constant * RBF + white noise
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

    gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr2 = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    gpr1.fit(X, y1)
    gpr2.fit(X, y2)

    mu1, std1 = gpr1.predict(Xgrid, return_std=True)
    mu2, std2 = gpr2.predict(Xgrid, return_std=True)

    S1 = std1.reshape(XX.shape)
    S2 = std2.reshape(XX.shape)

    # Pareto indices to highlight
    objs = [t.values for t in trials]
    pareto_idx = non_dominated_indices(objs)
    pareto_xy = np.array([[trials[i].params["x"], trials[i].params["y"]] for i in pareto_idx]) if pareto_idx else np.empty((0, 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].contourf(XX, YY, S1, levels=40, cmap="viridis")
    axes[0].scatter(X[:, 0], X[:, 1], c="white", s=20, edgecolor="black", label="Samples")
    if pareto_xy.size:
        axes[0].scatter(pareto_xy[:, 0], pareto_xy[:, 1], c="red", s=80, marker="*", label="Pareto")
    axes[0].set_title("Predictive std (Objective 1)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], label="std")

    im1 = axes[1].contourf(XX, YY, S2, levels=40, cmap="viridis")
    axes[1].scatter(X[:, 0], X[:, 1], c="white", s=20, edgecolor="black", label="Samples")
    if pareto_xy.size:
        axes[1].scatter(pareto_xy[:, 0], pareto_xy[:, 1], c="red", s=80, marker="*", label="Pareto")
    axes[1].set_title("Predictive std (Objective 2)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], label="std")

    for ax in axes:
        ax.legend(loc="upper right")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.suptitle("GPR Predictive Uncertainty (std) for Each Objective", fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved GPR predictive std figure to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Optuna multi-objective example")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--output", type=str, default="pareto.png", help="Output image file for Pareto front")
    parser.add_argument("--sensitivity-output", type=str, default="sensitivity.png", help="Output image file for GPR predictive std (sensitivity)")
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
    try:
        plot_gpr_uncertainty(study, output=args.sensitivity_output)
    except Exception as e:
        print("GPR sensitivity plot failed:", e)


if __name__ == "__main__":
    main()
