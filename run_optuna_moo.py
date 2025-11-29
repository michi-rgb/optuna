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


def main():
    parser = argparse.ArgumentParser(description="Optuna multi-objective example")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--output", type=str, default="pareto.png", help="Output image file for Pareto front")
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


if __name__ == "__main__":
    main()
