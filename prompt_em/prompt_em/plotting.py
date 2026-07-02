import matplotlib.pyplot as plt
import json
import os
import numpy as np

# TODO: Handle edge cases (diag_template, em_v2_template)
# TODO: Change color palette.
# TODO: Add wording_sequence result once appropriate branches are merged.

# ------------------------------- PLOT PARAMS -------------------------------
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ACTION_COLORS = {
    "maintain":   "#56B4E9",   # Sky Blue
    "accelerate": "#009E73",   # Bluish Green
    "decelerate": "#D55E00",   # Vermillion
    "left":       "#0072B2",   # Blue
    "right":      "#CC79A7",   # Reddish Purple
}

COLOR_POSITIVE = "tab:blue"
COLOR_NEGATIVE = "tab:orange"
# ------------------------------ PLOT PARAMS END ------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)                              # prompt_em/
#REPO = os.path.dirname(SUITE)                              # DeepTrafficByLLM/
RESULT_NAMES = ["em_v1", "diag_template", "em_v2_template", "em_v3_large"]

def get_dirs(name: str) -> tuple[str, str]:
    in_dir = os.path.join(SUITE, "results", name)
    out_dir = os.path.join(in_dir, "figures")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return in_dir, out_dir

def load_summary(in_dir: str, name: str):
    path = os.path.join(in_dir, "summary.json")
    if not os.path.exists(path):
        print(f"Summary file is missing in {in_dir}. Skipping...")
        return None
    return json.load(open(path))

def load_iterations(in_dir: str):
    with open(os.path.join(in_dir, "iterations.jsonl")) as jsons:
        return [json.loads(line) for line in jsons]

def plot_trajectories(summary, iterations, name, out_dir):
    """
    Plots mean and median trajectories of fleet speed across iterations. Cold Start and Exploit phases are highlighted.
    """
    phases = [run["optimizer"]["phase"] for run in iterations]
    exploit_begin = None

    # Find index for start of exploit phase.
    for i, phase in enumerate(phases):
        if phase == "exploit":
            exploit_begin = i + 1
            break

    if exploit_begin is None:
        exploit_begin = len(phases)

    mean_trajectory = summary["mean_trajectory"]
    median_trajectory = summary["median_trajectory"]

    n = len(mean_trajectory)
    x_ticks = range(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(0, exploit_begin - 0.05, color="gray", alpha=0.1, label="Cold Start Phase")
    ax.axvspan(exploit_begin - 0.05, len(iterations), color="green", alpha=0.1, label="Exploit Phase")
    ax.plot(x_ticks, mean_trajectory, label="Mean Trajectory", color="tab:blue")
    ax.plot(x_ticks, median_trajectory, label="Median Trajectory", color="tab:orange")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fleet Speed [mph]")
    ax.set_xticks(x_ticks)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_yticks(np.arange(np.floor(min(min(mean_trajectory), min(median_trajectory))), np.ceil(max(max(mean_trajectory), max(median_trajectory))) + 1, 1))

    ax.legend()

    ax.set_title(f"Mean and Median Trajectories - {name}")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "mean_median_trajectories.pdf"))

def plot_action_fractions(summary, out_dir, name):
    """
    Plots the fraction of each action taken across iterations as a stacked bar chart.
    """
    action_fraction_trajectory = summary["action_fraction_trajectory"]
    actions = ["maintain", "accelerate", "decelerate", "left", "right"]

    # Map each action to a list of the fractions it was used in each iteration.
    action_fractions = {a: [run[a] for run in action_fraction_trajectory] for a in actions}
    
    n = len(action_fraction_trajectory)
    x_ticks = range(1, n + 1)
    bottom = np.zeros(n)

    fig, ax = plt.subplots(figsize=(10,6))
    for a in actions:
        values = action_fractions[a]
        ax.bar(x_ticks, values, bottom=bottom, label=a, color=ACTION_COLORS[a])
        bottom += np.array(values)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of Action in Iteration")
    ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.legend(fontsize=10, title="Actions", loc="upper right")

    ax.grid(False, axis="x")


    ax.set_title(f"Fraction of Actions Taken across Iterations - {name}")
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "action_fractions.pdf"))

def plot_action_fractions_stackplot(summary, out_dir, name):
    """
    Plots the fraction of each action taken across iterations as a stackplot (filled area plot).
    """
    action_fraction_trajectory = summary["action_fraction_trajectory"]
    actions = ["maintain", "accelerate", "decelerate", "left", "right"]

    # Map each action to a list of the fractions it was used in each iteration.
    action_fractions = {a: [run[a] for run in action_fraction_trajectory] for a in actions}
    
    n = len(action_fraction_trajectory)
    x_ticks = range(1, n + 1)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.stackplot(x_ticks, *[action_fractions[a] for a in actions], labels=actions, colors=[ACTION_COLORS[a] for a in actions])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of Action in Iteration")
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_ylim(0, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_title(f"Fraction of Actions Taken across Iterations - {name}")

    ax.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "action_fractions_stackplot.pdf"))

def plot_contributions(summary, out_dir, name, kind):
    """
    Plots either the weight or order contributions of heuristics as a bar chart. The kind parameter should be either "weight" or "order".
    """
    if kind == "weight":
        contributions = summary["final_weight_contributions"]

    elif kind == "order":
        contributions = summary["final_order_contributions"]

    else:
        raise ValueError(f"Invalid kind: {kind}. Must be either 'weight' or 'order'.")

    if contributions is None:
        print(f"No {kind} contributions found in summary for {name}. Skipping...")
        return

    ids = list(contributions.keys())
    values = np.array([contributions[i] for i in ids])

    order = np.argsort(values)
    ids_sorted = [ids[i] for i in order]
    values_sorted = values[order]

    colors = [COLOR_NEGATIVE if value < 0 else COLOR_POSITIVE for value in values_sorted]

    max_abs = max(abs(values_sorted.min()), abs(values_sorted.max()))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ids_sorted, values_sorted, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Contribution Value")
    ax.set_xlim((-max_abs)*1.05, max_abs*1.05)
    ax.set_ylabel("Heuristic")

    ax.grid(False, axis="y")


    ax.set_title(f"Final {kind.capitalize()} Contributions - {name}")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{kind}_contributions.pdf"))

def main():
    for name in RESULT_NAMES:
        in_dir, out_dir = get_dirs(name)
        summary = load_summary(in_dir, name)

        if summary is None:
            continue

        iterations = load_iterations(in_dir)

        plot_trajectories(summary, iterations, name, out_dir)
        plot_action_fractions(summary, out_dir, name)
        plot_action_fractions_stackplot(summary, out_dir, name)
        plot_contributions(summary, out_dir, name, "weight")
        plot_contributions(summary, out_dir, name, "order")

if __name__ == "__main__":
    main()

