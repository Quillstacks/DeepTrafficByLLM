import matplotlib.pyplot as plt
import json
import os
import numpy as np

# TODO: Check if plotting is needed for each result and add according identifier in plot titles.

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)                              # prompt_em/
#REPO = os.path.dirname(SUITE)                              # DeepTrafficByLLM/

NAME = "em_v3_large"

def get_dirs(name: str) -> tuple[str, str]:
    in_dir = os.path.join(SUITE, "results", name)
    out_dir = os.path.join(in_dir, "figures")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return in_dir, out_dir

def load_iterations(in_dir: str):
    with open(os.path.join(in_dir, "iterations.jsonl")) as jsons:
        return [json.loads(line) for line in jsons]

IN_DIR, OUT_DIR = get_dirs(NAME)
summary = json.load(open(os.path.join(IN_DIR, "summary.json")))
iterations = load_iterations(IN_DIR)

def plot_trajectories():
    phases = [run["optimizer"]["phase"] for run in iterations]
    print(phases)
    cold_start_end = None

    for i, phase in enumerate(phases):
        if phase == "exploit":
            cold_start_end = i
            break

    mean_trajectory = summary["mean_trajectory"]
    median_trajectory = summary["median_trajectory"]

    n = len(mean_trajectory)
    x_ticks = range(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(0, cold_start_end - 0.001, color="gray", alpha=0.1, label="Cold Start Phase")
    ax.axvspan(cold_start_end-0.001, len(iterations), color="green", alpha=0.1, label="Exploit Phase")
    ax.plot(x_ticks, mean_trajectory, label="Mean Trajectory", color="tab:blue")
    ax.plot(x_ticks, median_trajectory, label="Median Trajectory", color="tab:orange")

    ax.set_xticks(x_ticks)
    ax.set_xlim(x_ticks[0], x_ticks[-1])

    ax.legend(fontsize=10, loc="lower right")

    ax.set_title("Mean and Median Trajectories")

    plt.tight_layout()
    #fig.savefig(os.path.join(OUT_DIR, "action_fractions_stackplot.pdf"), bbox_inches='tight')
    plt.show()

#plot_trajectories()

def plot_action_fractions():
    action_fraction_trajectory = summary["action_fraction_trajectory"]
    actions = ["maintain", "accelerate", "decelerate", "left", "right"]

    action_fractions = {a: [run[a] for run in action_fraction_trajectory] for a in actions}

    print(action_fractions)
    
    n = len(action_fraction_trajectory)
    x_ticks = range(1, n + 1)
    bottom = np.zeros(n)

    fig, ax = plt.subplots(figsize=(10,6))
    for a in actions:
        values = action_fractions[a]
        ax.bar(x_ticks, values, bottom=bottom, label=a)
        bottom += np.array(values)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of Action in Iteration")
    ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.legend(fontsize=10, title="Actions", loc="upper right")

    ax.set_title("Fraction of Actions Taken across Iterations")
    
    plt.tight_layout()
    #fig.savefig(os.path.join(OUT_DIR, "action_fractions_stackplot.pdf"), bbox_inches='tight')
    plt.show()

#plot_action_fractions()

def plot_action_fractions_stackplot():
    action_fraction_trajectory = summary["action_fraction_trajectory"]
    actions = ["maintain", "accelerate", "decelerate", "left", "right"]

    action_fractions = {a: [run[a] for run in action_fraction_trajectory] for a in actions}

    print(action_fractions)
    
    n = len(action_fraction_trajectory)
    x_ticks = range(1, n + 1)
    bottom = np.zeros(n)

    fig, ax = plt.subplots(figsize=(10,6))
    for a in actions:
        values = action_fractions[a]
        ax.stackplot(x_ticks, *[action_fractions[a] for a in actions], labels=actions)
        bottom += np.array(values)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of Action in Iteration")
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_ylim(0, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Actions", loc="upper right")

    ax.set_title("Fraction of Actions Taken across Iterations")
    
    plt.tight_layout()
    #fig.savefig(os.path.join(OUT_DIR, "action_fractions_stackplot.pdf"), bbox_inches='tight')
    plt.show()

#plot_action_fractions_stackplot()

weight_contributions = summary["final_weight_contributions"]

def plot_contributions(kind):
    if kind == "weight":
        contributions = summary["final_weight_contributions"]

    elif kind == "order":
        contributions = summary["final_order_contributions"]

    ids = list(weight_contributions.keys())
    values = np.array([contributions[i] for i in ids])

    order = np.argsort(values)
    ids_sorted = [ids[i] for i in order]
    values_sorted = values[order]

    colors = ["tab:red" if value < 0 else "tab:green" for value in values_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ids_sorted, values_sorted, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution Value")
    ax.set_ylabel("Heuristic")
    ax.set_title(f"Final {kind.capitalize()} Contributions")

    plt.tight_layout()
    #fig.savefig(os.path.join(OUT_DIR, f"{kind}_contributions.pdf"), bbox_inches='tight')
    plt.show()


#plot_contributions("weight")
plot_contributions("order")   

def plot_weight_contributions():
    pass

def plot_order_contributions():
    pass

