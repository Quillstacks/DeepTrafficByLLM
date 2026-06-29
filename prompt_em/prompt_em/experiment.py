"""Orchestrate the prompt-EM loop: synthesize -> evaluate -> reweight, for N
iterations, logging everything and writing a final report.

Per iteration we record: the weight vector, the synthesised system prompt, the
median (objective) and mean fleet mph, and the per-run scores. The optimiser's
regression-EM then updates the weights for the next round. At the end we report
the median AND mean for every iteration and the best prompt found.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import urllib.request
from typing import Dict, List

import numpy as np

from .heuristics import load_config, load_heuristics
from .optimizer import JointOptimizer
from .runner import evaluate_prompt
from .synthesize import synthesize


def _resolve(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _provenance(cfg: Dict, heuristics: List[Dict], hpath: str,
                config_path: str) -> Dict:
    """Everything needed to reproduce and cite the run: exact config, the full
    heuristic texts, seeds, and tool/model/library versions + git commit."""
    import numpy as _np
    import yaml as _yaml
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model = cfg["policy"]["model"]
    ver = _safe(lambda: json.load(urllib.request.urlopen(
        "http://127.0.0.1:11434/api/version", timeout=5)).get("version"))
    model_info = _safe(lambda: {
        k: json.load(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:11434/api/show",
            data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"}), timeout=10)).get(k)
        for k in ("details",)})
    return {
        "created_unix": int(time.time()),
        "config_path": os.path.abspath(config_path),
        "heuristics_path": os.path.abspath(hpath),
        "config": cfg,
        "heuristics": heuristics,                 # full id + text + priors
        "objective": cfg["objective"],
        "policy_model": model,
        "synthesis": cfg["synthesis"],
        "engine": cfg["engine"],
        "seeds": {"optimizer": cfg["optimizer"]["seed"],
                  "engine": "deterministic (official seed sequence)"},
        "versions": {
            "python": platform.python_version(),
            "numpy": _np.__version__, "pyyaml": _yaml.__version__,
            "ollama": ver, "platform": platform.platform(),
            "gpu": _safe(lambda: subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version",
                 "--format=csv,noheader"], text=True).strip()),
        },
        "model_details": model_info,
        "git_commit": _safe(lambda: subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()),
        "git_dirty": _safe(lambda: bool(subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"], text=True).strip())),
    }


def run_experiment(config_path: str, heuristics_path: str = None) -> Dict:
    base = os.path.dirname(os.path.abspath(config_path))
    cfg = load_config(config_path)
    hpath = heuristics_path or _resolve(base, "heuristics.yaml")
    heuristics, init_w, init_p = load_heuristics(hpath)
    ids = [h["id"] for h in heuristics]

    exp = cfg["experiment"]
    opt_cfg = cfg["optimizer"]
    syn_cfg = cfg["synthesis"]
    pol_cfg = cfg["policy"]
    eng_cfg = cfg["engine"]

    suite_root = os.path.dirname(base)            # prompt_em/
    out_dir = os.path.join(suite_root, "results", exp["name"])
    os.makedirs(out_dir, exist_ok=True)
    # provenance manifest (reproducibility for the paper)
    manifest = _provenance(cfg, heuristics, hpath, config_path)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    jsonl = open(os.path.join(out_dir, "iterations.jsonl"), "w")

    opt = JointOptimizer(
        n=len(heuristics), alpha=opt_cfg["alpha"],
        ridge_lambda=opt_cfg["ridge_lambda"], explore_std=opt_cfg["explore_std"],
        explore_decay=opt_cfg["explore_decay"],
        cold_start_rounds=opt_cfg["cold_start_rounds"], seed=opt_cfg["seed"],
        init_weights=init_w, init_priority=init_p,
        freeze_emphasis=opt_cfg.get("freeze_emphasis", False),
        freeze_order=opt_cfg.get("freeze_order", False))

    print(f"=== prompt-EM '{exp['name']}': {exp['iterations']} iterations x "
          f"{exp['runs_per_iter']} runs | objective={cfg['objective']} | "
          f"policy={pol_cfg['model']} | synth={syn_cfg['mode']} | "
          f"co-optimising emphasis + sequence ===", flush=True)
    print(f"heuristics: {', '.join(ids)}", flush=True)

    history: List[Dict] = []
    t0 = time.time()
    drop_below = syn_cfg.get("drop_below", 0.04)
    fcfg = syn_cfg.get("freedom", {}) or {}
    for it in range(exp["iterations"]):
        freedom = _freedom_at(it, fcfg)
        frozen = bool(fcfg) and freedom >= fcfg.get("freeze_structure", 1.0) \
            and len(opt._scores) > 0
        if frozen:
            # STAGE B: structure search done -> freeze weights+order at the best
            # found, and only the WORDING varies (clean wording attribution).
            w, p, order, _ = opt.best()
        else:
            w, p, order = opt.propose()
        diag = opt.diagnostics()
        dropped = [ids[i] for i in range(len(ids)) if w[i] < drop_below]

        def progress(g, n, scores, _it=it):
            sc = sorted(scores)
            print("  it %d run %2d/%d score=%.2f running_median=%.2f (%.0fs)"
                  % (_it, g, n, scores[-1], sc[len(sc) // 2], time.time() - t0),
                  flush=True)

        # synthesize `candidates` wording variants (only differ when freedom>0),
        # evaluate each, keep the best-scoring prompt.
        n_cand = max(1, int(fcfg.get("candidates", 1))) if freedom > 0 else 1
        prompt = res = objective = None
        for v in range(n_cand):
            cand = synthesize(heuristics, w, order, mode=syn_cfg["mode"],
                              model=syn_cfg.get("model", pol_cfg["model"]),
                              temperature=syn_cfg.get("temperature", 0.0),
                              drop_below=drop_below, freedom=freedom, variant=v)
            cres = evaluate_prompt(cand, model=pol_cfg["model"],
                                   engine_cfg=eng_cfg, runs=exp["runs_per_iter"],
                                   frames=exp["frames"],
                                   num_predict=pol_cfg.get("num_predict", 96),
                                   progress=progress)
            if res is None or cres[cfg["objective"]] > objective:
                prompt, res, objective = cand, cres, cres[cfg["objective"]]
        # the optimizer learns structure only while it is still searching it
        if not frozen:
            opt.observe(w, p, objective)
        else:
            opt._scores.append(objective)   # keep best-tracking current
            opt._wp.append((np.asarray(w), np.asarray(p)))

        emph_c, order_c = opt.contributions()
        rec = {
            "iteration": it,
            "timestamp_unix": int(time.time()),
            # --- proposal (the genome) ---
            "weights": {ids[i]: round(float(w[i]), 4) for i in range(len(ids))},
            "priority": {ids[i]: round(float(p[i]), 4) for i in range(len(ids))},
            "order": [ids[i] for i in order],          # prompt sequence this iter
            "dropped": dropped,                        # below the inclusion floor
            "freedom": round(freedom, 4),              # synthesizer wording latitude
            "structure_frozen": frozen,                # stage B (only wording varies)
            "candidates_tried": n_cand,
            "optimizer": diag,                         # phase, explore scale, responsibilities
            # --- objective & distribution ---
            "objective": cfg["objective"], "objective_value": objective,
            "median": res["median"], "mean": res["mean"], "std": res["std"],
            "min": res["min"], "max": res["max"],
            "scores": res["scores"],                   # per-run, seed order
            # --- behaviour & cost ---
            "action_fractions": res["action_fractions"],
            "action_counts": res["action_counts"], "decisions": res["decisions"],
            "parse_fail": res["parse_fail"], "llm_calls": res["llm_calls"],
            "cache_hits": res["cache_hits"],
            # --- credit assignment ---
            "weight_contributions": (None if emph_c is None else
                                     {ids[i]: round(float(emph_c[i]), 4)
                                      for i in range(len(ids))}),
            "order_contributions": (None if order_c is None else
                                    {ids[i]: round(float(order_c[i]), 4)
                                     for i in range(len(ids))}),
            "system_prompt": prompt,
            "elapsed_s": round(time.time() - t0),
        }
        history.append(rec)
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        print("ITER %d: median=%.2f mean=%.2f (objective %s=%.2f)  [%.0fs]"
              % (it, res["median"], res["mean"], cfg["objective"], objective,
                 time.time() - t0), flush=True)

    jsonl.close()
    report = _write_report(out_dir, exp, cfg, ids, history, opt)

    # machine-readable summary for analysis/plotting (the paper's figures)
    best = _best(history, cfg["objective"])
    emph_c, order_c = opt.contributions()
    summary = {
        "name": exp["name"], "objective": cfg["objective"],
        "iterations": exp["iterations"], "runs_per_iter": exp["runs_per_iter"],
        "heuristic_ids": ids,
        "median_trajectory": [r["median"] for r in history],
        "mean_trajectory": [r["mean"] for r in history],
        "objective_trajectory": [r["objective_value"] for r in history],
        "best_iteration": best["iteration"],
        "best_median": best["median"], "best_mean": best["mean"],
        "best_order": best["order"], "best_weights": best["weights"],
        "best_system_prompt": best["system_prompt"],
        "final_weight_contributions": (None if emph_c is None else
            {ids[i]: round(float(emph_c[i]), 4) for i in range(len(ids))}),
        "final_order_contributions": (None if order_c is None else
            {ids[i]: round(float(order_c[i]), 4) for i in range(len(ids))}),
        "action_fraction_trajectory": [r["action_fractions"] for r in history],
        "total_llm_calls": sum(r["llm_calls"] for r in history),
        "total_parse_fail": sum(r["parse_fail"] for r in history),
        "wall_seconds": history[-1]["elapsed_s"] if history else 0,
        "ceiling_ref": {"all_time": 76.3, "best_heuristic_H6": 76.40,
                        "hand_tuned_P4_24run": 76.50, "real_DQN": 74.13},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote report -> %s\nWrote summary -> %s\nWrote manifest -> %s"
          % (report, os.path.join(out_dir, "summary.json"),
             os.path.join(out_dir, "manifest.json")), flush=True)
    return {"out_dir": out_dir, "history": history, "best": best,
            "summary": summary}


def _best(history, objective):
    return max(history, key=lambda r: r["objective_value"])


def _freedom_at(it: int, f: Dict) -> float:
    """Scheduled synthesizer freedom: flat at `start` until `ramp_after`, then
    linearly to `end` over `ramp_span` iterations."""
    if not f:
        return 0.0
    if it <= f.get("ramp_after", 0):
        return float(f.get("start", 0.0))
    t = min(1.0, (it - f["ramp_after"]) / max(1, f.get("ramp_span", 1)))
    return float(f.get("start", 0.0) + t * (f.get("end", 0.0) - f.get("start", 0.0)))


def _write_report(out_dir, exp, cfg, ids, history, opt) -> str:
    best = _best(history, cfg["objective"])
    path = os.path.join(out_dir, "report.md")
    L = [f"# prompt-EM report — {exp['name']}", "",
         f"- objective: **{cfg['objective']} fleet mph** "
         f"(target: beat 76.3 ceiling)",
         f"- iterations: {exp['iterations']} x {exp['runs_per_iter']} runs",
         f"- policy model: {cfg['policy']['model']}; "
         f"synthesis: {cfg['synthesis']['mode']}; co-optimising emphasis + "
         f"sequence", "",
         "## Trajectory (median & mean fleet mph per iteration)", "",
         "| iter | median | mean | min | max | parse_fail |",
         "|---|---|---|---|---|---|"]
    for r in history:
        L.append("| %d | **%.2f** | %.2f | %.2f | %.2f | %d |"
                 % (r["iteration"], r["median"], r["mean"], r["min"],
                    r["max"], r["parse_fail"]))
    L += ["",
          f"**Best iteration: {best['iteration']} — "
          f"median {best['median']:.2f}, mean {best['mean']:.2f}.**", ""]
    emph_c, order_c = opt.contributions()
    bw = best["weights"]
    if emph_c is not None:
        L += ["## Emphasis contributions (higher weight => helps the objective)",
              "", "| heuristic | weight-contribution | best-iter weight |",
              "|---|---|---|"]
        for i in np.argsort(-emph_c):
            L.append("| %s | %+.3f | %.3f |" % (ids[i], emph_c[i], bw[ids[i]]))
        L.append("")
    if order_c is not None:
        L += ["## Sequence contributions (higher => helps to state EARLIER)", "",
              "| heuristic | order-contribution |", "|---|---|"]
        for i in np.argsort(-order_c):
            L.append("| %s | %+.3f |" % (ids[i], order_c[i]))
        L.append("")
    L += ["## Best prompt — sequence (heuristics in order)", "",
          " -> ".join(best["order"]), "",
          "## Best system prompt", "", "```", best["system_prompt"], "```", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path
