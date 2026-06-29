"""Command-line entry point for the prompt-EM suite.

  python -m prompt_em.cli run     --config config/default.yaml
  python -m prompt_em.cli synth   --config config/default.yaml   # preview a prompt
  python -m prompt_em.cli report  --name em_v1                   # reprint a report

The DeepTraffic engine must be importable (PYTHONPATH=../src) and `ollama serve`
must be running with the configured model pulled.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)                              # prompt_em/
REPO = os.path.dirname(SUITE)                              # DeepTrafficByLLM/
for p in (os.path.join(REPO, "src"), SUITE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _default_config():
    return os.path.join(SUITE, "config", "default.yaml")


def cmd_run(args):
    from prompt_em.experiment import run_experiment
    run_experiment(args.config, args.heuristics)


def cmd_synth(args):
    """Preview the prompt synthesised from the heuristics' initial weights."""
    from prompt_em.heuristics import load_config, load_heuristics
    from prompt_em.synthesize import synthesize
    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    hpath = args.heuristics or os.path.join(base, "heuristics.yaml")
    heuristics, w, p = load_heuristics(hpath)
    import numpy as np
    order = [int(i) for i in np.argsort(-p)]
    s = cfg["synthesis"]
    print(synthesize(heuristics, w, order, mode=args.mode or s["mode"],
                     model=s.get("model", "qwen2.5:3b"),
                     temperature=s.get("temperature", 0.0),
                     drop_below=s.get("drop_below", 0.04)))


def cmd_report(args):
    path = os.path.join(SUITE, "results", args.name, "report.md")
    if not os.path.exists(path):
        sys.exit(f"no report at {path}")
    print(open(path).read())


def main(argv=None):
    p = argparse.ArgumentParser(prog="prompt_em")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full EM experiment")
    r.add_argument("--config", default=_default_config())
    r.add_argument("--heuristics", default=None)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("synth", help="preview a synthesised prompt")
    s.add_argument("--config", default=_default_config())
    s.add_argument("--heuristics", default=None)
    s.add_argument("--mode", default=None, choices=["llm", "template"])
    s.set_defaults(func=cmd_synth)

    rp = sub.add_parser("report", help="print a finished run's report")
    rp.add_argument("--name", required=True)
    rp.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
