# Ablation Study

## Purpose

The goal of this study is to identify the effects of purposefully omitting a single heuristic at a time over the standard suite of 16 iterations, each with 10 runs.

For this purpose, each suite needs a new copy of default.yaml, e. g. omit_accelerate_when_clear.yaml, where the only thing that has to be changed is the experiment.name field.

Furthermore, each suite needs a new copy of heuristics.yaml, where the corresponding heuristic is deleted from "heuristics".

## How to run
1. Create a new copy of default.yaml, omit_[HEURISTIC].yaml
2. Create a new copy of heuristics.yaml, heuristics_omit_[HEURISTIC].yaml
3. Run the following command(s), assuming your terminal path is currently the project root:
3a. Windows: 
    - cd prompt_em
    - $env:PYTHONPATH = "../src;."
    - ../.venv/Scripts/python.exe -m prompt_em.cli run --config/omit_[HEURISTIC].yaml --heuristics config/heuristics_omit_[HEURISTIC].yaml

    3b. Linux:
    - cd prompt_em
    - PYTHONPATH=../src:. ../.venv/bin/python -m prompt_em.cli run --config config/omit_[HEURISTIC].yaml --heuristics config/heuristics_omit_[HEURISTIC].yaml

## Current State
As of this commit, 4 out of 16 heuristics have been tested. The tested heuristics are:
- "accelerate_when_clear"
- "overtake_when_blocked"
- "pick_faster_side"
- "prefer_centre_on_tie"

## Further Work
The remaining heuristics or rather suites with the remaining heuristics missing, still need to be tested. Currently, there are 12 left to be tested.

For that purpose it may be beneficial to implement one of the following, or something alongs the lines of it:
- Add a --ablation flag to the cli.py to have results of the ablation study be saved not directly in /results but in /results/ablation so that results does not get flooded with each directory. Likewise, the corresponding copies of heuristics.yaml and default.yaml should probably also be saved in a subdirectory of config as to not clog it.
- Add some sort of automization so that not every copy of default.yaml and heuristics.yaml has to be generated manually by hand.

## Known Issue
The 4 ablation suites were run in parallel, 2 ablation suites at a time, in different terminals for all of the following if not stated otherwise. It is not believed that this caused the following issue:

When running a check of the resultsw of omit_accelerate_when_clear, non-deterministic results were observed. Said suite was first run on Sat Jul 04 2026 13:52:21 GMT+0000 (/prompt_em/results/omit_accelerate_when_clear/manifest.json). On Sun Jul 05 2026 10:43:38 GMT+0000 it was run again with identical settings / parameters under the name omit_accelerate_when_clear_VAL. Both suites produced different results, which should not happen because of the fixed seed for the optimizer and temperature=0 for the LLM. 

Similar behaviour was observed with the inital and second suite of omit_pick_faster_side(_VAL), with the results being different from one another. 

A third, isolated run of the suite for omit_accelerate_when_clear (omit_accelerate_when_clear_VAL2, stopped after 6 iterations because of a lack of time) was only able to reproduce the results of the second run from Sunday for Iterations 0 to 4. Iteration 5 however was different from both the Saturday and the Sunday run.

omit_overtake_when_blocked and omit_prefer_centre_on_tie were not tested for this behaviour because a lack of time, but it's not unreasonable to assume that this behaviour will also occur here.

It is unclear what exactly caused the issue as of right now, so please take these results with a grain of salt and try to reproduce them.

See also: https://github.com/ollama/ollama/issues/586 ; https://github.com/ollama/ollama/issues/16860
