#!/usr/bin/env python3
"""Run 20 diverse CFD prompts through the agent and report a scored summary."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoam_agent.agent import OpenFOAMAgent

PROMPTS = [
    # --- lid-driven cavity ---
    "2D lid-driven cavity Re=100,  water, 1m square",
    "2D lid-driven cavity Re=400,  water, 1m square",
    "2D lid-driven cavity Re=1000, water, 1m square",
    "2D lid-driven cavity Re=3200, water, 1m square, turbulent",
    # --- pipe flow ---
    "2D pipe flow Re=500,  water, diameter 0.05 m, length 1 m",
    "2D pipe flow Re=2000, water, diameter 0.05 m, length 1 m",
    "2D pipe flow Re=5000, water, diameter 0.1 m, length 2 m, turbulent k-omega SST",
    # --- channel ---
    "2D channel flow Re=1000, water, length 5 m, height 0.1 m",
    "2D channel flow Re=5000, air,   length 2 m, height 0.1 m, turbulent",
    # --- cylinder ---
    "2D flow past a cylinder Re=40,  water, diameter 0.1 m",
    "2D flow past a cylinder Re=200, water, diameter 0.1 m",
    # --- backward facing step ---
    "2D backward facing step Re=200, water",
    "2D backward facing step Re=800, air",
    # --- airfoil ---
    "2D NACA 0012 airfoil Re=1e6, air, angle of attack 5 degrees",
    "2D airfoil Re=500000, air, angle of attack 10 degrees",
    # --- different fluids ---
    "2D lid-driven cavity Re=200, glycerine, 0.5 m square",
    "2D pipe flow Re=100,  engine oil, diameter 0.02 m, length 0.5 m",
    # --- 3D ---
    "3D pipe flow Re=2000, water, diameter 0.05 m, length 1 m, turbulent",
    "3D lid-driven cavity Re=100, water, 0.5 m cube",
    # --- transient ---
    "2D lid-driven cavity Re=1000, water, 1m square, transient simulation",
]

MAX_RETRIES = 2
SIM_TIMEOUT = 180   # seconds per simulation

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def colour(score):
    if score >= 0.80:
        return GREEN
    if score >= 0.55:
        return YELLOW
    return RED


def main():
    print(f"\n{BOLD}AutoFOAM 3B Batch Test — {len(PROMPTS)} prompts, "
          f"max {MAX_RETRIES} retries each{RESET}\n")

    agent = OpenFOAMAgent(use_llm=True)
    agent._init_components()          # load model once, reuse for all runs

    rows = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i:02d}/{len(PROMPTS)}] {prompt[:70]}")
        t0 = time.time()
        try:
            result  = agent.run(prompt, max_retries=MAX_RETRIES, sim_timeout=SIM_TIMEOUT)
            score   = result.score
            solver  = result.solver or "unknown"
            feedback = result.feedback[:50]
            n_tries = result.attempt + 1
        except Exception as e:
            score, solver, feedback, n_tries = 0.0, "error", str(e)[:50], 1
        elapsed = time.time() - t0

        c = colour(score)
        tag = "OK" if score >= 0.80 else ("PARTIAL" if score >= 0.55 else "FAIL")
        print(f"       {c}{tag}{RESET}  score={score:.2f}  solver={solver}  "
              f"tries={n_tries}  t={elapsed:.0f}s")
        rows.append((i, prompt[:60], score, solver, feedback, n_tries, elapsed))

    # ── summary table ──────────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print(f"{'#':>2}  {'Prompt':<62}  {'Score':>5}  {'Solver':<14}  {'Tries':>5}")
    print(f"{'─'*90}")
    ok = partial = fail = 0
    for i, prompt, score, solver, status, n_tries, elapsed in rows:
        c = colour(score)
        tag = "OK" if score >= 0.80 else ("~" if score >= 0.55 else "✗")
        print(f"{i:>2}  {prompt:<62}  {c}{score:>5.2f}{RESET}  {solver:<14}  {n_tries:>5}  {c}{tag}{RESET}")
        if score >= 0.80:   ok += 1
        elif score >= 0.55: partial += 1
        else:               fail += 1

    print(f"{'─'*90}")
    print(f"  {GREEN}OK (≥0.80){RESET}: {ok}   "
          f"{YELLOW}PARTIAL (0.55–0.80){RESET}: {partial}   "
          f"{RED}FAIL (<0.55){RESET}: {fail}   "
          f"Total: {len(rows)}")


if __name__ == "__main__":
    main()
