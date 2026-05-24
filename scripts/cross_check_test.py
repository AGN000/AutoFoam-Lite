"""Comprehensive cross-check: 23 diverse CFD prompts across all geometry/solver types."""
import os, sys, time, traceback
sys.stdout.reconfigure(line_buffering=True)   # unbuffered so output appears live
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["USE_CPU_INFERENCE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from openfoam_agent.agent import OpenFOAMAgent

CASES = [
    # ── Lid-driven cavity ─────────────────────────────────────────────────
    ("LDC laminar",      "2D lid-driven cavity Re=100, water, 1m square"),
    ("LDC mid-Re",       "2D lid-driven cavity Re=1000, water, 1m square"),
    # ── Channel / pipe ───────────────────────────────────────────────────
    ("Channel laminar",  "2D channel flow Re=500, water, length 2m, width 0.1m"),
    ("Pipe turbulent",   "2D pipe flow Re=5000, air, diameter 0.1m, turbulent k-omega SST"),
    ("Pipe 3D",          "3D pipe flow Re=3000, water, diameter 0.05m, length 0.5m"),
    # ── External flow ────────────────────────────────────────────────────
    ("Cylinder 2D",      "2D flow over a cylinder Re=200, water, diameter 0.1m"),
    ("Sphere 3D lam",    "3D flow past a sphere Re=300, water, diameter 0.1m"),
    # ── Complex geometries ───────────────────────────────────────────────
    ("BFS",              "2D backward-facing step Re=800, water, step height 0.05m"),
    ("Airfoil",          "2D flow over NACA 0012 airfoil Re=500000, air, chord 1m, angle of attack 5 degrees"),
    ("Diffuser",         "2D planar diffuser Re=1000, air, inlet width 0.1m, length 0.5m"),
    # ── Turbulent high-Re ────────────────────────────────────────────────
    ("Channel turb",     "2D turbulent channel Re=10000, air, length 2m, width 0.1m, k-omega SST"),
    # ── Special physics ──────────────────────────────────────────────────
    ("Heat transfer",    "2D natural convection in a square cavity, hot wall 350K cold wall 300K, air, 1m square"),
    ("Compressible",     "2D compressible flow in a channel Mach 0.5, air, length 1m, width 0.1m"),
    # ── 3D cases ─────────────────────────────────────────────────────────
    ("Box 3D",           "3D flow in a box Re=200, water, 0.5m cube"),
    ("Wedge axisym",     "Axisymmetric pipe flow Re=1000, water, diameter 0.05m, length 0.5m, wedge"),
    # ── Transient incompressible (icoFoam / pimpleFoam) ──────────────────
    ("Transient lam cyl","2D transient laminar flow over a cylinder Re=100, water, diameter 0.1m, end time 2s"),
    ("Transient turb pip","2D transient turbulent pipe Re=8000, air, diameter 0.1m, length 0.5m, k-omega SST"),
    # ── Transient heat transfer (buoyantPimpleFoam) ──────────────────────
    ("Transient buoy cav","2D transient natural convection 0.5m square air cavity, hot wall 370K, cold wall 290K"),
    # ── Multiphase VOF (interFoam) ────────────────────────────────────────
    ("Dam break VOF",    "2D dam break 2m wide 1m tall water column collapse, VOF two-phase air-water"),
    # ── Transient compressible (rhoPimpleFoam) ───────────────────────────
    ("Transient compress","2D transient compressible channel Mach 0.3, air, length 1m, width 0.1m"),
    # ── Complex geometry types ────────────────────────────────────────────
    ("Pipe elbow",       "2D pipe elbow Re=2000, water, pipe diameter 0.05m, 90-degree bend"),
    ("T-junction",       "2D T-junction pipe Re=1500, water, main pipe diameter 0.05m, branching flow"),
    ("S-bend",           "2D S-bend pipe Re=3000, air, pipe diameter 0.05m, double bend"),
]

START_FROM = int(os.environ.get("START_FROM", "1"))  # set to resume after crash

agent = OpenFOAMAgent(use_llm=True)
agent._init_components()

results = []
t_start = time.time()

for i, (label, prompt) in enumerate(CASES, 1):
    if i < START_FROM:
        print(f"[{i:02d}/{len(CASES)}] {label} — SKIPPED (already ran)")
        results.append((label, prompt, 0.0, "skipped", 0, "⏭", 0, "skipped"))
        continue
    print(f"\n{'='*65}")
    print(f"[{i:02d}/{len(CASES)}] {label}")
    print(f"  Prompt: {prompt[:70]}")
    print('='*65)
    t0 = time.time()
    try:
        r = agent.run(prompt=prompt, use_gmsh=True, max_retries=2, sim_timeout=600)
        elapsed = time.time() - t0
        emoji = "✅" if r.score >= 0.80 else ("⚠" if r.score >= 0.55 else "❌")
        results.append((label, prompt, r.score, r.solver, r.attempt + 1, emoji, elapsed, ""))
        print(f"  → {emoji} score={r.score:.2f}  solver={r.solver}  "
              f"attempt={r.attempt+1}  t={elapsed:.0f}s  case={r.case_dir}")
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        elapsed = time.time() - t0
        err_msg = traceback.format_exc()[-300:]
        results.append((label, prompt, 0.0, "error", 0, "❌", elapsed, str(exc)[:80]))
        print(f"  → ❌ EXCEPTION ({elapsed:.0f}s): {exc}")
        print(f"     {err_msg}")
    sys.stdout.flush()

total_time = time.time() - t_start

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("CROSS-CHECK SUMMARY")
print("="*65)
print(f"{'#':<3} {'Label':<16} {'Score':>6} {'Att':>4} {'Solver':<16} {'Time':>6}  {'Status'}")
print("-"*65)
for i, (label, prompt, score, solver, attempt, emoji, elapsed, err) in enumerate(results, 1):
    print(f"{i:<3} {label:<16} {score:>6.2f} {attempt:>4}  {solver:<16} {elapsed:>5.0f}s  {emoji}")
    if err:
        print(f"    ERR: {err}")
print("="*65)

passed  = sum(1 for r in results if r[2] >= 0.80)
partial = sum(1 for r in results if 0.55 <= r[2] < 0.80)
failed  = sum(1 for r in results if r[2] < 0.55)
avg     = sum(r[2] for r in results) / len(results)

print(f"✅ Pass (≥0.80): {passed}/{len(CASES)}   "
      f"⚠ Partial (0.55-0.80): {partial}/{len(CASES)}   "
      f"❌ Fail (<0.55): {failed}/{len(CASES)}")
print(f"Average score : {avg:.2f}   Total time: {total_time/60:.1f} min")
print("="*65)
