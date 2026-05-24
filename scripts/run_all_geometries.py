"""Run one case per geometry type and save contour images to figures/geometry_showcase/."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_CPU_INFERENCE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path
from openfoam_agent.agent import OpenFOAMAgent
from openfoam_agent import foam_plotter
from openfoam_agent.config import OPENFOAM_BASHRC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = [
    ("01_box",           "3D flow in a box Re=200, water, 0.5m cube"),
    ("02_channel",       "2D channel flow Re=500, water, length 2m, width 0.1m"),
    ("03_lid_cavity",    "2D lid-driven cavity Re=1000, water, 1m square"),
    ("04_cylinder",      "2D flow over a cylinder Re=200, water, diameter 0.1m"),
    ("05_pipe",          "2D pipe flow Re=5000, air, diameter 0.1m, turbulent k-omega SST"),
    ("06_bfs",           "2D backward-facing step Re=800, water, step height 0.05m"),
    ("07_airfoil",       "2D flow over NACA 0012 airfoil Re=500000, air, chord 1m, angle of attack 5 degrees"),
    ("08_sphere",        "3D flow past a sphere Re=300, water, diameter 0.1m"),
    ("09_wedge",         "Axisymmetric pipe flow Re=1000, water, diameter 0.05m, length 0.5m, wedge"),
    ("10_periodic_hill", "2D flow over periodic hills Re=1000, water, hill height 0.05m"),
    ("11_sbend",         "2D S-bend pipe Re=3000, air, pipe diameter 0.05m, double bend"),
    ("12_diffuser",      "2D planar diffuser Re=1000, air, inlet width 0.1m, length 0.5m"),
    ("13_ahmed_body",    "3D flow over Ahmed body Re=100000, air, length 0.5m, width 0.2m"),
    ("14_multi_hill",    "2D flow over multiple hills Re=500, water, hill height 0.05m, multiple hills"),
    ("15_tjunction",     "2D T-junction pipe Re=1500, water, main pipe diameter 0.05m, branching flow"),
    ("16_cd_nozzle",     "2D converging-diverging nozzle flow Re=1000, air, length 0.5m, throat diameter 0.02m"),
    ("17_elbow",         "2D pipe elbow Re=2000, water, pipe diameter 0.05m, 90-degree bend"),
]

OUT_DIR = Path(__file__).parent.parent / "figures" / "geometry_showcase"
OUT_DIR.mkdir(parents=True, exist_ok=True)

agent = OpenFOAMAgent(use_llm=True)
agent._init_components()

results = []
t_total = time.time()

for tag, prompt in CASES:
    print(f"\n{'='*65}")
    print(f"[{tag}]  {prompt[:65]}")
    print('='*65)
    t0 = time.time()
    try:
        r = agent.run(prompt=prompt, use_gmsh=True, max_retries=2, sim_timeout=600)
        elapsed = time.time() - t0
        status = "PASS" if r.score >= 0.80 else ("WARN" if r.score >= 0.55 else "FAIL")
        print(f"  [{status}] score={r.score:.2f}  solver={r.solver}  t={elapsed:.0f}s")

        case_dir = Path(r.case_dir)

        # Contour plot
        fig = foam_plotter.make_contour_figs(case_dir, OPENFOAM_BASHRC)
        fig.suptitle(f"{tag.replace('_', ' ').title()}  |  score={r.score:.2f}  solver={r.solver}",
                     fontsize=12, fontweight="bold")
        out = OUT_DIR / f"{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  saved → {out.name}")
        results.append((tag, r.score, r.solver, status, str(out)))

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        print(f"  [ERROR] {exc}  ({elapsed:.0f}s)")
        traceback.print_exc()
        results.append((tag, 0.0, "error", "ERROR", ""))

print(f"\n{'='*65}")
print(f"Done in {(time.time()-t_total)/60:.1f} min")
print(f"{'Tag':<22} {'Score':>6}  {'Solver':<22} {'Status'}")
print('-'*65)
for tag, score, solver, status, _ in results:
    print(f"{tag:<22} {score:>6.2f}  {solver:<22} {status}")
print(f"\nImages saved to: {OUT_DIR}")
