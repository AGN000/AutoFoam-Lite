"""Test the 3 UI placeholder prompts and print scores."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["USE_CPU_INFERENCE"] = "0"   # GPU with bitsandbytes 4-bit
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from openfoam_agent.agent import OpenFOAMAgent

PROMPTS = [
    "2D lid-driven cavity Re=1000, water, 1m square",
    "2D pipe flow Re=5000, air, diameter 0.1 m, turbulent k-omega SST",
    "3D flow past a sphere Re=300, water, diameter 0.1 m",
]

agent = OpenFOAMAgent(use_llm=True)
agent._init_components()

results = []
for i, prompt in enumerate(PROMPTS, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(PROMPTS)}] {prompt}")
    print('='*60)
    try:
        r = agent.run(prompt=prompt, use_gmsh=True, max_retries=2, sim_timeout=600)
        emoji = "✅" if r.score >= 0.80 else ("⚠" if r.score >= 0.55 else "❌")
        results.append((prompt, r.score, r.solver, r.attempt + 1, emoji))
        print(f"  → {emoji} score={r.score:.2f}  solver={r.solver}  attempt={r.attempt+1}  case={r.case_dir}")
    except Exception as exc:
        results.append((prompt, 0.0, "error", 0, "❌"))
        print(f"  → ❌ EXCEPTION: {exc}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for prompt, score, solver, attempt, emoji in results:
    print(f"{emoji} [{score:.2f}] (attempt {attempt})  {prompt[:55]}")
    print(f"       solver={solver}")
print("="*60)
avg = sum(s for _, s, *_ in results) / len(results)
print(f"Average score: {avg:.2f}")
