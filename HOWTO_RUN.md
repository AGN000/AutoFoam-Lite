# AutoFOAM — How to Run

## Prerequisites

| Requirement | Notes |
|---|---|
| OpenFOAM v2412 | Installed at `/usr/lib/openfoam/openfoam2412` (default config) |
| Python ≥ 3.10 | Via the `autofoam` conda environment |
| NVIDIA GPU | Any CUDA-capable GPU; model fits on ≥ 4 GB VRAM at 4-bit NF4 |
| conda | Miniconda/Anaconda |

---

## 1. Environment Setup (first time only)

### Create the conda environment

```bash
conda create -n autofoam python=3.10 -y
conda activate autofoam
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers bitsandbytes accelerate
pip install gmsh chromadb pydantic sentence-transformers huggingface_hub
```

### Clone the repo

```bash
git clone https://github.com/AGN000/AutoFOAM.git
cd AutoFOAM
```

### Configure OpenFOAM path

Edit `openfoam_agent/config.py` line 6 if your OpenFOAM is not at the default path:

```python
OPENFOAM_BASHRC = "/usr/lib/openfoam/openfoam2412/etc/bashrc"
```

### Model weights

The default model is downloaded automatically on first run from HuggingFace:

```
Qwen/Qwen2.5-Coder-3B-Instruct  (4-bit NF4, ~2 GB VRAM)
```

To use a fine-tuned or custom model, set the override before running:

```bash
export OPENFOAM_AGENT_LLM_OVERRIDE=/path/to/your/merged_model
```

### Build the RAG index (first time only)

```bash
conda activate autofoam
cd AutoFOAM
python scripts/index_tutorials.py
python scripts/index_knowledge_base.py
```

---

## 2. Running a Single Simulation

### From the command line

```bash
conda activate autofoam
cd AutoFOAM
python scripts/run_agent.py run "2D lid-driven cavity Re=1000, water, 1m square"
```

Common options:

```bash
# Use blockMesh instead of gmsh (faster for simple geometries)
python scripts/run_agent.py run "..." --no-gmsh

# Set solver timeout and retry count
python scripts/run_agent.py run "..." --timeout 600 --retries 3

# Run without LLM (uses rule-based fallback params — useful for debugging)
python scripts/run_agent.py run "..." --no-llm
```

### From Python

```python
from openfoam_agent.agent import OpenFOAMAgent

agent = OpenFOAMAgent(use_llm=True)

result = agent.run(
    "turbulent pipe flow Re=5000, air, diameter 0.1m, k-omega SST",
    use_gmsh=True,
    max_retries=2,
    sim_timeout=300,
)

print(f"Score  : {result.score:.2f}")     # 0.0 – 1.0
print(f"Solver : {result.solver}")         # e.g. simpleFoam
print(f"Case   : {result.case_dir}")       # path to OpenFOAM case directory
print(f"Notes  : {result.feedback}")       # scoring breakdown
```

---

## 3. Cross-Check (23-case benchmark)

Runs all 23 standard cases and prints a summary table. Cases 1–15 cover the
original geometry/solver set; cases 16–23 extend coverage to all transient
solvers (icoFoam, pimpleFoam, buoyantPimpleFoam, rhoPimpleFoam), multiphase
VOF (interFoam), and three new geometry types (elbow, T-junction, S-bend).

```bash
conda activate autofoam
cd AutoFOAM
python scripts/cross_check_test.py
```

Expected result (original 15 cases): **15/15 pass, average score ≥ 0.87**.

To resume after a crash (skips already-completed cases):

```bash
START_FROM=7 python scripts/cross_check_test.py
```

---

## 4. Interactive REPL

Loads the model once and accepts prompts in a loop — no reload between runs.

```bash
conda activate autofoam
cd AutoFOAM
python scripts/repl.py
```

```
prompt> 2D lid-driven cavity Re=1000, 2m square, water
prompt> NACA0012 airfoil AoA 5 deg Re=1e6 chord=1m
prompt> turbulent pipe flow Re=50000 diameter=0.05m
prompt> quit
```

REPL commands:

| Command | Effect |
|---|---|
| `last` | Re-print the last result |
| `cases` | List 10 most recent case directories |
| `timeout=N` | Set solver timeout in seconds (default 300) |
| `retries=N` | Set max self-correction retries (default 1) |
| `quit` / Ctrl-D | Exit |

---

## 5. Docker

### GPU (recommended)

```bash
# Build and start (Gradio UI on http://localhost:7861)
docker compose up --build

# Use a larger model at build time
docker compose build --build-arg MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct
docker compose up
```

### CPU-only

```bash
docker compose --profile cpu up --build
```

### Other container commands

```bash
# One-off simulation
docker exec autofoam python3.11 scripts/run_agent.py run "2D lid-driven cavity Re=1000"

# Interactive REPL inside container
docker compose run --rm autofoam repl

# Run the 15-case cross-check
docker compose run --rm autofoam crosscheck

# Build RAG index (first time only)
docker compose run --rm autofoam index

# Open a shell
docker compose run --rm autofoam bash
```

### Persistent data

Named volumes keep cases and the model cache across rebuilds:

| Volume | Contents |
|---|---|
| `autofoam_data` | `data/cases/`, `data/dataset/`, `data/checkpoints/` |
| `hf_cache` | Downloaded HuggingFace model weights |

---

## 6. Self-Evolution Loop

### Collect training data

```bash
python scripts/generate_training_data.py
```

### Full evolution cycle (curate → SFT → merge → eval → swap)

```bash
bash scripts/evolve.sh

# Dry run (validates without swapping the model)
EVOLVE_DRY_RUN=1 bash scripts/evolve.sh
```

---

## 6. Case Output

Every run writes a complete OpenFOAM case to `data/cases/`:

```
data/cases/case_<hash>_attempt0/
├── 0/           # initial conditions
├── constant/    # mesh + physical properties
├── system/      # controlDict, fvSchemes, fvSolution
└── agent.log    # full solver log (checkMesh + solver output)
```

Open in ParaView:

```bash
paraview data/cases/case_<hash>_attempt0/
```

---

## 7. Key Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `OPENFOAM_AGENT_LLM_OVERRIDE` | *(config.py)* | Override model path without editing config |
| `USE_CPU_INFERENCE` | `0` | Set to `1` to run model on CPU (slow) |
| `TORCHDYNAMO_DISABLE` | *(unset)* | Set to `1` to disable torch.compile (avoids compile errors) |
| `VLLM_GPU_MEM_FRAC` | `0.85` | Fraction of GPU memory for the model |
| `START_FROM` | `1` | Resume cross-check from case N |

---

## 8. Scorer Components

The score (0–1) reported for each run breaks down as:

| Component | Points | Condition |
|---|---|---|
| Converged | +0.40 | Simulation completed without fatal error |
| Residuals < 1e-4 | +0.20 | Inner linear solver final residuals all < 1e-4 |
| Residuals < 1e-3 | +0.10 | (partial, if not < 1e-4) |
| Trend quality | 0–0.10 | How well outer SIMPLE residuals decrease |
| Mass conservation | +0.05 | Continuity error < 1e-3 |
| Correct solver | +0.10 | Matches physics-based solver selection rules |
| Valid BCs | +0.05 | Boundary conditions present in `0/` directory |
| Non-ortho > 70° | −0.10 | Poor mesh quality |
| Runtime > 300 s | −0.10 | Solver wall time exceeded |
| Plateau | −0.05 | Residuals stuck at a high value |
