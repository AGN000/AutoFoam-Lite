#!/bin/bash
set -e

# Source OpenFOAM so every subprocess can call simpleFoam, blockMesh, gmsh, etc.
source /usr/lib/openfoam/openfoam2412/etc/bashrc 2>/dev/null || true

CMD="${1:-ui}"

case "$CMD" in
  ui)
    echo "[AutoFoam-Lite] Starting Gradio UI → http://localhost:7861"
    exec python3.11 -c "
from openfoam_agent.ui import launch_ui
launch_ui(host='0.0.0.0', port=7861, share=False)
"
    ;;
  run)
    shift
    exec python3.11 scripts/run_agent.py run "$@"
    ;;
  repl)
    exec python3.11 scripts/repl.py
    ;;
  crosscheck)
    exec python3.11 scripts/cross_check_test.py
    ;;
  index)
    python3.11 scripts/index_tutorials.py
    exec python3.11 scripts/index_knowledge_base.py
    ;;
  bash)
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac
