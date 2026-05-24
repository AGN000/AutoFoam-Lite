"""Gradio GUI for AutoFOAM — prompt → mesh → solve → convergence + contour plots."""
from __future__ import annotations

import os
from pathlib import Path

from .config import TUTORIALS_DIR, OPENFOAM_BASHRC
from .agent import SEED_PROMPTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_singleton(model_id: str, device: str = "CPU"):
    """Return (and cache) an OpenFOAMAgent for the given model_id + device."""
    from . import config as _cfg
    from .agent import OpenFOAMAgent

    if not hasattr(_agent_singleton, "_cache"):
        _agent_singleton._cache = {}

    key = (model_id, device)
    if key not in _agent_singleton._cache:
        os.environ["OPENFOAM_AGENT_LLM_OVERRIDE"] = model_id
        os.environ["USE_CPU_INFERENCE"] = "1" if device == "CPU" else "0"
        _cfg._llm_instance = None          # reset so next get_llm() uses new model/device
        agent = OpenFOAMAgent(use_llm=True)
        agent._init_components()
        _agent_singleton._cache[key] = agent

    return _agent_singleton._cache[key]


def _list_cases() -> list[str]:
    from .config import CASES_DIR
    if not CASES_DIR.exists():
        return []
    return sorted(
        [d.name for d in CASES_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

def launch_ui(host: str = "0.0.0.0", port: int = 7861, share: bool = False):
    import gradio as gr
    from .foam_plotter import (
        parse_residuals,
        make_convergence_fig,
        make_contour_figs,
    )
    from .config import CASES_DIR

    # ── Simulation runner ─────────────────────────────────────────────────────
    def run_simulation(
        prompt: str,
        model_id: str,
        device: str,
        use_gmsh: bool,
        max_retries: int,
        sim_timeout: int,
    ):
        import threading
        if not prompt.strip():
            yield (
                "",
                gr.update(value="⚠ Please enter a simulation prompt."),
                None, None, gr.update(value=""), gr.update(value="{}"),
                gr.update(value=0.0), gr.update(choices=_list_cases()),
            )
            return

        yield (
            _prog("Loading model…", 4),
            gr.update(value=f"⏳ Loading model on {device}…"),
            None, None, gr.update(value=""), gr.update(value="{}"),
            gr.update(value=0.0), gr.update(choices=_list_cases()),
        )

        try:
            agent = _agent_singleton(
                model_id.strip() or "Qwen/Qwen2.5-Coder-3B-Instruct",
                device=device,
            )
        except Exception as exc:
            yield (
                _prog("Error loading model", 0, error=True),
                gr.update(value=f"❌ Error loading model: {exc}"),
                None, None, gr.update(value=""), gr.update(value="{}"),
                gr.update(value=0.0), gr.update(choices=_list_cases()),
            )
            return

        # Run agent in a background thread so we can stream residuals live
        from . import runner as _runner
        import time as _time

        # Clear previous case state so the poll loop doesn't see stale data.
        agent._current_case_dir = None

        state: dict = {"result": None, "error": None, "done": False}

        def _run():
            try:
                state["result"] = agent.run(
                    prompt=prompt,
                    use_gmsh=use_gmsh,
                    max_retries=int(max_retries),
                    sim_timeout=int(sim_timeout),
                )
            except Exception as exc:
                state["error"] = exc
            finally:
                state["done"] = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # Poll loop — update residual plot every 2 s while solver runs.
        # Wrapped in try/finally so clicking Stop kills the solver immediately.
        _prev_hist: dict = {}
        try:
            while not state["done"]:
                _time.sleep(2)
                stage = agent._status
                pct = _STAGE_PCT.get(stage, 4)
                case_dir = agent._current_case_dir

                if case_dir is None:
                    yield (
                        _prog(stage, pct),
                        gr.update(value=stage),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    )
                    continue

                live_log = Path(case_dir) / "log.solver"
                plot_update = gr.update()
                status_msg = stage
                if live_log.exists():
                    hist = parse_residuals(live_log)
                    if hist:
                        last_iter = max(len(v) for v in hist.values())
                        # Progress climbs from 60 % → 90 % as iterations accumulate
                        pct = int(60 + 28 * (1 - 1 / (1 + last_iter / 300)))
                        status_msg = f"⏳ Solver — iteration {last_iter}"
                        if hist != _prev_hist:
                            _prev_hist = {k: list(v) for k, v in hist.items()}
                            plot_update = make_convergence_fig(hist)

                yield (
                    _prog(status_msg, pct),
                    gr.update(value=status_msg),
                    plot_update, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                )
        except GeneratorExit:
            _runner.kill_running()
            agent._current_case_dir = None
            return

        thread.join()

        if state["error"]:
            yield (
                _prog(f"Error: {state['error']}", 0, error=True),
                gr.update(value=f"❌ Error: {state['error']}"),
                None, None, gr.update(value=""), gr.update(value="{}"),
                gr.update(value=0.0), gr.update(choices=_list_cases()),
            )
            return

        result = state["result"]

        # ── Build final outputs ───────────────────────────────────────────────
        params_json = result.params.model_dump_json(indent=2) if result.params else "{}"
        log_text = ""
        conv_fig = None
        contour_fig = None

        if result.case_dir:
            log_file = Path(result.case_dir) / "agent.log"
            if log_file.exists():
                log_text = log_file.read_text()[-8000:]
                hist = parse_residuals(log_file)
                if hist:
                    conv_fig = make_convergence_fig(hist)
            if Path(result.case_dir).exists():
                try:
                    contour_fig = make_contour_figs(result.case_dir, OPENFOAM_BASHRC)
                except Exception:
                    contour_fig = None

        score_emoji = "✅" if result.score >= 0.80 else ("⚠" if result.score >= 0.55 else "❌")
        status = (
            f"{score_emoji}  Score: {result.score:.2f}  |  Solver: {result.solver}  "
            f"|  Attempt {result.attempt + 1}  |  Case: {result.case_dir}"
        )
        done_label = f"{score_emoji} Complete — Score {result.score:.2f}  |  {result.solver}"

        yield (
            _prog(done_label, 100, done=(result.score >= 0.5), error=(result.score < 0.5)),
            gr.update(value=status),
            conv_fig,
            contour_fig,
            gr.update(value=log_text),
            gr.update(value=params_json),
            gr.update(value=result.score),
            gr.update(choices=_list_cases(), value=Path(result.case_dir).name if result.case_dir else None),
        )

    # ── ParaView launcher ─────────────────────────────────────────────────────
    def _open_paraview(case_name: str):
        import subprocess
        if not case_name:
            return gr.update(value="⚠ No case selected.", visible=True)
        case_dir = CASES_DIR / case_name
        if not case_dir.exists():
            return gr.update(value=f"⚠ Case not found: {case_dir}", visible=True)
        foam_file = case_dir / f"{case_name}.foam"
        foam_file.touch(exist_ok=True)
        try:
            subprocess.Popen(["paraview", str(foam_file)])
            return gr.update(value=f"✅ ParaView launched — {foam_file.name}", visible=True)
        except FileNotFoundError:
            return gr.update(value="⚠ paraview not found in PATH. Install ParaView first.", visible=True)

    # ── Case browser callbacks ────────────────────────────────────────────────
    def load_case_plots(case_name: str):
        if not case_name:
            return None, None, ""
        case_dir = CASES_DIR / case_name
        conv_fig = None
        contour_fig = None
        log_text = ""
        log_file = case_dir / "agent.log"
        if log_file.exists():
            log_text = log_file.read_text()[-8000:]
            hist = parse_residuals(log_file)
            if hist:
                conv_fig = make_convergence_fig(hist)
        try:
            contour_fig = make_contour_figs(case_dir, OPENFOAM_BASHRC)
        except Exception:
            pass
        return conv_fig, contour_fig, log_text

    # ── Tutorial browser ──────────────────────────────────────────────────────
    def browse_tutorial(case_name: str):
        case_dir = TUTORIALS_DIR / case_name
        if not case_dir.exists():
            return "Case not found."
        parts = []
        for fname in ("README.md", "README.txt", "README"):
            fp = case_dir / fname
            if fp.exists():
                parts.append(fp.read_text(errors="ignore")[:2000])
                break
        ctrl = case_dir / "system" / "controlDict"
        if ctrl.exists():
            parts.append(f"\n--- system/controlDict ---\n{ctrl.read_text(errors='ignore')[:1000]}")
        return "\n".join(parts) if parts else "No README found."

    tutorial_names = (
        sorted([d.name for d in TUTORIALS_DIR.iterdir() if d.is_dir()])
        if TUTORIALS_DIR.exists() else []
    )

    # ── Progress bar helpers ──────────────────────────────────────────────────
    _STAGE_PCT: dict[str, int] = {
        "⏳ Loading model…":                          6,
        "⏳ Refining prompt…":                        20,
        "⏳ Extracting CFD parameters…":              36,
        "⏳ Generating mesh (gmsh)…":                 52,
        "⏳ Building blockMesh…":                     52,
        "⏳ gmsh failed — falling back to blockMesh…": 52,
        "⏳ Scoring results…":                        92,
    }

    def _prog(label: str, pct: int, *, done: bool = False, error: bool = False) -> str:
        if not label:
            return ""
        if done:
            color, bg = "#2e7d32", "#e8f5e9"
        elif error:
            color, bg = "#c62828", "#ffebee"
        else:
            color, bg = "#1565C0", "#e8eaf6"
        pct = max(0, min(100, pct))
        return (
            f'<div style="background:{bg};border-radius:8px;padding:10px 18px;'
            f'margin:0 0 6px 0;border-left:5px solid {color};font-family:sans-serif">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:6px">'
            f'<span style="font-weight:700;color:{color};font-size:14px">{label}</span>'
            f'<span style="color:#555;font-size:12px;font-weight:600">{pct}%</span></div>'
            f'<div style="background:rgba(0,0,0,0.12);border-radius:4px;height:7px;overflow:hidden">'
            f'<div style="background:{color};height:100%;width:{pct}%;'
            f'transition:width 0.5s ease;border-radius:4px"></div>'
            f'</div></div>'
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    ACCENT = "#1565C0"
    css = """
    .status-box textarea { font-size: 14px !important; }
    .score-number input { font-size: 22px !important; font-weight: bold; }
    .stop-btn { background: #c62828 !important; color: white !important; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="AutoFOAM — AI CFD Agent") as demo:

        gr.Markdown(
            "# 🌊 AutoFOAM — AI-Driven CFD Simulation\n"
            "_Natural language → Mesh → OpenFOAM → Results_"
        )

        # ── Top-level progress bar (visible across all tabs while running) ────
        progress_bar = gr.HTML(value="")

        # ── Tab 1: Run Simulation ────────────────────────────────────────────
        with gr.Tab("🚀 Run Simulation"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt_in = gr.Textbox(
                        label="Simulation Prompt",
                        lines=4,
                        placeholder=(
                            "e.g.  2D lid-driven cavity Re=1000, water, 1m square\n"
                            "      2D pipe flow Re=5000, air, diameter 0.1 m, turbulent k-omega SST\n"
                            "      3D flow past a sphere Re=300, water"
                        ),
                    )

                    # ── Model location row with folder-browse button ──────────
                    with gr.Row():
                        model_id_in = gr.Textbox(
                            label="🤖 Model (HuggingFace ID or local path)",
                            value="Qwen/Qwen2.5-Coder-3B-Instruct",
                            scale=5,
                        )
                        browse_model_btn = gr.Button("📂", scale=0, min_width=55,
                                                     elem_id="browse-model-btn")
                        use_gmsh_in = gr.Checkbox(label="Use gmsh", value=True, scale=1)

                    # FileExplorer — hidden until browse button clicked
                    model_explorer = gr.FileExplorer(
                        root_dir=str(Path.home()),
                        glob="**",
                        file_count="single",
                        label="Browse — select a local model folder, then click ✔ Use selected path",
                        visible=False,
                        height=280,
                    )
                    use_selected_btn = gr.Button("✔ Use selected path", visible=False, size="sm")

                    with gr.Row():
                        device_in = gr.Radio(
                            choices=["CPU", "GPU"],
                            value="GPU",
                            label="Inference device",
                            scale=1,
                        )
                        retries_in = gr.Slider(1, 5, value=2, step=1, label="Max retries", scale=2)
                        timeout_in = gr.Slider(60, 600, value=300, step=30, label="Timeout (s)", scale=2)

                    # ── Run / Stop buttons ────────────────────────────────────
                    with gr.Row():
                        run_btn = gr.Button("▶  Generate & Run", variant="primary",
                                            size="lg", scale=4)
                        stop_btn = gr.Button("⏹  Stop", variant="stop",
                                             size="lg", scale=1, elem_classes="stop-btn")

                with gr.Column(scale=1):
                    score_out = gr.Number(label="Score (0–1)", elem_classes="score-number")
                    status_out = gr.Textbox(
                        label="Status", lines=4, interactive=False, elem_classes="status-box"
                    )
                    params_out = gr.Code(label="Extracted Parameters", language="json", lines=14)

            gr.Markdown("### 📈 Convergence & Contours")
            with gr.Row():
                conv_plot = gr.Plot(label="Residual Convergence")
                contour_plot = gr.Plot(label="Field Contours (U, p)")

            with gr.Row():
                pv_btn = gr.Button("🔬  Open in ParaView", scale=1)
                pv_status = gr.Textbox(visible=False, label="", scale=4, interactive=False)

            log_out = gr.Textbox(label="Solver Log (last 8 000 chars)", lines=12, max_lines=20)

            # ── Browse past cases ─────────────────────────────────────────────
            with gr.Accordion("📂 Browse past cases", open=False):
                cases_dd = gr.Dropdown(choices=_list_cases(), label="Select case", interactive=True)
                with gr.Row():
                    load_case_btn = gr.Button("Load plots", scale=3)
                    open_folder_btn = gr.Button("📂 Open folder", scale=1)
                    browse_pv_btn = gr.Button("🔬 ParaView", scale=1)
                open_folder_status = gr.Textbox(visible=False, label="")
                load_case_btn.click(
                    load_case_plots,
                    inputs=[cases_dd],
                    outputs=[conv_plot, contour_plot, log_out],
                )

                def _open_case_folder(case_name: str):
                    import subprocess
                    if not case_name:
                        return gr.update(value="⚠ No case selected.", visible=True)
                    path = CASES_DIR / case_name
                    if not path.exists():
                        return gr.update(value=f"⚠ Path not found: {path}", visible=True)
                    subprocess.Popen(["xdg-open", str(path)])
                    return gr.update(value=f"Opened: {path}", visible=True)

                open_folder_btn.click(
                    _open_case_folder,
                    inputs=[cases_dd],
                    outputs=[open_folder_status],
                )
                browse_pv_btn.click(
                    _open_paraview,
                    inputs=[cases_dd],
                    outputs=[pv_status],
                )

            # ── Wire up folder browser ────────────────────────────────────────
            def _toggle_explorer(current_vis):
                new_vis = not current_vis
                return gr.update(visible=new_vis), gr.update(visible=new_vis)

            _explorer_vis = gr.State(False)
            browse_model_btn.click(
                _toggle_explorer,
                inputs=[_explorer_vis],
                outputs=[model_explorer, use_selected_btn],
            ).then(lambda v: not v, inputs=[_explorer_vis], outputs=[_explorer_vis])

            use_selected_btn.click(
                lambda path: (path if path else gr.update(), gr.update(visible=False),
                              gr.update(visible=False), False),
                inputs=[model_explorer],
                outputs=[model_id_in, model_explorer, use_selected_btn, _explorer_vis],
            )

            # ── Wire up Run / Stop / ParaView ─────────────────────────────────
            run_event = run_btn.click(
                run_simulation,
                inputs=[prompt_in, model_id_in, device_in, use_gmsh_in, retries_in, timeout_in],
                outputs=[progress_bar, status_out, conv_plot, contour_plot, log_out, params_out, score_out, cases_dd],
            )
            stop_btn.click(fn=None, cancels=[run_event])
            pv_btn.click(
                _open_paraview,
                inputs=[cases_dd],
                outputs=[pv_status],
            )

        # ── Tab 2: Batch & Results ───────────────────────────────────────────
        with gr.Tab("📊 Results Explorer"):
            gr.Markdown("Load any previously run case to inspect its plots and log.")
            all_cases_dd = gr.Dropdown(choices=_list_cases(), label="Select case", interactive=True)
            refresh_btn = gr.Button("🔄 Refresh list")
            refresh_btn.click(lambda: gr.update(choices=_list_cases()), outputs=[all_cases_dd])

            with gr.Row():
                res_conv = gr.Plot(label="Convergence")
                res_contour = gr.Plot(label="Contours")
            res_log = gr.Textbox(label="Log", lines=14, max_lines=25)
            all_cases_dd.change(
                load_case_plots,
                inputs=[all_cases_dd],
                outputs=[res_conv, res_contour, res_log],
            )

        # ── Tab 3: Training ──────────────────────────────────────────────────
        with gr.Tab("🏋 Train Model"):
            gr.Markdown("### Collect data from seed prompts and fine-tune via QLoRA.")

            with gr.Row():
                n_prompts_in = gr.Slider(1, max(len(SEED_PROMPTS), 10), value=5, step=1,
                                         label="Seed prompts to collect")
                min_score_collect_in = gr.Slider(0.3, 0.9, value=0.5, step=0.05,
                                                  label="Min score to collect")
            collect_btn = gr.Button("Collect Training Data")
            collect_out = gr.Textbox(label="Collection result")

            gr.Markdown("---")
            with gr.Row():
                min_score_train_in = gr.Slider(0.3, 0.9, value=0.6, step=0.05,
                                                label="Min score for training")
                max_ex_in = gr.Slider(10, 500, value=100, step=10, label="Max examples")
                epochs_in_t = gr.Slider(1, 5, value=2, step=1, label="Epochs")
            train_btn = gr.Button("Start QLoRA Fine-tuning", variant="primary")
            train_out = gr.Textbox(label="Training result")

            def collect_data(n: int, min_s: float):
                from .agent import SEED_PROMPTS as SP
                from .training import collect_training_episodes
                agent = _agent_singleton("Qwen/Qwen2.5-Coder-3B-Instruct")
                prompts = SP[: int(n)]
                examples = collect_training_episodes(agent, prompts, min_score=min_s)
                return f"Collected {len(examples)} examples (score ≥ {min_s})"

            def start_training(min_s: float, max_ex: int, epochs: int):
                from .training import train_qlora
                try:
                    train_qlora(min_score=min_s, max_examples=int(max_ex), num_epochs=int(epochs))
                    return "Training complete — adapter saved."
                except Exception as exc:
                    return f"Training failed: {exc}"

            collect_btn.click(collect_data, inputs=[n_prompts_in, min_score_collect_in], outputs=[collect_out])
            train_btn.click(start_training, inputs=[min_score_train_in, max_ex_in, epochs_in_t], outputs=[train_out])

        # ── Tab 4: Tutorial Browser ──────────────────────────────────────────
        with gr.Tab("📚 Tutorial Browser"):
            gr.Markdown("### Browse OpenFOAM tutorial cases")
            tut_dd = gr.Dropdown(choices=tutorial_names, label="Tutorial case", interactive=True)
            browse_btn = gr.Button("Load")
            tut_out = gr.Textbox(label="Case info", lines=20, max_lines=40)
            browse_btn.click(browse_tutorial, inputs=[tut_dd], outputs=[tut_out])

        # ── Tab 5: Setup / Info ──────────────────────────────────────────────
        with gr.Tab("⚙ Setup"):
            gr.Markdown(f"""
### System paths
| Setting | Value |
|---|---|
| OpenFOAM bashrc | `{OPENFOAM_BASHRC}` |
| Cases directory | `{CASES_DIR}` |
| Tutorials | `{TUTORIALS_DIR}` |

### Changing the model
Set the **Model** field in the *Run Simulation* tab to any HuggingFace ID or local path:
- `Qwen/Qwen2.5-Coder-3B-Instruct` — default (4-bit, ~2 GB VRAM)
- `Qwen/Qwen2.5-Coder-7B-Instruct` — larger (4-bit, ~4 GB VRAM)
- `/path/to/local/checkpoint` — local weights

### Docker quick-start
```bash
# GPU (recommended)
docker compose up --build

# CPU-only
docker compose -f docker-compose.cpu.yml up --build
```
""")

    demo.queue()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
        css=css,
    )


if __name__ == "__main__":
    launch_ui()
