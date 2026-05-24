"""Utilities for parsing OpenFOAM logs and generating plots."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Residual parsing ─────────────────────────────────────────────────────────

def parse_residuals(log_path: str | Path) -> dict[str, list[float]]:
    """Return {field: [initial_residual, ...]} from an OpenFOAM solver log."""
    history: dict[str, list[float]] = {}
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return history
    for line in text.splitlines():
        if "Solving for" not in line or "Initial residual" not in line:
            continue
        parts = line.split(",")
        try:
            field = parts[0].split("Solving for")[-1].strip().split()[0]
            for part in parts:
                if "Initial residual" in part:
                    val = float(part.split("=")[-1].strip())
                    history.setdefault(field, []).append(val)
        except (IndexError, ValueError):
            pass
    return history


def make_convergence_fig(history: dict[str, list[float]]) -> plt.Figure:
    """Semilogy convergence plot from residual history dict."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if not history:
        ax.text(0.5, 0.5, "No residual data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return fig

    COLORS = plt.cm.tab10.colors
    for idx, (field, values) in enumerate(sorted(history.items())):
        ax.semilogy(values, label=field, color=COLORS[idx % len(COLORS)], linewidth=1.6)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Initial Residual", fontsize=11)
    ax.set_title("Convergence History", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_facecolor("#f7f7f7")
    fig.tight_layout()
    return fig


# ── OpenFOAM field readers ────────────────────────────────────────────────────

def _read_foam_points(points_file: Path) -> Optional[np.ndarray]:
    """Read constant/polyMesh/points → (N, 3) float array."""
    text = points_file.read_text(errors="ignore")
    m = re.search(r"(\d+)\s*\n\(", text)
    if not m:
        return None
    n = int(m.group(1))
    pts: list[list[float]] = []
    for mp in re.finditer(r"\(([^)]+)\)", text[m.end():]):
        nums = list(map(float, mp.group(1).split()))
        pts.append(nums)
        if len(pts) >= n:
            break
    return np.array(pts) if pts else None


def _read_foam_scalar(field_file: Path) -> Optional[np.ndarray]:
    """Read an OpenFOAM internalField scalar → (N,) array or scalar float."""
    text = field_file.read_text(errors="ignore")
    m_uni = re.search(r"internalField\s+uniform\s+([\d.eE+\-]+)", text)
    if m_uni:
        return np.array([float(m_uni.group(1))])
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\(", text)
    if not m:
        return None
    n = int(m.group(1))
    vals: list[float] = []
    for mv in re.finditer(r"([\d.eE+\-]+)", text[m.end():]):
        try:
            vals.append(float(mv.group(1)))
        except ValueError:
            pass
        if len(vals) >= n:
            break
    return np.array(vals) if vals else None


def _read_foam_vector(field_file: Path) -> Optional[np.ndarray]:
    """Read an OpenFOAM internalField vector → (N, 3) array."""
    text = field_file.read_text(errors="ignore")
    m_uni = re.search(r"internalField\s+uniform\s+\(([^)]+)\)", text)
    if m_uni:
        return np.array([list(map(float, m_uni.group(1).split()))])
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\(", text)
    if not m:
        return None
    n = int(m.group(1))
    vecs: list[list[float]] = []
    for mv in re.finditer(r"\(([^)]+)\)", text[m.end():]):
        nums = list(map(float, mv.group(1).split()))
        vecs.append(nums)
        if len(vecs) >= n:
            break
    return np.array(vecs) if vecs else None


def _latest_time_dir(case_dir: Path) -> Optional[Path]:
    """Return the highest-numbered time directory in a case."""
    candidates = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                candidates.append((float(d.name), d))
            except ValueError:
                pass
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _cell_centres_from_foamproc(case_dir: Path, of_bashrc: str) -> Optional[np.ndarray]:
    """Run postProcess -func writeCellCentres and read resulting C field."""
    cmd = (
        f"bash -c 'source {of_bashrc} && "
        f"postProcess -func writeCellCentres -latestTime -case {case_dir} > /dev/null 2>&1'"
    )
    subprocess.run(cmd, shell=True, timeout=60)
    t_dir = _latest_time_dir(case_dir)
    if t_dir is None:
        return None
    c_file = t_dir / "C"
    if not c_file.exists():
        return None
    return _read_foam_vector(c_file)


# ── Contour plots ─────────────────────────────────────────────────────────────

def make_contour_figs(case_dir: str | Path, of_bashrc: str = "") -> plt.Figure:
    """
    Generate U-magnitude and p contour plots for a completed OpenFOAM case.
    Returns a matplotlib Figure with two subplots.
    """
    case_dir = Path(case_dir)
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_u = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])

    try:
        _fill_contours(case_dir, ax_u, ax_p, of_bashrc)
    except Exception as exc:
        for ax, title in [(ax_u, "Velocity magnitude |U|"), (ax_p, "Pressure p")]:
            ax.text(0.5, 0.5, f"Plot unavailable\n{exc}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            ax.set_title(title)
            ax.axis("off")

    fig.suptitle(f"Field contours — {case_dir.name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _fill_contours(case_dir: Path, ax_u: plt.Axes, ax_p: plt.Axes, of_bashrc: str):
    from scipy.spatial import KDTree  # type: ignore

    # 1. Get cell centres
    if of_bashrc:
        centres = _cell_centres_from_foamproc(case_dir, of_bashrc)
    else:
        centres = None

    if centres is None or len(centres) == 0:
        pts_file = case_dir / "constant" / "polyMesh" / "points"
        if not pts_file.exists():
            raise FileNotFoundError("polyMesh/points not found")
        centres = _read_foam_points(pts_file)
        if centres is None:
            raise ValueError("Could not parse mesh points")

    # 2. Project to 2D
    z_range = centres[:, 2].max() - centres[:, 2].min()
    xy_range = max(
        centres[:, 0].max() - centres[:, 0].min(),
        centres[:, 1].max() - centres[:, 1].min(),
        1e-9,
    )
    if z_range < 0.05 * xy_range:
        x, y = centres[:, 0], centres[:, 1]
    else:
        # 3D — take mid-z slice
        z_mid = (centres[:, 2].max() + centres[:, 2].min()) / 2
        mask = np.abs(centres[:, 2] - z_mid) < 0.05 * (z_range + 1e-9)
        if mask.sum() < 4:
            mask = np.ones(len(centres), dtype=bool)
        centres = centres[mask]
        x, y = centres[:, 0], centres[:, 1]

    # 3. KDTree + local cell spacing (nearest-neighbour distance)
    pts = np.column_stack([x, y])
    tree = KDTree(pts)
    nn_dist, _ = tree.query(pts, k=2)
    local_cs = nn_dist[:, 1]  # NN spacing per cell centre

    # 4. Regular output grid (250×250)
    GRID = 250
    xi = np.linspace(x.min(), x.max(), GRID)
    yi = np.linspace(y.min(), y.max(), GRID)
    XX, YY = np.meshgrid(xi, yi)
    gpts = np.column_stack([XX.ravel(), YY.ravel()])

    # 5 & 6. Combined: query K nearest cells for both domain masking and IDW.
    #   Domain mask: grid point is "inside the fluid domain" when its distance
    #   to the nearest cell is < 0.7× the MEAN local spacing of the K nearest
    #   cells.  Using the mean (instead of just the nearest cell) makes the
    #   threshold robust on highly non-uniform meshes (e.g. gmsh airfoil where
    #   far-field cells are huge and near-wall cells are tiny).
    # Domain mask: nearest cell with adaptive threshold.
    #   Tight (0.8×) for uniform blockMesh; loose (1.5×) for highly non-uniform
    #   gmsh meshes.  Using a single nearest cell avoids the K-NN averaging
    #   artefact where nearby fine cells pull the threshold down too far.
    cs_cv = float(local_cs.std() / (local_cs.mean() + 1e-12))
    thr = 0.8 + 0.7 * min(cs_cv, 1.0)
    gd1, gi1 = tree.query(gpts, k=1)
    in_dom = gd1 < thr * local_cs[gi1]

    # IDW interpolation from 4 nearest cells
    K = min(4, len(pts))
    gd_k, gi_k = tree.query(gpts, k=K)
    d_mat = gd_k + 1e-12
    w = 1.0 / d_mat
    w /= w.sum(axis=1, keepdims=True)

    def _to_grid(vals: np.ndarray) -> np.ndarray:
        zi = (w * vals[gi_k]).sum(axis=1)
        return np.where(in_dom, zi, np.nan).reshape(GRID, GRID)

    # 7. Read latest field files
    t_dir = _latest_time_dir(case_dir)
    if t_dir is None:
        raise FileNotFoundError("No time directories found")

    u_file = t_dir / "U"
    p_file = t_dir / "p"

    def _trim(values: np.ndarray, n: int) -> np.ndarray:
        if len(values) > n:
            return values[:n]
        if len(values) < n:
            return np.pad(values, (0, n - len(values)), constant_values=np.nan)
        return values

    n = len(x)

    # ── Velocity ──
    if u_file.exists():
        U = _read_foam_vector(u_file)
        if U is not None and len(U) > 0:
            if U.shape[0] == 1:
                Umag = np.full(n, np.linalg.norm(U[0, :2]))
            else:
                Umag = _trim(np.linalg.norm(U[:, :2], axis=1), n)
            p99 = np.nanpercentile(Umag, 99)
            Umag = np.clip(Umag, 0, p99 * 1.05)
            cf = ax_u.contourf(XX, YY, np.ma.masked_invalid(_to_grid(Umag)), levels=30, cmap="viridis")
            plt.colorbar(cf, ax=ax_u, label="|U| (m/s)")
            ax_u.set_title("Velocity magnitude |U|", fontsize=11)
            ax_u.set_xlabel("x (m)")
            ax_u.set_ylabel("y (m)")
            ax_u.set_aspect("equal", adjustable="box")
        else:
            ax_u.text(0.5, 0.5, "U field empty", ha="center", va="center", transform=ax_u.transAxes)
    else:
        ax_u.text(0.5, 0.5, "U field not found", ha="center", va="center", transform=ax_u.transAxes)

    # ── Pressure ──
    if p_file.exists():
        p_vals = _read_foam_scalar(p_file)
        if p_vals is not None and len(p_vals) > 0:
            if len(p_vals) == 1:
                p_arr = np.full(n, p_vals[0])
            else:
                p_arr = _trim(p_vals, n)
            p1 = np.nanpercentile(p_arr, 1)
            p99 = np.nanpercentile(p_arr, 99)
            p_arr = np.clip(p_arr, p1, p99)
            cf2 = ax_p.contourf(XX, YY, np.ma.masked_invalid(_to_grid(p_arr)), levels=30, cmap="coolwarm")
            plt.colorbar(cf2, ax=ax_p, label="p (m²/s²)")
            ax_p.set_title("Pressure p", fontsize=11)
            ax_p.set_xlabel("x (m)")
            ax_p.set_ylabel("y (m)")
            ax_p.set_aspect("equal", adjustable="box")
        else:
            ax_p.text(0.5, 0.5, "p field empty", ha="center", va="center", transform=ax_p.transAxes)
    else:
        ax_p.text(0.5, 0.5, "p field not found", ha="center", va="center", transform=ax_p.transAxes)
