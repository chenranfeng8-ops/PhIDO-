"""Tidy3D integration for PhIDO.

This module provides FDTD simulation capabilities using Tidy3D cloud API.
Supports various photonic components including waveguide crossings.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
import time
import re
import shutil
import matplotlib
matplotlib.use("Agg")  # BUG-7 fix: thread-safe backend, avoids Tkinter crash
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix Windows console encoding for Unicode output (Tidy3D uses bullet points •)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Force UTF-8 mode for all I/O
    os.environ["PYTHONUTF8"] = "1"
    # Also ensure stdout can handle Unicode
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    # Try to set console code page to UTF-8
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass

from PhotonicsAI.config import PATH


def _log_path() -> Path:
    PATH.build.mkdir(parents=True, exist_ok=True)
    return PATH.build / "tidy3d.log"


def _append_log(lines: list[str]) -> None:
    p = _log_path()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"\n[{ts}] Tidy3D integration\n")
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")


def _extract_metrics(sim_data: Any) -> Dict[str, Any]:
    """Extract quantitative metrics from a completed Tidy3D simulation.

    Iterates over flux monitors in *sim_data* and computes per-port mean
    transmitted power.  A composite ``score`` (sum of all flux values) is
    returned as the single scalar used by the optimisation loop.
    """
    metrics: Dict[str, Any] = {}
    if sim_data is None:
        return metrics

    try:
        monitor_names: list[str] = []
        if hasattr(sim_data, "monitor_data"):
            monitor_names = list(sim_data.monitor_data.keys())
        elif hasattr(sim_data, "keys"):
            monitor_names = list(sim_data.keys())

        flux_values: Dict[str, float] = {}
        for name in monitor_names:
            if not name.startswith("flux_"):
                continue
            try:
                mon = sim_data[name]
                flux = getattr(mon, "flux", None)
                if flux is not None:
                    flux_values[name] = float(flux.values.mean())
                else:
                    flux_values[name] = 0.0
            except Exception:
                flux_values[name] = 0.0

        mode_power: Dict[str, List[float]] = {}
        for name in monitor_names:
            if not str(name).startswith("mode_"):
                continue
            try:
                mon = sim_data[name]
                amps = getattr(mon, "amps", None)
                if amps is None:
                    continue
                _preferred_dir = "-" if str(name) == "mode_port_o1" else "+"
                collapsed_arr = _collapse_mode_power_from_amps(amps, preferred_direction=_preferred_dir)
                if collapsed_arr is None:
                    continue
                if collapsed_arr.size == 0:
                    continue
                mode_power[str(name)] = [float(item) for item in collapsed_arr.tolist()]
            except Exception:
                continue

        metrics["flux"] = flux_values
        if mode_power:
            metrics["mode_power"] = mode_power
        metrics["score"] = sum(flux_values.values()) if flux_values else 0.0
    except Exception:
        metrics["score"] = 0.0
    return metrics


def _wavelength_artifact_tag(wavelength_nm: float | None) -> str:
    """Build a stable artifact tag such as ``1300nm`` from wavelength."""
    if wavelength_nm is None:
        return ""
    try:
        value = float(wavelength_nm)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}nm"
    return f"{value:.3f}".rstrip("0").rstrip(".") + "nm"


def _sanitize_artifact_tag(tag: str | None) -> str:
    raw = str(tag or "").strip()
    if not raw:
        return ""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", raw).strip("_")[:80]


def _save_tagged_copy(path: Path, artifact_tag: str) -> Path | None:
    tag = _sanitize_artifact_tag(artifact_tag)
    if not tag or not path.exists():
        return None
    tagged = path.with_name(f"{path.stem}_{tag}{path.suffix}")
    try:
        shutil.copy2(path, tagged)
        return tagged
    except Exception:
        return None


def _record_artifact(artifacts: list[str], path: Path | None) -> None:
    """Record a generated artifact path when it exists on disk."""
    if path is None:
        return
    try:
        if path.exists():
            artifacts.append(str(path))
    except Exception:
        return


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _collapse_mode_power_from_amps(amps: Any, preferred_direction: str = "+") -> Any:
    """Collapse mode amplitudes into per-mode power, preferring outgoing direction."""
    import numpy as np

    data = amps
    dims = tuple(getattr(data, "dims", ()))
    if "direction" in dims:
        try:
            data = data.sel(direction=preferred_direction)
        except Exception:
            try:
                data = data.isel(direction=0)
            except Exception:
                pass

    values = getattr(data, "values", data)
    arr = np.asarray(values)
    if arr.size == 0:
        return None

    dims = tuple(getattr(data, "dims", ()))
    if "mode_index" in dims:
        mode_axis = int(dims.index("mode_index"))
    elif arr.ndim == 1:
        mode_axis = 0
    elif arr.ndim >= 2:
        mode_axis = 1
    else:
        mode_axis = 0
    mode_axis = max(0, min(mode_axis, arr.ndim - 1))

    power = np.abs(arr) ** 2
    reduce_axes = tuple(axis for axis in range(power.ndim) if axis != mode_axis)
    collapsed = power.mean(axis=reduce_axes) if reduce_axes else power
    collapsed_arr = np.ravel(np.asarray(collapsed, dtype=float))
    if collapsed_arr.size == 0:
        return None
    return collapsed_arr


def _load_density_replay_payload(
    density_path: str | None,
    density_meta_path: str | None,
    lines: list[str],
) -> tuple[Any, float | None, float | None]:
    """Load topology density array and optional region size for replay."""
    import numpy as np

    region_sx: float | None = None
    region_sy: float | None = None
    density_shape: tuple[int, int] | None = None
    if density_meta_path:
        try:
            meta = json.loads(Path(density_meta_path).read_text(encoding="utf-8"))
            region = meta.get("region_size_um") or []
            if isinstance(region, list) and len(region) >= 2:
                region_sx = float(region[0])
                region_sy = float(region[1])
            shape = meta.get("density_shape") or []
            if isinstance(shape, list) and len(shape) >= 2:
                density_shape = (int(shape[0]), int(shape[1]))
        except Exception as exc:
            lines.append(f"Replay meta load failed: {exc}")

    if not density_path:
        return None, region_sx, region_sy

    p = Path(str(density_path))
    if not p.exists():
        lines.append(f"Replay density path not found: {p}")
        return None, region_sx, region_sy

    try:
        if p.suffix.lower() == ".npy":
            density = np.asarray(np.load(p))
        else:
            # PNG fallback is best-effort only; exact replay requires .npy.
            import matplotlib.image as mpimg

            img = mpimg.imread(str(p))
            if img.ndim == 3:
                density = img[:, :, 0]
            else:
                density = img
            lines.append("Replay loaded from image fallback (approximate).")
        if density.ndim != 2:
            if density.ndim == 1 and density_shape is not None:
                expected = int(density_shape[0]) * int(density_shape[1])
                if int(density.size) == expected:
                    density = density.reshape(density_shape)
                    lines.append(
                        "Replay reshaped flat density using meta "
                        f"density_shape={density_shape}."
                    )
                else:
                    lines.append(
                        "Replay flat density size does not match meta "
                        f"density_shape={density_shape}: size={density.size}."
                    )
                    return None, region_sx, region_sy
            else:
                lines.append("Replay density is not 2D and no usable density_shape meta was supplied.")
                return None, region_sx, region_sy
        density = np.nan_to_num(density, nan=0.0, posinf=1.0, neginf=0.0)
        density = np.clip(density, 0.0, 1.0)
        return density, region_sx, region_sy
    except Exception as exc:
        lines.append(f"Replay density load failed: {exc}")
        return None, region_sx, region_sy


def _density_pixel_structures(
    td,
    density: Any,
    *,
    region_sx: float,
    region_sy: float,
    wg_height: float,
    threshold: float = 0.5,
    max_pixels: int = 4096,
) -> list[Any]:
    """Convert a 2D density map into silicon pixel structures."""
    import numpy as np

    arr = np.asarray(density, dtype=float)
    nx, ny = int(arr.shape[0]), int(arr.shape[1])
    if nx <= 0 or ny <= 0:
        return []

    total = nx * ny
    if total > max_pixels:
        step = int(np.ceil(np.sqrt(total / max_pixels)))
        arr = arr[::step, ::step]
        nx, ny = int(arr.shape[0]), int(arr.shape[1])

    dx = float(region_sx) / float(nx)
    dy = float(region_sy) / float(ny)
    x0 = -float(region_sx) / 2.0 + dx / 2.0
    y0 = -float(region_sy) / 2.0 + dy / 2.0
    si = td.Medium(permittivity=3.45**2, name="silicon")

    structs = []
    for i in range(nx):
        x = x0 + i * dx
        for j in range(ny):
            val = float(arr[i, j])
            if val < threshold:
                continue
            y = y0 + j * dy
            structs.append(
                td.Structure(
                    geometry=td.Box(
                        center=(x, y, wg_height / 2.0),
                        size=(dx, dy, wg_height),
                    ),
                    medium=si,
                    name=f"topo_px_{i}_{j}",
                )
            )
    return structs


def _plot_sim_results(
    sim_data: Any,
    component_type: str,
    wg_height: float,
    lines: list[str],
    wavelength_nm: float | None = None,
    artifact_tag: str = "",
    artifacts: list[str] | None = None,
    source_flux_key: str = "flux_port_o1",
) -> None:
    """Generate field distribution and flux plots from completed simulation data."""
    wl_tag = _wavelength_artifact_tag(wavelength_nm)
    artifact_sink = artifacts if artifacts is not None else []

    def _collapse_field_slice(values: Any) -> Any:
        import numpy as np

        arr = np.asarray(values)
        if arr.size == 0:
            return None
        arr = np.squeeze(arr)
        while arr.ndim > 2:
            arr = np.take(arr, indices=0, axis=-1)
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr

    def _phase_align_array(arr: Any) -> Any:
        import numpy as np

        data = np.asarray(arr, dtype=complex)
        if data.size == 0:
            return data
        ref_idx = int(np.argmax(np.abs(data)))
        ref = data.reshape(-1)[ref_idx]
        phase_ref = float(np.angle(ref)) if np.abs(ref) > 0 else 0.0
        return data * np.exp(-1j * phase_ref)

    def _plot_port_field_monitor(monitor_name: str) -> None:
        import numpy as np

        mon = sim_data[monitor_name]
        ey_obj = getattr(mon, "Ey", None)
        if ey_obj is None:
            return
        raw_values = getattr(ey_obj, "values", ey_obj)
        arr = _collapse_field_slice(raw_values)
        if arr is None:
            return
        aligned = _phase_align_array(arr)
        real_map = np.real(aligned)
        abs_map = np.abs(arr)
        max_abs_real = float(np.max(np.abs(real_map))) if real_map.size else 0.0
        vlim = max(1e-12, max_abs_real)

        port_name = monitor_name.replace("field_", "", 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(real_map, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-vlim, vmax=vlim)
        axes[0].set_title(f"{component_type} - {port_name} Ey (phase-aligned)")
        axes[1].imshow(abs_map, cmap="magma", origin="lower", aspect="auto")
        axes[1].set_title(f"{component_type} - {port_name} |Ey|")
        out = PATH.build / f"tidy3d_field_{port_name}_{component_type}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        _record_artifact(artifact_sink, out)
        tagged = _save_tagged_copy(out, artifact_tag)
        if tagged is not None:
            _record_artifact(artifact_sink, tagged)
            lines.append(f"Saved tagged output-port field plot: {tagged}")
        if wl_tag:
            out_wl = PATH.build / f"tidy3d_field_{port_name}_{wl_tag}_{component_type}.png"
            fig.savefig(out_wl, dpi=150, bbox_inches="tight")
            _record_artifact(artifact_sink, out_wl)
            tagged_wl = _save_tagged_copy(out_wl, artifact_tag)
            if tagged_wl is not None:
                _record_artifact(artifact_sink, tagged_wl)
                lines.append(f"Saved tagged wavelength output-port field plot: {tagged_wl}")
            lines.append(f"Saved wavelength output-port field plot: {out_wl}")
        plt.close(fig)
        lines.append(f"Saved output-port field plot: {out}")

    def _plot_mode_profile_monitor(monitor_name: str) -> None:
        # ModeMonitor data only has .amps (complex modal amplitudes), not field
        # distributions (.Ey etc.).  The previous approach of reading .Ey from
        # ModeMonitor data always returned None and silently skipped the plot.
        # Fix: use a standalone ModeSolver.plot_field() at the monitor cross-section
        # per the official Tidy3D API (docs.flexcompute.com: "how-can-i-plot-the-
        # mode-field-distribution").
        try:
            import tidy3d as td
            from tidy3d.plugins.mode import ModeSolver
        except ImportError as _imp_err:
            lines.append(f"ModeSolver import failed for {monitor_name}: {_imp_err}")
            return

        # Retrieve the Simulation object stored inside the SimulationData result.
        base_sim = getattr(sim_data, "simulation", None)
        if base_sim is None:
            lines.append(f"ModeSolver skipped for {monitor_name}: sim_data.simulation unavailable.")
            return

        # Locate the matching ModeMonitor in the simulation's monitor list to
        # obtain its cross-section center and size.
        mon_geom = None
        for _mon in getattr(base_sim, "monitors", []):
            if getattr(_mon, "name", None) == monitor_name:
                mon_geom = _mon
                break
        if mon_geom is None:
            lines.append(f"ModeSolver skipped for {monitor_name}: monitor not found in simulation.")
            return

        mon_center = tuple(float(c) for c in getattr(mon_geom, "center", (0.0, 0.0, 0.0)))
        mon_size = list(float(s) for s in getattr(mon_geom, "size", (0.0, 2.0, 1.5)))

        # The propagation axis has size == 0 in the ModeMonitor; ensure that.
        prop_axis = next((i for i, s in enumerate(mon_size) if s == 0.0), 0)
        plane_size = list(mon_size)
        plane_size[prop_axis] = 0.0
        # Ensure the cross-section is wide enough to capture evanescent field.
        for i, s in enumerate(plane_size):
            if i != prop_axis and s < 1.5:
                plane_size[i] = max(s * 3.0, 2.0)
        plane = td.Box(center=mon_center, size=tuple(plane_size))

        # Frequency from wavelength_nm parameter of the enclosing function.
        try:
            _wl_um = float(wavelength_nm) / 1000.0 if wavelength_nm else 1.55
        except (TypeError, ValueError):
            _wl_um = 1.55
        freq0_ms = td.C_0 / _wl_um

        # ModeSpec: no target_neff so modes are sorted by highest n_eff (fundamental
        # first), matching the ordering in the production ModeMonitor / optimizer.
        # Visualise at most 3 modes regardless of how many the FDTD monitor tracked.
        _num_vis = 3
        mode_spec_ms = td.ModeSpec(num_modes=_num_vis, filter_pol="te")

        try:
            mode_solver = ModeSolver(
                simulation=base_sim,
                plane=plane,
                mode_spec=mode_spec_ms,
                freqs=[freq0_ms],
            )
        except Exception as _ms_exc:
            lines.append(f"ModeSolver construction failed for {monitor_name}: {_ms_exc}")
            return

        # Plot Re(Ey) and |E| for each TE mode using ModeSolver.plot_field() —
        # the correct method per Tidy3D docs; ModeSolverData.plot_field() must NOT
        # be used here.
        port_name = monitor_name.replace("mode_", "", 1)
        fig, axes = plt.subplots(
            _num_vis, 3,
            figsize=(13.5, 3.8 * _num_vis),
            tight_layout=True,
        )
        axes = axes.reshape(_num_vis, 3)
        try:
            for _mi in range(_num_vis):
                mode_solver.plot_field("E", "abs", mode_index=_mi, f=freq0_ms, ax=axes[_mi, 0])
                axes[_mi, 0].set_title(f"TE{_mi} |E| total", fontsize=9)
                mode_solver.plot_field("Ey", "real", mode_index=_mi, f=freq0_ms, ax=axes[_mi, 1])
                axes[_mi, 1].set_title(f"TE{_mi} Re(Ey) — dominant TE component", fontsize=9)
                mode_solver.plot_field("Ez", "abs", mode_index=_mi, f=freq0_ms, ax=axes[_mi, 2])
                axes[_mi, 2].set_title(f"TE{_mi} |Ez| — crosspol check", fontsize=9)
        except Exception as _plot_exc:
            plt.close(fig)
            lines.append(f"ModeSolver.plot_field() failed for {monitor_name}: {_plot_exc}")
            return

        fig.suptitle(
            f"{monitor_name} – TE Mode Profiles @ {_wl_um:.3f} µm\n"
            f"({component_type}, local ModeSolver)",
            fontsize=11, fontweight="bold",
        )

        out = PATH.build / f"tidy3d_mode_profile_{port_name}_{component_type}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        _record_artifact(artifact_sink, out)
        tagged = _save_tagged_copy(out, artifact_tag)
        if tagged is not None:
            _record_artifact(artifact_sink, tagged)
            lines.append(f"Saved tagged mode-profile plot: {tagged}")
        if wl_tag:
            out_wl = PATH.build / f"tidy3d_mode_profile_{port_name}_{wl_tag}_{component_type}.png"
            fig.savefig(out_wl, dpi=150, bbox_inches="tight")
            _record_artifact(artifact_sink, out_wl)
            tagged_wl = _save_tagged_copy(out_wl, artifact_tag)
            if tagged_wl is not None:
                _record_artifact(artifact_sink, tagged_wl)
                lines.append(f"Saved tagged wavelength mode-profile plot: {tagged_wl}")
            lines.append(f"Saved wavelength mode-profile plot: {out_wl}")
        plt.close(fig)
        lines.append(f"Saved mode-profile (ModeSolver) plot: {out}")

    def _extract_phase_aligned_y_profile(monitor_name: str) -> tuple[Any, Any] | None:
        import numpy as np

        mon = sim_data[monitor_name]
        ey_obj = getattr(mon, "Ey", None)
        if ey_obj is None:
            return None
        try:
            data = ey_obj
            for dim in ("f", "freq", "frequency", "t", "time", "mode_index"):
                if hasattr(data, "dims") and dim in tuple(data.dims):
                    data = data.isel({dim: 0})
            for dim in ("x", "z"):
                if hasattr(data, "dims") and dim in tuple(data.dims):
                    data = data.mean(dim=dim)
            if not hasattr(data, "dims"):
                return None
            dims = tuple(data.dims)
            y_dim = "y" if "y" in dims else (dims[0] if dims else "")
            if not y_dim:
                return None
            y_values = np.asarray(data.coords[y_dim].values, dtype=float)
            profile_complex = np.asarray(data.values, dtype=complex).reshape(-1)
            if y_values.size == 0 or profile_complex.size == 0:
                return None
            if y_values.size != profile_complex.size:
                count = min(int(y_values.size), int(profile_complex.size))
                y_values = y_values[:count]
                profile_complex = profile_complex[:count]
            ref_idx = int(np.argmax(np.abs(profile_complex)))
            phase_ref = float(np.angle(profile_complex[ref_idx])) if profile_complex.size else 0.0
            aligned = profile_complex * np.exp(-1j * phase_ref)
            return y_values, aligned
        except Exception:
            return None

    def _plot_port_y_profiles(port_field_monitors: List[str]) -> None:
        import numpy as np

        profiles: Dict[str, tuple[Any, Any]] = {}
        for monitor_name in port_field_monitors:
            payload = _extract_phase_aligned_y_profile(monitor_name)
            if payload is None:
                continue
            port_name = monitor_name.replace("field_", "", 1)
            profiles[port_name] = payload
        if not profiles:
            return

        ports = sorted(profiles.keys())
        fig, axes = plt.subplots(len(ports), 1, figsize=(8.0, max(3.5, 2.8 * len(ports))), squeeze=False)
        for row_idx, port_name in enumerate(ports):
            y_values, aligned = profiles[port_name]
            ax = axes[row_idx, 0]
            ax.plot(y_values, np.real(aligned), color="tab:blue", label="Re(Ey, phase-aligned)")
            ax.plot(y_values, np.abs(aligned), color="tab:orange", linestyle="--", label="|Ey|")
            ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
            ax.set_ylabel("Field (a.u.)")
            ax.set_title(f"{component_type} - {port_name} y-profile (phase-aligned)")
            ax.legend(loc="best")
        axes[-1, 0].set_xlabel("y (um)")
        fig.tight_layout()

        out = PATH.build / f"tidy3d_field_profile_y_phase_aligned_{component_type}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        _record_artifact(artifact_sink, out)
        tagged = _save_tagged_copy(out, artifact_tag)
        if tagged is not None:
            _record_artifact(artifact_sink, tagged)
            lines.append(f"Saved tagged port y-profile plot: {tagged}")
        if wl_tag:
            out_wl = PATH.build / f"tidy3d_field_profile_y_phase_aligned_{wl_tag}_{component_type}.png"
            fig.savefig(out_wl, dpi=150, bbox_inches="tight")
            _record_artifact(artifact_sink, out_wl)
            tagged_wl = _save_tagged_copy(out_wl, artifact_tag)
            if tagged_wl is not None:
                _record_artifact(artifact_sink, tagged_wl)
                lines.append(f"Saved tagged wavelength port y-profile plot: {tagged_wl}")
            lines.append(f"Saved wavelength port y-profile plot: {out_wl}")
        plt.close(fig)
        lines.append(f"Saved port y-profile plot: {out}")

    # Field monitor — Ey in the XY plane
    try:
        if "field_monitor" in sim_data.monitor_data:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sim_data["field_monitor"].Ey.real.plot(x="x", y="y", ax=axes[0], cmap="RdBu")
            axes[0].set_title(f"{component_type} — Ey (real)")
            sim_data["field_monitor"].Ey.abs.plot(x="x", y="y", ax=axes[1], cmap="magma")
            axes[1].set_title(f"{component_type} — |Ey|")
            out = PATH.build / f"tidy3d_field_Ey_{component_type}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            _record_artifact(artifact_sink, out)
            tagged = _save_tagged_copy(out, artifact_tag)
            if tagged is not None:
                _record_artifact(artifact_sink, tagged)
                lines.append(f"Saved tagged field plot: {tagged}")
            if wl_tag:
                out_wl = PATH.build / f"tidy3d_field_Ey_{wl_tag}_{component_type}.png"
                fig.savefig(out_wl, dpi=150, bbox_inches="tight")
                _record_artifact(artifact_sink, out_wl)
                tagged_wl = _save_tagged_copy(out_wl, artifact_tag)
                if tagged_wl is not None:
                    _record_artifact(artifact_sink, tagged_wl)
                    lines.append(f"Saved tagged wavelength field plot: {tagged_wl}")
                lines.append(f"Saved wavelength field plot: {out_wl}")
            plt.close(fig)
            lines.append(f"Saved field plot: {out}")
    except Exception as fe:
        lines.append(f"Ey field plot failed: {fe}")

    # Explicit global z-normal field monitor plot for inspection requests.
    try:
        if "field_z" in sim_data.monitor_data:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sim_data["field_z"].Ey.real.plot(x="x", y="y", ax=axes[0], cmap="RdBu")
            axes[0].set_title(f"{component_type} — field_z Ey (real)")
            sim_data["field_z"].Ey.abs.plot(x="x", y="y", ax=axes[1], cmap="magma")
            axes[1].set_title(f"{component_type} — field_z |Ey|")
            out = PATH.build / f"tidy3d_field_Ey_field_z_{component_type}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            _record_artifact(artifact_sink, out)
            tagged = _save_tagged_copy(out, artifact_tag)
            if tagged is not None:
                _record_artifact(artifact_sink, tagged)
                lines.append(f"Saved tagged field_z plot: {tagged}")
            if wl_tag:
                out_wl = PATH.build / f"tidy3d_field_Ey_field_z_{wl_tag}_{component_type}.png"
                fig.savefig(out_wl, dpi=150, bbox_inches="tight")
                _record_artifact(artifact_sink, out_wl)
                tagged_wl = _save_tagged_copy(out_wl, artifact_tag)
                if tagged_wl is not None:
                    _record_artifact(artifact_sink, tagged_wl)
                    lines.append(f"Saved tagged wavelength field_z plot: {tagged_wl}")
                lines.append(f"Saved wavelength field_z plot: {out_wl}")
            plt.close(fig)
            lines.append(f"Saved field_z plot: {out}")
    except Exception as fe:
        lines.append(f"field_z plot failed: {fe}")

    try:
        port_field_monitors = sorted(
            name
            for name in sim_data.monitor_data
            if str(name).startswith("field_port_o")
        )
        for monitor_name in port_field_monitors:
            _plot_port_field_monitor(monitor_name)
        if port_field_monitors:
            _plot_port_y_profiles(port_field_monitors)
    except Exception as pe:
        lines.append(f"Output-port field plotting failed: {pe}")

    # Flux monitor bar chart
    try:
        flux_names = [n for n in sim_data.monitor_data if n.startswith("flux_")]
        if flux_names:
            flux_vals = {}
            for name in flux_names:
                flux_obj = getattr(sim_data[name], "flux", None)
                if flux_obj is not None:
                    flux_vals[name] = float(flux_obj.values.mean())
            if flux_vals:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(flux_vals.keys(), flux_vals.values(), color="steelblue")
                ax.set_ylabel("Mean flux (a.u.)")
                ax.set_title(f"{component_type} — Port flux")
                out = PATH.build / f"tidy3d_flux_{component_type}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                _record_artifact(artifact_sink, out)
                tagged = _save_tagged_copy(out, artifact_tag)
                if tagged is not None:
                    _record_artifact(artifact_sink, tagged)
                    lines.append(f"Saved tagged flux plot: {tagged}")
                if wl_tag:
                    out_wl = PATH.build / f"tidy3d_flux_{wl_tag}_{component_type}.png"
                    fig.savefig(out_wl, dpi=150, bbox_inches="tight")
                    _record_artifact(artifact_sink, out_wl)
                    tagged_wl = _save_tagged_copy(out_wl, artifact_tag)
                    if tagged_wl is not None:
                        _record_artifact(artifact_sink, tagged_wl)
                        lines.append(f"Saved tagged wavelength flux plot: {tagged_wl}")
                    lines.append(f"Saved wavelength flux plot: {out_wl}")
                plt.close(fig)
                lines.append(f"Saved flux plot: {out}")

                # Output-only flux chart (exclude the injected source port monitor).
                output_flux = {
                    name: value
                    for name, value in flux_vals.items()
                    if name.startswith("flux_port_o") and name != source_flux_key
                }
                if len(output_flux) >= 2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(output_flux.keys(), output_flux.values(), color="darkorange")
                    ax.set_ylabel("Mean flux (a.u.)")
                    ax.set_title(f"{component_type} — Output-port flux")
                    out_split = PATH.build / f"tidy3d_flux_output_only_{component_type}.png"
                    fig.savefig(out_split, dpi=150, bbox_inches="tight")
                    _record_artifact(artifact_sink, out_split)
                    tagged_split = _save_tagged_copy(out_split, artifact_tag)
                    if tagged_split is not None:
                        _record_artifact(artifact_sink, tagged_split)
                        lines.append(f"Saved tagged output-only flux plot: {tagged_split}")
                    if wl_tag:
                        out_split_wl = PATH.build / f"tidy3d_flux_output_only_{wl_tag}_{component_type}.png"
                        fig.savefig(out_split_wl, dpi=150, bbox_inches="tight")
                        _record_artifact(artifact_sink, out_split_wl)
                        tagged_split_wl = _save_tagged_copy(out_split_wl, artifact_tag)
                        if tagged_split_wl is not None:
                            _record_artifact(artifact_sink, tagged_split_wl)
                            lines.append(f"Saved tagged wavelength output-only flux plot: {tagged_split_wl}")
                        lines.append(f"Saved wavelength output-only flux plot: {out_split_wl}")
                    plt.close(fig)
                    lines.append(f"Saved output-only flux plot: {out_split}")

                # Input-normalized output coupling chart for acceptance semantics.
                input_flux = flux_vals.get(source_flux_key)
                if input_flux is None:
                    input_flux = flux_vals.get("flux_port_o1")
                if input_flux is not None and abs(float(input_flux)) > 1e-12 and len(output_flux) >= 1:
                    ratio_map = {
                        name: float(val) / float(input_flux)
                        for name, val in output_flux.items()
                    }
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(ratio_map.keys(), ratio_map.values(), color="seagreen")
                    ax.set_ylabel("Output/Input flux ratio")
                    ax.set_title(f"{component_type} - Output coupling ratio to input")
                    out_ratio = PATH.build / f"tidy3d_flux_ratio_to_input_{component_type}.png"
                    fig.savefig(out_ratio, dpi=150, bbox_inches="tight")
                    _record_artifact(artifact_sink, out_ratio)
                    tagged_ratio = _save_tagged_copy(out_ratio, artifact_tag)
                    if tagged_ratio is not None:
                        _record_artifact(artifact_sink, tagged_ratio)
                        lines.append(f"Saved tagged output/input ratio plot: {tagged_ratio}")
                    if wl_tag:
                        out_ratio_wl = PATH.build / f"tidy3d_flux_ratio_to_input_{wl_tag}_{component_type}.png"
                        fig.savefig(out_ratio_wl, dpi=150, bbox_inches="tight")
                        _record_artifact(artifact_sink, out_ratio_wl)
                        tagged_ratio_wl = _save_tagged_copy(out_ratio_wl, artifact_tag)
                        if tagged_ratio_wl is not None:
                            _record_artifact(artifact_sink, tagged_ratio_wl)
                            lines.append(f"Saved tagged wavelength output/input ratio plot: {tagged_ratio_wl}")
                        lines.append(f"Saved wavelength output/input ratio plot: {out_ratio_wl}")
                    plt.close(fig)
                    lines.append(f"Saved output/input ratio plot: {out_ratio}")
    except Exception as fe:
        lines.append(f"Flux bar chart failed: {fe}")

    try:
        import numpy as np

        mode_names = [name for name in sim_data.monitor_data if name.startswith("mode_")]
        mode_power_map: Dict[str, np.ndarray] = {}

        for name in mode_names:
            mon = sim_data[name]
            amps = getattr(mon, "amps", None)
            if amps is None:
                continue
            _preferred_dir = "-" if str(name) == "mode_port_o1" else "+"
            collapsed = _collapse_mode_power_from_amps(amps, preferred_direction=_preferred_dir)
            if collapsed is None:
                continue
            if collapsed.size == 0:
                continue
            mode_power_map[name] = collapsed

        if mode_power_map:
            monitor_names = sorted(mode_power_map.keys())
            max_modes = max(1, min(max(len(mode_power_map[name]) for name in monitor_names), 6))
            x = np.arange(len(monitor_names), dtype=float)
            width = 0.8 / float(max_modes)

            fig, ax = plt.subplots(figsize=(max(8.0, len(monitor_names) * 2.2), 4.8))
            for mode_idx in range(max_modes):
                values = [
                    float(mode_power_map[name][mode_idx]) if mode_idx < len(mode_power_map[name]) else 0.0
                    for name in monitor_names
                ]
                offset = (mode_idx - (max_modes - 1) / 2.0) * width
                ax.bar(x + offset, values, width=width, label=f"mode {mode_idx}")

            ax.set_xticks(x)
            ax.set_xticklabels(monitor_names, rotation=20, ha="right")
            ax.set_ylabel("Relative modal power (a.u.)")
            ax.set_title(f"{component_type} - Mode expansion")
            ax.legend(loc="best")
            out_mode = PATH.build / f"tidy3d_mode_expansion_{component_type}.png"
            fig.savefig(out_mode, dpi=150, bbox_inches="tight")
            _record_artifact(artifact_sink, out_mode)
            tagged_mode = _save_tagged_copy(out_mode, artifact_tag)
            if tagged_mode is not None:
                _record_artifact(artifact_sink, tagged_mode)
                lines.append(f"Saved tagged mode-expansion plot: {tagged_mode}")
            if wl_tag:
                out_mode_wl = PATH.build / f"tidy3d_mode_expansion_{wl_tag}_{component_type}.png"
                fig.savefig(out_mode_wl, dpi=150, bbox_inches="tight")
                _record_artifact(artifact_sink, out_mode_wl)
                tagged_mode_wl = _save_tagged_copy(out_mode_wl, artifact_tag)
                if tagged_mode_wl is not None:
                    _record_artifact(artifact_sink, tagged_mode_wl)
                    lines.append(f"Saved tagged wavelength mode-expansion plot: {tagged_mode_wl}")
                lines.append(f"Saved wavelength mode-expansion plot: {out_mode_wl}")
            plt.close(fig)
            lines.append(f"Saved mode-expansion plot: {out_mode}")
            mode_port_o1 = mode_power_map.get("mode_port_o1")
            if mode_port_o1 is not None and len(mode_port_o1) >= 1:
                mode_count_o1 = min(len(mode_port_o1), 6)
                x_o1 = np.arange(mode_count_o1, dtype=float)
                values_o1 = [float(mode_port_o1[idx]) for idx in range(mode_count_o1)]
                fig, ax = plt.subplots(figsize=(7.2, 4.2))
                ax.bar([f"mode {idx}" for idx in x_o1.astype(int)], values_o1, color="teal")
                ax.set_ylabel("Relative modal power (a.u.)")
                ax.set_title(f"{component_type} - mode_port_o1 decomposition")
                out_o1 = PATH.build / f"tidy3d_mode_expansion_port_o1_{component_type}.png"
                fig.savefig(out_o1, dpi=150, bbox_inches="tight")
                _record_artifact(artifact_sink, out_o1)
                tagged_o1 = _save_tagged_copy(out_o1, artifact_tag)
                if tagged_o1 is not None:
                    _record_artifact(artifact_sink, tagged_o1)
                    lines.append(f"Saved tagged mode-port_o1 decomposition plot: {tagged_o1}")
                if wl_tag:
                    out_o1_wl = PATH.build / f"tidy3d_mode_expansion_port_o1_{wl_tag}_{component_type}.png"
                    fig.savefig(out_o1_wl, dpi=150, bbox_inches="tight")
                    _record_artifact(artifact_sink, out_o1_wl)
                    tagged_o1_wl = _save_tagged_copy(out_o1_wl, artifact_tag)
                    if tagged_o1_wl is not None:
                        _record_artifact(artifact_sink, tagged_o1_wl)
                        lines.append(f"Saved tagged wavelength mode-port_o1 decomposition plot: {tagged_o1_wl}")
                    lines.append(f"Saved wavelength mode-port_o1 decomposition plot: {out_o1_wl}")
                plt.close(fig)
                lines.append(f"Saved mode-port_o1 decomposition plot: {out_o1}")

            # Per-port TE0/TE1 non-superposed components (outgoing direction only).
            focus_modes = max(1, min(max_modes, 2))
            if focus_modes >= 1:
                width_comp = 0.7 / float(focus_modes)
                fig, axes = plt.subplots(1, 2, figsize=(max(9.0, len(monitor_names) * 2.8), 4.6))

                for mode_idx in range(focus_modes):
                    values = [
                        float(mode_power_map[name][mode_idx]) if mode_idx < len(mode_power_map[name]) else 0.0
                        for name in monitor_names
                    ]
                    offset = (mode_idx - (focus_modes - 1) / 2.0) * width_comp
                    axes[0].bar(x + offset, values, width=width_comp, label=f"TE{mode_idx}")

                axes[0].set_xticks(x)
                axes[0].set_xticklabels(monitor_names, rotation=20, ha="right")
                axes[0].set_ylabel("Mode power (a.u.)")
                axes[0].set_title(f"{component_type} - Port TE components (output direction)")
                axes[0].legend(loc="best")

                purity_map: Dict[str, List[float]] = {}
                for name in monitor_names:
                    vals = [
                        float(mode_power_map[name][idx]) if idx < len(mode_power_map[name]) else 0.0
                        for idx in range(focus_modes)
                    ]
                    total = float(sum(vals))
                    if total <= 1e-30:
                        purity_map[name] = [0.0 for _ in range(focus_modes)]
                    else:
                        purity_map[name] = [float(v / total) for v in vals]

                for mode_idx in range(focus_modes):
                    values = [purity_map[name][mode_idx] for name in monitor_names]
                    offset = (mode_idx - (focus_modes - 1) / 2.0) * width_comp
                    axes[1].bar(x + offset, values, width=width_comp, label=f"TE{mode_idx} purity")

                axes[1].set_xticks(x)
                axes[1].set_xticklabels(monitor_names, rotation=20, ha="right")
                axes[1].set_ylim(0.0, 1.05)
                axes[1].set_ylabel("Purity")
                axes[1].set_title(f"{component_type} - Port TE purity (output direction)")
                axes[1].legend(loc="best")
                fig.tight_layout()

                out_mode_components = PATH.build / f"tidy3d_mode_components_{component_type}.png"
                fig.savefig(out_mode_components, dpi=150, bbox_inches="tight")
                _record_artifact(artifact_sink, out_mode_components)
                tagged_mode_components = _save_tagged_copy(out_mode_components, artifact_tag)
                if tagged_mode_components is not None:
                    _record_artifact(artifact_sink, tagged_mode_components)
                    lines.append(f"Saved tagged mode-components plot: {tagged_mode_components}")
                if wl_tag:
                    out_mode_components_wl = PATH.build / f"tidy3d_mode_components_{wl_tag}_{component_type}.png"
                    fig.savefig(out_mode_components_wl, dpi=150, bbox_inches="tight")
                    _record_artifact(artifact_sink, out_mode_components_wl)
                    tagged_mode_components_wl = _save_tagged_copy(out_mode_components_wl, artifact_tag)
                    if tagged_mode_components_wl is not None:
                        _record_artifact(artifact_sink, tagged_mode_components_wl)
                        lines.append(f"Saved tagged wavelength mode-components plot: {tagged_mode_components_wl}")
                    lines.append(f"Saved wavelength mode-components plot: {out_mode_components_wl}")
                plt.close(fig)
                lines.append(f"Saved mode-components plot: {out_mode_components}")

        for name in mode_names:
            _plot_mode_profile_monitor(str(name))
    except Exception as me:
        lines.append(f"Mode expansion plot failed: {me}")


def _sbend_polyslab(
    td,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    wg_width: float,
    wg_height: float,
    name: str,
    num_points: int = 40,
):
    """Build a smooth S-bend waveguide as a single td.PolySlab.

    The centreline follows a raised-cosine ``y(x) = y_start + dy*(1-cos(pi*t))/2``
    (t in [0,1]) so its slope is zero at both ends and joins horizontal I/O
    waveguides tangentially.  The polygon is the top edge traced forward plus the
    bottom edge traced back, offset by wg_width/2 — a non-self-intersecting closed
    loop extruded to (0, wg_height) along z (axis=2).
    """
    import numpy as np

    si = td.Medium(permittivity=3.45**2, name="silicon")
    t = np.linspace(0.0, 1.0, num_points)
    xs = x_start + (x_end - x_start) * t
    ys = y_start + (y_end - y_start) * (1.0 - np.cos(np.pi * t)) / 2.0

    half = wg_width / 2.0
    top = [(float(x), float(y) + half) for x, y in zip(xs, ys)]
    bottom = [(float(x), float(y) - half) for x, y in zip(xs, ys)]
    vertices = top + bottom[::-1]  # forward along top, back along bottom

    return td.Structure(
        geometry=td.PolySlab(
            vertices=vertices,
            axis=2,
            slab_bounds=(0.0, wg_height),
        ),
        medium=si,
        name=name,
    )


def create_waveguide_crossing(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    wg_length: float = 10.0,
) -> tuple:
    """Create a waveguide crossing structure.
    
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material (n ≈ 3.45 at 1550nm)
    si = td.Medium(permittivity=3.45**2, name="silicon")
    
    structures = []
    
    # Horizontal waveguide
    wg_h = td.Structure(
        geometry=td.Box(
            center=(0, 0, wg_height / 2),
            size=(wg_length * 2, wg_width, wg_height),
        ),
        medium=si,
        name="wg_horizontal",
    )
    structures.append(wg_h)
    
    # Vertical waveguide
    wg_v = td.Structure(
        geometry=td.Box(
            center=(0, 0, wg_height / 2),
            size=(wg_width, wg_length * 2, wg_height),
        ),
        medium=si,
        name="wg_vertical",
    )
    structures.append(wg_v)
    
    # Simulation domain size (with PML padding)
    Lx = wg_length * 2 + 2.0
    Ly = wg_length * 2 + 2.0
    Lz = 4.0
    
    # Source position (at left end of horizontal waveguide)
    src_center = (-wg_length + 1.0, 0, wg_height / 2)
    
    # Monitor positions (at all 4 ports)
    monitor_positions = [
        (-wg_length + 1.0, 0, "port_o1"),  # Left
        (wg_length - 1.0, 0, "port_o2"),   # Right
        (0, -wg_length + 1.0, "port_o3"),  # Bottom
        (0, wg_length - 1.0, "port_o4"),   # Top
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def create_simple_waveguide(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    wg_length: float = 10.0,
) -> tuple:
    """Create a simple straight waveguide.
    
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")
    
    structures = []
    
    # Straight waveguide
    wg = td.Structure(
        geometry=td.Box(
            center=(0, 0, wg_height / 2),
            size=(wg_length * 2, wg_width, wg_height),
        ),
        medium=si,
        name="waveguide",
    )
    structures.append(wg)
    
    # Simulation domain size
    Lx = wg_length * 2 + 2.0
    Ly = 4.0
    Lz = 3.0
    
    # Source position
    src_center = (-wg_length + 1.0, 0, wg_height / 2)
    
    # Monitor positions
    monitor_positions = [
        (-wg_length + 1.0, 0, "port_o1"),  # Input
        (wg_length - 1.0, 0, "port_o2"),   # Output
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


# ---------------------------------------------------------------------------
# Shared monitor-orientation helpers (single source of truth)
# ---------------------------------------------------------------------------

def _port_prop_axis(component_type: str, port_name: str) -> str:
    """Return the propagation axis of a port.

    ``'x'`` means light propagates along *x* (monitor plane perpendicular to x,
    i.e. ``size_x = 0``).  ``'y'`` means light propagates along *y*
    (``size_y = 0``).

    Only waveguide crossings have y-propagating ports (port_o3 / port_o4).
    All other component types propagate exclusively along x.
    """
    if component_type == "crossing" and port_name not in ("port_o1", "port_o2"):
        return "y"
    return "x"


def monitor_size_for_port(
    prop_axis: str, wg_width: float, wg_height: float,
) -> tuple:
    """Monitor size perpendicular to *prop_axis*.

    This is the **single source of truth** for flux / mode monitor sizing.
    ``tidy3d_runner``, ``_build_invdes_simulation``, and Step3 config
    generation all consume this helper to guarantee consistent orientation.
    """
    w = max(float(wg_width), 1e-3)
    h = max(float(wg_height), 1e-3)
    if prop_axis == "x":
        return (0.0, w, h)
    return (w, 0.0, h)


def source_size_for_port(
    prop_axis: str, wg_width: float, wg_height: float,
) -> tuple:
    """Mode source size perpendicular to *prop_axis*.

    Unlike monitors, the source cross-section should match the local port
    waveguide cross-section to avoid injecting energy into adjacent ports.
    """
    w = max(float(wg_width), 1e-3)
    h = max(float(wg_height), 1e-3)
    if prop_axis == "x":
        return (0.0, w, h)
    return (w, 0.0, h)


def _port_position_map(monitor_positions: list[tuple[float, float, str]]) -> Dict[str, tuple[float, float]]:
    mapping: Dict[str, tuple[float, float]] = {}
    for mx, my, name in monitor_positions:
        mapping[str(name).strip().lower()] = (float(mx), float(my))
    return mapping


def _normalize_port_name(raw_port: str | None, *, default: str = "port_o1") -> str:
    lowered = str(raw_port or "").strip().lower()
    if not lowered:
        return default
    if lowered.startswith("port_o"):
        return lowered
    m = re.search(r"(\d+)", lowered)
    if not m:
        return default
    try:
        return f"port_o{int(m.group(1))}"
    except ValueError:
        return default


def _default_source_direction_for_port(
    component_type: str,
    port_name: str,
    port_xy: tuple[float, float],
) -> str:
    axis = _port_prop_axis(component_type, port_name)
    if axis == "y":
        return "+" if float(port_xy[1]) <= 0 else "-"
    return "+" if float(port_xy[0]) <= 0 else "-"


def _port_waveguide_width(
    *,
    component_type: str,
    port_name: str,
    objective_metric: str,
    wg_width: float,
    mmi_width: float,
    params: Dict[str, Any] | None = None,
) -> float:
    cfg = params or {}
    lowered_metric = str(objective_metric or "").strip().lower()
    lowered_port = str(port_name or "").strip().lower()
    explicit_o1 = cfg.get("port_o1_wg_width")
    explicit_side = cfg.get("side_port_wg_width")
    try:
        if lowered_port == "port_o1" and explicit_o1 is not None:
            return max(float(explicit_o1), 1e-3)
        if lowered_port != "port_o1" and explicit_side is not None:
            return max(float(explicit_side), 1e-3)
    except (TypeError, ValueError):
        pass
    if component_type in {"mmi", "splitter"} and lowered_metric == "mux_routing" and lowered_port == "port_o1":
        candidate = max(float(wg_width) * 2.0, 1.0)
        upper = max(float(wg_width) + 0.05, float(mmi_width) - 0.2)
        return min(candidate, upper)
    return float(wg_width)


def run_tidy3d_simulation(
    component_type: str = "unknown",
    wavelength_nm: float = 1550.0,
    **kwargs  # Accept all parameters as keyword arguments
) -> Dict[str, Any]:
    """Run Tidy3D simulation for a photonic component.
    
    Args:
        component_type: Type of component (from component_detector)
        wavelength_nm: Central wavelength in nm
        **kwargs: Additional parameters (wg_width, wg_height, ring_radius, etc.)

    Returns:
        Dict with extracted metrics (flux per port + composite score).
        Empty dict on failure.
    """
    lines: list[str] = []
    generated_artifacts: list[str] = []
    sim_data: Any = None
    
    # Extract common parameters with defaults
    wg_width = kwargs.get("wg_width", 0.5)
    wg_height = kwargs.get("wg_height", 0.22)
    wg_length = kwargs.get("wg_length", 2.0)
    ring_radius = kwargs.get("ring_radius", 5.0)
    gap = kwargs.get("gap", 0.2)
    mmi_width = kwargs.get("mmi_width", 2.5)
    mmi_length = kwargs.get("mmi_length", 10.0)
    arm_length = kwargs.get("arm_length", 20.0)
    arm_separation = kwargs.get("arm_separation", 5.0)
    coupler_length = kwargs.get("coupler_length", 10.0)
    grating_period = kwargs.get("grating_period", 0.62)
    num_periods = kwargs.get("num_periods", 20)
    rotation_length = kwargs.get("rotation_length", 30.0)
    swg_period = kwargs.get("swg_period", 0.4)
    run_time_s = float(kwargs.get("run_time_s", 1e-12))
    min_steps_per_wvl = int(kwargs.get("min_steps_per_wvl", 20))
    artifact_tag = _sanitize_artifact_tag(kwargs.get("artifact_tag"))
    topology_density_path = kwargs.get("topology_density_path")
    topology_density_meta_path = kwargs.get("topology_density_meta_path")
    require_field_plot = bool(kwargs.get("require_field_plot", False))
    objective_metric = str(kwargs.get("objective_metric", "") or "").strip().lower()
    target_ports_raw = kwargs.get("target_ports", [])
    target_mode_indices_raw = kwargs.get("target_mode_indices", [])
    source_port_raw = kwargs.get("source_port")
    source_mode_index_raw = kwargs.get("source_mode_index")
    source_direction_raw = kwargs.get("source_direction")
    skip_cloud = bool(kwargs.get("skip_cloud", False))

    target_ports: List[str] = []
    if isinstance(target_ports_raw, (list, tuple)):
        for item in target_ports_raw:
            port_name = str(item or "").strip().lower()
            if port_name and port_name not in target_ports:
                target_ports.append(port_name)

    target_mode_indices: List[int] = []
    if isinstance(target_mode_indices_raw, (list, tuple)):
        for item in target_mode_indices_raw:
            try:
                target_mode_indices.append(max(int(item), 0))
            except (TypeError, ValueError):
                continue
    required_mode_count = max(1, max(target_mode_indices) + 1) if target_mode_indices else 1
    source_port = _normalize_port_name(source_port_raw, default="port_o1")
    try:
        source_mode_index = max(int(source_mode_index_raw), 0)
    except (TypeError, ValueError):
        source_mode_index = 0
    source_direction = str(source_direction_raw or "").strip()
    
    try:
        import tidy3d as td
        
        lines.append("import tidy3d as td  # OK")
        
        wavelength_um = wavelength_nm / 1000.0
        port_o1_width = _port_waveguide_width(
            component_type=component_type,
            port_name="port_o1",
            objective_metric=objective_metric,
            wg_width=wg_width,
            mmi_width=mmi_width,
            params=kwargs,
        )
        mmi_num_outputs = max(int(round(float(kwargs.get("mmi_num_outputs", 2) or 2))), 2)
        output_wg_widths = [
            _port_waveguide_width(
                component_type=component_type,
                port_name=f"port_o{idx + 2}",
                objective_metric=objective_metric,
                wg_width=wg_width,
                mmi_width=mmi_width,
                params=kwargs,
            )
            for idx in range(mmi_num_outputs)
        ]
        
        # Create component geometry based on type
        replay_density, replay_sx, replay_sy = _load_density_replay_payload(
            topology_density_path,
            topology_density_meta_path,
            lines,
        )
        if topology_density_path and replay_density is None:
            lines.append("Topology density replay failed; refusing to run fixed-geometry diagnostic.")
            _append_log(lines)
            payload = {
                "_error": "topology_density_replay_failed",
                "_artifacts": sorted(set(generated_artifacts)),
            }
            if artifact_tag:
                payload["_artifact_tag"] = artifact_tag
            return payload
        if component_type in {"mmi", "splitter"} and replay_density is not None:
            base_structures, base_sim_size, src_center, monitor_positions = create_mmi(
                td,
                wavelength_um,
                wg_width,
                wg_height,
                mmi_width,
                mmi_length,
                num_outputs=mmi_num_outputs,
                input_wg_width=port_o1_width,
                output_wg_widths=output_wg_widths,
            )
            structures = [s for s in base_structures if getattr(s, "name", "") != "mmi_section"]
            region_sx = float(replay_sx) if replay_sx is not None else float(max(mmi_length, 2.0))
            region_sy = float(replay_sy) if replay_sy is not None else float(max(mmi_width, 2.0))
            topo_structs = _density_pixel_structures(
                td,
                replay_density,
                region_sx=region_sx,
                region_sy=region_sy,
                wg_height=wg_height,
                threshold=float(kwargs.get("topology_threshold", 0.5)),
                max_pixels=int(kwargs.get("topology_max_pixels", 4096)),
            )
            structures.extend(topo_structs)
            Lx, Ly, Lz = base_sim_size
            Ly = max(float(Ly), region_sy + 4.0)
            sim_size = (Lx, Ly, Lz)
            lines.append(
                f"Created topology replay MMI: {len(topo_structs)} density pixels, "
                f"region=({region_sx:.3f},{region_sy:.3f})um"
            )
        elif component_type == "crossing":
            structures, sim_size, src_center, monitor_positions = create_waveguide_crossing(
                td, wavelength_um, wg_width, wg_height, wg_length
            )
            lines.append(f"Created waveguide crossing: {len(structures)} structures")
        elif component_type == "ring_resonator":
            structures, sim_size, src_center, monitor_positions = create_ring_resonator(
                td, wavelength_um, wg_width, wg_height, ring_radius, gap
            )
            lines.append(f"Created ring resonator (R={ring_radius}um, gap={gap}um): {len(structures)} structures")
        elif component_type == "mmi" or component_type == "splitter":
            structures, sim_size, src_center, monitor_positions = create_mmi(
                td,
                wavelength_um,
                wg_width,
                wg_height,
                mmi_width,
                mmi_length,
                num_outputs=mmi_num_outputs,
                input_wg_width=port_o1_width,
                output_wg_widths=output_wg_widths,
            )
            lines.append(f"Created MMI (W={mmi_width}um, L={mmi_length}um): {len(structures)} structures")
        elif component_type == "mzi":
            structures, sim_size, src_center, monitor_positions = create_mzi(
                td, wavelength_um, wg_width, wg_height, arm_length, arm_separation
            )
            lines.append(f"Created MZI (arm={arm_length}um): {len(structures)} structures")
        elif component_type == "directional_coupler":
            structures, sim_size, src_center, monitor_positions = create_coupler(
                td, wavelength_um, wg_width, wg_height, coupler_length, gap
            )
            lines.append(f"Created directional coupler (L={coupler_length}um, gap={gap}um): {len(structures)} structures")
        elif component_type == "grating_coupler":
            structures, sim_size, src_center, monitor_positions = create_grating_coupler(
                td, wavelength_um, wg_width, wg_height, grating_period, num_periods
            )
            lines.append(f"Created grating coupler: {len(structures)} structures")
        elif component_type == "polarization_rotator":
            structures, sim_size, src_center, monitor_positions = create_polarization_rotator(
                td, wavelength_um, wg_width, wg_height, rotation_length, swg_period
            )
            lines.append(f"Created polarization rotator: {len(structures)} structures")
        elif component_type == "y_branch":
            structures, sim_size, src_center, monitor_positions = create_y_branch(
                td, wavelength_um, wg_width, wg_height, arm_length, arm_separation
            )
            lines.append(f"Created y-branch splitter: {len(structures)} structures")
        elif component_type == "waveguide":
            structures, sim_size, src_center, monitor_positions = create_simple_waveguide(
                td, wavelength_um, wg_width, wg_height, wg_length
            )
            lines.append(f"Created simple waveguide: {len(structures)} structures")
        else:
            # Default: create simple waveguide
            structures, sim_size, src_center, monitor_positions = create_simple_waveguide(
                td, wavelength_um, wg_width, wg_height, wg_length
            )
            lines.append(f"Created default waveguide (type={component_type}): {len(structures)} structures")
        
        Lx, Ly, Lz = sim_size
        port_positions = _port_position_map(monitor_positions)
        if source_port not in port_positions and "port_o1" in port_positions:
            source_port = "port_o1"
        source_xy = port_positions.get(source_port, (float(src_center[0]), float(src_center[1])))
        if source_direction not in {"+", "-"}:
            source_direction = _default_source_direction_for_port(component_type, source_port, source_xy)
        source_axis = _port_prop_axis(component_type, source_port)
        source_width = _port_waveguide_width(
            component_type=component_type,
            port_name=source_port,
            objective_metric=objective_metric,
            wg_width=wg_width,
            mmi_width=mmi_width,
            params=kwargs,
        )
        source_size = source_size_for_port(source_axis, source_width, wg_height)
        src_center = (float(source_xy[0]), float(source_xy[1]), wg_height / 2)
        source_flux_key = f"flux_{source_port}"

        # Grid specification
        grid_spec = td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=max(min_steps_per_wvl, 10),
        )
        
        # Mode source
        try:
            # Create mode source
            # filter_pol='te' guarantees mode_index=0 is TE0 regardless of cross-section;
            # without it, substrate or TM modes may sort before TE0 for non-standard geometries.
            mode_spec = td.ModeSpec(
                num_modes=required_mode_count,
                filter_pol="te",
            )
            
            # Calculate frequency from wavelength (freq = c / wavelength)
            # td.C_0 is speed of light in um/s (~2.998e14), wavelength already in um
            freq0 = td.C_0 / wavelength_um
            fwidth = freq0 * 0.1  # 10% bandwidth
            
            pulse = td.GaussianPulse(
                freq0=freq0,
                fwidth=fwidth,
            )
            
            source = td.ModeSource(
                source_time=pulse,
                center=src_center,
                size=source_size,
                mode_spec=mode_spec,
                mode_index=source_mode_index,
                direction=source_direction,
                name="mode_source",
            )
            lines.append("td.ModeSource created")
        except Exception as se:
            lines.append(f"Source setup failed: {se}")
            source = None
        
        # Monitors
        monitors = []
        
        # Field monitor at z=0
        try:
            freqs = [td.C_0 / wavelength_um]
            
            field_mon = td.FieldMonitor(
                center=(0, 0, wg_height / 2),
                size=(Lx * 0.9, Ly * 0.9, 0),
                freqs=freqs,
                name="field_monitor",
            )
            monitors.append(field_mon)
            lines.append("td.FieldMonitor created")

            field_z = td.FieldMonitor(
                center=(0, 0, wg_height / 2),
                size=(Lx * 0.9, Ly * 0.9, 0),
                freqs=freqs,
                name="field_z",
            )
            monitors.append(field_z)
            lines.append("td.FieldMonitor(field_z) created")
        except Exception as me:
            lines.append(f"Field monitor failed: {me}")
        
        # Flux monitors at each port — use shared helper for orientation
        for mx, my, mname in monitor_positions:
            try:
                port_name = str(mname).strip().lower()
                axis = _port_prop_axis(component_type, port_name)
                port_width = _port_waveguide_width(
                    component_type=component_type,
                    port_name=port_name,
                    objective_metric=objective_metric,
                    wg_width=wg_width,
                    mmi_width=mmi_width,
                    params=kwargs,
                )
                msize = monitor_size_for_port(axis, port_width, wg_height)

                flux_mon = td.FluxMonitor(
                    center=(mx, my, wg_height / 2),
                    size=msize,
                    freqs=freqs,
                    name=f"flux_{mname}",
                )
                monitors.append(flux_mon)
                lines.append(f"td.FluxMonitor({mname}) axis={axis} created")
            except Exception as fe:
                lines.append(f"Flux monitor {mname} failed: {fe}")

        demux_like_metrics = {"demux_routing", "mode_demux", "wdm_routing", "mux_routing"}
        force_output_observability = objective_metric in demux_like_metrics
        output_observability_targets: set[str] = set()
        if force_output_observability:
            for _, _, name in monitor_positions:
                port_name = str(name).strip().lower()
                if not port_name.startswith("port_o"):
                    continue
                if objective_metric == "mux_routing":
                    if port_name == source_port:
                        continue
                    output_observability_targets.add(port_name)
                elif port_name != "port_o1":
                    output_observability_targets.add(port_name)
        if target_ports:
            output_observability_targets.update(
                item for item in target_ports if item.startswith("port_o")
            )
        if any(str(name).strip().lower() == "port_o1" for _, _, name in monitor_positions):
            output_observability_targets.add("port_o1")
        if objective_metric == "mux_routing":
            output_observability_targets.add("port_o1")

        if output_observability_targets:
            for mx, my, mname in monitor_positions:
                name = str(mname)
                if name not in output_observability_targets:
                    continue
                try:
                    axis = _port_prop_axis(component_type, name)
                    port_width = _port_waveguide_width(
                        component_type=component_type,
                        port_name=name,
                        objective_metric=objective_metric,
                        wg_width=wg_width,
                        mmi_width=mmi_width,
                        params=kwargs,
                    )
                    msize = monitor_size_for_port(axis, port_width, wg_height)
                    port_field_mon = td.FieldMonitor(
                        center=(mx, my, wg_height / 2),
                        size=msize,
                        freqs=freqs,
                        name=f"field_{name}",
                    )
                    monitors.append(port_field_mon)
                    lines.append(f"td.FieldMonitor({name}) axis={axis} created")
                except Exception as fme:
                    lines.append(f"Output field monitor {name} failed: {fme}")

        # Mode monitors on output ports are required for mode-demux observability.
        # When objective asks for demux/mode-demux, we always include per-output
        # mode monitors so TE0/TE1 evidence can be rendered as mode-expansion.
        if force_output_observability or output_observability_targets:
            store_mode_fields = _env_bool("TIDY3D_STORE_MODE_FIELDS", default=bool(force_output_observability))
            for mx, my, mname in monitor_positions:
                name = str(mname)
                if name not in output_observability_targets:
                    continue
                try:
                    axis = _port_prop_axis(component_type, name)
                    port_width = _port_waveguide_width(
                        component_type=component_type,
                        port_name=name,
                        objective_metric=objective_metric,
                        wg_width=wg_width,
                        mmi_width=mmi_width,
                        params=kwargs,
                    )
                    msize = monitor_size_for_port(axis, port_width, wg_height)
                    mode_mon = td.ModeMonitor(
                        center=(mx, my, wg_height / 2),
                        size=msize,
                        freqs=freqs,
                        mode_spec=td.ModeSpec(num_modes=required_mode_count, filter_pol="te"),
                        store_fields_direction="+" if store_mode_fields else None,
                        name=f"mode_{name}",
                    )
                    monitors.append(mode_mon)
                    lines.append(
                        f"td.ModeMonitor({name}) axis={axis} created "
                        f"(num_modes={required_mode_count}, "
                        f"store_fields_direction={'+' if store_mode_fields else 'none'})"
                    )
                except Exception as me:
                    lines.append(f"Mode monitor {name} failed: {me}")
        
        # Create simulation
        sim = td.Simulation(
            size=(Lx, Ly, Lz),
            run_time=run_time_s,
            boundary_spec=td.BoundarySpec.all_sides(td.PML()),
            medium=td.Medium(permittivity=1.44**2, name="silica"),  # SiO2 substrate
            grid_spec=grid_spec,
            structures=structures,
            sources=[source] if source else [],
            monitors=monitors,
        )
        lines.append("td.Simulation created with structures")

        # Persist the Simulation object so MCP viewer / post-hoc plotting can load it
        sim_hdf5 = PATH.build / f"tidy3d_sim_{component_type}.hdf5"
        try:
            sim.to_hdf5(str(sim_hdf5))
            _record_artifact(generated_artifacts, sim_hdf5)
            tagged_sim_hdf5 = _save_tagged_copy(sim_hdf5, artifact_tag)
            if tagged_sim_hdf5 is not None:
                _record_artifact(generated_artifacts, tagged_sim_hdf5)
                lines.append(f"Saved tagged simulation object: {tagged_sim_hdf5}")
            lines.append(f"Saved simulation object: {sim_hdf5}")
        except Exception as se:
            lines.append(f"Simulation HDF5 save failed: {se}")

        # Generate a tiny Python loader script for the MCP 3D viewer
        viewer_script = PATH.build / f"tidy3d_viewer_{component_type}.py"
        try:
            viewer_script.write_text(
                f'"""Auto-generated viewer script for {component_type}."""\n'
                f"import tidy3d as td\n\n"
                f'sim = td.Simulation.from_hdf5(r"{sim_hdf5}")\n',
                encoding="utf-8",
            )
            _record_artifact(generated_artifacts, viewer_script)
            tagged_viewer_script = _save_tagged_copy(viewer_script, artifact_tag)
            if tagged_viewer_script is not None:
                _record_artifact(generated_artifacts, tagged_viewer_script)
                lines.append(f"Saved tagged viewer script: {tagged_viewer_script}")
            lines.append(f"Saved viewer script: {viewer_script}")
        except Exception as ve:
            lines.append(f"Viewer script save failed: {ve}")

        # Plot simulation cross-sections with component-specific filenames
        sim_time = datetime.now().strftime('%H:%M:%S')
        
        try:
            # XY plane at waveguide center
            fig, ax = plt.subplots(figsize=(10, 8))
            sim.plot(z=wg_height / 2, ax=ax)
            ax.set_title(f"{component_type.upper()} - XY Plane (z={wg_height/2:.2f}μm) [{sim_time}]")
            out_png = PATH.build / f"tidy3d_sim_z0_{component_type}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            _record_artifact(generated_artifacts, out_png)
            tagged_z = _save_tagged_copy(out_png, artifact_tag)
            if tagged_z is not None:
                _record_artifact(generated_artifacts, tagged_z)
                lines.append(f"Saved tagged {tagged_z}")
            plt.close(fig)
            lines.append(f"Saved {out_png}")
        except Exception as pe:
            lines.append(f"XY plot failed: {pe}")
        
        try:
            # XZ plane (y=0)
            fig, ax = plt.subplots(figsize=(12, 4))
            sim.plot(y=0, ax=ax)
            ax.set_title(f"{component_type.upper()} - XZ Plane (y=0) [{sim_time}]")
            out_png = PATH.build / f"tidy3d_sim_x0_{component_type}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            _record_artifact(generated_artifacts, out_png)
            tagged_x = _save_tagged_copy(out_png, artifact_tag)
            if tagged_x is not None:
                _record_artifact(generated_artifacts, tagged_x)
                lines.append(f"Saved tagged {tagged_x}")
            plt.close(fig)
            lines.append(f"Saved {out_png}")
        except Exception as pe:
            lines.append(f"XZ plot failed: {pe}")
        
        try:
            # YZ plane (x=0)
            fig, ax = plt.subplots(figsize=(4, 8))
            sim.plot(x=0, ax=ax)
            ax.set_title(f"{component_type.upper()} - YZ Plane (x=0) [{sim_time}]")
            out_png = PATH.build / f"tidy3d_sim_y0_{component_type}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            _record_artifact(generated_artifacts, out_png)
            tagged_y = _save_tagged_copy(out_png, artifact_tag)
            if tagged_y is not None:
                _record_artifact(generated_artifacts, tagged_y)
                lines.append(f"Saved tagged {tagged_y}")
            plt.close(fig)
            lines.append(f"Saved {out_png}")
        except Exception as pe:
            lines.append(f"YZ plot failed: {pe}")
        
        # Run simulation - try local mode first, then cloud
        use_local = os.getenv("TIDY3D_LOCAL", "1") == "1"  # Default to local mode
        
        if use_local:
            # Local FDTD simulation (no API key needed)
            lines.append("Running local FDTD simulation...")
            try:
                import tidy3d as td
                from tidy3d import web as _td_web
                
                # Run locally — API changed between tidy3d versions:
                # ≥2.7: web.run_local / web.run (with local kwarg)
                # ≤2.6: td.webapi.webapi.run_local
                _data_path = str(PATH.build / "tidy3d_data.hdf5")
                _task_name = f"local_{component_type}"
                _local_ran = False
                for _try_fn in (
                    lambda: _td_web.run_local(sim, task_name=_task_name, path=_data_path),
                    lambda: _td_web.run(sim, task_name=_task_name, path=_data_path, local=True),
                    lambda: getattr(td.webapi, "webapi", td.webapi).run_local(
                        sim, task_name=_task_name, path=_data_path
                    ),
                ):
                    try:
                        sim_data = _try_fn()
                        _local_ran = True
                        break
                    except (AttributeError, TypeError):
                        continue
                    except Exception:
                        raise
                if not _local_ran:
                    raise RuntimeError("No local tidy3d solver available in this installation.")
                lines.append("Local simulation completed!")
                result = "success"
                data = sim_data
                    
            except Exception as le:
                lines.append(f"Local run failed: {le}")
                lines.append("Falling back to cloud mode...")
                use_local = False
        
        if not use_local:
            # Cloud simulation (requires API key)
            # Refresh ONLY Tidy3D auth keys from .env. Do NOT call
            # ``load_dotenv(override=True)`` here — it would silently
            # clobber launcher-level toggles (e.g. INVERSE_STEP5_*)
            # and reintroduce the M48-class side-effect bug where
            # auto-resume / completed-state flags came back from .env
            # after the launcher cleared them.
            try:
                from dotenv import dotenv_values
                _vals = dotenv_values() or {}
                for _k in ("TIDY3D_API_KEY", "SIMCLOUD_APIKEY", "FLEXCOMPUTE_API_KEY"):
                    _v = _vals.get(_k)
                    if _v:
                        os.environ[_k] = str(_v).strip()
            except ImportError:
                pass
            api_key = os.getenv("TIDY3D_API_KEY")
            if skip_cloud:
                lines.append("skip_cloud=True - cloud simulation skipped")
            elif api_key:
                try:
                    from tidy3d import web
                    
                    web.configure(apikey=api_key)
                    lines.append("Configured Tidy3D API")
                    
                    task_name = f"PhIDO-{component_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    lines.append(f"Starting cloud run: {task_name}")
                    data_path = str(PATH.build / "tidy3d_data.hdf5")
                    max_load_retries = max(1, int(os.getenv("TIDY3D_LOAD_RETRIES", "3")))
                    load_retry_sleep_s = max(
                        0.5,
                        float(os.getenv("TIDY3D_LOAD_RETRY_SLEEP_S", "8.0")),
                    )
                    
                    # On Windows, rich/tqdm progress bars can hang indefinitely when
                    # stdout is not a real TTY (e.g. CMD file redirect).  The old
                    # StringIO-redirect workaround caused web.load() to also hang.
                    # Simplest fix: disable verbose for all web calls on Windows so
                    # rich never tries to render anything.
                    _web_verbose = sys.platform != "win32"

                    # Use explicit task lifecycle so data-download retries do
                    # not re-submit the simulation task and burn extra credits.
                    task_id = web.upload(
                        simulation=sim,
                        task_name=task_name,
                        folder_name="default",
                        verbose=_web_verbose,
                    )
                    lines.append(f"Uploaded cloud task: {task_id}")
                    web.start(task_id=task_id)
                    lines.append("Cloud task started.")
                    web.monitor(task_id=task_id, verbose=_web_verbose)
                    lines.append("Monitor returned.")

                    # Confirm the task reached a terminal state before downloading.
                    _TERMINAL_STATUSES = {"success", "error", "diverged", "deleted", "timeout"}
                    _STATUS_POLL_SLEEP_S = 15
                    _STATUS_POLL_MAX = 400  # up to ~100 min
                    _task_status = "unknown"
                    for _sp in range(_STATUS_POLL_MAX):
                        try:
                            _task_info = web.get_info(task_id=task_id, verbose=False)
                            _task_status = (getattr(_task_info, "status", None) or "").lower()
                        except Exception:
                            _task_status = "unknown"
                        if _task_status in _TERMINAL_STATUSES or _task_status == "unknown":
                            break
                        if _sp == 0:
                            lines.append(
                                f"Task {task_id} still running after monitor() returned "
                                f"(status={_task_status}); polling until done."
                            )
                        time.sleep(_STATUS_POLL_SLEEP_S)
                    lines.append(f"Task final status: {_task_status}")
                    if _task_status not in {"success", "unknown"}:
                        raise RuntimeError(
                            f"Cloud task did not succeed: task_id={task_id}, status={_task_status}"
                        )

                    sim_data = None
                    for attempt in range(1, max_load_retries + 1):
                        try:
                            sim_data = web.load(
                                task_id=task_id,
                                path=data_path,
                                replace_existing=True,
                                verbose=_web_verbose,
                            )
                            break
                        except Exception as load_exc:
                            msg = str(load_exc)
                            lines.append(
                                f"Cloud data load attempt {attempt}/{max_load_retries} failed: {msg}"
                            )
                            msg_lower = msg.lower()
                            is_download_failure = (
                                "failed to download the simulation data file" in msg_lower
                                or "max retries exceeded" in msg_lower
                                or "connectionerror" in msg_lower
                                or "connection aborted" in msg_lower
                                or "read timed out" in msg_lower
                            )
                            if is_download_failure and attempt < max_load_retries:
                                backoff_s = load_retry_sleep_s * attempt
                                lines.append(
                                    f"Retrying cloud data download in {backoff_s:.1f}s "
                                    f"(task_id={task_id})."
                                )
                                time.sleep(backoff_s)
                                continue
                            raise
                    if sim_data is None:
                        raise RuntimeError("Cloud task finished but no SimulationData was loaded.")
                    data = sim_data
                    result = "success"
                    lines.append("Cloud simulation completed!")
                    
                except Exception as we:
                    lines.append(f"Cloud run failed: {we}")
            else:
                lines.append("No TIDY3D_API_KEY - simulation skipped")
        
        # Save config
        config = {
            "component_type": component_type,
            "wavelength_nm": wavelength_nm,
            "waveguide": {
                "width_um": wg_width,
                "height_um": wg_height,
                "length_um": wg_length,
            },
            "simulation_size_um": [Lx, Ly, Lz],
        }
        config_path = PATH.build / "tidy3d_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        _record_artifact(generated_artifacts, config_path)
        lines.append(f"Saved config: {config_path}")
        
        # Extract quantitative metrics from simulation results
        if sim_data is None:
            lines.append("Simulation produced no SimulationData object; metrics unavailable.")
            _append_log(lines)
            payload = {
                "_error": "simulation_data_unavailable",
                "_artifacts": sorted(set(generated_artifacts)),
            }
            if artifact_tag:
                payload["_artifact_tag"] = artifact_tag
            return payload

        metrics = _extract_metrics(sim_data)
        if metrics:
            lines.append(f"Metrics: score={metrics.get('score', 0):.4f}, flux={metrics.get('flux', {})}")
        data_hdf5 = PATH.build / "tidy3d_data.hdf5"
        _record_artifact(generated_artifacts, data_hdf5)
        tagged_data_hdf5 = _save_tagged_copy(data_hdf5, artifact_tag)
        if tagged_data_hdf5 is not None:
            _record_artifact(generated_artifacts, tagged_data_hdf5)
            lines.append(f"Saved tagged simulation data: {tagged_data_hdf5}")

        # Generate richer field/flux plots from sim_data for Streamlit display
        # Only when require_field_plot=True to avoid expensive mode-profile
        # plots (local ModeSolver on complex topologies can take 10+ min).
        if sim_data is not None and require_field_plot:
            _plot_sim_results(
                sim_data,
                component_type,
                wg_height,
                lines,
                wavelength_nm=wavelength_nm,
                artifact_tag=artifact_tag,
                artifacts=generated_artifacts,
                source_flux_key=source_flux_key,
            )

        _append_log(lines)
        metrics["_artifacts"] = sorted(set(generated_artifacts))
        if artifact_tag:
            metrics["_artifact_tag"] = artifact_tag
        return metrics
        
    except Exception as e:
        lines.append(f"Error: {type(e).__name__}: {e}")
        _append_log(lines)
        return {}


def run_tidy3d_from_config(config: Dict[str, Any]) -> None:
    """Run Tidy3D simulation from config dict."""
    component_type = config.get("component_type", "crossing")
    wavelength_nm = config.get("wavelengths", {}).get("central_nm", 1550.0)
    
    run_tidy3d_simulation(
        component_type=component_type,
        wavelength_nm=wavelength_nm,
    )


def create_ring_resonator(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    ring_radius: float = 5.0,
    gap: float = 0.2,
) -> tuple:
    """Create a ring resonator structure.
    
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    import numpy as np
    
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")
    
    structures = []
    
    # Straight bus waveguide (horizontal)
    bus_wg = td.Structure(
        geometry=td.Box(
            center=(0, -ring_radius - gap - wg_width/2, wg_height / 2),
            size=(ring_radius * 4, wg_width, wg_height),
        ),
        medium=si,
        name="bus_waveguide",
    )
    structures.append(bus_wg)
    
    # Ring resonator (approximated as a bent waveguide - we create 4 arc segments)
    # For simplicity, we'll create a ring using a cylinder
    ring = td.Structure(
        geometry=td.Cylinder(
            center=(0, 0, wg_height / 2),
            radius=ring_radius + wg_width/2,
            length=wg_height,
            axis=2,  # Z axis
        ),
        medium=si,
        name="ring_outer",
    )
    structures.append(ring)
    
    # Create inner hole to make it a ring
    inner_ring = td.Structure(
        geometry=td.Cylinder(
            center=(0, 0, wg_height / 2),
            radius=ring_radius - wg_width/2,
            length=wg_height * 1.1,
            axis=2,
        ),
        medium=td.Medium(permittivity=1.44**2, name="silica"),  # SiO2 for cladding
        name="ring_inner",
    )
    structures.append(inner_ring)
    
    # Simulation domain size
    Lx = ring_radius * 4 + 4.0
    Ly = ring_radius * 3 + 4.0
    Lz = 4.0
    
    # Source position (at left end of bus waveguide)
    src_center = (-ring_radius * 2 + 1.0, -ring_radius - gap - wg_width/2, wg_height / 2)
    
    # Monitor positions
    monitor_positions = [
        (-ring_radius * 2 + 1.0, -ring_radius - gap - wg_width/2, "port_o1"),  # Input
        (ring_radius * 2 - 1.0, -ring_radius - gap - wg_width/2, "port_o2"),   # Through
        (ring_radius * 2 - 1.0, ring_radius + gap + wg_width/2, "port_o3"),    # Drop (if exists)
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def create_mmi(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    mmi_width: float = 2.5,
    mmi_length: float = 10.0,
    num_inputs: int = 1,
    num_outputs: int = 2,
    input_wg_width: float | None = None,
    output_wg_widths: List[float] | None = None,
) -> tuple:
    """Create an MMI (Multi-Mode Interference) splitter.
    
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")
    
    structures = []
    
    # MMI section
    mmi = td.Structure(
        geometry=td.Box(
            center=(0, 0, wg_height / 2),
            size=(mmi_length, mmi_width, wg_height),
        ),
        medium=si,
        name="mmi_section",
    )
    structures.append(mmi)
    
    input_width = float(input_wg_width) if input_wg_width is not None else float(wg_width)

    # Input waveguide
    input_wg = td.Structure(
        geometry=td.Box(
            center=(-mmi_length/2 - 5, 0, wg_height / 2),
            size=(10, input_width, wg_height),
        ),
        medium=si,
        name="input_wg",
    )
    structures.append(input_wg)
    
    # Output waveguides
    output_spacing = mmi_width / (num_outputs + 1)
    monitor_positions = []
    
    for i in range(num_outputs):
        y_pos = -mmi_width/2 + output_spacing * (i + 1)
        if output_wg_widths and i < len(output_wg_widths):
            output_width = float(output_wg_widths[i])
        else:
            output_width = float(wg_width)
        out_wg = td.Structure(
            geometry=td.Box(
                center=(mmi_length/2 + 5, y_pos, wg_height / 2),
                size=(10, output_width, wg_height),
            ),
            medium=si,
            name=f"output_wg_{i}",
        )
        structures.append(out_wg)
        monitor_positions.append((mmi_length/2 + 9, y_pos, f"port_o{i+2}"))
    
    # Source position
    src_center = (-mmi_length/2 - 9, 0, wg_height / 2)
    monitor_positions.insert(0, (-mmi_length/2 - 9, 0, "port_o1"))  # Input monitor
    
    # Simulation domain size
    Lx = mmi_length + 24
    Ly = mmi_width + 4
    Lz = 4.0

    # SiO2 BOX/substrate slab — same permittivity as background medium (silica);
    # adding as an explicit named structure ensures the SOI buried-oxide layer
    # appears unambiguously in viewer cross-sections and structure lists.
    sio2 = td.Medium(permittivity=1.44**2, name="SiO2")
    sio2_substrate = td.Structure(
        geometry=td.Box(
            center=(0, 0, -Lz / 4),
            size=(Lx + 4, Ly + 4, Lz / 2),
        ),
        medium=sio2,
        name="SiO2_substrate",
    )
    structures.insert(0, sio2_substrate)

    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def create_mzi(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    arm_length: float = 20.0,
    arm_separation: float = 5.0,
) -> tuple:
    """Create an MZI (Mach-Zehnder Interferometer) structure.
    
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")

    structures = []

    # Input waveguide (y=0), right edge at x = -arm_length/2
    input_wg = td.Structure(
        geometry=td.Box(
            center=(-arm_length/2 - 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="input_wg",
    )
    structures.append(input_wg)

    # Layout along x: splitter S-bends | straight parallel arms | combiner
    # S-bends over the outer 30% of the arm span each, straight middle 40%.
    x_in = -arm_length / 2       # junction with input waveguide
    x_out = arm_length / 2       # junction with output waveguide
    taper = arm_length * 0.3
    x_split_end = x_in + taper   # splitter S-bend ends, straight arms begin
    x_comb_start = x_out - taper  # straight arms end, combiner S-bend begins
    y_arm = arm_separation / 2

    # Splitter: two smooth S-bends from input (y=0) to y=±y_arm.
    # Overlap the input waveguide by 0.5um for a continuous material path.
    structures.append(_sbend_polyslab(
        td, x_in - 0.5, x_split_end, 0.0, y_arm,
        wg_width, wg_height, name="split_upper",
    ))
    structures.append(_sbend_polyslab(
        td, x_in - 0.5, x_split_end, 0.0, -y_arm,
        wg_width, wg_height, name="split_lower",
    ))

    # Straight parallel arms (y=±y_arm), overlapping the S-bend ends by 0.1um.
    arm_len_straight = max(x_comb_start - x_split_end, 0.0) + 0.2
    arm_center_x = (x_split_end + x_comb_start) / 2
    upper_arm = td.Structure(
        geometry=td.Box(
            center=(arm_center_x, y_arm, wg_height / 2),
            size=(arm_len_straight, wg_width, wg_height),
        ),
        medium=si,
        name="upper_arm",
    )
    structures.append(upper_arm)

    lower_arm = td.Structure(
        geometry=td.Box(
            center=(arm_center_x, -y_arm, wg_height / 2),
            size=(arm_len_straight, wg_width, wg_height),
        ),
        medium=si,
        name="lower_arm",
    )
    structures.append(lower_arm)

    # Combiner: two smooth S-bends from y=±y_arm back to output (y=0).
    structures.append(_sbend_polyslab(
        td, x_comb_start, x_out + 0.5, y_arm, 0.0,
        wg_width, wg_height, name="combine_upper",
    ))
    structures.append(_sbend_polyslab(
        td, x_comb_start, x_out + 0.5, -y_arm, 0.0,
        wg_width, wg_height, name="combine_lower",
    ))

    # Output waveguide (y=0), left edge at x = arm_length/2
    output_wg = td.Structure(
        geometry=td.Box(
            center=(arm_length/2 + 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="output_wg",
    )
    structures.append(output_wg)

    # Simulation domain size
    Lx = arm_length + 30
    Ly = arm_separation + 6
    Lz = 4.0
    
    # Source position
    src_center = (-arm_length/2 - 9, 0, wg_height / 2)
    
    # Monitor positions
    monitor_positions = [
        (-arm_length/2 - 9, 0, "port_o1"),  # Input
        (arm_length/2 + 9, 0, "port_o2"),   # Output
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def create_coupler(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    coupler_length: float = 10.0,
    gap: float = 0.2,
) -> tuple:
    """Create a directional coupler structure.
    
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")
    
    structures = []
    
    # Upper waveguide (through)
    upper_wg = td.Structure(
        geometry=td.Box(
            center=(0, gap/2 + wg_width/2, wg_height / 2),
            size=(coupler_length + 20, wg_width, wg_height),
        ),
        medium=si,
        name="upper_wg",
    )
    structures.append(upper_wg)
    
    # Lower waveguide (cross)
    lower_wg = td.Structure(
        geometry=td.Box(
            center=(0, -gap/2 - wg_width/2, wg_height / 2),
            size=(coupler_length + 20, wg_width, wg_height),
        ),
        medium=si,
        name="lower_wg",
    )
    structures.append(lower_wg)
    
    # Simulation domain size
    Lx = coupler_length + 24
    Ly = gap + wg_width * 2 + 4
    Lz = 4.0
    
    # Source position (upper left)
    src_center = (-coupler_length/2 - 9, gap/2 + wg_width/2, wg_height / 2)
    
    # Monitor positions
    y_upper = gap/2 + wg_width/2
    y_lower = -gap/2 - wg_width/2
    monitor_positions = [
        (-coupler_length/2 - 9, y_upper, "port_o1"),  # Upper input
        (coupler_length/2 + 9, y_upper, "port_o2"),   # Upper output (through)
        (coupler_length/2 + 9, y_lower, "port_o3"),   # Lower output (cross)
        (-coupler_length/2 - 9, y_lower, "port_o4"),  # Lower input
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def create_grating_coupler(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    grating_period: float = 0.62,
    grating_duty_cycle: float = 0.5,
    num_periods: int = 20,
    etch_depth: float = 0.07,
) -> tuple:
    """Create a grating coupler structure.
    
    Args:
        td: Tidy3D module
        wavelength_um: Central wavelength in um
        wg_width: Waveguide width in um
        wg_height: Waveguide height in um
        grating_period: Grating period in um (default 0.62 for 1550nm)
        grating_duty_cycle: Duty cycle of grating (default 0.5)
        num_periods: Number of grating periods
        etch_depth: Etch depth in um
        
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")
    
    structures = []
    
    grating_length = num_periods * grating_period
    
    # Input waveguide (before grating)
    input_wg = td.Structure(
        geometry=td.Box(
            center=(-grating_length/2 - 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="input_wg",
    )
    structures.append(input_wg)
    
    # Grating teeth
    tooth_width = grating_period * grating_duty_cycle
    for i in range(num_periods):
        x_center = -grating_length/2 + grating_period * (i + 0.5)
        # Unetched part (taller)
        tooth = td.Structure(
            geometry=td.Box(
                center=(x_center, 0, wg_height / 2),
                size=(tooth_width, wg_width, wg_height),
            ),
            medium=si,
            name=f"tooth_{i}",
        )
        structures.append(tooth)
    
    # Taper section (optional, to expand mode)
    taper_length = 10.0
    taper_end_width = 8.0  # Wide end
    taper = td.Structure(
        geometry=td.Box(
            center=(grating_length/2 + taper_length/2, 0, wg_height / 2),
            size=(taper_length, taper_end_width, wg_height),
        ),
        medium=si,
        name="taper",
    )
    structures.append(taper)
    
    # Simulation domain size - make sure monitors are inside
    Lx = grating_length + taper_length + 20  # Extra padding
    Ly = max(wg_width, taper_end_width) + 4
    Lz = 5.0
    
    # Source position (in input waveguide, away from edge)
    src_x = -Lx/2 + 3  # 3 units from left edge
    src_center = (src_x, 0, wg_height / 2)
    
    # Monitor positions - ensure they are inside simulation domain
    mon_in_x = -Lx/2 + 3   # Input monitor
    mon_out_x = Lx/2 - 3   # Output monitor (3 units from right edge)
    monitor_positions = [
        (mon_in_x, 0, "port_o1"),   # Input
        (mon_out_x, 0, "port_o2"),  # Output
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def create_polarization_rotator(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    rotation_length: float = 30.0,
    swg_period: float = 0.4,
) -> tuple:
    """Create a polarization rotator using subwavelength gratings.
    
    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")

    structures = []

    num_periods = int(rotation_length / swg_period)
    # SWG teeth span: first tooth centred at -rotation_length/2 + swg_period*0.5,
    # last near +rotation_length/2.  Extend the I/O waveguides so their inner
    # edges reach the first/last tooth centre (0.5um overlap), removing the
    # ~0.1um gaps that left the teeth disconnected from the I/O waveguides.
    first_tooth_x = -rotation_length / 2 + swg_period * 0.5
    last_tooth_x = -rotation_length / 2 + swg_period * (num_periods - 1 + 0.5)
    in_outer = -rotation_length / 2 - 10.0   # left edge of input waveguide
    in_len = (first_tooth_x - in_outer)
    out_outer = rotation_length / 2 + 10.0   # right edge of output waveguide
    out_len = (out_outer - last_tooth_x)

    # Input waveguide (wider for TE mode), inner edge overlaps the first tooth
    input_wg = td.Structure(
        geometry=td.Box(
            center=((in_outer + first_tooth_x) / 2, 0, wg_height / 2),
            size=(in_len, wg_width, wg_height),
        ),
        medium=si,
        name="input_wg",
    )
    structures.append(input_wg)

    # Subwavelength grating section (alternating wide/narrow sections)
    for i in range(num_periods):
        x_center = -rotation_length/2 + swg_period * (i + 0.5)
        # Alternating widths for polarization rotation
        width = wg_width * 1.5 if i % 2 == 0 else wg_width * 0.7
        tooth = td.Structure(
            geometry=td.Box(
                center=(x_center, 0, wg_height / 2),
                size=(swg_period * 0.5, width, wg_height),
            ),
            medium=si,
            name=f"swg_{i}",
        )
        structures.append(tooth)

    # Output waveguide (narrower for TM mode), inner edge overlaps the last tooth
    output_wg = td.Structure(
        geometry=td.Box(
            center=((last_tooth_x + out_outer) / 2, 0, wg_height / 2),
            size=(out_len, wg_width * 0.8, wg_height),
        ),
        medium=si,
        name="output_wg",
    )
    structures.append(output_wg)

    # Simulation domain size - ensure monitors are inside
    Lx = rotation_length + 30  # Extra padding
    Ly = wg_width * 3 + 4
    Lz = 4.0
    
    # Source position (inside simulation domain)
    src_x = -Lx/2 + 5  # 5 units from left edge
    src_center = (src_x, 0, wg_height / 2)
    
    # Monitor positions - ensure they are inside simulation domain
    mon_in_x = -Lx/2 + 5    # Input monitor
    mon_out_x = Lx/2 - 5    # Output monitor
    monitor_positions = [
        (mon_in_x, 0, "port_o1"),   # Input (TE)
        (mon_out_x, 0, "port_o2"),  # Output (TM)
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def create_y_branch(
    td,
    wavelength_um: float = 1.55,
    wg_width: float = 0.5,
    wg_height: float = 0.22,
    arm_length: float = 15.0,
    arm_separation: float = 3.0,
) -> tuple:
    """Create a Y-branch 1x2 splitter.

    Returns:
        Tuple of (structures, simulation_size, source_center, monitor_positions)
    """
    # Silicon material
    si = td.Medium(permittivity=3.45**2, name="silicon")

    structures = []

    # Input waveguide
    input_wg = td.Structure(
        geometry=td.Box(
            center=(-arm_length - 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="input_wg",
    )
    structures.append(input_wg)

    # S-bend arms as single smooth PolySlabs.  Both arms emanate from the
    # junction (x=-arm_length, y=0) and curve to y=±arm_separation/2 at the
    # output.  Overlap the I/O waveguides by 0.5um so the material path is
    # continuous (no stamped-segment gaps).
    x_junction = -arm_length
    x_arm_end = arm_length
    upper_arm = _sbend_polyslab(
        td, x_junction - 0.5, x_arm_end + 0.5,
        0.0, arm_separation / 2,
        wg_width, wg_height, name="upper_arm",
    )
    structures.append(upper_arm)

    lower_arm = _sbend_polyslab(
        td, x_junction - 0.5, x_arm_end + 0.5,
        0.0, -arm_separation / 2,
        wg_width, wg_height, name="lower_arm",
    )
    structures.append(lower_arm)

    # Output waveguides - connect properly to arm ends
    upper_out = td.Structure(
        geometry=td.Box(
            center=(arm_length + 5, arm_separation/2, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="upper_out",
    )
    structures.append(upper_out)
    
    lower_out = td.Structure(
        geometry=td.Box(
            center=(arm_length + 5, -arm_separation/2, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="lower_out",
    )
    structures.append(lower_out)
    
    # Simulation domain size
    Lx = arm_length * 3 + 25
    Ly = arm_separation + 8
    Lz = 4.0
    
    # Source position (inside input waveguide)
    src_center = (-arm_length - 9, 0, wg_height / 2)
    
    # Monitor positions
    monitor_positions = [
        (-arm_length - 9, 0, "port_o1"),            # Input
        (arm_length + 9, arm_separation/2, "port_o2"),   # Upper output
        (arm_length + 9, -arm_separation/2, "port_o3"),  # Lower output
    ]
    
    return structures, (Lx, Ly, Lz), src_center, monitor_positions


def try_log_tidy3d(session: Any) -> None:
    """Build config from session and run Tidy3D simulation."""
    # Import unified component detector
    from component_detector import detect_component_type, get_component_sim_params
    
    # Extract component type from session
    component_type = "unknown"  # default
    
    # Extract parameters from generated template if available
    wg_width = 0.5
    wg_height = 0.22
    
    if isinstance(session, dict):
        # Use unified component detection
        comp_list = session.get("p200_pretemplate", {}).get("components_list", [])
        if comp_list:
            first_comp = str(comp_list[0])
            component_type, confidence = detect_component_type(first_comp)
            print(f"Detected component type: {component_type} (confidence: {confidence:.1f})")
        
        # Try to extract parameters from template if available
        template_path = session.get("generated_template_path")
        if template_path:
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                    import re
                    radius_match = re.search(r'radius["\s:=]+(\d+\.?\d*)', template_content)
                    if radius_match:
                        wg_width = float(radius_match.group(1))
                    gap_match = re.search(r'gap["\s:=]+(\d+\.?\d*)', template_content)
                    if gap_match:
                        wg_height = float(gap_match.group(1))
            except Exception as e:
                print(f"Failed to extract parameters from template: {e}")
    
    # Get default parameters for this component type
    sim_params = get_component_sim_params(component_type)
    wg_width = sim_params.get("wg_width", wg_width)
    wg_height = sim_params.get("wg_height", wg_height)
    
    # Run simulation with the detected component type
    run_tidy3d_simulation(
        component_type=component_type,
        wg_width=wg_width,
        wg_height=wg_height,
        **{k: v for k, v in sim_params.items() if k not in ["wg_width", "wg_height"]}
    )
