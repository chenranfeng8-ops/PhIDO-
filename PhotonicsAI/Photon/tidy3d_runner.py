"""Tidy3D integration for PhIDO.

This module provides FDTD simulation capabilities using Tidy3D cloud API.
Supports various photonic components including waveguide crossings.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
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


def run_tidy3d_simulation(
    component_type: str = "unknown",
    wavelength_nm: float = 1550.0,
    **kwargs  # Accept all parameters as keyword arguments
) -> None:
    """Run Tidy3D simulation for a photonic component.
    
    Args:
        component_type: Type of component (from component_detector)
        wavelength_nm: Central wavelength in nm
        **kwargs: Additional parameters (wg_width, wg_height, ring_radius, etc.)
    """
    lines: list[str] = []
    
    # Extract common parameters with defaults
    wg_width = kwargs.get("wg_width", 0.5)
    wg_height = kwargs.get("wg_height", 0.22)
    wg_length = kwargs.get("wg_length", 10.0)
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
    
    try:
        import tidy3d as td
        
        lines.append("import tidy3d as td  # OK")
        
        wavelength_um = wavelength_nm / 1000.0
        
        # Create component geometry based on type
        if component_type == "crossing":
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
                td, wavelength_um, wg_width, wg_height, mmi_width, mmi_length
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
        
        # Grid specification
        grid_spec = td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=20,
        )
        
        # Mode source
        try:
            # Create mode source
            mode_spec = td.ModeSpec(
                num_modes=1,
                target_neff=3.45,  # Silicon effective index
            )
            
            # Calculate frequency from wavelength (freq = c / wavelength)
            freq0 = td.C_0 / (wavelength_um * 1e-6)  # Convert wavelength to frequency
            fwidth = freq0 * 0.1  # 10% bandwidth
            
            pulse = td.GaussianPulse(
                freq0=freq0,
                fwidth=fwidth,
            )
            
            source = td.ModeSource(
                source_time=pulse,
                center=src_center,
                size=(0, wg_width * 3, wg_height * 3),
                mode_spec=mode_spec,
                mode_index=0,
                direction="+",
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
            freqs = [td.C_0 / (wavelength_um * 1e-6)]
            
            field_mon = td.FieldMonitor(
                center=(0, 0, wg_height / 2),
                size=(Lx * 0.9, Ly * 0.9, 0),
                freqs=freqs,
                name="field_monitor",
            )
            monitors.append(field_mon)
            lines.append("td.FieldMonitor created")
        except Exception as me:
            lines.append(f"Field monitor failed: {me}")
        
        # Flux monitors at each port
        for mx, my, mname in monitor_positions:
            try:
                if "o1" in mname or "o2" in mname:
                    msize = (0, wg_width * 3, wg_height * 3)
                else:
                    msize = (wg_width * 3, 0, wg_height * 3)
                
                flux_mon = td.FluxMonitor(
                    center=(mx, my, wg_height / 2),
                    size=msize,
                    freqs=freqs,
                    name=f"flux_{mname}",
                )
                monitors.append(flux_mon)
                lines.append(f"td.FluxMonitor({mname}) created")
            except Exception as fe:
                lines.append(f"Flux monitor {mname} failed: {fe}")
        
        # Create simulation
        sim = td.Simulation(
            size=(Lx, Ly, Lz),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(td.PML()),
            medium=td.Medium(permittivity=1.44**2, name="silica"),  # SiO2 substrate
            grid_spec=grid_spec,
            structures=structures,
            sources=[source] if source else [],
            monitors=monitors,
        )
        lines.append("td.Simulation created with structures")
        
        # Plot simulation cross-sections with component-specific filenames
        sim_time = datetime.now().strftime('%H:%M:%S')
        
        try:
            # XY plane at waveguide center
            fig, ax = plt.subplots(figsize=(10, 8))
            sim.plot(z=wg_height / 2, ax=ax)
            ax.set_title(f"{component_type.upper()} - XY Plane (z={wg_height/2:.2f}μm) [{sim_time}]")
            out_png = PATH.build / f"tidy3d_sim_z0_{component_type}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
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
                
                # Run locally
                sim_data = td.webapi.webapi.run_local(
                    sim,
                    task_name=f"local_{component_type}",
                    path=str(PATH.build / "tidy3d_data.hdf5"),
                )
                lines.append("Local simulation completed!")
                result = "success"
                data = sim_data
                
                # Plot field data
                try:
                    if hasattr(data, 'simulation_data'):
                        for mon_name in ['field_monitor', 'field_z', 'field_xy']:
                            if mon_name in [m.name for m in sim.monitors]:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                data[mon_name].plot(field="Ey", ax=ax)
                                out_png = PATH.build / f"tidy3d_field_{mon_name}.png"
                                fig.savefig(out_png, dpi=150, bbox_inches="tight")
                                plt.close(fig)
                                lines.append(f"Saved field plot: {out_png}")
                except Exception as fe:
                    lines.append(f"Field plot: {fe}")
                    
            except Exception as le:
                lines.append(f"Local run failed: {le}")
                lines.append("Falling back to cloud mode...")
                use_local = False
        
        if not use_local:
            # Cloud simulation (requires API key)
            api_key = os.getenv("TIDY3D_API_KEY")
            if api_key:
                try:
                    from tidy3d import web
                    
                    web.configure(apikey=api_key)
                    lines.append("Configured Tidy3D API")
                    
                    task_name = f"PhIDO-{component_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    lines.append(f"Starting cloud run: {task_name}")
                    
                    # Run simulation with encoding fix for Windows
                    import io
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    try:
                        if sys.platform == "win32":
                            sys.stdout = io.StringIO()
                            sys.stderr = io.StringIO()
                        
                        data = web.run(
                            simulation=sim,
                            task_name=task_name,
                            path=str(PATH.build / "tidy3d_data.hdf5"),
                        )
                        result = "success"
                    except UnicodeEncodeError as ue:
                        lines.append(f"Note: Unicode output issue")
                        result = "unicode_warning"
                        data = None
                    finally:
                        if sys.platform == "win32":
                            try:
                                stdout_capture = sys.stdout.getvalue() if hasattr(sys.stdout, 'getvalue') else ""
                                stderr_capture = sys.stderr.getvalue() if hasattr(sys.stderr, 'getvalue') else ""
                            except:
                                pass
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr
                    
                    if result == "success":
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
        lines.append(f"Saved config: {config_path}")
        
        _append_log(lines)
        
    except Exception as e:
        lines.append(f"Error: {type(e).__name__}: {e}")
        _append_log(lines)


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
    
    # Input waveguide
    input_wg = td.Structure(
        geometry=td.Box(
            center=(-mmi_length/2 - 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
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
        out_wg = td.Structure(
            geometry=td.Box(
                center=(mmi_length/2 + 5, y_pos, wg_height / 2),
                size=(10, wg_width, wg_height),
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
    
    # Input waveguide
    input_wg = td.Structure(
        geometry=td.Box(
            center=(-arm_length/2 - 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="input_wg",
    )
    structures.append(input_wg)
    
    # Y-junction input (simplified as straight sections)
    y_in_left = td.Structure(
        geometry=td.Box(
            center=(-arm_length/2, arm_separation/4, wg_height / 2),
            size=(5, wg_width, wg_height),
        ),
        medium=si,
        name="y_in_left",
    )
    structures.append(y_in_left)
    
    y_in_right = td.Structure(
        geometry=td.Box(
            center=(-arm_length/2, -arm_separation/4, wg_height / 2),
            size=(5, wg_width, wg_height),
        ),
        medium=si,
        name="y_in_right",
    )
    structures.append(y_in_right)
    
    # Upper arm
    upper_arm = td.Structure(
        geometry=td.Box(
            center=(0, arm_separation/2, wg_height / 2),
            size=(arm_length, wg_width, wg_height),
        ),
        medium=si,
        name="upper_arm",
    )
    structures.append(upper_arm)
    
    # Lower arm
    lower_arm = td.Structure(
        geometry=td.Box(
            center=(0, -arm_separation/2, wg_height / 2),
            size=(arm_length, wg_width, wg_height),
        ),
        medium=si,
        name="lower_arm",
    )
    structures.append(lower_arm)
    
    # Output waveguide
    output_wg = td.Structure(
        geometry=td.Box(
            center=(arm_length/2 + 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="output_wg",
    )
    structures.append(output_wg)
    
    # Y-junction output
    y_out_left = td.Structure(
        geometry=td.Box(
            center=(arm_length/2, arm_separation/4, wg_height / 2),
            size=(5, wg_width, wg_height),
        ),
        medium=si,
        name="y_out_left",
    )
    structures.append(y_out_left)
    
    y_out_right = td.Structure(
        geometry=td.Box(
            center=(arm_length/2, -arm_separation/4, wg_height / 2),
            size=(5, wg_width, wg_height),
        ),
        medium=si,
        name="y_out_right",
    )
    structures.append(y_out_right)
    
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
    
    # Input waveguide (wider for TE mode)
    input_wg = td.Structure(
        geometry=td.Box(
            center=(-rotation_length/2 - 5, 0, wg_height / 2),
            size=(10, wg_width, wg_height),
        ),
        medium=si,
        name="input_wg",
    )
    structures.append(input_wg)
    
    # Subwavelength grating section (alternating wide/narrow sections)
    num_periods = int(rotation_length / swg_period)
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
    
    # Output waveguide (narrower for TM mode)
    output_wg = td.Structure(
        geometry=td.Box(
            center=(rotation_length/2 + 5, 0, wg_height / 2),
            size=(10, wg_width * 0.8, wg_height),
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
    import numpy as np
    
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
    
    # S-bend upper arm
    # Create using multiple segments with overlap to ensure connectivity
    num_segments = 15  # More segments for smoother curve
    segment_length = arm_length / num_segments * 1.5  # Overlap factor > 1 ensures no gaps
    
    for i in range(num_segments):
        t = i / (num_segments - 1)  # t from 0 to 1
        x = -arm_length + t * arm_length * 2  # Cover full length
        y = t * arm_separation / 2
        segment = td.Structure(
            geometry=td.Box(
                center=(x, y, wg_height / 2),
                size=(segment_length, wg_width, wg_height),
            ),
            medium=si,
            name=f"upper_arm_{i}",
        )
        structures.append(segment)
    
    # S-bend lower arm
    for i in range(num_segments):
        t = i / (num_segments - 1)
        x = -arm_length + t * arm_length * 2
        y = -t * arm_separation / 2
        segment = td.Structure(
            geometry=td.Box(
                center=(x, y, wg_height / 2),
                size=(segment_length, wg_width, wg_height),
            ),
            medium=si,
            name=f"lower_arm_{i}",
        )
        structures.append(segment)
    
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