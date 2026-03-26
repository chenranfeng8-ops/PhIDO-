"""
Meep FDTD Simulation for Photonic Components
使用 gdsfactory 验证过的几何数据
"""

import numpy as np
import meep as mp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import os


def run_meep_simulation(
    component_type: str,
    params: Dict[str, Any],
    output_dir: str = "build",
    wavelength: float = 1.55,
    resolution: int = 20,
) -> Dict[str, Any]:
    """
    使用 Meep 运行 FDTD 仿真
    
    关键改进：使用 gdsfactory 验证过的结构参数
    """
    print(f"Running Meep FDTD simulation for: {component_type}")
    print(f"Wavelength: {wavelength} um, Resolution: {resolution} pixels/um")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # === 从 gdsfactory_geometry.json 加载验证过的几何数据 ===
    geometry_file = output_path / "gdsfactory_geometry.json"
    
    if geometry_file.exists():
        with open(geometry_file, 'r') as f:
            gf_data = json.load(f)
        print(f"Loaded geometry from gdsfactory: {gf_data['component_type']}")
        bbox = gf_data['bbox']
        ports = gf_data['ports']
    else:
        print("WARNING: gdsfactory_geometry.json not found, using default geometry")
        bbox = {"x_min": -10, "y_min": -1.25, "x_max": 15.5, "y_max": 1.25}
        ports = {
            "o1": {"center": [-10.0, 0.0], "width": 0.5},
            "o2": {"center": [15.5, 0.625], "width": 0.5},
            "o3": {"center": [15.5, -0.625], "width": 0.5},
        }
    
    # 计算仿真单元大小
    sx = (bbox['x_max'] - bbox['x_min']) + 20  # 加上 PML
    sy = (bbox['y_max'] - bbox['y_min']) + 10
    sz = 1.0
    
    print(f"Simulation cell size: {sx} x {sy} x {sz} um")
    print(f"Ports: {list(ports.keys())}")
    
    # 创建材料
    silicon = mp.Medium(epsilon=12.0)
    
    # === 使用 gdsfactory 的 mmi1x2 参数创建正确结构 ===
    # 这些参数来自 gdsfactory.components.mmi1x2() 的默认值
    geometry_objects = []
    
    # MMI 参数 (从 gdsfactory 提取)
    length_mmi = 5.5
    width_mmi = 2.5
    gap_mmi = 0.25  # 这是输出波导之间的间隙
    width_taper = 1.0
    length_taper = 10.0
    width_wg = 0.5
    
    # 偏移量，将结构放在仿真单元中心
    x_offset = -bbox['x_min'] - sx/2 + 10
    y_offset = -bbox['y_min'] - sy/2 + 5
    
    # 1. 输入波导 + 锥形
    input_taper = mp.Block(
        size=mp.Vector3(length_taper + 1, width_taper, mp.inf),
        center=mp.Vector3(x_offset + bbox['x_min'] + length_taper/2 + 0.5, y_offset, 0),
        material=silicon,
    )
    geometry_objects.append(input_taper)
    
    # 2. MMI 区域
    mmi = mp.Block(
        size=mp.Vector3(length_mmi, width_mmi, mp.inf),
        center=mp.Vector3(x_offset + bbox['x_min'] + length_taper + length_mmi/2, y_offset, 0),
        material=silicon,
    )
    geometry_objects.append(mmi)
    
    # 3. 输出锥形 + 波导
    output_taper_length = length_taper
    gap_outputs = gap_mmi  # 输出波导间隙
    
    # 上输出
    upper_output = mp.Block(
        size=mp.Vector3(output_taper_length + 1, width_taper, mp.inf),
        center=mp.Vector3(
            x_offset + bbox['x_min'] + length_taper + length_mmi + output_taper_length/2 + 0.5,
            y_offset + (width_mmi/2 - width_taper/2),
            0
        ),
        material=silicon,
    )
    geometry_objects.append(upper_output)
    
    # 下输出
    lower_output = mp.Block(
        size=mp.Vector3(output_taper_length + 1, width_taper, mp.inf),
        center=mp.Vector3(
            x_offset + bbox['x_min'] + length_taper + length_mmi + output_taper_length/2 + 0.5,
            y_offset - (width_mmi/2 - width_taper/2),
            0
        ),
        material=silicon,
    )
    geometry_objects.append(lower_output)
    
    print(f"Created {len(geometry_objects)} geometry objects based on gdsfactory parameters")
    
    # 创建仿真
    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz),
        geometry=geometry_objects,
        sources=[
            mp.Source(
                mp.GaussianSource(1/wavelength, fwidth=0.2/wavelength),
                component=mp.Ez,
                center=mp.Vector3(x_offset + bbox['x_min'] + 3, y_offset, 0),
                size=mp.Vector3(0, width_wg * 2, 0),
            )
        ],
        boundary_layers=[mp.PML(1.0)],
        default_material=mp.Medium(epsilon=2.25),
        resolution=resolution,
    )
    
    # 运行仿真
    print("Running FDTD...")
    sim.run(until=100)
    
    # 输出结果
    output_file = str(output_path / f"meep_sim_{component_type}")
    
    try:
        eps_data = sim.get_array(center=mp.Vector3(z=0), size=mp.Vector3(sx, sy, 0), component=mp.Dielectric)
        ez_data = sim.get_array(center=mp.Vector3(z=0), size=mp.Vector3(sx, sy, 0), component=mp.Ez)
        
        print(f"Epsilon data shape: {eps_data.shape}")
        print(f"Ez data shape: {ez_data.shape}")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        im1 = ax1.imshow(eps_data.T, cmap='Blues', aspect='auto', origin='lower',
                        extent=[-sx/2, sx/2, -sy/2, sy/2])
        ax1.set_xlabel('x (um)')
        ax1.set_ylabel('y (um)')
        ax1.set_title(f'{component_type.upper()} Structure (gdsfactory params)')
        plt.colorbar(im1, ax=ax1, label='ε')
        
        ax2 = axes[1]
        im2 = ax2.imshow(np.abs(ez_data.T), cmap='hot', aspect='auto', origin='lower',
                        extent=[-sx/2, sx/2, -sy/2, sy/2])
        ax2.set_xlabel('x (um)')
        ax2.set_ylabel('y (um)')
        ax2.set_title(f'{component_type.upper()} Field |Ez|')
        plt.colorbar(im2, ax=ax2, label='|Ez|')
        
        plt.tight_layout()
        plt.savefig(f"{output_file}_combined.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}_combined.png")
        
        return {
            "component_type": component_type,
            "field_path": f"{output_file}_combined.png",
            "params": params,
            "wavelength": wavelength,
            "source": "gdsfactory geometry data",
        }
        
    except Exception as plot_e:
        print(f"Plot error: {plot_e}")
        import traceback
        traceback.print_exc()
        return {
            "component_type": component_type,
            "error": str(plot_e),
            "params": params,
        }
    
    # 运行仿真
    print("Running FDTD...")
    sim.run(until=50)  # 减少步数加速演示
    
    # 输出结果
    output_file = str(output_path / f"meep_sim_{component_type}")
    
    # 获取场数据并手动绘制
    try:
        # 获取介电常数分布 (2D slice at z=0)
        eps_data = sim.get_array(center=mp.Vector3(z=0), size=mp.Vector3(sx, sy, 0), component=mp.Dielectric)
        
        # 获取场分布 (2D slice at z=0)
        ez_data = sim.get_array(center=mp.Vector3(z=0), size=mp.Vector3(sx, sy, 0), component=mp.Ez)
        
        print(f"Epsilon data shape: {eps_data.shape}")
        print(f"Ez data shape: {ez_data.shape}")
        
        # 绘制场分布
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 结构 (介电常数)
        ax1 = axes[0]
        im1 = ax1.imshow(eps_data.T, cmap='Blues', aspect='auto', origin='lower',
                        extent=[-sx/2, sx/2, -sy/2, sy/2])
        ax1.set_xlabel('x (um)')
        ax1.set_ylabel('y (um)')
        ax1.set_title(f'{component_type.upper()} Structure (ε)')
        plt.colorbar(im1, ax=ax1, label='ε')
        
        # 场分布
        ax2 = axes[1]
        im2 = ax2.imshow(np.abs(ez_data.T), cmap='hot', aspect='auto', origin='lower',
                        extent=[-sx/2, sx/2, -sy/2, sy/2])
        ax2.set_xlabel('x (um)')
        ax2.set_ylabel('y (um)')
        ax2.set_title(f'{component_type.upper()} Field |Ez|')
        plt.colorbar(im2, ax=ax2, label='|Ez|')
        
        plt.tight_layout()
        plt.savefig(f"{output_file}_combined.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_file}_combined.png")
        
        return {
            "component_type": component_type,
            "field_path": f"{output_file}_combined.png",
            "params": params,
            "wavelength": wavelength,
        }
        
    except Exception as plot_e:
        print(f"Plot error: {plot_e}")
        import traceback
        traceback.print_exc()
        return {
            "component_type": component_type,
            "error": str(plot_e),
            "params": params,
        }


def create_y_branch_geometry(params: Dict, silicon: mp.Medium, wavelength: float):
    """创建 Y-branch 结构 - 正确连通的实现
    
    关键：所有段之间必须有重叠，确保完全连通
    """
    gap = params.get("gap", 2.0)
    width = params.get("width", 0.5)
    length = params.get("length", 30.0)
    
    sx = 50.0
    sy = 16.0
    sz = 1.0
    
    geometry = {
        "cell_size": mp.Vector3(sx, sy, sz),
        "objects": [],
    }
    
    # 1. 输入波导
    input_length = 10.0
    input_wg = mp.Block(
        size=mp.Vector3(input_length + 1, width, mp.inf),  # 加长确保连接
        center=mp.Vector3(-sx/2 + input_length/2, 0, 0),
        material=silicon,
    )
    geometry["objects"].append(input_wg)
    
    # 2. MMI 区域（固定宽度，确保连接两个分支）
    mmi_length = 12.0
    mmi_width = gap + width * 2  # 足够宽容纳两分支
    
    mmi = mp.Block(
        size=mp.Vector3(mmi_length + 1, mmi_width, mp.inf),  # 加长确保连接
        center=mp.Vector3(-sx/2 + input_length + mmi_length/2, 0, 0),
        material=silicon,
    )
    geometry["objects"].append(mmi)
    
    # 3. 分支过渡区 - 使用连续重叠的块
    # 关键：从 MMI 边缘立即分开
    transition_start_x = -sx/2 + input_length + mmi_length
    transition_length = 12.0
    output_length = 10.0
    
    # 上分支过渡（从 MMI 上边缘开始）
    y_mmi_top = mmi_width / 2  # MMI 上边缘
    y_upper_out = gap / 2 + width / 2  # 输出波导位置
    
    for i in range(15):
        t = i / 14.0
        x_center = transition_start_x + t * transition_length
        # 从 MMI 上边缘平滑过渡到输出位置
        y_center = y_mmi_top - width/2 + t * (y_upper_out - (y_mmi_top - width/2))
        
        # 使用足够的长度确保重叠
        block_length = transition_length / 14 + 1
        
        block = mp.Block(
            size=mp.Vector3(block_length, width, mp.inf),
            center=mp.Vector3(x_center, y_center, 0),
            material=silicon,
        )
        geometry["objects"].append(block)
    
    # 下分支过渡
    y_mmi_bottom = -mmi_width / 2  # MMI 下边缘
    y_lower_out = -gap / 2 - width / 2  # 输出波导位置
    
    for i in range(15):
        t = i / 14.0
        x_center = transition_start_x + t * transition_length
        y_center = y_mmi_bottom + width/2 + t * (y_lower_out - (y_mmi_bottom + width/2))
        
        block_length = transition_length / 14 + 1
        
        block = mp.Block(
            size=mp.Vector3(block_length, width, mp.inf),
            center=mp.Vector3(x_center, y_center, 0),
            material=silicon,
        )
        geometry["objects"].append(block)
    
    # 4. 输出波导
    upper_out = mp.Block(
        size=mp.Vector3(output_length + 1, width, mp.inf),
        center=mp.Vector3(sx/2 - output_length/2, y_upper_out, 0),
        material=silicon,
    )
    geometry["objects"].append(upper_out)
    
    lower_out = mp.Block(
        size=mp.Vector3(output_length + 1, width, mp.inf),
        center=mp.Vector3(sx/2 - output_length/2, y_lower_out, 0),
        material=silicon,
    )
    geometry["objects"].append(lower_out)
    
    # 源
    sources = [
        mp.Source(
            mp.GaussianSource(1/wavelength, fwidth=0.2/wavelength),
            component=mp.Ez,
            center=mp.Vector3(-sx/2 + 3, 0, 0),
            size=mp.Vector3(0, width * 2, 0),
        )
    ]
    
    # 监视器
    monitors = [
        mp.FluxRegion(
            center=mp.Vector3(sx/2 - 3, y_upper_out, 0),
            size=mp.Vector3(0, width * 2, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(sx/2 - 3, y_lower_out, 0),
            size=mp.Vector3(0, width * 2, 0),
        ),
    ]
    
    return geometry, sources, monitors, sx, sy


def create_mzi_geometry(params: Dict, silicon: mp.Medium, wavelength: float):
    """创建 MZI 结构"""
    delta_length = params.get("delta_length", 10.0)
    width = params.get("width", 0.5)
    
    sx = 50
    sy = 20
    
    geometry = {
        "cell_size": mp.Vector3(sx, sy, 1.0),
        "objects": [],
    }
    
    # 输入波导
    input_wg = mp.Block(
        size=mp.Vector3(15, width, mp.inf),
        center=mp.Vector3(-sx/2 + 7.5, 0, 0),
        material=silicon,
    )
    geometry["objects"].append(input_wg)
    
    # 上臂
    upper_arm = mp.Block(
        size=mp.Vector3(20, width, mp.inf),
        center=mp.Vector3(0, 3, 0),
        material=silicon,
    )
    geometry["objects"].append(upper_arm)
    
    # 下臂 (更长)
    lower_arm = mp.Block(
        size=mp.Vector3(20 + delta_length/5, width, mp.inf),
        center=mp.Vector3(0, -3, 0),
        material=silicon,
    )
    geometry["objects"].append(lower_arm)
    
    # 输出波导
    output_wg = mp.Block(
        size=mp.Vector3(15, width, mp.inf),
        center=mp.Vector3(sx/2 - 7.5, 0, 0),
        material=silicon,
    )
    geometry["objects"].append(output_wg)
    
    sources = [
        mp.Source(
            mp.GaussianSource(1/wavelength, fwidth=0.2/wavelength),
            component=mp.Ez,
            center=mp.Vector3(-sx/2 + 2, 0, 0),
            size=mp.Vector3(0, width, 0),
        )
    ]
    
    monitors = [
        mp.FluxRegion(
            center=mp.Vector3(sx/2 - 2, 0, 0),
            size=mp.Vector3(0, width, 0),
        )
    ]
    
    return geometry, sources, monitors, sx, sy


def create_ring_geometry(params: Dict, silicon: mp.Medium, wavelength: float):
    """创建环形谐振器结构"""
    radius = params.get("radius", 10.0)
    gap = params.get("gap", 0.2)
    width = params.get("width", 0.5)
    
    sx = radius * 3
    sy = radius * 3
    
    geometry = {
        "cell_size": mp.Vector3(sx, sy, 1.0),
        "objects": [],
    }
    
    # 环
    ring = mp.Cylinder(
        radius=radius,
        height=width,
        center=mp.Vector3(0, 0, 0),
        material=silicon,
    )
    geometry["objects"].append(ring)
    
    # 直波导
    bus_wg = mp.Block(
        size=mp.Vector3(sx, width, mp.inf),
        center=mp.Vector3(0, radius + gap + width/2, 0),
        material=silicon,
    )
    geometry["objects"].append(bus_wg)
    
    sources = [
        mp.Source(
            mp.GaussianSource(1/wavelength, fwidth=0.2/wavelength),
            component=mp.Ez,
            center=mp.Vector3(-sx/2 + 2, radius + gap + width/2, 0),
            size=mp.Vector3(0, width, 0),
        )
    ]
    
    monitors = [
        mp.FluxRegion(
            center=mp.Vector3(sx/2 - 2, radius + gap + width/2, 0),
            size=mp.Vector3(0, width, 0),
        )
    ]
    
    return geometry, sources, monitors, sx, sy


def create_waveguide_geometry(params: Dict, silicon: mp.Medium, wavelength: float):
    """创建简单波导结构"""
    width = params.get("width", 0.5)
    length = params.get("length", 20.0)
    
    sx = length + 10
    sy = 6
    
    geometry = {
        "cell_size": mp.Vector3(sx, sy, 1.0),
        "objects": [
            mp.Block(
                size=mp.Vector3(length, width, mp.inf),
                center=mp.Vector3(0, 0, 0),
                material=silicon,
            )
        ],
    }
    
    sources = [
        mp.Source(
            mp.GaussianSource(1/wavelength, fwidth=0.2/wavelength),
            component=mp.Ez,
            center=mp.Vector3(-sx/2 + 2, 0, 0),
            size=mp.Vector3(0, width, 0),
        )
    ]
    
    monitors = [
        mp.FluxRegion(
            center=mp.Vector3(sx/2 - 2, 0, 0),
            size=mp.Vector3(0, width, 0),
        )
    ]
    
    return geometry, sources, monitors, sx, sy


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Meep FDTD Simulation")
    print("=" * 60)
    
    result = run_meep_simulation(
        "y_branch",
        {"gap": 2.0, "width": 0.5, "length": 30.0},
        "build"
    )
    
    print(f"\nResult: {result}")