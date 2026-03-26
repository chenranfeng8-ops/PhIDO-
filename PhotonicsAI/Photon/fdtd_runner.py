"""
2D FDTD Simulation for Photonic Components
真正的 FDTD 仿真（使用 numpy）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FDTDParams:
    """FDTD 仿真参数"""
    wavelength: float = 1.55  # um
    dx: float = 0.02  # 空间步长 (um)
    dt: float = None  # 时间步长 (自动计算)
    nx: int = 200  # x 方向网格数
    ny: int = 100  # y 方向网格数
    pml_layers: int = 10  # PML 吸收边界层数
    n_steps: int = 2000  # 时间步数
    source_y: int = None  # 源的 y 位置
    monitor_x: int = None  # 监视器 x 位置


class FDTD2D:
    """2D FDTD 仿真器（TM 模式）"""
    
    def __init__(self, params: FDTDParams):
        self.params = params
        self.c = 3e8  # 光速 (m/s)
        self.c_um = 3e14  # 光速 (um/s)
        
        # 计算时间步长（CFL 条件）
        if params.dt is None:
            self.dt = params.dx / (2 * self.c_um)  # 简化的 CFL 条件
        else:
            self.dt = params.dt
        
        # 网格
        self.nx = params.nx
        self.ny = params.ny
        self.dx = params.dx
        
        # 场分量 (TM 模式: Ez, Hx, Hy)
        self.Ez = np.zeros((self.nx, self.ny))
        self.Hx = np.zeros((self.nx, self.ny))
        self.Hy = np.zeros((self.nx, self.ny))
        
        # 材料折射率分布
        self.n = np.ones((self.nx, self.ny))  # 背景为空气 (n=1)
        
        # PML 参数
        self.pml = params.pml_layers
        self.sigma_max = 0.8 * (4 + 1) / (150 * np.pi * params.dx)
        
        # 设置源和监视器位置
        self.source_y = params.source_y if params.source_y else self.ny // 2
        self.monitor_x = params.monitor_x if params.monitor_x else self.nx - 2 * self.pml
        
        # 记录场分布
        self.field_history = []
        self.transmission = []
        
    def set_material(self, structure: np.ndarray, n: float):
        """设置材料区域
        
        Args:
            structure: 布尔数组，True 表示该位置有材料
            n: 折射率
        """
        self.n[structure] = n
        
    def add_waveguide(self, y_center: float, width: float, n: float = 3.45, 
                      x_start: int = None, x_end: int = None):
        """添加直波导
        
        Args:
            y_center: 波导中心 y 坐标
            width: 波导宽度
            n: 折射率（默认硅 n=3.45）
            x_start, x_end: 波导的 x 范围
        """
        y_idx = int(y_center / self.dx)
        w_idx = int(width / (2 * self.dx))
        
        x_s = x_start if x_start else self.pml
        x_e = x_end if x_end else self.nx - self.pml
        
        for i in range(x_s, x_e):
            for j in range(max(0, y_idx - w_idx), min(self.ny, y_idx + w_idx + 1)):
                self.n[i, j] = n
                
    def add_mmi(self, x_center: float, y_center: float, 
                length: float, width: float, n: float = 3.45):
        """添加 MMI 区域"""
        x_idx = int(x_center / self.dx)
        y_idx = int(y_center / self.dx)
        l_idx = int(length / (2 * self.dx))
        w_idx = int(width / (2 * self.dx))
        
        for i in range(max(0, x_idx - l_idx), min(self.nx, x_idx + l_idx)):
            for j in range(max(0, y_idx - w_idx), min(self.ny, y_idx + w_idx)):
                self.n[i, j] = n
                
    def add_y_branch(self, x_split: float, y_center: float,
                     input_width: float, output_width: float, 
                     separation: float, n: float = 3.45):
        """添加 Y 分支结构"""
        x_split_idx = int(x_split / self.dx)
        y_center_idx = int(y_center / self.dx)
        sep_idx = int(separation / (2 * self.dx))
        
        # 输入波导
        in_w = int(input_width / (2 * self.dx))
        for i in range(self.pml, x_split_idx):
            for j in range(y_center_idx - in_w, y_center_idx + in_w):
                if 0 <= j < self.ny:
                    self.n[i, j] = n
        
        # Y 分支过渡区
        out_w = int(output_width / (2 * self.dx))
        transition_length = int(self.nx * 0.2)
        
        for i in range(x_split_idx, x_split_idx + transition_length):
            t = (i - x_split_idx) / transition_length
            # 上分支
            y_upper = y_center_idx + int(t * sep_idx)
            for j in range(y_upper - out_w, y_upper + out_w):
                if 0 <= j < self.ny:
                    self.n[i, j] = n
            # 下分支
            y_lower = y_center_idx - int(t * sep_idx)
            for j in range(y_lower - out_w, y_lower + out_w):
                if 0 <= j < self.ny:
                    self.n[i, j] = n
        
        # 输出波导
        y_upper_out = y_center_idx + sep_idx
        y_lower_out = y_center_idx - sep_idx
        
        for i in range(x_split_idx + transition_length, self.nx - self.pml):
            for j in range(y_upper_out - out_w, y_upper_out + out_w):
                if 0 <= j < self.ny:
                    self.n[i, j] = n
            for j in range(y_lower_out - out_w, y_lower_out + out_w):
                if 0 <= j < self.ny:
                    self.n[i, j] = n
    
    def gaussian_source(self, t: float, x0: int, y0: int, 
                        wavelength: float, pulse_width: float = 50):
        """高斯脉冲源"""
        omega = 2 * np.pi * self.c_um / wavelength
        t0 = pulse_width
        return np.exp(-((t - t0) / pulse_width) ** 2) * np.sin(omega * t * self.dt)
    
    def run(self, wavelength: float = 1.55, verbose: bool = True):
        """运行 FDTD 仿真"""
        # 预计算系数
        c1 = self.c_um * self.dt / self.dx
        
        # 源位置
        src_x = self.pml + 5
        
        # 运行时间步进
        for n_step in range(self.params.n_steps):
            # 更新磁场
            self.Hx -= 0.5 * c1 * np.roll(self.Ez, -1, axis=1)
            self.Hy += 0.5 * c1 * np.roll(self.Ez, -1, axis=0)
            
            # 更新电场
            n_squared = self.n ** 2
            self.Ez += c1 / n_squared * (
                np.roll(self.Hy, 1, axis=0) - self.Hy -
                np.roll(self.Hx, 1, axis=1) + self.Hx
            )
            
            # 添加源
            t = n_step
            source_val = self.gaussian_source(t, src_x, self.source_y, wavelength)
            self.Ez[src_x, self.source_y] += source_val
            
            # PML 边界条件（简化版）
            # 左边界
            self.Ez[:self.pml, :] *= 0.95
            # 右边界
            self.Ez[-self.pml:, :] *= 0.95
            # 上边界
            self.Ez[:, -self.pml:] *= 0.95
            # 下边界
            self.Ez[:, :self.pml] *= 0.95
            
            # 数值稳定性：限制场值
            max_val = 1e10
            self.Ez = np.clip(self.Ez, -max_val, max_val)
            self.Hx = np.clip(self.Hx, -max_val, max_val)
            self.Hy = np.clip(self.Hy, -max_val, max_val)
            
            # 记录场分布
            if n_step % 100 == 0:
                self.field_history.append(self.Ez.copy())
                if verbose:
                    print(f"Step {n_step}/{self.params.n_steps}")
            
            # 记录传输
            self.transmission.append(np.abs(self.Ez[self.monitor_x, :]).sum())
        
        if verbose:
            print("FDTD simulation completed!")
            
    def plot_field(self, save_path: str = None):
        """绘制场分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 最终场分布
        ax1 = axes[0]
        im1 = ax1.imshow(self.Ez.T, cmap='RdBu', aspect='auto',
                         extent=[0, self.nx * self.dx, 0, self.ny * self.dx])
        ax1.set_xlabel('x (um)')
        ax1.set_ylabel('y (um)')
        ax1.set_title('Ez Field Distribution')
        plt.colorbar(im1, ax=ax1, label='Ez')
        
        # 材料分布
        ax2 = axes[1]
        im2 = ax2.imshow(self.n.T, cmap='Blues', aspect='auto',
                         extent=[0, self.nx * self.dx, 0, self.ny * self.dx])
        ax2.set_xlabel('x (um)')
        ax2.set_ylabel('y (um)')
        ax2.set_title('Material Distribution (n)')
        plt.colorbar(im2, ax=ax2, label='n')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
        
    def plot_transmission(self, save_path: str = None):
        """绘制传输曲线"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.transmission)
        plt.xlabel('Time Step')
        plt.ylabel('Transmission (a.u.)')
        plt.title('Output Transmission')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()


def run_fdtd_simulation(
    component_type: str,
    params: Dict[str, Any],
    output_dir: str = "build"
) -> Dict[str, Any]:
    """运行 FDTD 仿真的主函数"""
    
    print(f"Running 2D FDTD simulation for: {component_type}")
    
    # 设置仿真参数
    fdtd_params = FDTDParams(
        wavelength=params.get("wavelength", 1.55),
        nx=300,
        ny=150,
        dx=0.02,
        n_steps=1500,
    )
    
    # 创建仿真器
    sim = FDTD2D(fdtd_params)
    
    # 根据组件类型设置结构
    if component_type in ["y_branch", "y-branch", "splitter", "mmi", "mmi1x2"]:
        # Y-branch / MMI 结构
        sim.add_y_branch(
            x_split=2.0,
            y_center=1.5,
            input_width=0.5,
            output_width=0.5,
            separation=params.get("gap", 2.0),
            n=3.45
        )
    elif component_type in ["mzi"]:
        # MZI 结构
        # 输入波导
        sim.add_waveguide(y_center=1.5, width=0.5, n=3.45, 
                         x_start=10, x_end=50)
        # 臂
        sim.add_waveguide(y_center=2.5, width=0.5, n=3.45,
                         x_start=50, x_end=100)
        sim.add_waveguide(y_center=0.5, width=0.5, n=3.45,
                         x_start=50, x_end=100)
        # 输出波导
        sim.add_waveguide(y_center=1.5, width=0.5, n=3.45,
                         x_start=100, x_end=140)
    elif component_type in ["ring", "resonator"]:
        # Ring resonator (简化为弯曲波导)
        sim.add_waveguide(y_center=1.5, width=0.5, n=3.45)
    else:
        # 默认：直波导
        sim.add_waveguide(y_center=1.5, width=0.5, n=3.45)
    
    # 运行仿真
    sim.run(wavelength=fdtd_params.wavelength)
    
    # 保存结果
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    field_path = f"{output_dir}/fdtd_field_{component_type}.png"
    trans_path = f"{output_dir}/fdtd_transmission_{component_type}.png"
    
    sim.plot_field(field_path)
    sim.plot_transmission(trans_path)
    
    return {
        "component_type": component_type,
        "field_path": field_path,
        "transmission_path": trans_path,
        "params": params,
        "max_field": float(np.max(np.abs(sim.Ez))),
    }


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("Testing 2D FDTD Simulation")
    print("=" * 60)
    
    # 测试 Y-branch
    result = run_fdtd_simulation(
        "y_branch",
        {"gap": 2.0, "wavelength": 1.55},
        "build"
    )
    print(f"\nResult: {result}")