"""
从 gdsfactory 提取几何数据，供 Meep 使用
在 Windows 端运行
"""
import gdsfactory as gf
import json
import numpy as np
from pathlib import Path

gf.gpdk.PDK.activate()

# 生成组件
component_type = "mmi1x2"
c = gf.components.mmi1x2(
    length_mmi=5.5,
    width_mmi=2.5,
    gap_mmi=0.25,
    width_taper=1.0,
    length_taper=10.0,
)

# 获取边界框
bbox = c.bbox()
print(f"Bounding box: {bbox}")
print(f"X range: {bbox.left} to {bbox.right}")
print(f"Y range: {bbox.bottom} to {bbox.top}")

# 获取端口
ports = {p.name: {"center": [float(p.center[0]), float(p.center[1])], "width": float(p.width)} for p in c.ports}
print(f"Ports: {ports}")

# 写入 GDS 文件，然后提取几何
gds_path = Path("build/component.gds")
gds_path.parent.mkdir(parents=True, exist_ok=True)
c.write_gds(str(gds_path))
print(f"\nSaved GDS to: {gds_path}")

# 保存端口和边界信息
geometry_data = {
    "component_type": component_type,
    "bbox": {
        "x_min": float(bbox.left),
        "y_min": float(bbox.bottom),
        "x_max": float(bbox.right),
        "y_max": float(bbox.top),
    },
    "ports": ports,
    "gds_path": str(gds_path),
}

output_path = Path("build/gdsfactory_geometry.json")
with open(output_path, 'w') as f:
    json.dump(geometry_data, f, indent=2)

print(f"Saved geometry info to: {output_path}")