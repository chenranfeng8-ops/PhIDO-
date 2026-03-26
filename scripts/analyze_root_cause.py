"""
分析 Y-branch 结构生成的完整流程
找出根本性错误
"""

print("=" * 70)
print("Y-branch 结构生成流程分析")
print("=" * 70)

# Step 1: Entity Extraction
print("\n[Step 1] Entity Extraction")
print("用户输入: 'Y-branch splitter'")
print("输出: {components_list: ['Y-branch splitter'], design_type: 'single_component'}")

# Step 2: Component Selection
print("\n[Step 2] Component Selection")
print("匹配: y_branch (from keyword_map)")
print("问题: 这只是名称匹配，没有提取任何参数！")

# Step 3: Schematic Generation
print("\n[Step 3] Schematic Generation")
print("生成原理图: 使用 gdsfactory 组件")
print("问题: 这个阶段生成的参数是否传递给了 Meep？")

# Step 4: Layout Generation
print("\n[Step 4] Layout Generation")
print("生成 GDS: 使用模板文件")
print("问题: 模板中的参数是否被提取？")

# Step 5: Meep Simulation
print("\n[Step 5] Meep Simulation")
print("参数来源: 从模板文件用正则表达式提取")
print("""
params = {}
template_content = Path(session.generated_template_path).read_text()
param_matches = re.findall(r'(\w+):\s*float\s*=\s*([\d.]+)', template_content)
for name, value in param_matches:
    params[name] = float(value)
""")
print("问题: 如果模板没有生成或格式不对，params 会是空的！")

print("\n" + "=" * 70)
print("根本问题诊断")
print("=" * 70)

print("\n问题 1: 参数传递断链")
print("  Entity Extraction → 没有提取结构参数")
print("  Component Selection → 只有名称匹配")
print("  Schematic Generation → 生成的参数没有传递给 Meep")
print("  Meep Simulation → params 可能是空的或错误的")

print("\n问题 2: Meep 结构定义是硬编码的")
print("  create_y_branch_geometry(params, ...) 中的 params 没有被正确使用")
print("  结构的坐标和尺寸是硬编码的常量")
print("  params 中的 gap, width, length 被读取但可能没有影响最终结构")

print("\n问题 3: 缺少从 gdsfactory 到 Meep 的直接映射")
print("  Tidy3D 使用 gdsfactory 组件 → 结构正确")
print("  Meep 手动定义结构 → 可能出错")
print("  正确做法: 使用 gdsfactory 的几何数据驱动 Meep")

print("\n" + "=" * 70)
print("正确的解决方案")
print("=" * 70)

print("\n方案 1: 使用 gdsfactory 几何数据")
print("""
import gdsfactory as gf
c = gf.components.mmi1x2()  # 使用验证过的组件
# 提取几何数据
polygons = c.get_polygons()
ports = c.ports
# 将几何数据传递给 Meep
for poly in polygons:
    meep_block = mp.Block(...)
""")

print("\n方案 2: 使用 gmeep (gdsfactory + meep 集成)")
print("""
from gmeep import get_simulation
c = gf.components.mmi1x2()
sim = get_simulation(c)
sim.run()
""")

print("\n方案 3: 修复参数传递链")
print("""
Entity Extraction 时提取参数:
  - gap: 2.0
  - width: 0.5
  - length: 30.0

传递给 Meep 时验证参数:
  - 检查 params 是否为空
  - 使用默认值填充缺失参数
  - 确保参数在合理范围内
""")