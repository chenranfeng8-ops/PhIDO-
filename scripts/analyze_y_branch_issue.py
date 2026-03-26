"""
分析 Y-branch 结构生成问题

正确的 Y-branch 结构：
1. 输入波导 → 2. MMI区域 → 3. S-bend过渡 → 4. 两输出波导

问题分析：
"""

# 从仿真输出看到的结构：
print("=== 实际生成的结构 ===")
print("1. 输入波导: center=(-17.5, 0), size=(15, 0.5)")
print("2. MMI: center=(-5, 0), size=(10, 3)")  
print("3. 过渡第一段: center=(0.5, 0), size=(2, 0.5)  <- 问题！")
print("4. 上分支: center=(1.5, 0.07), size=(2, 0.5)")
print("5. 下分支: center=(1.5, -0.07), size=(2, 0.5)")

print("\n=== 问题诊断 ===")
print("问题1: 过渡第一段 center=(0.5, 0) 在 y=0")
print("  → 这意味着上下分支起始点重叠在中心！")
print("  → 正确应该从 MMI 边缘开始分开")

print("\n问题2: 代码逻辑错误")
print("  for i in range(15):")
print("    t = i / 14.0  # t 从 0 到 1")
print("    y_pos = t * gap / 2  # y 从 0 到 gap/2")
print("  → 当 t=0 时，y=0，上下分支都在中心！")

print("\n=== 正确逻辑 ===")
print("过渡区应该：")
print("1. 从 MMI 结束位置开始")
print("2. 立即分开，不要重叠")
print("3. 使用 S-bend 曲线，不是直线")

print("\n=== 修复方案 ===")
print("方案1: 从 MMI 边缘分开（最简单）")
print("  - MMI 宽度要足够宽，能容纳两个分支")
print("  - 过渡区从 MMI 边缘开始，不要从中心开始")

print("\n方案2: 使用 S-bend 波导（最正确）")
print("  - 使用 gdsfactory 的 s-bend 函数")
print("  - 创建平滑的曲线过渡")