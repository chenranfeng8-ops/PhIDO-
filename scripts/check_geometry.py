# 检查分支过渡逻辑
print("=== 分支过渡位置检查 ===")

sx = 50.0
gap = 2.0
width = 0.5
input_length = 10.0
taper_length = 15.0

# 锥形结束位置
taper_end_x = -sx/2 + input_length + taper_length
print(f"锥形结束位置: x = {taper_end_x}")

# 分支点位置
split_x = taper_end_x
print(f"分支点位置: x = {split_x}")

# 上输出波导位置
y_upper = gap / 2 + width / 4
print(f"上输出 y = {y_upper}")

# 下输出波导位置
y_lower = -gap / 2 - width / 4
print(f"下输出 y = {y_lower}")

# S-bend 过渡块位置
print("\n=== 上分支 S-bend ===")
for i in range(10):
    t = i / 9.0
    x = split_x + 2 + t * (sx/2 - 10 - split_x - 2)
    y = t * y_upper
    print(f"  块 {i}: x={x:.1f}, y={y:.3f}")

print("\n=== 下分支 S-bend ===")
for i in range(10):
    t = i / 9.0
    x = split_x + 2 + t * (sx/2 - 10 - split_x - 2)
    y = t * y_lower
    print(f"  块 {i}: x={x:.1f}, y={y:.3f}")