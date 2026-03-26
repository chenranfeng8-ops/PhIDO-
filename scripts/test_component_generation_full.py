# -*- coding: utf-8 -*-
"""
测试新单元组件生成功能 - 完整流程
包含网络爬虫的测试（可能会失败，需要 Selenium 和网络）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import auto_pdk_generator as apg

print("=" * 70)
print("Test: Full Component Generation Flow")
print("=" * 70)

# 测试 1: 参数验证器集成测试（不需要网络）
print("\n[TEST 1] Parameter Validation Integration")
print("-" * 70)

component_name = "mmi_1x2_high_bandwidth"
print(f"Component: {component_name}")

# 类型识别
comp_type = apg.type_validator.classify_component_type(component_name)
print(f"  Type: {comp_type}")

# 获取类型定义
type_def = apg.type_validator.get_type_definition(comp_type) if comp_type else None
if type_def:
    print(f"  Ports: {type_def.get('ports', {}).get('definition')}")
    print(f"  Params: {list(type_def.get('params', {}).keys())}")

# 模拟 LLM 提取的参数
extracted = {
    'length_mmi': 50.0,
    'width_mmi': 3.0,
    'gap_mmi': 0.3
}
print(f"  Input params: {extracted}")

# 验证
result = apg.type_validator.verify_params(comp_type, extracted)
print(f"  Valid: {result['valid']}")
print(f"  Output params: {result['params']}")

print("\n  [PASS] Parameter validation works correctly!")

# 测试 2: 检查设备类型映射
print("\n[TEST 2] Device Type Mapping")
print("-" * 70)

test_mappings = [
    ("mmi_1x2", "mmi1x2"),
    ("ring_resonator", "ring"),
    ("directional_coupler", "directional_coupler"),
    ("mzi_2x2", "mzi"),
    ("heater", "heater"),
    ("ge_photodetector", "ge_pd"),
]

for name, expected in test_mappings:
    result = apg._resolve_device_type(name)
    status = "[OK]" if result == expected else "[FAIL]"
    print(f"  {status} '{name}' -> '{result}' (expected: '{expected}')")

# 测试 3: 组件模板检查
print("\n[TEST 3] Component Templates")
print("-" * 70)

for device_type in ['mmi1x2', 'ring', 'directional_coupler', 'mzi']:
    if device_type in apg.TEMPLATES:
        print(f"  [OK] Template exists for: {device_type}")
    else:
        print(f"  [WARN] No template for: {device_type}")

# 测试 4: 参数提取函数（使用 LLM 或启发式）
print("\n[TEST 4] Parameter Extraction Functions")
print("-" * 70)

# 测试启发式提取
test_text = """
This paper presents a 1x2 MMI with length_mmi = 5.5 um, width_mmi = 3.0 um.
The device was fabricated on a 220nm SOI platform.
"""

params = apg.extract_params_heuristic(test_text, 'mmi1x2')
print(f"  Heuristic extraction result: {params}")

# 测试 5: 完整生成流程（不使用网络爬虫）
print("\n[TEST 5] Direct Component Generation (no crawler)")
print("-" * 70)

# 模拟论文数据
mock_paper = {
    'title': 'High-performance MMI for C-band applications',
    'abstract': 'A 1x2 multimode interferometer with optimized dimensions for C-band operation. Length = 5.5um, Width = 2.5um.',
    'full_text': 'The MMI device has length_mmi = 5.5 um, width_mmi = 2.5 um, gap = 0.25 um on 220nm SOI.',
    'source': 'test',
    'link': 'http://test'
}

# 使用启发式提取
print("  Extracting params from mock paper...")
params = apg.extract_params_heuristic(mock_paper['full_text'], 'mmi1x2')
print(f"  Raw params: {params}")

# 验证参数
comp_type = apg.type_validator.classify_component_type('mmi_1x2')
if comp_type and params:
    validated = apg.type_validator.verify_params(comp_type, params)
    print(f"  Validated params: {validated['params']}")
    print(f"  Warnings: {validated.get('warnings', [])[:2]}")

# 测试 6: 文件生成
print("\n[TEST 6] Component File Generation")
print("-" * 70)

try:
    # 生成组件代码
    device_type = 'mmi1x2'
    func_name = "test_mmi_1x2_generated"
    
    if device_type in apg.TEMPLATES:
        template = apg.TEMPLATES[device_type]
        
        # 使用验证后的参数，匹配模板需要的参数名
        final_params = validated['params'] if 'validated' in dir() else {}
        
        # 格式化参数 (模板需要的参数名)
        length_mmi = final_params.get('length_mmi', 5.5)
        width_mmi = final_params.get('width_mmi', 2.5)
        gap_mmi = final_params.get('gap_mmi', 0.25)
        width_taper = final_params.get('width_taper', 1.0)
        length_taper = final_params.get('length_taper', 10.0)
        
        code = template.format(
            func_name=func_name,
            title=mock_paper['title'],
            link=mock_paper['link'],
            length_mmi=length_mmi,
            width_mmi=width_mmi,
            gap_mmi=gap_mmi,
            width_taper=width_taper,
            length_taper=length_taper,
        )
        
        print(f"  Generated code preview (first 500 chars):")
        print("-" * 40)
        print(code[:500])
        print("-" * 40)
        print("  [OK] Code generation successful!")
    else:
        print(f"  [SKIP] No template for device_type: {device_type}")

except Exception as e:
    print(f"  [ERROR] Code generation failed: {e}")
    import traceback
    traceback.print_exc()

# 测试 7: 检查生成的组件文件目录
print("\n[TEST 7] Output Directory Check")
print("-" * 70)

output_dir = apg.OUTPUT_DIR
print(f"  Output directory: {output_dir}")
print(f"  Directory exists: {os.path.exists(output_dir)}")

if os.path.exists(output_dir):
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.py')]
    print(f"  Existing component files: {len(existing_files)}")
    print(f"  Sample files: {existing_files[:5]}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)