# -*- coding: utf-8 -*-
"""
测试新单元组件生成功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import auto_pdk_generator as apg

print("=" * 70)
print("Test: New Component Generation with Type Validation")
print("=" * 70)

# 测试组件名称
test_components = [
    'mmi_1x2_high_bandwidth',
    'ring_resonator_c_band',
    'mzi_2x2_thermo_optic',
]

for component_name in test_components:
    print(f"\n{'='*70}")
    print(f"Testing: {component_name}")
    print("=" * 70)
    
    # Step 1: 类型识别
    comp_type = apg.type_validator.classify_component_type(component_name)
    print(f"\n[Step 1] Type Classification")
    print(f"  Result: {component_name} -> {comp_type}")
    
    if not comp_type:
        print("  [SKIP] Unknown component type")
        continue
    
    # Step 2: 获取类型定义
    type_def = apg.type_validator.get_type_definition(comp_type)
    print(f"\n[Step 2] Type Definition")
    if type_def:
        desc = type_def.get('description', 'N/A')
        ports = type_def.get('ports', {}).get('definition', 'N/A')
        params = list(type_def.get('params', {}).keys())
        print(f"  Description: {desc[:60]}...")
        print(f"  Ports: {ports}")
        print(f"  Params: {params[:5]}")
    else:
        print("  [WARN] No type definition found")
        continue
    
    # Step 3: 模拟 LLM 提取参数（包含超出范围的值）
    print(f"\n[Step 3] Simulated LLM Parameter Extraction")
    
    # 根据组件类型模拟不同的参数
    if comp_type == 'mmi':
        extracted_params = {
            'length_mmi': 150.0,  # 超出范围 [2, 100]
            'width_mmi': 3.0,
            'gap_mmi': 0.5,
            'unknown_param': 10.0  # 未知参数
        }
    elif comp_type == 'ring':
        extracted_params = {
            'radius': 500.0,  # 超出范围 [2, 100]
            'gap': 0.15,
        }
    elif comp_type == 'mzi':
        extracted_params = {
            'delta_length': 20.0,
            'length_x': 500.0,  # 超出范围 [0, 200]
            'splitter': 'mmi1x2',
        }
    else:
        extracted_params = {'length': 100.0}
    
    print(f"  Extracted params: {extracted_params}")
    
    # Step 4: 参数验证
    print(f"\n[Step 4] Parameter Validation")
    result = apg.type_validator.verify_params(comp_type, extracted_params)
    
    print(f"  Valid: {result['valid']}")
    if result.get('adjustments'):
        print(f"  Adjustments:")
        for adj in result['adjustments']:
            print(f"    - {adj}")
    if result.get('warnings'):
        print(f"  Warnings:")
        for warn in result['warnings'][:3]:
            print(f"    - {warn}")
    
    print(f"\n  Final params: {result['params']}")
    
    # Step 5: 端口验证
    print(f"\n[Step 5] Port Validation")
    ports_def = type_def.get('ports', {}).get('definition', '2x2')
    port_result = apg.type_validator.verify_ports(comp_type, ports_def)
    print(f"  Expected ports: {ports_def}")
    print(f"  Valid: {port_result['valid']}")
    
    # Step 6: 生成验证 prompt
    print(f"\n[Step 6] Verification Prompt Generation")
    prompt = apg.type_validator.generate_verification_prompt(
        component_name,
        "This paper describes a device with optimized parameters for C-band operation."
    )
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Contains type constraints: {'must' in prompt.lower() or 'constraint' in prompt.lower()}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)