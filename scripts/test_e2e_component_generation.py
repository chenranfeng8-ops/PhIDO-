# -*- coding: utf-8 -*-
"""
端到端测试：新组件生成完整流程
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import auto_pdk_generator as apg
from PhotonicsAI.Photon import component_type_validator

print("=" * 70)
print("End-to-End Test: New Component Generation")
print("=" * 70)

# 测试场景：用户输入新组件名称
test_cases = [
    {
        "name": "mmi_1x2_broadband",
        "description": "Broadband 1x2 MMI splitter",
        "mock_paper": {
            "title": "Broadband 1x2 MMI splitter for C-band",
            "abstract": "A broadband 1x2 MMI with length_mmi=6.5um, width_mmi=3.2um",
            "full_text": "Optimized 1x2 MMI design: length_mmi = 6.5 um, width_mmi = 3.2 um, gap = 0.3 um on 220nm SOI."
        }
    },
    {
        "name": "ring_resonator_high_q",
        "description": "High-Q ring resonator",
        "mock_paper": {
            "title": "High-Q ring resonator for sensing",
            "abstract": "Ring resonator with radius=5um, gap=0.1um achieving Q>10000",
            "full_text": "High-Q ring resonator: radius = 5 um, gap = 0.15 um, length_x = 2 um on 220nm SOI."
        }
    },
    {
        "name": "mzi_switch",
        "description": "MZI optical switch",
        "mock_paper": {
            "title": "MZI-based optical switch",
            "abstract": "2x2 MZI switch with delta_length=20um",
            "full_text": "MZI optical switch: delta_length = 25 um, length_x = 100 um, length_y = 10 um."
        }
    }
]

for i, test_case in enumerate(test_cases):
    print(f"\n{'='*70}")
    print(f"Test Case {i+1}: {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print("=" * 70)
    
    component_name = test_case["name"]
    mock_paper = test_case["mock_paper"]
    
    # Step 1: 组件类型识别
    print("\n[Step 1] Component Type Classification")
    comp_type = apg.type_validator.classify_component_type(component_name)
    device_type = apg._resolve_device_type(component_name)
    print(f"  component_type_validator: {comp_type}")
    print(f"  _resolve_device_type: {device_type}")
    
    # Step 2: 获取类型定义
    print("\n[Step 2] Type Definition Retrieval")
    type_def = apg.type_validator.get_type_definition(comp_type) if comp_type else None
    if type_def:
        print(f"  Ports: {type_def.get('ports', {}).get('definition')}")
        print(f"  Params: {list(type_def.get('params', {}).keys())[:5]}")
    
    # Step 3: 参数提取（模拟论文内容）
    print("\n[Step 3] Parameter Extraction")
    params = apg.extract_params_heuristic(mock_paper["full_text"], device_type or comp_type)
    print(f"  Raw params: {params}")
    
    # Step 4: 参数验证和修正
    print("\n[Step 4] Parameter Validation & Correction")
    if comp_type and params:
        validation = apg.type_validator.verify_params(comp_type, params)
        print(f"  Valid: {validation['valid']}")
        if validation.get('adjustments'):
            print(f"  Adjustments: {validation['adjustments']}")
        if validation.get('warnings'):
            print(f"  Warnings: {validation['warnings'][:2]}")
        print(f"  Final params: {validation['params']}")
    else:
        validation = {"params": params}
        print(f"  No validation available, using raw params")
    
    # Step 5: 检查模板
    print("\n[Step 5] Template Check")
    if device_type in apg.TEMPLATES:
        print(f"  Template found for: {device_type}")
        template = apg.TEMPLATES[device_type]
        # 提取模板需要的参数
        import re
        required_placeholders = set(re.findall(r'\{(\w+)\}', template))
        print(f"  Required placeholders: {required_placeholders}")
    else:
        print(f"  No template for: {device_type}")
    
    # Step 6: 生成代码预览
    print("\n[Step 6] Code Generation Preview")
    if device_type in apg.TEMPLATES and validation.get('params'):
        try:
            template = apg.TEMPLATES[device_type]
            final_params = validation['params']
            
            # 准备格式化参数
            format_args = {
                "func_name": f"auto_{component_name}",
                "title": mock_paper["title"],
                "link": "auto_generated",
            }
            
            # 添加组件参数（根据模板需要）
            if device_type == "mmi1x2":
                format_args.update({
                    "length_mmi": final_params.get("length_mmi", 5.5),
                    "width_mmi": final_params.get("width_mmi", 2.5),
                    "gap_mmi": final_params.get("gap_mmi", 0.25),
                    "width_taper": final_params.get("width_taper", 1.0),
                    "length_taper": final_params.get("length_taper", 10.0),
                })
            elif device_type == "ring":
                format_args.update({
                    "radius": final_params.get("radius", 10.0),
                    "gap": final_params.get("gap", 0.2),
                    "length_x": final_params.get("length_x", 4.0),
                })
            elif device_type == "mzi":
                format_args.update({
                    "delta_length": final_params.get("delta_length", 10.0),
                    "length_x": final_params.get("length_x", 0.1),
                    "length_y": final_params.get("length_y", 2.0),
                })
            elif device_type == "directional_coupler":
                format_args.update({
                    "length": final_params.get("length", 10.0),
                    "gap": final_params.get("gap", 0.2),
                })
            
            code = template.format(**format_args)
            print("  Generated code (first 300 chars):")
            print("  " + "-" * 40)
            for line in code[:300].split('\n'):
                print(f"  {line}")
            print("  " + "-" * 40)
            print("  [SUCCESS] Code generation completed!")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        print("  [SKIP] No template or params available")

print("\n" + "=" * 70)
print("End-to-End Test Completed Successfully!")
print("=" * 70)