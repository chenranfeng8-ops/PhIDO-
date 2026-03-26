# -*- coding: utf-8 -*-
"""
测试动态组件生成功能（模拟论文数据，无需网络爬虫）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import auto_pdk_generator as apg
from PhotonicsAI.Photon import component_type_validator

print("=" * 70)
print("Test: Dynamic Component Generation (Simulated Papers)")
print("=" * 70)

# 模拟论文数据
MOCK_PAPERS = {
    "y_branch": [
        {
            "title": "Low-loss Y-branch splitter for silicon photonics",
            "abstract": "A compact Y-branch splitter with insertion loss < 0.2 dB and splitting ratio 50:50.",
            "full_text": """
            Y-branch splitter design with optimized S-bend transition.
            Parameters: length = 25 um, angle = 8 degrees, gap = 1.5 um.
            The device achieves 0.15 dB insertion loss and 50:50 splitting ratio over 100 nm bandwidth.
            Fabricated on 220nm SOI platform with strip waveguides of 0.5 um width.
            """,
            "source": "mock_arxiv",
            "link": "http://mock-paper-1"
        },
        {
            "title": "Broadband Y-junction power splitter",
            "abstract": "Ultra-broadband Y-junction with flat response across C and L bands.",
            "full_text": """
            Y-junction splitter with adiabatic transition.
            Total length = 30 um, output separation = 2.0 um, bend radius = 15 um.
            Insertion loss < 0.3 dB, imbalance < 0.1 dB over 120 nm bandwidth.
            """,
            "source": "mock_optica",
            "link": "http://mock-paper-2"
        }
    ],
    "mzi": [
        {
            "title": "Compact MZI switch for optical interconnects",
            "abstract": "High-speed MZI switch with 10 GHz bandwidth.",
            "full_text": """
            Mach-Zehnder interferometer switch.
            delta_length = 50 um, length_x = 100 um, length_y = 5 um.
            Uses mmi1x2 as splitter and combiner.
            Switching voltage = 3V, extinction ratio = 20 dB.
            """,
            "source": "mock_ieee",
            "link": "http://mock-paper-3"
        }
    ]
}

def test_dynamic_generation(component_name: str, mock_papers: list):
    """测试单个组件的动态生成"""
    print(f"\n{'='*70}")
    print(f"Testing: {component_name}")
    print("=" * 70)
    
    # Step 1: 类型识别
    print("\n[Step 1] Component Type Classification")
    comp_type = component_type_validator.classify_component_type(component_name)
    device_type = apg._resolve_device_type(component_name)
    print(f"  component_type: {comp_type}")
    print(f"  device_type: {device_type}")
    
    if not comp_type or not device_type:
        print("  [ERROR] Cannot identify component type!")
        return None
    
    # Step 2: 获取类型定义
    print("\n[Step 2] Type Definition")
    type_def = component_type_validator.get_type_definition(comp_type)
    if type_def:
        print(f"  Ports: {type_def.get('ports', {}).get('definition')}")
        print(f"  Required params: {list(type_def.get('params', {}).keys())[:5]}")
    
    # Step 3: 模拟论文参数提取
    print("\n[Step 3] Parameter Extraction from Papers")
    all_params = []
    for paper in mock_papers:
        params = apg.extract_params_heuristic(paper['full_text'], device_type)
        if params:
            print(f"  Paper '{paper['title'][:30]}...': {params}")
            all_params.append(params)
    
    # Step 4: 参数综合（取平均值或中位数）
    print("\n[Step 4] Parameter Aggregation")
    if all_params:
        # 合并所有参数
        combined_params = {}
        param_counts = {}
        
        for params in all_params:
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if key not in combined_params:
                        combined_params[key] = []
                        param_counts[key] = 0
                    combined_params[key].append(value)
                    param_counts[key] += 1
        
        # 计算平均值
        final_params = {}
        for key, values in combined_params.items():
            if values:
                final_params[key] = sum(values) / len(values)
        
        print(f"  Aggregated params: {final_params}")
    else:
        final_params = {}
        print("  [WARN] No params extracted, using defaults")
    
    # Step 5: 参数验证
    print("\n[Step 5] Parameter Validation")
    validation = component_type_validator.verify_params(comp_type, final_params)
    print(f"  Valid: {validation['valid']}")
    if validation.get('adjustments'):
        print(f"  Adjustments: {validation['adjustments']}")
    if validation.get('warnings'):
        print(f"  Warnings: {validation['warnings'][:2]}")
    print(f"  Final params: {validation['params']}")
    
    # Step 6: 代码生成
    print("\n[Step 6] Code Generation")
    if device_type in apg.TEMPLATES:
        template = apg.TEMPLATES[device_type]
        
        # 准备参数
        format_args = {
            "func_name": f"auto_{component_name.replace('-', '_')}",
            "title": mock_papers[0]['title'] if mock_papers else "Auto-generated",
            "link": mock_papers[0]['link'] if mock_papers else "auto",
        }
        
        # 添加组件参数
        final_params = validation['params']
        if device_type == "y_branch":
            format_args.update({
                "length": final_params.get("length", 25.0),
                "angle": final_params.get("angle", 10.0),
                "gap": final_params.get("gap", 2.0),
            })
        elif device_type == "mzi":
            format_args.update({
                "delta_length": final_params.get("delta_length", 10.0),
                "length_x": final_params.get("length_x", 0.1),
                "length_y": final_params.get("length_y", 2.0),
            })
        elif device_type == "mmi1x2":
            format_args.update({
                "length_mmi": final_params.get("length_mmi", 5.5),
                "width_mmi": final_params.get("width_mmi", 2.5),
                "gap_mmi": final_params.get("gap_mmi", 0.25),
                "width_taper": final_params.get("width_taper", 1.0),
                "length_taper": final_params.get("length_taper", 10.0),
            })
        
        code = template.format(**format_args)
        
        print("  Generated code:")
        print("  " + "-" * 40)
        for line in code.split('\n')[:20]:
            print(f"  {line}")
        print("  " + "-" * 40)
        
        return code
    else:
        print(f"  [ERROR] No template for device_type: {device_type}")
        return None


# 测试 Y-branch
print("\n" + "=" * 70)
print("Test Case 1: Y-branch Splitter")
print("=" * 70)

y_branch_code = test_dynamic_generation(
    "y_branch_splitter",
    MOCK_PAPERS["y_branch"]
)

# 测试 MZI
print("\n" + "=" * 70)
print("Test Case 2: MZI Switch")
print("=" * 70)

mzi_code = test_dynamic_generation(
    "mzi_switch",
    MOCK_PAPERS["mzi"]
)

# 验证生成的代码
print("\n" + "=" * 70)
print("Verification: Execute Generated Code")
print("=" * 70)

if y_branch_code:
    print("\n[Y-branch] Checking syntax...")
    try:
        compile(y_branch_code, '<string>', 'exec')
        print("  [OK] Syntax valid")
    except SyntaxError as e:
        print(f"  [ERROR] Syntax error: {e}")

if mzi_code:
    print("\n[MZI] Checking syntax...")
    try:
        compile(mzi_code, '<string>', 'exec')
        print("  [OK] Syntax valid")
    except SyntaxError as e:
        print(f"  [ERROR] Syntax error: {e}")

print("\n" + "=" * 70)
print("Dynamic Generation Test Complete!")
print("=" * 70)