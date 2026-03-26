# -*- coding: utf-8 -*-
"""
测试真实动态生成 - 使用常见组件名称
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import auto_pdk_generator as apg

print("=" * 70)
print("Test: Real Discovery with Common Component Names")
print("=" * 70)

# 测试多个组件
test_cases = [
    ("1x2 MMI", "mmi1x2"),      # 常见组件，应该能找到论文
    ("directional coupler", "directional_coupler"),  # 常见组件
    ("ring resonator", "ring"),  # 常见组件
]

for component_name, expected_type in test_cases:
    print(f"\n{'='*70}")
    print(f"Testing: {component_name} (expected: {expected_type})")
    print("=" * 70)
    
    try:
        result = apg.discover_and_generate(
            component_name=component_name,
            max_papers=5
        )
        
        print(f"\n[Result]")
        print(f"  Device type: {result.get('device_type', 'N/A')}")
        print(f"  Papers found: {result.get('papers_found', 0)}")
        print(f"  Parameters: {result.get('params', {})}")
        
        if result.get('filepath'):
            print(f"  Generated: {result['filepath']}")
            print("  [SUCCESS]")
        else:
            print(f"  Error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "=" * 70)
print("Done")
print("=" * 70)