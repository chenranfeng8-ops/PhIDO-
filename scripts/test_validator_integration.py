# -*- coding: utf-8 -*-
"""测试组件类型验证器与 auto_pdk_generator 的衔接"""

import sys
sys.path.insert(0, 'scripts')

# 测试 1: 导入模块
print("=" * 60)
print("Test 1: Module Import")
print("=" * 60)

try:
    import auto_pdk_generator as apg
    print("[OK] auto_pdk_generator imported")
except Exception as e:
    print(f"[FAIL] auto_pdk_generator import failed: {e}")
    sys.exit(1)

try:
    from PhotonicsAI.Photon import component_type_validator
    print("[OK] component_type_validator imported")
except Exception as e:
    print(f"[FAIL] component_type_validator import failed: {e}")
    sys.exit(1)

# 测试 2: 类型识别
print("\n" + "=" * 60)
print("Test 2: Component Type Classification")
print("=" * 60)

test_names = [
    'mzi_2x2_heater',
    'ring_resonator', 
    'mmi1x2',
    'grating_coupler',
    'heater_tin_cband',
    'directional_coupler',
    'bend_euler',
    'straight_waveguide',
    'unknown_component'
]

for name in test_names:
    t = apg.type_validator.classify_component_type(name)
    status = "[OK]" if t else "[?]"
    print(f"  {status} {name} -> {t}")

# 测试 3: 类型定义获取
print("\n" + "=" * 60)
print("Test 3: Type Definition Retrieval")
print("=" * 60)

for comp_type in ['mzi', 'mmi', 'ring', 'coupler', 'heater']:
    type_def = apg.type_validator.get_type_definition(comp_type)
    if type_def:
        ports = type_def.get('ports', {}).get('definition', 'N/A')
        params = list(type_def.get('params', {}).keys())[:3]
        print(f"  [OK] {comp_type}: ports={ports}, params={params}")
    else:
        print(f"  [FAIL] {comp_type}: not found")

# 测试 4: 参数范围验证
print("\n" + "=" * 60)
print("Test 4: Parameter Range Validation")
print("=" * 60)

# 测试正常范围
result = apg.type_validator.verify_params('ring', {'radius': 10.0, 'gap': 0.2})
print(f"  Ring (radius=10, gap=0.2): valid={result['valid']}")
print(f"    params: {result['params']}")

# 测试超出范围
result = apg.type_validator.verify_params('ring', {'radius': 500.0})
print(f"\n  Ring (radius=500): valid={result['valid']}")
print(f"    adjustments: {result.get('adjustments', [])}")
print(f"    corrected: {result['params']}")

# 测试缺失参数
result = apg.type_validator.verify_params('mzi', {'delta_length': 100.0})
print(f"\n  MZI (only delta_length=100):")
print(f"    filled params: {list(result['params'].keys())}")

# 测试 5: 端口验证
print("\n" + "=" * 60)
print("Test 5: Port Configuration Validation")
print("=" * 60)

test_ports = [
    ('mzi', '2x2'),
    ('mzi', '1x2'),
    ('mzi', '1x3'),
    ('ring', '1x2'),
    ('ring', '2x2'),
    ('mmi', '1x2'),
    ('mmi', '2x2'),
]

for comp_type, ports in test_ports:
    result = apg.type_validator.verify_ports(comp_type, ports)
    status = "[OK]" if result['valid'] else "[FAIL]"
    print(f"  {status} {comp_type} ports={ports}: {result.get('message', '')}")

# 测试 6: 完整验证流程
print("\n" + "=" * 60)
print("Test 6: Complete Validation Flow")
print("=" * 60)

result = apg.type_validator.validate_component(
    'mzi_high_speed',
    ports='2x2',
    params={'delta_length': 50.0, 'length_x': 100.0}
)
print(f"  Component: mzi_high_speed")
print(f"  Type: {result['component_type']}")
print(f"  Valid: {result['valid']}")
if result.get('port_check'):
    print(f"  Port check: {result['port_check']}")
if result.get('param_check'):
    print(f"  Param check: valid={result['param_check']['valid']}")

# 测试 7: LLM prompt 生成
print("\n" + "=" * 60)
print("Test 7: Constrained LLM Prompt Generation")
print("=" * 60)

prompt = apg.type_validator.generate_verification_prompt(
    'ring_resonator',
    'This paper describes a ring resonator with radius 5um and gap 0.15um...'
)
print(f"  Prompt length: {len(prompt)} chars")
print(f"  Contains type definition: {'type' in prompt.lower()}")
print(f"  Contains parameter range: {'range' in prompt.lower() or 'range' in prompt}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)