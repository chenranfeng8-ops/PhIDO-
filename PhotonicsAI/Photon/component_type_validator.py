"""
组件类型验证模块

用于验证新组件的参数是否符合类型基础定义
"""

import yaml
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


# 加载类型定义
_TYPES_CACHE = None

def load_component_types() -> Dict[str, Any]:
    """加载组件类型定义"""
    global _TYPES_CACHE
    
    if _TYPES_CACHE is not None:
        return _TYPES_CACHE
    
    config_path = Path(__file__).parent.parent / "config" / "component_types.yaml"
    
    if not config_path.exists():
        print(f"Warning: Component types config not found at {config_path}")
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        _TYPES_CACHE = yaml.safe_load(f)
    
    return _TYPES_CACHE or {}


# 组件类型关键词映射 (按优先级排序，更具体的类型放在前面)
_TYPE_KEYWORDS = {
    # 高优先级 - 具体组件类型
    "grating": ["grating", "gc_", "coupler_grating"],  # 放在 coupler 前面
    "y_branch": ["y_branch", "y branch", "y-branch", "ybranch"],  # Y 分支
    "mzi": ["mzi", "mach_zehnder", "interferometer"],
    "modulator": ["modulator", "mod_", "mzm", "pn_", "pindiode"],
    "heater": ["heater", "thermal", "tin_", "phase_shifter"],
    "mmi": ["mmi", "multimode"],
    "ring": ["ring", "mrr", "resonator"],
    "crossing": ["crossing", "cross"],
    "taper": ["taper"],
    # 低优先级 - 通用类型
    "coupler": ["coupler", "dc_", "directional"],
    "bend": ["bend", "curve"],
    "straight": ["straight", "waveguide", "wg_"],
    "via": ["via"],
    "pad": ["pad"],
}


def classify_component_type(component_name: str) -> Optional[str]:
    """从组件名识别类型
    
    Args:
        component_name: 组件名称，如 "mzi_2x2_heater", "ring_resonator"
        
    Returns:
        组件类型，如 "mzi", "ring"
    """
    name_lower = component_name.lower()
    
    # 按优先级顺序匹配 (字典顺序即为优先级)
    for comp_type, keywords in _TYPE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return comp_type
    
    return None


def get_type_definition(component_type: str) -> Optional[Dict[str, Any]]:
    """获取组件类型定义
    
    Args:
        component_type: 组件类型
        
    Returns:
        类型定义字典
    """
    types = load_component_types()
    return types.get(component_type)


def verify_ports(component_type: str, expected_ports: str) -> Dict[str, Any]:
    """验证端口配置
    
    Args:
        component_type: 组件类型
        expected_ports: 期望的端口配置，如 "2x2"
        
    Returns:
        验证结果
    """
    type_def = get_type_definition(component_type)
    
    if not type_def:
        return {
            "valid": False,
            "error": f"Unknown component type: {component_type}"
        }
    
    ports_def = type_def.get("ports", {})
    definition = ports_def.get("definition", "unknown")
    variants = ports_def.get("variants", [])
    
    # 检查是否匹配
    is_valid = (
        expected_ports == definition or 
        expected_ports in variants or
        re.match(r"^\d+x\d+$", expected_ports)  # 通用 NxM 格式
    )
    
    return {
        "valid": is_valid,
        "expected": expected_ports,
        "definition": definition,
        "variants": variants,
        "message": "Ports match" if is_valid else f"Expected {definition} or {variants}"
    }


def verify_param_range(param_name: str, value: float, type_def: Dict[str, Any]) -> Dict[str, Any]:
    """验证参数范围
    
    Args:
        param_name: 参数名
        value: 参数值
        type_def: 类型定义
        
    Returns:
        验证结果
    """
    params_def = type_def.get("params", {})
    
    if param_name not in params_def:
        return {
            "valid": True,
            "warning": f"Parameter '{param_name}' not defined in type specification"
        }
    
    param_def = params_def[param_name]
    range_def = param_def.get("range")
    
    if range_def:
        min_val, max_val = range_def
        
        if min_val <= value <= max_val:
            return {
                "valid": True,
                "value": value,
                "range": range_def
            }
        else:
            # 自动修正到有效范围
            adjusted = max(min_val, min(max_val, value))
            return {
                "valid": False,
                "value": value,
                "range": range_def,
                "adjusted": adjusted,
                "message": f"Value {value} out of range [{min_val}, {max_val}], adjusted to {adjusted}"
            }
    
    return {"valid": True, "value": value}


def verify_params(component_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """验证所有参数
    
    Args:
        component_type: 组件类型
        params: 参数字典
        
    Returns:
        验证结果，包含修正后的参数
    """
    type_def = get_type_definition(component_type)
    
    if not type_def:
        return {
            "valid": False,
            "error": f"Unknown component type: {component_type}",
            "params": params
        }
    
    results = {
        "valid": True,
        "params": {},
        "warnings": [],
        "adjustments": []
    }
    
    params_def = type_def.get("params", {})
    
    # 验证提供的参数
    for param_name, value in params.items():
        if param_name in params_def:
            check = verify_param_range(param_name, value, type_def)
            
            if check["valid"]:
                results["params"][param_name] = value
            else:
                results["valid"] = False
                results["params"][param_name] = check.get("adjusted", value)
                results["adjustments"].append(check.get("message", ""))
        else:
            results["params"][param_name] = value
            results["warnings"].append(f"Unknown parameter: {param_name}")
    
    # 填充缺失的必需参数
    for param_name, param_def in params_def.items():
        if param_name not in params:
            default = param_def.get("default")
            if default is not None:
                results["params"][param_name] = default
                results["warnings"].append(f"Missing parameter '{param_name}', using default: {default}")
    
    return results


def get_required_params(component_type: str) -> List[str]:
    """获取组件类型的必需参数列表
    
    Args:
        component_type: 组件类型
        
    Returns:
        必需参数名列表
    """
    type_def = get_type_definition(component_type)
    
    if not type_def:
        return []
    
    params_def = type_def.get("params", {})
    required = []
    
    for param_name, param_def in params_def.items():
        if param_def.get("required", False) or param_def.get("default") is not None:
            required.append(param_name)
    
    return required


def get_physical_constraints(component_type: str) -> List[Dict[str, str]]:
    """获取组件类型的物理约束
    
    Args:
        component_type: 组件类型
        
    Returns:
        物理约束列表
    """
    type_def = get_type_definition(component_type)
    
    if not type_def:
        return []
    
    return type_def.get("physical_constraints", [])


def validate_component(
    component_name: str,
    ports: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """综合验证组件
    
    Args:
        component_name: 组件名称
        ports: 端口配置
        params: 参数字典
        
    Returns:
        验证结果
    """
    # 识别组件类型
    component_type = classify_component_type(component_name)
    
    if not component_type:
        return {
            "valid": False,
            "error": f"Cannot classify component: {component_name}",
            "suggestion": "Component type not recognized. Check naming or add to type keywords."
        }
    
    type_def = get_type_definition(component_type)
    
    results = {
        "valid": True,
        "component_type": component_type,
        "component_name": component_name,
        "type_definition": type_def,
        "port_check": None,
        "param_check": None
    }
    
    # 验证端口
    if ports:
        results["port_check"] = verify_ports(component_type, ports)
        if not results["port_check"]["valid"]:
            results["valid"] = False
    
    # 验证参数
    if params:
        results["param_check"] = verify_params(component_type, params)
        if not results["param_check"]["valid"]:
            results["valid"] = False
    
    return results


# =============================================================================
# LLM 集成函数
# =============================================================================

def generate_verification_prompt(component_name: str, papers_text: str) -> str:
    """生成带类型约束的 LLM prompt
    
    Args:
        component_name: 组件名称
        papers_text: 论文内容
        
    Returns:
        完整的 prompt
    """
    component_type = classify_component_type(component_name)
    
    if not component_type:
        return f"""
        无法识别组件类型: {component_name}
        请基于光子学标准设计规范提取参数。
        
        论文内容:
        {papers_text}
        """
    
    type_def = get_type_definition(component_type)
    
    # 构建参数规范
    params_spec = []
    for param_name, param_def in type_def.get("params", {}).items():
        range_str = ""
        if "range" in param_def:
            range_str = f" (范围: {param_def['range']} {param_def.get('unit', '')})"
        
        default_str = ""
        if "default" in param_def:
            default_str = f", 默认值: {param_def['default']}"
        
        params_spec.append(
            f"  - {param_name}: {param_def.get('description', '')}{range_str}{default_str}"
        )
    
    # 构建端口规范
    ports_def = type_def.get("ports", {})
    ports_spec = f"""
端口配置:
  - 定义: {ports_def.get('definition', 'unknown')}
  - 变体: {ports_def.get('variants', [])}
  - 规则: {ports_def.get('rules', '')}
"""
    
    # 构建物理约束
    constraints = type_def.get("physical_constraints", [])
    constraints_spec = []
    for c in constraints:
        constraints_spec.append(
            f"  - {c.get('name', '')}: {c.get('formula', '')} ({c.get('description', '')})"
        )
    
    prompt = f"""
你正在设计 {component_name} 组件，类型为 {component_type}。

=== 组件类型定义（必须遵守） ===
描述: {type_def.get('description', '')}

{ports_spec}

参数规范:
{chr(10).join(params_spec)}

物理约束:
{chr(10).join(constraints_spec) if constraints_spec else '  (无特定约束)'}

=== 论文内容 ===
{papers_text}

=== 提取要求 ===
1. 端口配置必须符合上述定义
2. 参数值必须在指定范围内
3. 缺失参数使用默认值
4. 输出 JSON 格式:
{{
  "ports": "端口配置",
  "params": {{
    "param1": value1,
    "param2": value2
  }},
  "confidence": "high/medium/low",
  "notes": "提取说明"
}}
"""
    
    return prompt


# 测试
if __name__ == "__main__":
    # 测试类型识别
    print("=== 类型识别测试 ===")
    test_names = ["mzi_2x2_heater", "ring_resonator", "mmi1x2", "grating_coupler", "unknown_component"]
    for name in test_names:
        comp_type = classify_component_type(name)
        print(f"  {name} → {comp_type}")
    
    # 测试端口验证
    print("\n=== 端口验证测试 ===")
    result = verify_ports("mzi", "2x2")
    print(f"  MZI 2x2: {result}")
    
    result = verify_ports("mzi", "1x3")
    print(f"  MZI 1x3: {result}")
    
    # 测试参数验证
    print("\n=== 参数验证测试 ===")
    result = verify_params("mzi", {"delta_length": 100.0, "unknown_param": 5.0})
    print(f"  Result: {result}")
    
    # 测试范围检查
    print("\n=== 范围检查测试 ===")
    result = verify_params("ring", {"radius": 500.0})  # 超出范围
    print(f"  Ring radius=500: {result}")
    
    # 测试完整验证
    print("\n=== 完整验证测试 ===")
    result = validate_component(
        "mzi_high_speed",
        ports="2x2",
        params={"delta_length": 50.0, "length_x": 100.0}
    )
    print(f"  Valid: {result['valid']}")
    print(f"  Type: {result['component_type']}")