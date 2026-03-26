"""
从 gdsfactory 组件库自动归纳组件类型基础定义

输出:
- config/component_types_extracted.yaml: 归纳的类型定义
"""

import inspect
import yaml
import json
from collections import defaultdict
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import gdsfactory as gf


# 组件类型关键词映射
TYPE_KEYWORDS = {
    "mzi": ["mzi", "mach_zehnder", "interferometer"],
    "mmi": ["mmi", "multimode"],
    "ring": ["ring", "mrr", "resonator"],
    "coupler": ["coupler", "dc_", "directional"],
    "grating": ["grating", "gc_"],
    "crossing": ["crossing", "cross"],
    "bend": ["bend", "curve"],
    "straight": ["straight", "waveguide", "wg_"],
    "taper": ["taper"],
    "heater": ["heater", "thermal", "tin_"],
    "modulator": ["modulator", "mod_", "mzm", "pn_", "pindiode"],
    "detector": ["detector", "pd_", "photodiode", "ge_"],
    "splitter": ["splitter", "1x2", "2x2", "y_branch"],
    "via": ["via"],
    "pad": ["pad"],
    "metal": ["metal", "wire"],
    "text": ["text", "label"],
    "corner": ["corner"],
    "termination": ["termination", "terminator"],
}


def classify_component_type(name):
    """从组件名提取类型"""
    name_lower = name.lower()
    
    for comp_type, keywords in TYPE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return comp_type
    
    return "other"


def extract_port_config(ports):
    """从端口列表推断端口配置"""
    if not ports:
        return "0x0"
    
    try:
        # 过滤光学端口
        optical_ports = [p for p in ports if hasattr(p, 'name') and p.name.startswith("o") and p.name[1:].isdigit()]
        
        if not optical_ports:
            # 检查是否有电气端口
            electrical_ports = [p for p in ports if hasattr(p, 'name') and p.name.startswith("e")]
            if electrical_ports:
                return "electrical"
            return "0x0"
        
        # 排序端口号
        port_nums = sorted([int(p.name[1:]) for p in optical_ports])
        
        if not port_nums:
            return "0x0"
        
        # 推断输入输出端口数
        max_port = max(port_nums)
        num_ports = len(port_nums)
        
        # 常见模式
        if num_ports == 1:
            return "1x0"
        elif num_ports == 2:
            return "1x1"
        elif num_ports == 3:
            return "1x2"
        elif num_ports == 4:
            # 可能是 2x2 或 1x3
            if port_nums == [1, 2, 3, 4]:
                return "2x2"
            else:
                return "1x3"
        elif num_ports == 5:
            return "1x4"
        elif num_ports == 6:
            return "2x4"
        else:
            return f"{num_ports}ports"
    except Exception as e:
        return "error"


def extract_params_from_signature(func):
    """从函数签名提取参数信息"""
    try:
        sig = inspect.signature(func)
        params = {}
        
        for name, param in sig.parameters.items():
            if name in ["self", "cls", "kwargs", "args"]:
                continue
            
            param_info = {
                "default": None,
                "annotation": None,
            }
            
            # 默认值
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            
            # 类型注解
            if param.annotation != inspect.Parameter.empty:
                param_info["annotation"] = str(param.annotation)
            
            params[name] = param_info
        
        return params
    except Exception as e:
        return {}


def extract_docstring_info(func):
    """从 docstring 提取信息"""
    doc = func.__doc__ or ""
    
    info = {
        "description": "",
        "args": {},
    }
    
    # 提取描述（第一段）
    lines = doc.split("\n")
    desc_lines = []
    for line in lines:
        if line.strip().startswith("Args:") or line.strip().startswith("Returns:"):
            break
        desc_lines.append(line)
    info["description"] = " ".join(desc_lines).strip()[:200]
    
    # 提取 Args
    if "Args:" in doc:
        args_section = doc.split("Args:")[1].split("Returns:")[0]
        for line in args_section.split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    arg_name = parts[0].strip().strip("- ")
                    arg_desc = parts[1].strip()
                    info["args"][arg_name] = arg_desc[:100]
    
    return info


def analyze_gdsfactory_components():
    """分析 gdsfactory 所有组件"""
    
    print("=" * 60)
    print("从 gdsfactory 归纳组件类型定义")
    print("=" * 60)
    
    all_components = [n for n in dir(gf.components) if not n.startswith("_")]
    print(f"总组件数: {len(all_components)}")
    
    type_data = defaultdict(lambda: {
        "ports_patterns": [],
        "params": {},
        "components": [],
        "descriptions": [],
    })
    
    success_count = 0
    error_count = 0
    
    for comp_name in all_components:
        try:
            comp_func = getattr(gf.components, comp_name)
            
            if not callable(comp_func):
                continue
            
            # 识别类型
            comp_type = classify_component_type(comp_name)
            
            # 提取参数
            params = extract_params_from_signature(comp_func)
            
            # 实例化获取端口
            try:
                c = comp_func()
                port_config = extract_port_config(list(c.ports))
            except Exception as e:
                port_config = "error"
            
            # 提取 docstring
            doc_info = extract_docstring_info(comp_func)
            
            # 归纳
            type_data[comp_type]["ports_patterns"].append(port_config)
            type_data[comp_type]["components"].append(comp_name)
            
            # 合并参数
            for param_name, param_info in params.items():
                if param_name not in type_data[comp_type]["params"]:
                    type_data[comp_type]["params"][param_name] = {
                        "default_values": [],
                        "occurrences": 0,
                    }
                type_data[comp_type]["params"][param_name]["default_values"].append(
                    param_info["default"]
                )
                type_data[comp_type]["params"][param_name]["occurrences"] += 1
            
            if doc_info["description"]:
                type_data[comp_type]["descriptions"].append(doc_info["description"])
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    print(f"\n成功分析: {success_count}")
    print(f"失败: {error_count}")
    
    return type_data


def sanitize_value(v):
    """将值转换为可序列化的格式"""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: sanitize_value(val) for k, val in v.items()}
    # 其他类型转字符串
    return str(v)


def generate_type_definitions(type_data):
    """生成类型定义"""
    
    definitions = {}
    
    for comp_type, data in type_data.items():
        if len(data["components"]) < 2:
            continue  # 至少2个样本
        
        # 找最常见的端口配置
        ports_counter = defaultdict(int)
        for p in data["ports_patterns"]:
            ports_counter[p] += 1
        most_common_port = max(ports_counter.keys(), key=lambda k: ports_counter[k])
        
        # 找常见参数
        common_params = {}
        for param_name, param_info in data["params"].items():
            if param_info["occurrences"] >= len(data["components"]) * 0.3:  # 出现在30%以上
                # 找最常见默认值
                valid_defaults = [v for v in param_info["default_values"] if v is not None]
                if valid_defaults:
                    # 取第一个有效值，并清理
                    default_val = sanitize_value(valid_defaults[0])
                    common_params[param_name] = {
                        "default": default_val,
                        "frequency": f"{param_info['occurrences']}/{len(data['components'])}",
                    }
        
        definitions[comp_type] = {
            "ports": most_common_port,
            "port_variants": list(set(data["ports_patterns"]))[:5],
            "params": common_params,
            "sample_count": len(data["components"]),
            "sample_components": data["components"][:10],
            "description": data["descriptions"][0] if data["descriptions"] else "",
        }
    
    return definitions


def main():
    # 分析组件
    type_data = analyze_gdsfactory_components()
    
    # 生成定义
    definitions = generate_type_definitions(type_data)
    
    # 输出统计
    print("\n" + "=" * 60)
    print("归纳结果:")
    print("=" * 60)
    
    for comp_type in sorted(definitions.keys(), key=lambda x: definitions[x]["sample_count"], reverse=True):
        d = definitions[comp_type]
        print(f"\n{comp_type.upper()}:")
        print(f"  端口: {d['ports']}")
        print(f"  样本数: {d['sample_count']}")
        print(f"  常见参数: {list(d['params'].keys())[:5]}")
        print(f"  示例组件: {d['sample_components'][:5]}")
    
    # 保存结果
    output_path = Path(__file__).parent.parent / "PhotonicsAI" / "config" / "component_types_extracted.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(definitions, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n\n已保存到: {output_path}")
    
    # 同时保存 JSON 方便查看
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)
    
    print(f"JSON 版本: {json_path}")
    
    return definitions


if __name__ == "__main__":
    main()