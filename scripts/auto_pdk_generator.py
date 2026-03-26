import time
import json
import os
import re
import random
import urllib.parse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import gdsfactory as gf

# 导入组件类型验证器
try:
    from PhotonicsAI.Photon import component_type_validator as type_validator
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from PhotonicsAI.Photon import component_type_validator as type_validator

# ==================== 配置区域 ====================

# 目标输出目录 (DesignLibrary)
# 假设脚本在 scripts/ 目录下，向上两级找到 PhotonicsAI
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "PhotonicsAI", "KnowledgeBase", "DesignLibrary"))

# 关键词配置
DEVICE_keywords = {
    # --- Active Devices ---
    "ge_pd": [
        "Germanium photodetector bandwidth",
        "GePD waveguide integrated",
    ],
    "si_modulator": [
        "Silicon Mach-Zehnder modulator",
        "silicon travelling wave modulator",
    ],
    "si_phase_shifter": [
        "thermo-optic phase shifter silicon",
        "silicon photonic heater",
    ],
    
    # --- Passive Devices (Si & SiN MMI) ---
    "mmi1x2": [
        "1x2 MMI silicon photonics splitter",
        "multimode interferometer 1x2 design",
    ],
    "mmi2x2": [
        "2x2 MMI coupler silicon",
        "multimode interferometer 2x2",
    ],
    
    # --- Crossings & Couplers ---
    "waveguide_crossing": [
        "silicon waveguide crossing loss",
    ],
    "directional_coupler": [
        "silicon directional coupler design",
        "directional coupler splitting ratio",
    ],
    
    # --- IO Couplers ---
    "grating_coupler": [
        "silicon grating coupler efficiency",
        "focusing grating coupler design",
    ],
    "edge_coupler": [
        "spot size converter silicon photonics",
        "inverse taper coupling efficiency",
    ],
    
    # --- Polarization ---
    "pbs_pbr": [
        "polarization beam splitter silicon",
        "polarization rotator integrated photonics",
    ]
}

MAX_RESULTS_PER_KEYWORD = 10  # 增加检索数量

# ==================== 组件代码模板 ====================

TEMPLATES = {
    # ================= Active Devices =================
    "ge_pd": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length: float = {length},
    width: float = {width},
) -> gf.Component:
    """Auto-generated Ge Photodetector from: {title}
    Source: {link}
    """
    c = gf.Component()
    # 模拟 GePD 结构 (PIN结)
    # 实际应用中需指定 doping layers
    ref = c << gf.components.straight_pin(length=length, cross_section="rib")
    c.add_ports(ref.ports)
    return c
''',

    "si_modulator": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length_phase_shifter: float = {length},
) -> gf.Component:
    """Auto-generated MZM Modulator from: {title}
    Source: {link}
    """
    c = gf.Component()
    # 模拟 Mach-Zehnder Modulator
    mzm = gf.components.mzi(
        delta_length=0,
        length_x=length_phase_shifter,
        splitter=gf.components.mmi1x2,
        combiner=gf.components.mmi1x2
    )
    ref = c << mzm
    c.add_ports(ref.ports)
    return c
''',
    
    "si_phase_shifter": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length: float = {length},
    heater_width: float = 2.0,
) -> gf.Component:
    """Auto-generated Thermo-optic Phase Shifter from: {title}
    Source: {link}
    """
    c = gf.Component()
    # 带加热器的直波导
    ref = c << gf.components.straight_heater_metal(
        length=length,
        heater_width=heater_width
    )
    c.add_ports(ref.ports)
    return c
''',

    # ================= Passive Devices =================
    "mmi1x2": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length_mmi: float = {length_mmi},
    width_mmi: float = {width_mmi},
    gap_mmi: float = {gap_mmi},
    width_taper: float = {width_taper},
    length_taper: float = {length_taper},
) -> gf.Component:
    """Auto-generated 1x2 MMI from: {title}
    Source: {link}
    """
    c = gf.Component()
    ref = c << gf.components.mmi1x2(
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        width_taper=width_taper,
        length_taper=length_taper,
    )
    c.add_ports(ref.ports)
    return c
''',

    "mmi2x2": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length_mmi: float = {length_mmi},
    width_mmi: float = {width_mmi},
    gap_mmi: float = {gap_mmi},
    width_taper: float = {width_taper},
    length_taper: float = {length_taper},
) -> gf.Component:
    """Auto-generated 2x2 MMI from: {title}
    Source: {link}
    """
    c = gf.Component()
    ref = c << gf.components.mmi2x2(
        length_mmi=length_mmi,
        width_mmi=width_mmi,
        gap_mmi=gap_mmi,
        width_taper=width_taper,
        length_taper=length_taper,
    )
    c.add_ports(ref.ports)
    return c
''',

    "ring": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    radius: float = {radius},
    gap: float = {gap},
    length_x: float = {length_x},
) -> gf.Component:
    """Auto-generated Ring Resonator from: {title}
    Source: {link}
    """
    c = gf.Component()
    ref = c << gf.components.ring_single(
        radius=radius,
        gap=gap,
        length_x=length_x,
    )
    c.add_ports(ref.ports)
    return c
''',

    "mzi": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    delta_length: float = {delta_length},
    length_x: float = {length_x},
    length_y: float = {length_y},
) -> gf.Component:
    """Auto-generated MZI from: {title}
    Source: {link}
    """
    c = gf.Component()
    ref = c << gf.components.mzi(
        delta_length=delta_length,
        length_x=length_x,
        length_y=length_y,
        splitter=gf.components.mmi1x2,
        combiner=gf.components.mmi1x2,
    )
    c.add_ports(ref.ports)
    return c
''',

    "y_branch": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length: float = {length},
    separation: float = {gap},
) -> gf.Component:
    """Auto-generated Y-Branch Splitter from: {title}
    Source: {link}
    
    Y-branch splitter using mmi1x2_with_sbend for smooth transition.
    """
    c = gf.Component()
    
    # Use mmi1x2_with_sbend which has proper S-bend outputs
    ref = c << gf.components.mmi1x2_with_sbend(
        length_mmi=length * 0.5,
        width_mmi=separation * 2.5,
        gap_mmi=separation,
    )
    c.add_ports(ref.ports)
    return c
''',

    "waveguide_crossing": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    width: float = 0.5,
) -> gf.Component:
    """Auto-generated Crossing from: {title}
    Source: {link}
    """
    c = gf.Component()
    ref = c << gf.components.crossing(width=width)
    c.add_ports(ref.ports)
    return c
''',

    "directional_coupler": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length: float = {length},
    gap: float = {gap},
) -> gf.Component:
    """Auto-generated Directional Coupler from: {title}
    Source: {link}
    """
    c = gf.Component()
    ref = c << gf.components.coupler(length=length, gap=gap)
    c.add_ports(ref.ports)
    return c
''',

    "grating_coupler": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    period: float = 0.63,
    fill_factor: float = 0.5,
) -> gf.Component:
    """Auto-generated Grating Coupler from: {title}
    Source: {link}
    """
    c = gf.Component()
    # 聚焦型光栅耦合器
    ref = c << gf.components.grating_coupler_elliptical(
        period=period,
        fill_factor=fill_factor
    )
    c.add_ports(ref.ports)
    return c
''',

    "edge_coupler": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    width_tip: float = 0.15,
    length_taper: float = {length},
) -> gf.Component:
    """Auto-generated Edge Coupler (SSC) from: {title}
    Source: {link}
    """
    c = gf.Component()
    # 倒锥形耦合器 (Inverse Taper)
    ref = c << gf.components.taper(
        width1=width_tip,
        width2=0.5,
        length=length_taper
    )
    c.add_ports(ref.ports)
    return c
''',
    
    "pbs_pbr": '''
import gdsfactory as gf

@gf.cell
def {func_name}(
    length: float = {length},
) -> gf.Component:
    """Auto-generated PBS/PBR from: {title}
    Source: {link}
    """
    c = gf.Component()
    # 示例使用非对称定向耦合器作为PBS
    ref = c << gf.components.coupler_asymmetric(length=length)
    c.add_ports(ref.ports)
    return c
'''
}

# ==================== 爬虫辅助函数 ====================

def init_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # 调试时可注释掉
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def extract_params_with_llm(text, device_type):
    """(可选) 使用 Zhipu AI (ChatGLM) 提取精准参数
    需要设置环境变量 ZHIPUAI_API_KEY 或直接在代码中填入
    """
    try:
        from zhipuai import ZhipuAI
        api_key = os.getenv("ZHIPUAI_API_KEY") # 或者填入 "your_zhipuai_api_key"
        if not api_key:
            return None
            
        client = ZhipuAI(api_key=api_key)
        
        # 使用类型验证器生成带约束的 prompt
        component_type = type_validator.classify_component_type(device_type)
        type_def = type_validator.get_type_definition(component_type) if component_type else None
        
        # 构建基础 prompt
        base_prompt = f"""
        You are an expert in silicon photonics PDK development. 
        Extract key geometry parameters for a "{device_type}" device from the following text.
        
        Return a JSON object ONLY, with keys matching standard GDSFactory parameters:
        - For MMI: length_mmi, width_mmi, gap_mmi, width_taper, length_taper
        - For Ring: radius, gap, length_x
        - For Directional Coupler: length, gap
        - For Modulator: length
        - For GePD: length, width
        
        If a parameter is not explicitly found, estimate a standard value for a 220nm SOI platform (C-band).
        Do not include markdown formatting like ```json.
        """
        
        # 如果有类型定义，添加约束
        if type_def:
            params_spec = []
            for param_name, param_def in type_def.get("params", {}).items():
                range_str = f" (范围: {param_def['range']})" if "range" in param_def else ""
                default_str = f", 默认: {param_def['default']}" if "default" in param_def else ""
                params_spec.append(f"  - {param_name}{range_str}{default_str}")
            
            ports_def = type_def.get("ports", {})
            
            base_prompt = f"""
        You are an expert in silicon photonics PDK development.
        Extract geometry parameters for a "{device_type}" ({component_type}) device.
        
        === 组件类型定义 (必须遵守) ===
        描述: {type_def.get('description', '')}
        端口: {ports_def.get('definition', 'unknown')}
        
        参数规范:
{chr(10).join(params_spec) if params_spec else '  (无特定参数定义)'}
        
        === 提取规则 ===
        1. 参数值必须在指定范围内
        2. 缺失参数使用默认值
        3. 输出 JSON 格式，不要 markdown 代码块
        """
        
        base_prompt += f"""
        
        Text:
        {text[:4000]}
        """
        
        response = client.chat.completions.create(
            model="glm-4-flash",  # 使用免费的 flash 模型
            messages=[{"role": "user", "content": base_prompt}],
        )
        # GLM-4 的返回格式可能包含 markdown 代码块，需要清理
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        params = json.loads(content)
        
        # 使用类型验证器验证参数
        if component_type and params:
            validation = type_validator.verify_params(component_type, params)
            if not validation["valid"]:
                print(f"  [Validation] 参数验证失败: {validation.get('adjustments', [])}")
            if validation.get("adjustments"):
                print(f"  [Validation] 参数调整: {validation['adjustments']}")
            # 使用验证/修正后的参数
            return validation["params"]
        
        return params
        
    except ImportError:
        print("ZhipuAI library not installed. Run `pip install zhipuai`.")
        return None
    except Exception as e:
        print(f"LLM Extraction failed: {e}")
        return None

def extract_params_heuristic(text, device_type):
    """启发式参数提取 (默认回退方案)"""
    
    # 1. 尝试使用 LLM (如果配置了 API Key)
    llm_params = extract_params_with_llm(text, device_type)
    if llm_params:
        print(f"  [LLM] Successfully extracted params for {device_type}")
        return llm_params
        
    print(f"  [Heuristic] Falling back to regex extraction...")
    params = {}
    
    # 简单的正则匹配
    #寻找类似 "length: 10 um" 或 "length of 10um" 的模式
    length_match = re.search(r'length.*?(\d+\.?\d*)', text, re.IGNORECASE)
    width_match = re.search(r'width.*?(\d+\.?\d*)', text, re.IGNORECASE)
    gap_match = re.search(r'gap.*?(\d+\.?\d*)', text, re.IGNORECASE)
    radius_match = re.search(r'radius.*?(\d+\.?\d*)', text, re.IGNORECASE)

    # 默认值 (防止生成失败，保证代码可运行)
    # 在真实场景中，如果提取失败应该跳过或标记
    # 这里我们为每种新设备添加了默认参数
    defaults = {
        # Active
        "ge_pd": {"length": 20.0, "width": 1.0},
        "si_modulator": {"length": 1000.0}, # 1mm default for modulators
        "si_phase_shifter": {"length": 100.0},
        
        # Passive
        "mmi1x2": {"length_mmi": 10.0, "width_mmi": 4.0, "gap_mmi": 0.25, "width_taper": 1.0, "length_taper": 5.0},
        "mmi2x2": {"length_mmi": 25.0, "width_mmi": 6.0, "gap_mmi": 0.25, "width_taper": 1.0, "length_taper": 5.0},
        "waveguide_crossing": {"width": 0.5},
        "directional_coupler": {"length": 20.0, "gap": 0.2},
        "grating_coupler": {"period": 0.63, "fill_factor": 0.5},
        "edge_coupler": {"length": 150.0},
        "pbs_pbr": {"length": 30.0},
        
        # Old
        "ring_resonator": {"radius": 10.0, "gap": 0.2, "length_x": 0.0, "length_y": 0.0},
        "y_branch": {"length": 15.0}
    }
    
    p = defaults.get(device_type, {}).copy()
    
    # 通用的参数提取逻辑
    if length_match: 
        # MMI 的长度通常叫 length_mmi
        if "mmi" in device_type: p["length_mmi"] = float(length_match.group(1))
        # 其他设备通常叫 length
        else: p["length"] = float(length_match.group(1))
        
    if width_match:
        if "mmi" in device_type: p["width_mmi"] = float(width_match.group(1))
        else: p["width"] = float(width_match.group(1))
        
    if gap_match:
        if "mmi" in device_type: p["gap_mmi"] = float(gap_match.group(1))
        else: p["gap"] = float(gap_match.group(1))

    return p

# ==================== 代码生成 ====================

def generate_component_file(paper_info):
    """根据爬取的论文信息生成 .py 文件 (旧接口, 每篇论文一个文件)"""
    device_type = paper_info['device_type']
    if device_type not in TEMPLATES:
        return None
        
    # 提取参数
    params = extract_params_heuristic(paper_info['full_text'], device_type)
    
    # 生成唯一函数名 (去除非法字符)
    safe_title = re.sub(r'[^a-zA-Z0-9]', '', paper_info['title'])[:20]
    func_name = f"auto_{device_type}_{safe_title}_{random.randint(100,999)}"
    filename = f"{func_name}.py"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 填充模板
    try:
        code = TEMPLATES[device_type].format(
            func_name=func_name,
            title=paper_info['title'],
            link=paper_info['link'],
            **params
        )
        
        # 确保目录存在
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
            
        print(f"[SUCCESS] Generated: {filename}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to generate code for {paper_info['title']}: {e}")
        return None


# ==================== 核心新接口: 多论文聚合 -> 单模板生成 ====================
def _generate_search_keywords_with_llm(component_name, device_type=None):
    """使用 LLM 动态生成学术搜索关键词。
    
    不再完全依赖硬编码的 DEVICE_keywords 字典，而是让 LLM 根据组件名
    智能生成更精准、更多样的搜索词（涵盖不同术语、缩写、变体）。
    
    Args:
        component_name: 用户描述的组件名 (e.g., "high-Q ring resonator")
        device_type: 已识别的设备类型 (可选, e.g., "ring_resonator")
    
    Returns:
        list[str]: 搜索关键词列表 (5-8个)
    """
    try:
        from zhipuai import ZhipuAI
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            return None

        client = ZhipuAI(api_key=api_key)

        prompt = f"""You are an expert in silicon photonics and integrated photonics research.

A user wants to design the following photonic component:
  Component: "{component_name}"
  {f'Internal type: "{device_type}"' if device_type else ''}

Generate 5-8 high-quality academic search keywords/phrases for finding relevant papers on 
arXiv, Google Scholar, and IEEE/Optica journals.

Requirements:
1. Include the EXACT component name and common synonyms/abbreviations
2. Include platform-specific terms (e.g., "silicon photonics", "SOI", "silicon nitride")
3. Include design-oriented terms (e.g., "design", "optimization", "fabrication")
4. Include performance-oriented terms (e.g., "low loss", "broadband", "high efficiency")
5. Vary the specificity: some broad ("MMI coupler design"), some narrow ("1x2 MMI 220nm SOI insertion loss")
6. Include terms that would appear in methodology/results sections of relevant papers

Return a JSON array of strings ONLY. No markdown, no explanation.
Example: ["1x2 MMI silicon photonics", "multimode interferometer coupler SOI", ...]
"""

        response = client.chat.completions.create(
            model="glm-4-flash",  # 使用免费的 flash 模型
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        keywords = json.loads(content)
        
        if isinstance(keywords, list) and len(keywords) > 0:
            print(f"  [LLM-Keywords] Generated {len(keywords)} search keywords for '{component_name}'")
            for i, kw in enumerate(keywords):
                print(f"    {i+1}. {kw}")
            return keywords[:8]  # 最多 8 个
        return None
        
    except Exception as e:
        print(f"  [LLM-Keywords] Failed: {e}")
        return None


def _resolve_device_type(component_name):
    """将用户自然语言组件名映射到内部 device_type 关键字。
    例如 'Ge photodetector' -> 'ge_pd', '1x2 MMI' -> 'mmi1x2'
    """
    name_lower = component_name.lower()
    mapping = {
        # Y 分支 (放在最前面，优先级最高)
        "y_branch": ["y_branch", "y branch", "y-branch", "ybranch", "y splitter"],
        # 光电探测器
        "ge_pd": ["germanium", "ge pd", "ge photodetector", "photodetector", "ge-pd"],
        # 调制器
        "si_modulator": ["modulator", "mzm", "mach-zehnder modulator", "si modulator", "pn diode", "pindiode"],
        # 相移器/加热器
        "si_phase_shifter": ["phase shifter", "thermo-optic", "thermal"],
        "heater": ["heater", "heater_tin", "tin heater"],
        # MMI
        "mmi1x2": ["mmi 1x2", "1x2 mmi", "mmi1x2", "1x2 splitter", "mmi_1x2"],
        "mmi2x2": ["mmi 2x2", "2x2 mmi", "mmi2x2", "2x2 coupler mmi", "mmi_2x2"],
        "mmi": ["mmi", "multimode interferometer"],
        # 环形谐振器
        "ring": ["ring resonator", "ring", "mrr", "micro-ring"],
        # MZI
        "mzi": ["mzi", "mach-zehnder interferometer", "mach_zehnder"],
        # 交叉
        "waveguide_crossing": ["crossing", "waveguide crossing"],
        # 光栅耦合器 (放在 coupler 前面，优先级更高)
        "grating_coupler": ["grating coupler", "grating", "gc ", "fiber coupler"],
        # 边缘耦合器
        "edge_coupler": ["edge coupler", "spot size converter", "ssc", "inverse taper"],
        # 定向耦合器
        "directional_coupler": ["directional coupler", "dc coupler", "coupler"],
        # 偏振
        "pbs_pbr": ["polarization", "pbs", "pbr", "polarization beam splitter"],
        # 波导
        "straight": ["straight waveguide", "straight"],
        "bend": ["bend", "bent waveguide"],
        "taper": ["taper"],
    }
    # 优先检查更具体的匹配（grating_coupler 要在 directional_coupler 之前）
    for dtype, keywords in mapping.items():
        for kw in keywords:
            if kw in name_lower:
                return dtype
    return None


def _aggregate_params_with_llm(papers, device_type):
    """使用 LLM 综合多篇论文的全文，提取一组共识基准参数。
    
    核心思想：不是每篇论文提取一次参数，而是把所有论文的全文一次性给 LLM，
    让它从中找出 most commonly reported / most reliable 的参数值。
    
    策略：
    - 每篇论文提供全文（或尽可能多的文本），而非仅摘要
    - GLM-4 支持 128K context，可以容纳大量文本
    - 优先使用 full_text 字段（全文），降级到 abstract
    """
    try:
        from zhipuai import ZhipuAI
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            print("[WARN] No Zhipu API key found, falling back to heuristic defaults.")
            return None

        client = ZhipuAI(api_key=api_key)

        # 拼接所有论文的全文内容
        # GLM-4 支持 128K tokens，我们可以放入更多文本
        MAX_TOTAL_CHARS = 100000  # ~25K tokens, 留余量给 prompt 和输出
        MAX_PER_PAPER = MAX_TOTAL_CHARS // max(len(papers), 1)  # 均分给每篇论文
        
        combined_text = ""
        full_text_count = 0
        abstract_only_count = 0
        
        for i, p in enumerate(papers):
            combined_text += f"\n{'='*60}\n"
            combined_text += f"--- Paper {i+1}: {p.get('title', 'N/A')} ---\n"
            combined_text += f"--- Source: {p.get('link', 'N/A')} ---\n"
            combined_text += f"{'='*60}\n"
            
            # 优先使用全文
            paper_text = p.get('full_text', '')
            has_full = p.get('has_full_text', False)
            
            if has_full and len(paper_text) > 500:
                combined_text += paper_text[:MAX_PER_PAPER]
                full_text_count += 1
            else:
                # 降级到摘要
                abstract = p.get('abstract', '')
                combined_text += abstract if abstract else paper_text[:2000]
                abstract_only_count += 1
            
            combined_text += "\n"
        
        print(f"  [LLM-Aggregate] Input: {full_text_count} full papers + {abstract_only_count} abstracts-only")
        print(f"  [LLM-Aggregate] Total text length: {len(combined_text)} chars")

        prompt = f"""You are an expert in silicon photonics PDK development.

I have collected {len(papers)} research papers about "{device_type}" devices on a 220nm SOI platform (C-band, ~1550nm).

Below are the FULL TEXTS (or abstracts when full text unavailable) of these papers.
You have access to complete paper contents. Pay special attention to ALL of the following areas:

[Methodology & Design]
- Device design methodology and simulation setup (FDTD, FDE, BPM, EME parameters)
- Design equations, analytical models, and transfer matrix methods used
- Optimization algorithms or parameter sweep strategies described

[Fabrication Details]
- Fabrication process flow (lithography, etching, deposition steps)
- Actual fabricated device dimensions (as-fabricated, not just designed)
- Process tolerances and critical dimension (CD) variations reported
- Waveguide cross-section: core thickness, slab thickness, etch depth, sidewall angle
- Material stack details: cladding material, BOX thickness

[Measurement Results]
- Measured performance metrics (insertion loss, extinction ratio, bandwidth, etc.)
- Optimized parameters from experimental characterization
- Specific numerical values in tables, figure captions, and result sections
- Comparison between simulated and measured performance

[Process Window & Tolerances]
- Fabrication tolerance analysis (sensitivity to width/length/gap variations)
- Wavelength dependence and operating bandwidth
- Temperature sensitivity and thermal tuning requirements

[Benchmarking]
- Comparisons with other reported devices in literature
- State-of-the-art performance numbers cited from references
- Any consensus values across multiple papers for the same parameter

{combined_text}

YOUR TASK:
1. Carefully read ALL papers above. Focus on experimentally reported geometry parameters.
2. Identify the CONSENSUS geometry parameters for a "{device_type}" device.
3. For each parameter, choose the value that:
   a) Appears most frequently across papers, OR
   b) Is reported in the most rigorous experimental study
4. If a parameter is only mentioned in one paper, use it but note it as less reliable.
5. If a parameter is not mentioned at all, provide a standard value for 220nm SOI C-band.

Return a JSON object ONLY with these standard GDSFactory parameter keys:
- For MMI (1x2 or 2x2): length_mmi, width_mmi, gap_mmi, width_taper, length_taper
- For Ring: radius, gap, length_x
- For Directional Coupler: length, gap
- For Modulator/Phase Shifter: length
- For Ge PD: length, width
- For Grating Coupler: period, fill_factor
- For Edge Coupler: length

Also include a key "confidence_note" (string) briefly explaining:
- Which papers contributed most to each parameter
- Whether values are from full-text analysis or abstract-only estimation
- Your confidence level (high/medium/low) for each parameter

Do NOT include markdown formatting. Output pure JSON only.
"""

        response = client.chat.completions.create(
            model="glm-4-flash",  # 使用免费的 flash 模型
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        # 移除非参数字段，保留 confidence_note 用于 docstring
        confidence_note = result.pop("confidence_note", "")
        print(f"  [LLM-Aggregate] Consensus params extracted from {len(papers)} papers.")
        if confidence_note:
            print(f"  [LLM-Aggregate] Note: {confidence_note[:120]}")
        
        return result, confidence_note

    except Exception as e:
        print(f"  [LLM-Aggregate] Failed: {e}")
        return None, ""


# ==================== 论文质量评判与排序 ====================
def _rank_papers_with_llm(papers, device_type, top_n=8):
    """使用 LLM 对论文进行质量评分和排序。
    
    评判维度:
    1. 相关性 (Relevance):     论文是否直接研究目标器件？
    2. 实验数据 (Experimental): 是否包含实际制造和测量数据（而非纯仿真/理论）？
    3. 平台匹配 (Platform):    是否使用 220nm SOI / C-band 平台？
    4. 参数完整性 (Params):     是否报告了完整的几何参数？
    5. 发表质量 (Venue):        期刊/会议的影响力
    
    Args:
        papers: 论文列表 (含 title, abstract, source, link 等)
        device_type: 目标器件类型
        top_n: 返回排名前 N 的论文
    
    Returns:
        list: 按质量得分降序排列的论文列表 (每篇新增 quality_score 和 quality_reason 字段)
    """
    if len(papers) <= top_n:
        # 论文数量不足，无需筛选
        for p in papers:
            p["quality_score"] = 50  # 默认分
            p["quality_reason"] = "Not enough papers to rank"
        return papers
    
    try:
        from zhipuai import ZhipuAI
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            # 无 API Key，按搜索顺序返回
            return papers[:top_n]

        client = ZhipuAI(api_key=api_key)

        # 构造论文摘要列表 (只发送标题+摘要+来源用于评判，不发全文以节省 token)
        papers_summary = ""
        for i, p in enumerate(papers):
            papers_summary += f"\n[Paper {i+1}]\n"
            papers_summary += f"  Title: {p.get('title', 'N/A')}\n"
            papers_summary += f"  Source: {p.get('source', 'N/A')} | Link: {p.get('link', 'N/A')}\n"
            # 摘要截断到 500 字符
            abstract = p.get('abstract', '')[:500]
            papers_summary += f"  Abstract: {abstract}\n"
            papers_summary += f"  Has Full Text: {p.get('has_full_text', False)}\n"

        prompt = f"""You are an expert reviewer in silicon photonics research.

I need to select the BEST papers for extracting design parameters of a "{device_type}" device 
on a 220nm SOI platform operating at C-band (~1550nm).

Here are {len(papers)} candidate papers:
{papers_summary}

Score each paper from 0-100 based on these criteria (weights shown):

1. RELEVANCE (30%): Does the paper directly study/design/fabricate a "{device_type}" device?
   - 100: Exact match (paper title/abstract is about this specific device)
   - 50: Related device (similar component category)
   - 0: Unrelated topic

2. EXPERIMENTAL DATA (25%): Does the paper contain real fabrication + measurement results?
   - 100: Fabricated device with measured performance data
   - 60: Simulation-only study with realistic parameters
   - 20: Purely theoretical/analytical
   - 0: Review paper / no original data

3. PLATFORM MATCH (20%): Is the device on 220nm SOI, C-band?
   - 100: Explicitly 220nm SOI, C-band
   - 70: Silicon photonics but different thickness (e.g., 340nm) or band
   - 40: Different material (SiN, InP) but similar device concept
   - 0: Completely different platform

4. PARAMETER COMPLETENESS (15%): Does the abstract mention specific geometry parameters?
   - 100: Multiple specific dimensions mentioned (e.g., "length=5.4μm, width=2.8μm")
   - 50: Some parameters mentioned
   - 0: No dimensions in abstract

5. VENUE QUALITY (10%): Publication venue prestige for photonics
   - 100: Nature Photonics, Optica, Light: Science & Applications
   - 80: Optics Express, Optics Letters, Photonics Research, JLT, APL Photonics
   - 60: IEEE journals (PTL, JSTQE), ACS Photonics
   - 40: Conference proceedings (CLEO, OFC, ECOC, GFP)
   - 20: arXiv preprint (not yet published)
   - 0: Cannot determine venue

Return a JSON array where each element has:
  "paper_index": (1-based index matching the paper list above)
  "score": (0-100 weighted total)
  "reason": (one sentence explaining the score)

Sort by score descending. Output pure JSON only, no markdown.
"""

        response = client.chat.completions.create(
            model="glm-4-flash",  # 使用免费的 flash 模型
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        rankings = json.loads(content)

        # 将评分映射回论文列表
        score_map = {}  # paper_index -> {score, reason}
        for r in rankings:
            idx = r.get("paper_index", 0)
            score_map[idx] = {
                "score": r.get("score", 0),
                "reason": r.get("reason", "")
            }

        # 给每篇论文添加评分
        for i, p in enumerate(papers):
            info = score_map.get(i + 1, {"score": 0, "reason": "Not evaluated"})
            p["quality_score"] = info["score"]
            p["quality_reason"] = info["reason"]

        # 按得分降序排序
        papers_sorted = sorted(papers, key=lambda x: x.get("quality_score", 0), reverse=True)
        
        # 打印排名
        print(f"  [Paper Ranking] Evaluated {len(papers)} papers:")
        for i, p in enumerate(papers_sorted[:top_n]):
            score = p.get("quality_score", 0)
            reason = p.get("quality_reason", "")
            title_short = p.get("title", "N/A")[:50]
            marker = "✅" if score >= 60 else "⚠️" if score >= 30 else "❌"
            print(f"    {marker} #{i+1} [{score}/100] {title_short}...")
            if reason:
                print(f"       → {reason[:80]}")
        
        if len(papers_sorted) > top_n:
            discarded = len(papers_sorted) - top_n
            print(f"    ... ({discarded} lower-quality papers discarded)")

        return papers_sorted[:top_n]

    except Exception as e:
        print(f"  [Paper Ranking] LLM ranking failed: {e}, using original order.")
        return papers[:top_n]


def resolve_device_type_and_keywords(component_name):
    """解析组件类型并生成搜索关键词（含降级策略）。

    Returns:
        dict: {
            "ok": bool,
            "data": {
                "device_type": str,
                "keywords": list[str],
                "fallback_reason": str,
            },
            "error": str | None,
        }
    """
    device_type = _resolve_device_type(component_name)
    if not device_type:
        return {
            "ok": False,
            "data": {},
            "error": f"Cannot map '{component_name}' to a known device type. Supported: {list(TEMPLATES.keys())}",
        }

    keywords = _generate_search_keywords_with_llm(component_name, device_type)
    fallback_reason = "llm_generated"

    if not keywords:
        keywords = DEVICE_keywords.get(device_type)
        fallback_reason = "hardcoded"
        if keywords:
            print(f"  [Keywords] Falling back to hardcoded keywords ({len(keywords)} keywords)")

    if not keywords:
        keywords = [f"{component_name} silicon photonics", f"{component_name} design optimization"]
        fallback_reason = "component_name"
        print("  [Keywords] Using component name as search keyword")

    return {
        "ok": True,
        "data": {
            "device_type": device_type,
            "keywords": keywords,
            "fallback_reason": fallback_reason,
        },
        "error": None,
    }


def retrieve_papers_multi_source(device_type, keywords, max_papers=8):
    """多源检索论文，返回原始结果与来源统计。"""
    driver = None
    all_papers = []
    source_stats = {
        "arxiv": 0,
        "google_scholar": 0,
        "optica": 0,
        "total_raw": 0,
    }
    error_msg = None

    try:
        driver = init_driver()

        print("  [Source: ArXiv]")
        for kw in keywords[:4]:
            papers = search_arxiv(driver, kw, device_type)
            all_papers.extend(papers)
            source_stats["arxiv"] += len(papers)
            print(f"    Found {len(papers)} papers for '{kw}'")
            if len(all_papers) >= max_papers:
                break

        if len(all_papers) < max_papers:
            print("  [Source: Google Scholar]")
            for kw in keywords[:3]:
                papers = search_google_scholar(driver, kw, device_type)
                all_papers.extend(papers)
                source_stats["google_scholar"] += len(papers)
                print(f"    Found {len(papers)} papers for '{kw}'")
                if len(all_papers) >= max_papers * 2:
                    break

        if len(all_papers) < max_papers:
            print("  [Source: Optica/OSA]")
            for kw in keywords[:2]:
                papers = search_optica(driver, kw, device_type)
                all_papers.extend(papers)
                source_stats["optica"] += len(papers)
                print(f"    Found {len(papers)} papers for '{kw}'")
                if len(all_papers) >= max_papers * 2:
                    break

    except Exception as e:
        error_msg = f"Crawler error: {e}"
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    source_stats["total_raw"] = len(all_papers)
    return {
        "ok": True,
        "data": {
            "papers": all_papers,
            "source_stats": source_stats,
        },
        "error": error_msg,
    }


def rank_and_dedup_papers(papers, device_type, top_n=8):
    """按标题去重并进行质量排序，输出 Top-N。"""
    seen_titles = set()
    unique_papers = []

    for p in papers:
        title = p.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(p)

    print(f"[Discovery] {len(unique_papers)} unique papers found across all sources.")

    llm_used = False
    if len(unique_papers) > top_n:
        print(f"[Discovery] Ranking {len(unique_papers)} papers to select top {top_n}...")
        ranked_papers = _rank_papers_with_llm(unique_papers, device_type, top_n=top_n)
        llm_used = True
    else:
        ranked_papers = unique_papers

    return {
        "ok": True,
        "data": {
            "ranked_papers": ranked_papers,
            "ranking_meta": {
                "total_input": len(papers),
                "total_unique": len(unique_papers),
                "total_output": len(ranked_papers),
                "llm_used": llm_used,
            },
        },
        "error": None,
    }


def extract_or_aggregate_params(papers, device_type):
    """统一参数策略节点：多篇聚合 / 单篇提取 / 默认降级。"""
    params = {}
    confidence_note = ""
    provenance = "heuristic_default"

    if len(papers) >= 2:
        agg_result = _aggregate_params_with_llm(papers, device_type)
        if agg_result and agg_result[0]:
            params, confidence_note = agg_result
            provenance = "llm_aggregate"

    elif len(papers) == 1:
        single_text = papers[0].get("full_text") or papers[0].get("abstract", "")
        params = extract_params_heuristic(single_text, device_type)
        confidence_note = f"Single paper: {papers[0].get('title', 'N/A')}"
        provenance = "heuristic_single"

    if not params:
        params = extract_params_heuristic("", device_type)
        confidence_note = "No papers found; using default 220nm SOI C-band values."
        provenance = "heuristic_default"

    return {
        "ok": True,
        "data": {
            "params": params,
            "confidence_note": confidence_note,
            "provenance": provenance,
        },
        "error": None,
    }


def generate_template_file(device_type, params, papers, confidence_note=""):
    """基于参数与论文来源生成单个共识模板文件。"""
    if device_type not in TEMPLATES:
        return {
            "ok": False,
            "data": {},
            "error": f"No code template defined for device_type='{device_type}'.",
        }

    func_name = f"auto_{device_type}_consensus"
    filename = f"{func_name}.py"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if papers:
        sources_cited = [p.get("title", "N/A")[:60] for p in papers[:5]]
        sources = "; ".join(sources_cited)
    else:
        sources_cited = ["Default parameters (no papers retrieved)"]
        sources = sources_cited[0]

    try:
        code = TEMPLATES[device_type].format(
            func_name=func_name,
            title=f"Consensus parameters from {len(papers)} papers",
            link=sources,
            **params,
        )

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        print(f"[Discovery] SUCCESS: Generated ONE template -> {filename}")
        return {
            "ok": True,
            "data": {
                "filepath": filepath,
                "render_meta": {
                    "func_name": func_name,
                    "filename": filename,
                    "sources_cited": sources_cited,
                    "device_type": device_type,
                    "confidence_note": confidence_note,
                },
            },
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "data": {},
            "error": f"Code generation failed: {e}",
        }


def discover_and_generate(component_name, max_papers=8):
    """核心新接口：根据组件名，执行完整的 Discovery 流程。
    
    流程：
    1. 将自然语言组件名映射到 device_type
    2. 使用爬虫搜索相关论文（ArXiv）
    3. 收集多篇论文摘要
    4. 用 LLM 综合所有论文，提取共识参数
    5. 生成 ONE 高质量基础模板文件
    
    Args:
        component_name: 用户自然语言描述的组件名 (e.g., "1x2 MMI", "Ge photodetector")
        max_papers: 最多使用多少篇论文进行参数聚合 (default: 8)
    
    Returns:
        dict: {
            "filepath": str or None,     # 生成文件的路径
            "device_type": str,           # 内部设备类型
            "papers_found": int,          # 找到的论文数
            "params": dict,              # 提取到的参数
            "confidence_note": str,       # LLM 的置信度说明
            "error": str or None          # 错误信息
        }
    """
    result = {
        "filepath": None,
        "device_type": None,
        "papers_found": 0,
        "paper_rankings": [],    # 论文评分详情
        "params": {},
        "confidence_note": "",
        "error": None,
    }

    # Step 1: 解析设备类型 + 关键词
    print(f"[Discovery] Generating search keywords for '{component_name}'...")
    resolved = resolve_device_type_and_keywords(component_name)
    if not resolved["ok"]:
        result["error"] = resolved["error"]
        return result

    device_type = resolved["data"]["device_type"]
    keywords = resolved["data"]["keywords"]
    result["device_type"] = device_type

    # Step 2: 多源检索论文
    print(f"[Discovery] Searching papers for '{component_name}' (type={device_type})...")
    retrieved = retrieve_papers_multi_source(device_type, keywords, max_papers=max_papers)
    raw_papers = retrieved["data"]["papers"]
    if retrieved["error"]:
        result["error"] = retrieved["error"]

    # Step 3: 去重与排序
    ranked = rank_and_dedup_papers(raw_papers, device_type, top_n=max_papers)
    all_papers = ranked["data"]["ranked_papers"]

    result["papers_found"] = len(all_papers)
    # 保存评分信息用于前端展示
    result["paper_rankings"] = [
        {
            "title": p.get("title", "N/A"),
            "source": p.get("source", "N/A"),
            "score": p.get("quality_score", "N/A"),
            "reason": p.get("quality_reason", ""),
            "has_full_text": p.get("has_full_text", False),
        }
        for p in all_papers
    ]
    print(f"[Discovery] Using {len(all_papers)} top-ranked papers for parameter aggregation.")

    # Step 4: 参数提取与聚合
    params_result = extract_or_aggregate_params(all_papers, device_type)
    result["params"] = params_result["data"]["params"]
    result["confidence_note"] = params_result["data"]["confidence_note"]

    # Step 5: 生成模板文件
    render_result = generate_template_file(
        device_type=device_type,
        params=result["params"],
        papers=all_papers,
        confidence_note=result["confidence_note"],
    )
    if render_result["ok"]:
        result["filepath"] = render_result["data"]["filepath"]
    else:
        result["error"] = render_result["error"]
        print(f"[Discovery] ERROR: {render_result['error']}")

    return result

# ==================== 全文抓取 ====================

def fetch_full_paper_text(driver, arxiv_link, max_chars=15000):
    """从 ArXiv HTML 版 (ar5iv) 获取论文全文。
    
    ArXiv 提供了 HTML 渲染版论文 (ar5iv.labs.arxiv.org)，
    可以直接解析文本内容，无需 PDF 处理。
    
    Args:
        driver: Selenium WebDriver 实例
        arxiv_link: ArXiv 论文链接 (e.g., https://arxiv.org/abs/2301.12345)
        max_chars: 最大字符数限制 (防止过长)
    
    Returns:
        str: 论文全文文本，失败时返回空字符串
    """
    try:
        # 从 arxiv.org/abs/XXXX 转换为 ar5iv.labs.arxiv.org/html/XXXX
        arxiv_id = ""
        if "/abs/" in arxiv_link:
            arxiv_id = arxiv_link.split("/abs/")[-1].strip("/")
        elif "/pdf/" in arxiv_link:
            arxiv_id = arxiv_link.split("/pdf/")[-1].replace(".pdf", "").strip("/")
        else:
            # 尝试直接取最后的 ID
            arxiv_id = arxiv_link.rstrip("/").split("/")[-1]
        
        if not arxiv_id:
            return ""
        
        # ar5iv 提供 HTML 版论文
        html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        print(f"    [FullText] Fetching: {html_url}")
        
        driver.get(html_url)
        time.sleep(3)  # 等待页面加载
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # 移除不需要的元素 (导航、脚注、参考文献)
        for tag in soup.find_all(['nav', 'footer', 'script', 'style']):
            tag.decompose()
        
        # 尝试移除参考文献部分 (通常在 id='bib' 或 class='ltx_bibliography')
        for ref_section in soup.find_all(class_='ltx_bibliography'):
            ref_section.decompose()
        for ref_section in soup.find_all(id='bib'):
            ref_section.decompose()
        
        # 提取正文文本
        # ar5iv 论文正文通常在 <article> 或 class='ltx_document' 中
        article = soup.find('article') or soup.find(class_='ltx_document')
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            # 降级：取整个 body
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else ""
        
        # 清理多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 200:
            print(f"    [FullText] Warning: Very short text ({len(text)} chars), page may not have loaded.")
            return ""
        
        print(f"    [FullText] Extracted {len(text)} chars (truncating to {max_chars}).")
        return text[:max_chars]
        
    except Exception as e:
        print(f"    [FullText] Failed for {arxiv_link}: {e}")
        return ""


# ==================== 爬虫逻辑 (ArXiv) ====================
def search_arxiv(driver, keyword, device_type, fetch_full=True):
    """搜索 ArXiv 论文并收集信息。
    
    Args:
        driver: Selenium WebDriver
        keyword: 搜索关键词
        device_type: 设备类型
        fetch_full: 是否抓取全文 (True=全文, False=仅摘要)
    """
    results = []
    url = f"https://arxiv.org/search/?searchtype=all&query={keyword.replace(' ', '+')}&start=0"
    try:
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        articles = soup.find_all('li', class_='arxiv-result')
        if not articles:
            print(f"  No results for keyword: {keyword}")
            return []
            
        for article in articles[:MAX_RESULTS_PER_KEYWORD]:
            try:
                title = article.find('p', class_='title').get_text(strip=True)
                
                abstract_span = article.find('span', class_='abstract-full')
                if not abstract_span:
                    abstract_span = article.find('span', class_='abstract-short')
                
                abstract = abstract_span.get_text(strip=True) if abstract_span else ""
                
                link_tag = article.find('p', class_='list-title').find('a')
                link = link_tag['href'] if link_tag else "N/A"
                
                # 抓取全文
                full_text = ""
                if fetch_full and link != "N/A":
                    full_text = fetch_full_paper_text(driver, link)
                
                # 如果全文获取失败，降级为 title + abstract
                if not full_text:
                    full_text = f"{title} {abstract}"
                
                results.append({
                    "source": "arxiv",
                    "title": title,
                    "link": link,
                    "abstract": abstract,
                    "full_text": full_text,
                    "has_full_text": len(full_text) > len(title) + len(abstract) + 10,
                    "device_type": device_type
                })
            except Exception as inner_e:
                print(f"  Skipping one article: {inner_e}")
                continue
    except Exception as e:
        print(f"  ArXiv connection error: {e}")
        
    return results


# ==================== 爬虫逻辑 (Google Scholar) ====================
def search_google_scholar(driver, keyword, device_type, fetch_full=False):
    """搜索 Google Scholar 论文。
    
    注意: Google Scholar 反爬较严，需要注意速率限制。
    Scholar 的全文通常需要跳转到出版商页面，因此默认不抓全文。
    
    Args:
        driver: Selenium WebDriver
        keyword: 搜索关键词
        device_type: 设备类型
        fetch_full: 是否尝试获取全文 (默认 False, Scholar 页面结构复杂)
    """
    results = []
    encoded_kw = urllib.parse.quote_plus(keyword)
    url = f"https://scholar.google.com/scholar?q={encoded_kw}&hl=en&as_sdt=0%2C5"
    
    try:
        driver.get(url)
        time.sleep(3 + random.uniform(1, 3))  # 随机延迟避免反爬
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # 检测是否被 CAPTCHA 拦截
        if "captcha" in driver.page_source.lower() or "unusual traffic" in driver.page_source.lower():
            print("    [Scholar] CAPTCHA detected, skipping Google Scholar.")
            return []
        
        articles = soup.find_all('div', class_='gs_r gs_or gs_scl')
        if not articles:
            # 尝试旧版选择器
            articles = soup.find_all('div', class_='gs_ri')
        
        if not articles:
            print(f"    No Scholar results for: {keyword}")
            return []
        
        for article in articles[:5]:  # Scholar 限制少取一些，避免触发反爬
            try:
                # 标题
                title_tag = article.find('h3', class_='gs_rt')
                if not title_tag:
                    title_tag = article.find('h3')
                if not title_tag:
                    continue
                    
                title = title_tag.get_text(strip=True)
                # 移除 [PDF] [HTML] 等标记
                title = re.sub(r'^\[.*?\]\s*', '', title)
                
                # 链接
                link_tag = title_tag.find('a')
                link = link_tag['href'] if link_tag else "N/A"
                
                # 摘要
                abstract_div = article.find('div', class_='gs_rs')
                abstract = abstract_div.get_text(strip=True) if abstract_div else ""
                
                # Scholar 通常只显示摘要片段
                full_text = f"{title} {abstract}"
                
                # 如果链接是 ArXiv 的，可以尝试抓全文
                has_full = False
                if fetch_full and link != "N/A" and "arxiv.org" in link:
                    full_content = fetch_full_paper_text(driver, link)
                    if full_content:
                        full_text = full_content
                        has_full = True
                
                results.append({
                    "source": "google_scholar",
                    "title": title,
                    "link": link,
                    "abstract": abstract,
                    "full_text": full_text,
                    "has_full_text": has_full,
                    "device_type": device_type
                })
            except Exception as inner_e:
                continue
        
        time.sleep(random.uniform(2, 4))  # 请求间隔
        
    except Exception as e:
        print(f"    Scholar connection error: {e}")
    
    return results


# ==================== 爬虫逻辑 (Optica / OSA) ====================
def search_optica(driver, keyword, device_type, fetch_full=True):
    """搜索 Optica Publishing Group (原 OSA) 论文。
    
    覆盖期刊: Optics Express, Optics Letters, Photonics Research, 
              Optica, JOSA B 等光学/光子学核心期刊。
    
    Args:
        driver: Selenium WebDriver
        keyword: 搜索关键词
        device_type: 设备类型
        fetch_full: 是否尝试抓取全文 (Optica 有部分开放获取)
    """
    results = []
    encoded_kw = urllib.parse.quote_plus(keyword)
    # Optica Publishing Group 搜索接口
    url = f"https://opg.optica.org/search.cfm?q={encoded_kw}&searchtype=all"
    
    try:
        driver.get(url)
        time.sleep(3)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Optica 搜索结果项
        articles = soup.find_all('div', class_='search-result-item')
        if not articles:
            # 尝试备用选择器
            articles = soup.find_all('li', class_='search-result')
        if not articles:
            # 再次尝试
            articles = soup.find_all('div', class_='resultItem')
        
        if not articles:
            print(f"    No Optica results for: {keyword}")
            return []
        
        for article in articles[:5]:
            try:
                # 标题
                title_tag = article.find('a', class_='title') or article.find('h3') or article.find('a')
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                
                # 链接
                link = title_tag.get('href', 'N/A')
                if link != 'N/A' and not link.startswith('http'):
                    link = f"https://opg.optica.org{link}"
                
                # 摘要
                abstract_tag = article.find('p', class_='abstract') or article.find('div', class_='abstract')
                abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""
                
                full_text = f"{title} {abstract}"
                has_full = False
                
                # 尝试获取全文 (Optica 部分论文开放获取)
                if fetch_full and link != "N/A":
                    try:
                        driver.get(link)
                        time.sleep(2)
                        page_soup = BeautifulSoup(driver.page_source, 'html.parser')
                        
                        # 移除无关内容
                        for tag in page_soup.find_all(['nav', 'footer', 'script', 'style']):
                            tag.decompose()
                        for ref in page_soup.find_all(class_='references'):
                            ref.decompose()
                        
                        # Optica 全文通常在 article-body 或 content-inner
                        body = (page_soup.find('div', class_='article-body') or
                                page_soup.find('div', class_='content-inner') or
                                page_soup.find('article'))
                        
                        if body:
                            page_text = body.get_text(separator=' ', strip=True)
                            page_text = re.sub(r'\s+', ' ', page_text).strip()
                            if len(page_text) > 500:
                                full_text = page_text[:15000]
                                has_full = True
                                print(f"      [Optica Full] {len(page_text)} chars from {title[:40]}...")
                    except Exception as e:
                        pass  # 全文获取失败，使用摘要
                
                results.append({
                    "source": "optica",
                    "title": title,
                    "link": link,
                    "abstract": abstract,
                    "full_text": full_text,
                    "has_full_text": has_full,
                    "device_type": device_type
                })
            except Exception as inner_e:
                continue
                
    except Exception as e:
        print(f"    Optica connection error: {e}")
    
    return results


# ==================== 主流程 ====================

def main():
    print("=== Auto-PDK Generator Started ===")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    
    driver = init_driver()
    all_papers = []

    try:
        for device_type, keywords in DEVICE_keywords.items():
            print(f"\nSearching for {device_type}...")
            for kw in keywords:
                papers = search_arxiv(driver, kw, device_type)
                all_papers.extend(papers)
                print(f"  Found {len(papers)} papers for '{kw}'")
    
    finally:
        try:
            driver.quit()
        except:
            pass

    print(f"\nProcessing {len(all_papers)} papers for code generation...")
    
    generated_count = 0
    for paper in all_papers:
        filepath = generate_component_file(paper)
        if filepath:
            generated_count += 1

    print("\n" + "="*50)
    print(f"COMPLETE: Generated {generated_count} new components in DesignLibrary.")
    print(f"Location: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
