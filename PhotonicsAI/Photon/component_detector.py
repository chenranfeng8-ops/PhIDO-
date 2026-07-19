"""
Component type detection module.
Provides unified component type detection for Tidy3D simulations.
"""

from typing import Tuple, Optional
import re


# Component type definitions with detection rules
# Format: (type_name, primary_keywords, exclude_keywords, priority)
COMPONENT_RULES = [
    # Priority 1: Most specific types (checked first)
    ("y_branch", ["y-branch", "y branch", "ybranch", "1x2 splitter"], ["mmi"], 1),
    ("polarization_rotator", ["polarization rotator", "polarization converter", "te-tm converter"], [], 1),
    ("subwavelength_grating", ["subwavelength grating", "swg", "sub-wavelength"], ["polarization"], 1),
    
    # Priority 2: Specific photonic components
    ("ring_resonator", ["ring resonator", "micro-ring", "microring", "mrr", "ring filter"], [], 2),
    ("grating_coupler", ["grating coupler", "fiber coupler", "gc ", "vertical coupler"], [], 2),
    ("mzi", ["mzi", "mach-zehnder", "mach zehnder", "interferometer"], [], 2),
    ("mmi", [
        "mmi", "multi-mode interferometer", "multimode",
        # WDM / demux common aliases (English + Chinese)
        "wdm", "demux", "mux", "wavelength division multiplexer", "wavelength division demultiplexer",
        "波分复用", "波分解复用", "波分复用器", "波分解复用器", "解复用", "复用器",
    ], [], 2),
    ("directional_coupler", ["directional coupler", "dc ", "coupler", "evanescent coupler"], ["grating", "y-branch"], 2),
    
    # Priority 3: General types
    ("crossing", ["crossing", "waveguide crossing", "wg crossing"], [], 3),
    ("waveguide", ["waveguide", "wg ", "straight waveguide", "wire"], [], 3),
    ("splitter", ["splitter", "power splitter", "分束器", "功分器"], [], 3),
    ("modulator", ["modulator", "mzm", "phase shifter", "heater"], [], 3),
    ("bragg", ["bragg", "bragg grating", "dbr"], [], 3),
    
    # Fallback
    ("unknown", [], [], 99),
]


def detect_component_type(component_description: str) -> Tuple[str, float]:
    """
    Detect component type from description.
    
    Args:
        component_description: Description of the component (e.g., "y-branch splitter with 50:50 ratio")
        
    Returns:
        Tuple of (component_type, confidence_score)
    """
    if not component_description:
        return ("unknown", 0.0)
    
    desc_lower = component_description.lower().strip()
    
    # Sort rules by priority
    sorted_rules = sorted(COMPONENT_RULES, key=lambda x: x[3])
    
    best_match = ("unknown", 0.0)
    
    for comp_type, keywords, excludes, priority in sorted_rules:
        if comp_type == "unknown":
            continue
            
        # Check if any exclude keyword is present
        has_exclude = any(ex in desc_lower for ex in excludes)
        if has_exclude:
            continue
        
        # Check for keyword matches
        match_count = 0
        matched_keywords = []
        for kw in keywords:
            if kw in desc_lower:
                match_count += 1
                matched_keywords.append(kw)
        
        if match_count > 0:
            # Calculate confidence based on:
            # - Number of matched keywords
            # - Priority (lower is better)
            # - Length of matched keywords (longer is more specific)
            keyword_length = sum(len(kw) for kw in matched_keywords)
            confidence = match_count * 10 + (100 - priority) + keyword_length / 10
            
            if confidence > best_match[1]:
                best_match = (comp_type, confidence)
    
    return best_match


def get_component_display_name(component_type: str) -> str:
    """Get human-readable display name for component type."""
    names = {
        "y_branch": "Y-Branch Splitter",
        "polarization_rotator": "Polarization Rotator",
        "subwavelength_grating": "Subwavelength Grating",
        "ring_resonator": "Ring Resonator",
        "grating_coupler": "Grating Coupler",
        "mzi": "Mach-Zehnder Interferometer",
        "mmi": "MMI Splitter",
        "directional_coupler": "Directional Coupler",
        "crossing": "Waveguide Crossing",
        "waveguide": "Waveguide",
        "splitter": "Power Splitter",
        "modulator": "Modulator",
        "bragg": "Bragg Grating",
        "unknown": "Unknown Component",
    }
    return names.get(component_type, component_type.replace("_", " ").title())


def get_component_sim_params(component_type: str) -> dict:
    """Get default simulation parameters for component type."""
    params = {
        "y_branch": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "arm_length": 15.0,
            "arm_separation": 3.0,
        },
        "polarization_rotator": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "rotation_length": 30.0,
            "swg_period": 0.4,
        },
        "ring_resonator": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "ring_radius": 5.0,
            "gap": 0.2,
        },
        "grating_coupler": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "grating_period": 0.62,
            "grating_duty_cycle": 0.5,
            "num_periods": 20,
        },
        "mzi": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "arm_length": 20.0,
            "arm_separation": 5.0,
        },
        "mmi": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "mmi_width": 2.5,
            "mmi_length": 10.0,
        },
        "directional_coupler": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "coupler_length": 10.0,
            "gap": 0.2,
        },
        "crossing": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "wg_length": 10.0,
        },
        "waveguide": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "wg_length": 10.0,
        },
        "splitter": {
            "wg_width": 0.5,
            "wg_height": 0.22,
            "mmi_width": 2.5,
            "mmi_length": 10.0,
        },
        "unknown": {
            "wg_width": 0.5,
            "wg_height": 0.22,
        },
    }
    return params.get(component_type, params["unknown"])


# Test function
if __name__ == "__main__":
    test_cases = [
        "y-branch splitter with 50:50 ratio",
        "polarization rotator using subwavelength gratings",
        "ring resonator with high Q factor",
        "grating coupler for fiber coupling",
        "1x2 MMI splitter",
        "Mach-Zehnder interferometer",
        "directional coupler with 50:50 splitting",
        "waveguide crossing",
        "subwavelength grating polarizer",
    ]
    
    print("Component Type Detection Tests:")
    print("-" * 60)
    for desc in test_cases:
        comp_type, confidence = detect_component_type(desc)
        display_name = get_component_display_name(comp_type)
        print(f"{desc[:40]:40} → {comp_type:20} ({confidence:.1f})")
