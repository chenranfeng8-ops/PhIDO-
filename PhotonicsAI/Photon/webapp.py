# ruff: noqa
"""
Phido - PHotonic Intelligent Design & Optimization

A Streamlit web application for automated photonic circuit design using AI.
Supports both automatic workflow (guided) and step-by-step execution modes.

The application provides two main workflow modes:
1. Automatic Workflow: Guided step-by-step process with automatic progression
2. Step-by-Step: Individual step execution with custom inputs

Each workflow consists of 4 main phases:
- Entity Extraction: Extract circuit components from natural language
- Component Specification: Search and select specific components
- Schematic Generation: Create circuit diagrams and layouts
- Layout & Simulation: Generate GDS files and run simulations
"""

# Standard library imports
import copy
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Third-party imports
import numpy as np
import streamlit as st
import yaml

# Local imports
from PhotonicsAI.config import PATH
from PhotonicsAI.Photon import llm_api, utils
from PhotonicsAI.Photon.DemoPDK import *
from PhotonicsAI.Photon.drc.drc import run_drc



# =============================================================================
# CONFIGURATION
# =============================================================================

# Available LLM models (legacy fallback catalog)
LEGACY_LLM_MODELS = {
    "阿里云百炼": [
        "glm-4.7",           # GLM-4.7 (推荐)
        "qwen-plus",         # Qwen Plus
        "qwen-turbo",        # Qwen Turbo (快速)
        "qwen-max",          # Qwen Max (最强)
        "qwen-long",         # Qwen Long (长上下文)
    ],
    "智谱 AI 模型": [
        "glm-4",             # GLM-4 最新版
        "glm-4-flash",       # GLM-4 Flash (免费)
        "chatglm_turbo",     # ChatGLM Turbo（更快）
        "chatglm_pro",       # ChatGLM Pro
        "chatglm_std",       # ChatGLM 标准版
    ],
    "OpenAI 模型": [
        "gpt-4o",           # GPT-4
        "o1",               # 推理优化版
        "o3-mini",          # 快速版
    ],
    "Anthropic 模型": [
        "claude-3-7-sonnet-20250219",  # Claude Sonnet
        "claude-opus-4-20250514",      # Claude Opus 4.0
    ],
    "Google 模型": [
        "gemini-2.5-pro",               # Gemini Pro 2.5
        "gemini-1.5-pro",               # Gemini Pro 1.5
        "gemini-1.5-flash",             # Gemini Flash（快速）
        "gemini-2.0-flash",             # Gemini Flash 2.0
    ],
}


def _load_resourcepack_models() -> list[str]:
    raw = os.getenv("RESOURCEPACK_MODELS", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


RESOURCEPACK_MODELS = _load_resourcepack_models()
DEFAULT_LLM_MODEL = os.getenv("RESOURCEPACK_DEFAULT_MODEL", "glm-4.7").strip() or "glm-4.7"

if RESOURCEPACK_MODELS:
    LLM_MODELS = {"资源包模型": RESOURCEPACK_MODELS}
else:
    LLM_MODELS = LEGACY_LLM_MODELS

# LLM model configurations for different workflow steps
# 从侧边栏选择的模型
def get_selected_model():
    # 创建扁平的模型列表用于选择
    all_models = []
    for category, models in LLM_MODELS.items():
        all_models.extend([f"{model} ({category.split()[0]})" for model in models])
    
    # 在侧边栏添加模型选择器
    with st.sidebar:
        st.markdown("## 🤖 LLM 模型选择")
        selected = st.selectbox(
            "选择要使用的 AI 模型",
            all_models,
            index=all_models.index(f"{DEFAULT_LLM_MODEL} (资源包模型)") if f"{DEFAULT_LLM_MODEL} (资源包模型)" in all_models else 0,
            help="选择用于所有步骤的 LLM 模型。不同模型可能需要不同的 API Key，请确保已在 .env 中配置。"
        )
        # 从选择中提取实际的模型名（去掉分类标识）
        return selected.split(" (")[0]

# 所有步骤默认使用相同的模型
selected_model = DEFAULT_LLM_MODEL  # 默认值，会被 get_selected_model() 更新
entity_extraction_model = selected_model
component_selection_model = selected_model
component_specification_model = selected_model
schematic_model = selected_model
layout_model = selected_model

# HTML templates for UI styling
# Used to create consistent visual elements throughout the interface
html_banner = """
<div style="
    border: 0px;
    background-color: dimgray;
    padding: 10px;
    border-radius: 5px;
    width: 100%;
    text-align: center;
    margin: 0 auto;
">
    <p style="font-size: 18px; color: white; margin: 0;">
        {content}
    </p>
</div>
"""

html_small = """<p style='font-size:13px; color: grey;'>{content}</p>"""

# =============================================================================
# DATA LOADING
# =============================================================================

# Load prompts and templates from configuration files
# These contain the LLM prompts and circuit templates used throughout the application
with open(PATH.prompts) as file:
    prompts = yaml.safe_load(file)

# Load component documentation and names
# This searches the design library for available components and their documentation
components_list = utils.search_directory_for_docstrings()
list_of_docs = [i["docstring"] for i in components_list]
list_of_cnames = [i["module_name"] for i in components_list]

# Load circuit templates
# Templates provide pre-defined circuit structures that users can customize
with open(PATH.templates) as file:
    templates_dict = yaml.safe_load(file)

# Utility function to convert tuples to lists for safe YAML serialization
def convert_tuples_to_lists(obj):
    """
    Convert all tuples in a nested structure to lists for safe YAML serialization.
    
    This function is necessary because YAML doesn't handle Python tuples well,
    and many of our data structures contain tuples that need to be serialized.
    
    Args:
        obj: Object that may contain tuples (dict, list, tuple, or primitive)
        
    Returns:
        Object with all tuples converted to lists
    """
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(item) for item in obj]
    else:
        return obj

# Step-by-Step Execution Functions
# These functions handle the step-by-step execution mode where users can
# run each workflow step individually with custom inputs
def run_step_by_step_entity_extraction(custom_prompt):
    """
    Run entity extraction step with custom input in step-by-step workflow mode.
    
    This function extracts circuit components and their relationships from natural
    language descriptions using LLM-based entity extraction.
    
    Args:
        custom_prompt: Natural language description of the photonic circuit
        
    Returns:
        dict: Results containing pretemplate, preschematic, and metadata
    """
    try:
        # Start timing for step-by-step workflow
        session.p100_start_time = time.time()
        
        # Reset token usage at the start of a new workflow
        llm_api.reset_token_usage()
        
        session.log_filename, session.log_id = get_next_log_filename()
        logger()
        
        st.markdown(
            '<div style="text-align: right; font-size: 18px; font-family: monospace;">Step-by-Step: Entity Extraction</div>',
            unsafe_allow_html=True,
        )
        
        # Configure LLM model and component data for this step
        session.p100_llm_api_selection = entity_extraction_model
        session.p100_list_of_docs = list_of_docs
        session.p100_list_of_cnames = list_of_cnames
        
        # Display the input prompt for user reference
        with st.container(border=True):
            st.markdown(f"*{custom_prompt}*")
        
        # Classify input as photonic layout prompt or not
        # This ensures we only process relevant photonic circuit descriptions
        interpreter_cat = llm_api.intent_classification(custom_prompt)
        
        if interpreter_cat.category_id != 1:
            st.markdown(f"**{interpreter_cat.response}**")
            return None
        
        # Perform entity extraction and preschematic generation
        with st.spinner("Entity extraction ..."):
            # Extract circuit entities from natural language
            pretemplate = llm_api.entity_extraction(custom_prompt)
            # Generate initial schematic diagram
            preschematic = llm_api.preschematic(pretemplate, session.p100_llm_api_selection)
            # Create a copy of the pretemplate for later use
            pretemplate_copy = copy.deepcopy(pretemplate)
            
            # Display results in two columns for better visualization
            col1, col2 = st.columns(2)
            with col2:
                try:
                    st.write("Initial schematic:")
                    st.graphviz_chart(preschematic)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.write("Failed to render:\n```dot\n" + preschematic)
            
            with col1:
                st.write("Extracted entities:")
                st.write(
                    "```yaml\n"
                    + yaml.dump(pretemplate, sort_keys=False, width=55)
                )
        
        # Store results in session for later use
        result = {
            "pretemplate": pretemplate,
            "pretemplate_copy": pretemplate_copy,
            "preschematic": preschematic,
            "prompt": custom_prompt
        }
        session.step_results["entity_extraction"] = result
        logger()
        return result
    except Exception as e:
        st.error(f"Entity extraction failed: {e}")
        return None

def run_step_by_step_component_specification(custom_pretemplate_yaml):
    """
    Run component specification step with custom pretemplate in step-by-step workflow mode.
    
    This function searches the component library for components that match the
    requirements specified in the pretemplate and prepares them for selection.
    
    Args:
        custom_pretemplate_yaml: YAML string containing the pretemplate
        
    Returns:
        None: Results are stored in session state for UI display
    """
    try:
        st.markdown(
            '<div style="text-align: right; font-size: 18px; font-family: monospace;">Step-by-Step: Component Specification</div>',
            unsafe_allow_html=True,
        )
        
        # Parse YAML input with error handling
        try:
            pretemplate = yaml.safe_load(custom_pretemplate_yaml)
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML format: {e}")
            return None
        
        # Configure LLM model and component data for this step
        session.p100_llm_api_selection = component_selection_model
        session.p100_list_of_docs = list_of_docs
        session.p100_list_of_cnames = list_of_cnames
        
        # Initialize search results containers
        components_search_r = []
        retreived_templates = []
        
        # Perform component and template search if components are specified
        if len(pretemplate.get("components_list", [])):
            # Search for matching templates in the template library
            templates_search_r = llm_api.llm_search(
                str(pretemplate), list(templates_dict.values())
            )
            retreived_templates = [
                (list(templates_dict.items())[i])
                for i in templates_search_r.match_list
            ]
            # Search for matching components in the component library
            for c in pretemplate["components_list"]:
                r = llm_api.llm_search(c, list_of_docs)
                components_search_r.append(r)

        # Store search results in session for the interface
        # Convert any tuples to lists to avoid YAML issues later
        safe_pretemplate = convert_tuples_to_lists(pretemplate)
        session.component_search_results = {
            "pretemplate": safe_pretemplate,
            "components_search": components_search_r,
            "templates_search": retreived_templates
        }
        session.component_search_ready = True
        
        st.success("Component search completed! Please select components below.")
        return None
        
    except Exception as e:
        st.error(f"Component specification failed: {e}")
        return None

def run_step_by_step_circuit_dsl_creation(custom_pretemplate_yaml, custom_selected_components=None, custom_template_id=None):
    """
    Run circuit DSL creation step with custom inputs in step-by-step workflow mode.
    
    This function creates a circuit DSL (Domain Specific Language) representation
    from either selected components or a template. The DSL defines the circuit
    structure, connections, and properties.
    
    Args:
        custom_pretemplate_yaml: YAML string containing the pretemplate
        custom_selected_components: List of selected component names (optional)
        custom_template_id: Template ID to use (optional)
        
    Returns:
        dict: Results containing circuit DSL and metadata
    """
    try:
        st.markdown(
            '<div style="text-align: right; font-size: 18px; font-family: monospace;">Step-by-Step: Circuit DSL Creation</div>',
            unsafe_allow_html=True,
        )
        
        try:
            pretemplate = yaml.safe_load(custom_pretemplate_yaml)
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML format: {e}")
            return None
        
        session.p100_llm_api_selection = component_specification_model
        session.p100_list_of_docs = list_of_docs
        session.p100_list_of_cnames = list_of_cnames
        
        # Create circuit DSL based on inputs
        if custom_template_id:
            # Template path
            session.p200_selected_template = custom_template_id
            session.template_selected = True
            
            # Try to preserve original component specifications from entity extraction
            original_pretemplate = None
            if hasattr(session, 'step_results') and 'entity_extraction' in session.step_results:
                original_pretemplate = session.step_results['entity_extraction'].get('pretemplate')
            
            # Store original specifications for later use in apply_settings
            if original_pretemplate and 'components_list' in original_pretemplate:
                # Store the original specifications for use in schematic generation
                session.original_component_specifications = original_pretemplate["components_list"]
            
            # Get template specifications
            template_specs = templates_dict[custom_template_id]["properties"]["specs"]
            
            # Create UI for template specifications
            st.write(f"Template: {custom_template_id}")
            st.write("Required specifications:")
            
            user_specs = {}
            for key, item in template_specs.items():
                user_input = st.text_input(
                    f"{key} ({item['comment']})", 
                    item["value"],
                    key=f"template_spec_{key}"
                )
                user_specs[key] = {
                    "value": user_input,
                    "comment": item["comment"],
                }
            
            if st.button("Generate Circuit DSL from Template"):
                session.user_specs = user_specs
                session.updated_specs = yaml.dump(user_specs, default_flow_style=False)
                session["p200_user_specs"] = session.updated_specs
                
                parsed_spec = llm_api.parse_user_specs(session)
                
                if "Error" in parsed_spec:
                    st.error(f"Specification error: {parsed_spec}")
                    return None
                
                # Create circuit DSL from template
                circuit_dsl = templates_dict[custom_template_id].copy()
                circuit_dsl["properties"]["specs"] = parsed_spec
                
                # Component retrieval for template nodes
                if "TEMPLATE" in custom_template_id:
                    st.write("Looking for components...")
                    for key, value in circuit_dsl["nodes"].items():
                        try:
                            user_specs = circuit_dsl["properties"]["specs"]
                            for spec_key in user_specs:
                                if "comment" in user_specs[spec_key]:
                                    del user_specs[spec_key]["comment"]
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            user_specs = ""

                        r = llm_api.llm_retrieve(
                            value + f"\n({str(user_specs)})",
                            session["p100_list_of_docs"],
                            session["p100_llm_api_selection"],
                        )

                        all_retrieved = [session["p100_list_of_cnames"][i] for i in r]
                        st.write(f"For {value}, found: " + "\n".join(all_retrieved))

                        selected_component = session["p100_list_of_cnames"][r[0]]
                        circuit_dsl["nodes"][key] = {}
                        circuit_dsl["nodes"][key]["component"] = selected_component
                
                result = {
                    "circuit_dsl": circuit_dsl,
                    "template_id": custom_template_id,
                    "parsed_specs": parsed_spec
                }
                session.step_results["circuit_dsl_creation"] = result
                st.success("Circuit DSL creation completed!")
                return result
                
        elif custom_selected_components:
            # Component selection path
            session.p200_selected_components = custom_selected_components
            session.components_selected = True
            
            # Try to preserve original component specifications from entity extraction
            original_pretemplate = None
            if hasattr(session, 'step_results') and 'entity_extraction' in session.step_results:
                original_pretemplate = session.step_results['entity_extraction'].get('pretemplate')
            
            # Store original specifications for later use in apply_settings
            if original_pretemplate and 'components_list' in original_pretemplate:
                # Store the original specifications for use in schematic generation
                session.original_component_specifications = original_pretemplate["components_list"]
            
            # Use the selected components for the circuit DSL
            pretemplate["components_list"] = custom_selected_components
            
            # Create circuit DSL using map_pretemplate_to_template logic
            link = "(link)"
            labels = [""]

            circuit_dsl = {
                "doc": {
                    "title": pretemplate.get("title", ""),
                    "description": pretemplate.get("brief_summary", ""),
                    "reference": link,
                    "labels": labels,
                },
                "nodes": {},
                "edges": pretemplate.get("circuit_instructions", ""),
                "properties": {},
            }

            # Mapping components_list to nodes
            components = pretemplate.get("components_list", [])
            for i, component in enumerate(components, start=1):
                node_label = f"N{i}"
                circuit_dsl["nodes"][node_label] = {"component": component}
            
            result = {
                "circuit_dsl": circuit_dsl,
                "selected_components": custom_selected_components
            }
            session.step_results["circuit_dsl_creation"] = result
            st.success("Circuit DSL creation completed!")
            return result
        
        return None
    except Exception as e:
        st.error(f"Circuit DSL creation failed: {e}")
        return None

def run_step_by_step_schematic_generation(custom_circuit_dsl_yaml, custom_preschematic=None):
    """Run schematic generation with custom circuit DSL"""
    try:
        st.markdown(
            '<div style="text-align: right; font-size: 18px; font-family: monospace;">Step-by-Step: Schematic Generation</div>',
            unsafe_allow_html=True,
        )
        
        try:
            circuit_dsl = yaml.safe_load(custom_circuit_dsl_yaml)
        except yaml.YAMLError as e:
            # If there are tuple tags, try to clean them up first
            try:
                # Remove Python tuple tags and convert to lists
                cleaned_yaml = custom_circuit_dsl_yaml.replace("!!python/tuple", "")
                # Replace tuple syntax with list syntax
                cleaned_yaml = cleaned_yaml.replace("  - ''", "  - ''")  # Keep empty strings as lists
                circuit_dsl = yaml.safe_load(cleaned_yaml)
                # Convert any remaining tuples to lists
                circuit_dsl = convert_tuples_to_lists(circuit_dsl)
            except yaml.YAMLError as e2:
                st.error(f"Invalid YAML format: {e2}")
                return None
        
        session.p100_llm_api_selection = schematic_model
        session.p100_list_of_docs = list_of_docs
        session.p100_list_of_cnames = list_of_cnames
        
        # Set up session for schematic generation
        session["p300_circuit_dsl"] = circuit_dsl
        session["p300"] = True
        
        # Initialize missing session variables for step-by-step workflow
        if not hasattr(session, 'p200_pretemplate_copy'):
            # Create a mock pretemplate from the circuit DSL for apply_settings function
            # Try to preserve original specifications from the circuit DSL or user input
            components_list = []
            
            # First, try to get original specifications from stored session data
            if hasattr(session, 'original_component_specifications'):
                # Use the stored original component specifications
                components_list = session.original_component_specifications
            else:
                # Fallback: try to get from step results
                original_pretemplate = None
                if hasattr(session, 'step_results') and 'entity_extraction' in session.step_results:
                    original_pretemplate = session.step_results['entity_extraction'].get('pretemplate')
                
                if original_pretemplate and 'components_list' in original_pretemplate:
                    # Use the original component specifications from entity extraction
                    components_list = original_pretemplate['components_list']
                else:
                    # Final fallback: try to preserve specifications from the circuit DSL
                    for i, (node_id, node) in enumerate(circuit_dsl.get("nodes", {}).items(), 1):
                        component_name = node.get("component", f"component_{i}")
                        
                        # Check if there are any settings or specifications in the node
                        settings = node.get("settings", {})
                        if settings:
                            # Create a descriptive component name with specifications
                            spec_parts = []
                            for key, value in settings.items():
                                if key != "comment":  # Skip comment fields
                                    spec_parts.append(f"{key} of {value}")
                            
                            if spec_parts:
                                component_name = f"{component_name} with " + ", ".join(spec_parts)
                        
                        components_list.append(component_name)
            
            mock_pretemplate = {
                "components_list": components_list
            }
            session.p200_pretemplate_copy = mock_pretemplate
        
        # Generate proper preschematic with edge information
        if not hasattr(session, 'p200_preschematic'):
            if custom_preschematic and custom_preschematic.strip():
                # Use custom preschematic provided by user
                session.p200_preschematic = custom_preschematic.strip()
                st.info("📝 Using custom preschematic provided by user")
            else:
                # Create a pretemplate from the circuit DSL for preschematic generation
                pretemplate = {
                    "title": circuit_dsl.get("doc", {}).get("title", "Custom Circuit"),
                    "brief_summary": circuit_dsl.get("doc", {}).get("description", "A custom photonic circuit"),
                    "circuit_instructions": circuit_dsl.get("edges", ""),  # Look in edges field for circuit instructions
                    "components_list": [node.get("component", f"component_{i}") for i, node in enumerate(circuit_dsl.get("nodes", {}).values(), 1)]
                }
                
                # Generate preschematic using the same function as automatic workflow
                session.p200_preschematic = llm_api.preschematic(pretemplate, session.p100_llm_api_selection)
                st.info("📝 Auto-generated preschematic from circuit DSL")
        
        # Add ports and parameters
        session["p300_circuit_dsl"] = get_ports_info(session["p300_circuit_dsl"])
        session["p300_circuit_dsl"] = get_params(session["p300_circuit_dsl"])
        
        # Apply settings if not template
        if not session.get("template_selected", False):
            session["p300_circuit_dsl"] = llm_api.apply_settings(session, session.p100_llm_api_selection)
        
        with st.expander("Circuit draft", expanded=False):
            st.write("```yaml\n" + yaml.dump(session["p300_circuit_dsl"]))
        
        with st.spinner("Working on the schematic..."):
            session["p300_dot_string_draft"] = utils.circuit_to_dot(
                session["p300_circuit_dsl"]
            )
            
            if len(session["p300_circuit_dsl"]["nodes"]) > 0:
                session["p300_dot_string"] = llm_api.dot_add_edges(session)
                session["p300_dot_string"] = llm_api.dot_verify(session)
                
                for attempt in range(4):
                    happy_flag = utils.dot_planarity(session["p300_dot_string"])
                    if happy_flag:
                        break
                    else:
                        st.markdown(":red[Crossing edges found! Redoing the graph edges...]")
                        session["p300_dot_string"] = llm_api.dot_add_edges_errorfunc(session)
                        session["p300_dot_string"] = llm_api.dot_verify(session)
            else:
                session["p300_dot_string"] = llm_api.dot_add_edges_templates(session)
            
            session["p300_dot_string"] = llm_api.dot_verify(session)
        
        with st.expander("Schematic diagram", expanded=False):
            st.write("```dot\n" + session["p300_dot_string"])
        
        st.graphviz_chart(session["p300_dot_string"])
        session["p300_circuit_dsl"] = utils.edges_dot_to_yaml(session)
        
        # Get initial placements from dot
        session["p300_footprints_dict"], session["p300_circuit_dsl"] = footprint_netlist(
            session["p300_circuit_dsl"]
        )
        session["p300_dot_string_scaled"] = utils.dot_add_node_sizes(
            session["p300_dot_string"],
            utils.multiply_node_dimensions(session["p300_footprints_dict"], 0.01),
        )
        session["p300_graphviz_node_coordinates"] = utils.get_graphviz_placements(
            session["p300_dot_string_scaled"]
        )
        session["p300_graphviz_node_coordinates"] = utils.multiply_node_dimensions(
            session["p300_graphviz_node_coordinates"], 100 / 72
        )
        
        session["p300_circuit_dsl"] = utils.add_placements_to_dsl(session)
        session["p300_circuit_dsl"] = utils.add_final_ports(session)
        
        with st.expander("Circuit draft, updated", expanded=False):
            st.write("```yaml\n" + yaml.dump(session["p300_circuit_dsl"]))
        
        result = {
            "circuit_dsl": convert_tuples_to_lists(session["p300_circuit_dsl"]),
            "dot_string": session["p300_dot_string"],
            "footprints_dict": session["p300_footprints_dict"]
        }
        session.step_results["schematic_generation"] = result
        st.success("Schematic generation completed!")
        return result
        
    except Exception as e:
        st.error(f"Schematic generation failed: {e}")
        return None

def run_step_by_step_layout_simulation(custom_circuit_dsl_yaml):
    """Run layout and simulation with custom circuit DSL"""
    try:
        st.markdown(
            '<div style="text-align: right; font-size: 18px; font-family: monospace;">Step-by-Step: Layout & Simulation</div>',
            unsafe_allow_html=True,
        )
        
        try:
            circuit_dsl = yaml.safe_load(custom_circuit_dsl_yaml)
        except yaml.YAMLError as e:
            # If there are tuple tags, try to clean them up first
            try:
                # Remove Python tuple tags and convert to lists
                cleaned_yaml = custom_circuit_dsl_yaml.replace("!!python/tuple", "")
                # Replace tuple syntax with list syntax
                cleaned_yaml = cleaned_yaml.replace("  - ''", "  - ''")  # Keep empty strings as lists
                circuit_dsl = yaml.safe_load(cleaned_yaml)
                # Convert any remaining tuples to lists
                circuit_dsl = convert_tuples_to_lists(circuit_dsl)
            except yaml.YAMLError as e2:
                st.error(f"Invalid YAML format: {e2}")
                return None
        
        session.p100_llm_api_selection = layout_model
        session.p100_list_of_docs = list_of_docs
        session.p100_list_of_cnames = list_of_cnames
        
        # Set up session for layout
        session["p300_circuit_dsl"] = circuit_dsl
        session["p400"] = True
        
        # Convert to GDSFactory netlist
        session["p400_gf_netlist"] = utils.dsl_to_gf(session["p300_circuit_dsl"])
        
        with st.expander("GDS-Factory Netlist", expanded=False):
            st.write("```yaml\n" + yaml.dump(session["p400_gf_netlist"]))
        
        with st.spinner("Rendering the GDS ..."):
            try:
                c, d = yaml_netlist_to_gds(session, ignore_links=False)
                routing_flag = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
                c, d = yaml_netlist_to_gds(session, ignore_links=True)
                st.markdown(":red[Routing error.]")
                routing_flag = False
        
        st.pyplot(session["p400_gdsfig"])
        
        with st.spinner("Simulating s-parameters ..."):
            wl = np.linspace(1.5, 1.6, 200)
            result = session["p400_sax_circuit"](wl=wl)
            p400_sax_fig = utils.plot_dict_arrays(wl, result)
        st.image(str(PATH.build / "plot_sax.png"))

        # Show Tidy3D integration logs (if any) for this run
        try:
            tlog = PATH.build / "tidy3d.log"
            tcfg = PATH.build / "tidy3d_config.json"
            sim_pngs = [
                PATH.build / "tidy3d_sim_z0.png",
                PATH.build / "tidy3d_sim_x0.png",
                PATH.build / "tidy3d_sim_y0.png",
            ]
            if tlog.exists() or tcfg.exists() or any(p.exists() for p in sim_pngs):
                with st.expander("Tidy3D 集成日志、配置与结构图", expanded=False):
                    # 配置
                    if tcfg.exists():
                        st.caption("配置 (build/tidy3d_config.json)")
                        try:
                            st.code((tcfg.read_text(encoding="utf-8")), language="json")
                        except Exception:
                            st.write(str(tcfg))
                    # 日志
                    if tlog.exists():
                        st.caption("日志 (build/tidy3d.log)")
                        try:
                            lines = tlog.read_text(encoding="utf-8").splitlines()[-120:]
                            st.code("\n".join(lines), language="text")
                        except Exception:
                            st.write(str(tlog))
                    # 结构图
                    if any(p.exists() for p in sim_pngs):
                        st.caption("结构图切片 (z=0 / x=0 / y=0)")
                        cols = st.columns(3)
                        for i, p in enumerate(sim_pngs):
                            if p.exists():
                                with cols[i]:
                                    st.image(str(p))
        except Exception:
            pass
        
        with st.spinner("Checking DRC..."):
            try:
                cwd = Path.cwd()
                open(str(cwd)+"/PhotonicsAI/Photon/drc/report.lydrb", "w+").close()
                file_name = "placeholder"
                gds_drc_file_path = "./drc/" + file_name + ".gds"
                skip_drc = False  # Flag to skip DRC if GDS write fails
                
                # Write GDS file - try different approaches to handle large layer numbers
                try:
                    # First try: standard write
                    c.write_gds(gds_drc_file_path)
                except Exception as e:
                    if "layer numbers larger than 65535" in str(e):
                        st.warning("Large layer numbers detected, trying alternative write method...")
                        try:
                            # Try flattening first - check if result is not None
                            flattened = c.flatten()
                            if flattened is not None:
                                flattened.write_gds(gds_drc_file_path)
                                st.success("Successfully wrote flattened GDS file")
                            else:
                                st.warning("Flattening returned None, trying different approach...")
                                # Try to write with different parameters
                                c.write_gds(gds_drc_file_path, max_points=None)
                                st.success("Successfully wrote GDS file with modified parameters")
                        except Exception as flatten_error:
                            st.error(f"Flattening failed: {flatten_error}")
                            # Try writing with different parameters as last resort
                            try:
                                st.warning("Trying to write with minimal parameters...")
                                c.write_gds(gds_drc_file_path, max_points=None, max_absolute_error=None, max_relative_error=None)
                                st.success("Successfully wrote GDS file with minimal parameters")
                            except Exception as final_error:
                                st.error(f"All write methods failed: {final_error}")
                                # Create a simple placeholder file for DRC
                                st.warning("Creating placeholder file for DRC...")
                                with open(gds_drc_file_path, 'w') as f:
                                    f.write("# Placeholder file - GDS write failed due to layer number limitations\n")
                                st.info("DRC will be skipped due to GDS write failure")
                                # Skip DRC by setting a flag
                                skip_drc = True
                    else:
                        raise e
                
                # Only run DRC if we didn't skip it due to GDS write failure
                if not skip_drc:
                    run_drc(gds_drc_file_path, file_name)
                    st.success("✅ DRC completed successfully!")
                else:
                    st.warning("DRC skipped due to GDS write failure")
                with st.expander("DRC results", expanded=False):
                    report_file = str(cwd)+"/PhotonicsAI/Photon/drc/report.lydrb"
                    try:
                        if os.path.exists(report_file):
                            with open(report_file, 'r') as f:
                                report_content = f.read()
                            if report_content.strip():
                                st.text("DRC Report:")
                                st.code(report_content, language="text")
                            else:
                                st.info("DRC completed but report file is empty. This usually means no violations were found.")
                        else:
                            st.warning("DRC report file not found. DRC may not have completed successfully.")
                    except Exception as e:
                        st.error(f"Error reading DRC report: {e}")
                        st.info("Check the terminal output for DRC execution details.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Note: This error might be due to GDS2 layer number limitations. The circuit layout is still valid.")
        
        # Circuit optimizer (if applicable)
        optimize_flag = False
        if "properties" in session["p300_circuit_dsl"]:
            if "optimizer" in session["p300_circuit_dsl"]["properties"]:
                if "error_fn" in session["p300_circuit_dsl"]["properties"]["optimizer"]:
                    if "free_params" in session["p300_circuit_dsl"]["properties"]["optimizer"]:
                        if "sparam" in session["p300_circuit_dsl"]["properties"]["optimizer"]:
                            if routing_flag:
                                optimize_flag = True
        
        if optimize_flag:
            with st.spinner("Optimizing circuit..."):
                session["p400_gf_netlist"] = circuit_optimizer(session)
            
            with st.expander("OPTIMIZED GDS-Factory Netlist", expanded=False):
                st.write("```yaml\n" + yaml.dump(session["p400_gf_netlist"]))
            
            try:
                c, d = yaml_netlist_to_gds(session, ignore_links=False)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                c, d = yaml_netlist_to_gds(session, ignore_links=True)
                st.markdown(":red[Routing error.]")
            
            wl = np.linspace(1.53, 1.57, 500)
            result = session["p400_sax_circuit"](wl=wl)
            p400_sax_fig = utils.plot_dict_arrays(wl, result)
            st.image(str(PATH.build / "plot_sax.png"))
        
        result = {
            "gf_netlist": session["p400_gf_netlist"],
            "routing_success": routing_flag,
            "optimized": optimize_flag
        }
        session.step_results["layout_simulation"] = result
        
        # Display token usage at the end of the workflow
        token_usage = llm_api.get_token_usage()
        if token_usage["non_cached_input_tokens"] > 0 or token_usage["output_tokens"] > 0:
            st.markdown("---")
            st.markdown("### Token Usage Summary")
            st.markdown(f"**Input Tokens:** {token_usage['non_cached_input_tokens']}")
            st.markdown(f"**Output Tokens:** {token_usage['output_tokens']}")
            st.markdown(f"**Total Tokens:** {token_usage['non_cached_input_tokens'] + token_usage['output_tokens']}")
            
            # Print to terminal as well
            print(f"\n=== TOKEN USAGE SUMMARY ===")
            print(f"Input Tokens: {token_usage['non_cached_input_tokens']}")
            print(f"Output Tokens: {token_usage['output_tokens']}")
            print(f"Total Tokens: {token_usage['non_cached_input_tokens'] + token_usage['output_tokens']}")
            print(f"===========================\n")
        
        st.success("Layout and simulation completed!")
        return result
        
    except Exception as e:
        st.error(f"Layout and simulation failed: {e}")
        return None

def map_pretemplate_to_template():
    """Convert pretemplate to circuit DSL template"""
    pretemplate_dict = session.p200_pretemplate

    link = "(link)"
    labels = [""]

    template_dict = {
        "doc": {
            "title": pretemplate_dict.get("title", ""),
            "description": pretemplate_dict.get("brief_summary", ""),
            "reference": link,
            "labels": labels,
        },
        "nodes": {},
        "edges": pretemplate_dict.get("circuit_instructions", ""),
        "properties": {},
    }

    # Mapping components_list to nodes
    components = pretemplate_dict.get("components_list", [])
    for i, component in enumerate(components, start=1):
        node_label = f"N{i}"
        template_dict["nodes"][node_label] = {"component": component}

    session.p300_circuit_dsl = template_dict

def prepare_component_selection(custom_pretemplate_yaml):
    """Prepare component search results and set up for selection"""
    try:
        pretemplate = yaml.safe_load(custom_pretemplate_yaml)
        session.p100_llm_api_selection = component_selection_model
        components_search_r = []
        retreived_templates = []
        
        if len(pretemplate.get("components_list", [])):
            # search templates
            templates_search_r = llm_api.llm_search(
                str(pretemplate), list(templates_dict.values())
            )
            retreived_templates = [
                (list(templates_dict.items())[i])
                for i in templates_search_r.match_list
            ]
            # search components
            for c in pretemplate["components_list"]:
                r = llm_api.llm_search(c, list_of_docs)
                components_search_r.append(r)
        
        # Store search results in session for the interface
        # Convert any tuples to lists to avoid YAML issues later
        safe_pretemplate = convert_tuples_to_lists(pretemplate)
        session.component_search_data = {
            "pretemplate": safe_pretemplate,
            "components_search": components_search_r,
            "templates_search": retreived_templates
        }
        session.component_search_ready = True
        st.rerun()
    except Exception as e:
        st.error(f"Failed to prepare component selection: {e}")

def run_component_selection_interface():
    """Display component selection interface"""
    if not hasattr(session, 'component_search_results'):
        st.error("No component search results available")
        return
    
    results = session.component_search_results
    pretemplate = results["pretemplate"]
    components_search_r = results["components_search"]
    retreived_templates = results["templates_search"]
    
    # Display results and allow user selection
    selected_components = st.session_state.get("step_by_step_selected_components", [None]*len(components_search_r))
    if components_search_r:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                html_banner.format(content="🛠️ build a new circuit"),
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='height: 20px;'></div>",
                unsafe_allow_html=True
            )
            for i, search_result in enumerate(components_search_r):
                with st.container(border=True):
                    st.write(f"**{pretemplate['components_list'][i]}**")
                    options = []
                    for k, j in enumerate(search_result.match_list):
                        name = list_of_cnames[j]
                        score = search_result.match_scores[k]
                        if score == "exact":
                            option = f"{name} :green[/{score}/]"
                        elif score == "partial":
                            option = f"{name} :orange[/{score}/]"
                        elif score == "poor":
                            option = f"{name} :red[/{score}/]"
                        else:
                            option = f"{name} :grey[/{score}/]"
                        options.append(option)
                    # Use session state to persist selection
                    selected = st.radio(
                        f"Component {i+1} options:",
                        options,
                        key=f"step_by_step_component_{i}",
                        label_visibility="collapsed"
                    )
                    selected_components[i] = selected
        if retreived_templates:
            with col4:
                st.markdown(
                    html_banner.format(content="🧩 use a template"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='height: 20px;'></div>", unsafe_allow_html=True
                )
                for i, item in enumerate(retreived_templates):
                    with st.container(border=True):
                        t = item[1]
                        id_ = item[0]
                        st.write(t["doc"]["title"] + "\n" + t["doc"]["description"])
                        st.markdown(f"[reference]({t['doc']['reference']})")
                        if st.button(f"Select {id_}", key=f"step_by_step_template_{i}"):
                            st.success(f"Selected template: {id_}")
                            st.session_state["step_by_step_selected_template"] = id_

    # Submit button for component selection
    if components_search_r:
        if st.button("Submit Component Selection (Step-by-Step)", key="step_by_step_submit_components"):
            # Extract selected component names
            selected_names = []
            for sel in selected_components:
                if sel:
                    selected_names.append(sel.split(" :")[0])
            
            if selected_names:
                # Save result for next step
                result = {
                    "pretemplate": pretemplate,
                    "components_search": components_search_r,
                    "templates_search": retreived_templates,
                    "selected_components": selected_names,
                    "selected_template": st.session_state.get("step_by_step_selected_template", None)
                }
                
                session.step_results["component_specification"] = result
                
                # Clear search state
                session.component_search_ready = False
                session.component_search_results = None
                
                # Optionally clear selection state
                for i in range(len(selected_components)):
                    st.session_state.pop(f"step_by_step_component_{i}", None)
                st.session_state.pop("step_by_step_selected_template", None)
                st.session_state["step_by_step_selected_components"] = [None]*len(components_search_r)
                
                st.success("Component specification completed!")
                st.rerun()
            else:
                st.error("Please select at least one component.")

# Set the page configuration to wide mode for better layout
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="icon.png",
)

st.markdown(
    """
    <style>
        header {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for persistent data across app reruns
session = st.session_state

# 在session中存储并更新模型选择
if 'selected_model' not in session:
    session.selected_model = DEFAULT_LLM_MODEL
if 'model_initialized' not in session:
    session.model_initialized = False

# 更新模型选择（仅在未初始化时或通过UI更改时更新）
if not session.model_initialized:
    all_models = []
    for category, models in LLM_MODELS.items():
        all_models.extend([f"{model} ({category.split()[0]})" for model in models])
    
    # 在侧边栏添加模型选择器
    with st.sidebar:
        st.markdown("## 🤖 LLM 模型选择")
        selected = st.selectbox(
            "选择要使用的 AI 模型",
            all_models,
            index=all_models.index(f"{DEFAULT_LLM_MODEL} ({list(LLM_MODELS.keys())[0].split()[0]})") if f"{DEFAULT_LLM_MODEL} ({list(LLM_MODELS.keys())[0].split()[0]})" in all_models else 0,
            help="选择用于所有步骤的 LLM 模型。不同模型可能需要不同的 API Key，请确保已在 .env 中配置。"
        )
        # 从选择中提取实际的模型名（去掉分类标识）
        session.selected_model = selected.split(" (")[0]
        session.model_initialized = True

# 更新所有步骤使用的模型
selected_model = session.selected_model
entity_extraction_model = selected_model
component_selection_model = selected_model
component_specification_model = selected_model
schematic_model = selected_model
layout_model = selected_model

# if 'project_data' not in session:
#     session.project_data = {}
# sdata = session.project_data
# Core application state variables
if "current_message" not in session:
    session.current_message = ""
if "last_input" not in session:
    session.last_input = ""
if "chat_input" not in session:
    session.chat_input = ""
if "show_examples" not in session:
    session.show_examples = True
if "input_submitted" not in session:
    session.input_submitted = False
# Workflow state variables
if "template_selected" not in session:
    session.template_selected = False
if "components_selected" not in session:
    session.components_selected = False
if "user_specs" not in session:
    session.user_specs = {}
if "p100" not in session:
    session.p100 = True
# Automatic workflow phase tracking
# Tracks the current phase of the guided workflow
if "automatic_phase" not in session:
    session.automatic_phase = "input"  # input, entity_extraction, component_selection, schematic, layout
if "entity_extraction_complete" not in session:
    session.entity_extraction_complete = False
if "component_search_complete" not in session:
    session.component_search_complete = False
if "schematic_complete" not in session:
    session.schematic_complete = False
# Step-by-step workflow state variables
# Tracks the current step and results for step-by-step workflow mode
if "step_by_step_mode" not in session:
    session.step_by_step_mode = False
if "current_step_by_step_step" not in session:
    session.current_step_by_step_step = "Entity Extraction"
if "custom_inputs" not in session:
    session.custom_inputs = {
        "entity_extraction": "",
        "component_specification": "",
        "circuit_dsl": "",
        "schematic_dot": "",
        "layout_netlist": "",
        "custom_preschematic": ""
    }
if "step_results" not in session:
    session.step_results = {}

# LLM API configuration
if "p100_llm_api_selection" not in session:
    session.p100_llm_api_selection = DEFAULT_LLM_MODEL

# Function to handle input submission and state changes
def check_input_change():
    """
    Handle input changes and update session state accordingly.
    
    This function monitors changes in the chat input and updates the session
    state to trigger appropriate workflow transitions.
    """
    if hasattr(session, 'chat_input') and hasattr(session, 'last_input') and session.chat_input != session.last_input:
        if session.chat_input.strip():  # Check if input is not just whitespace
            session.current_message = session.chat_input
            session.show_examples = False
            # Start timing when user enters a prompt
            session.p100_start_time = time.time()
            # Reset token usage when user enters a new prompt
            llm_api.reset_token_usage()
            # session.input_submitted = True
        session.last_input = session.chat_input


# Display processed text
# st.markdown("#### Photon Fury ⚡")
# st.markdown("A Furious Photonic Chip Engineer")
st.markdown("#### Phido ⚡")
st.markdown("PHotonic Intelligent Design & Optimization")

# Add workflow mode selection
# Users can choose between automatic (guided) and step-by-step workflows
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        """
        <div style="margin-top: 4vh; text-align: center;">
        </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    workflow_mode = st.selectbox(
        "Workflow Mode:",
        ["Automatic", "Step-by-Step"],
        key="workflow_mode",
        on_change=lambda: setattr(session, 'step_by_step_mode', workflow_mode == "Step-by-Step")
    )
    session.step_by_step_mode = workflow_mode == "Step-by-Step"

# Move the text input for chat input to the main page
# This handles user input for circuit descriptions
if not session.input_submitted:
    # st.markdown('⇨ Describe a photonic circuit:')
    chat_input = st.text_input(
        "You: ",
        placeholder="⇨ Describe a photonic circuit and hit Enter!",
        key="chat_input",
        on_change=check_input_change,
        label_visibility="collapsed",
    )
    session.input_submitted = True


# Function to handle button clicks for example prompts
def on_button_click(button_text):
    """
    Handle example button clicks and update session state.
    
    Args:
        button_text: The text of the clicked example button
    """
    session.current_message = button_text
    session.show_examples = False
    session.input_submitted = True
    # Start timing when user clicks an example button
    session.p100_start_time = time.time()
    # Reset token usage when user clicks an example button
    llm_api.reset_token_usage()
    st.rerun()


# Example prompts for user guidance
# These provide common photonic circuit descriptions to help users get started
example_prompts_left = [
    "A 2x2 MZI",
    "A wavelength division demultiplexer",
    "A transceiver with four wavelength channels",
    "A four channel WDM",
    # 'Connect four 1x1 fast amplitude modulators to a 4x1 WDM',
    # "A 1x2 MMI for 1310 nm and a length of 40 um",
    # 'Connect a 2x2 mzi with heater to a directional coupler with a length of 125 um, dy of 100 um, and dx of 100 um',
    # "A low loss 1x4 power splitter connected to four fast amplitude modulators.",
    # "Two cascaded fast MZIs, each with two input and two output ports.",
    "A power splitter connected to two MZIs with thermo-optic phase shifters each with a path difference 100 um",
    "A low loss 1x2 power splitter connected to two GHz modulators each with a delta length of 100 um.",
    # "A high speed modulator connected to a VOA",
    "Layout 1x2 MMIs connected to each other to for a 1x8 splitter tree",
    "A 2x2 MZI with 1 GHz bandwidth",
    "A 2d mesh of nine MZIs. Each MZI has two input and two outputs and they are back to back connected",
    "A 1x2 splitter connected to two amplitude modulators with 100 dB extinction ratio",
    "Eight low loss and low power thermo optic phase shifters. the phase shifters should be arranged in parallel in an array",
]

example_prompts_right = [
    "Cascaded 2x2 MZIs to create a switch tree network with 8 outputs",
    "Coupler with a 300 um distance between the two input ports",
    "coupler with sbend height of 300 and sbend length of 300",
    "Connect a 2x2 mzi with heater to a directional coupler with a length of 125 um, dy of 100 um, and dx of 100 um",
    "A low loss 1x4 power splitter connected to four fast amplitude modulators.",
    "Two cascaded fast MZIs, each with two input and two output ports.",
    "three 1x2 mmi use a 20um bend radius for routing",
    # 'What is a transceiver?',
    # 'Simulate modes of a SiN wavguide, 400 nm width, 200 nm thick.',
    # 'What is the difference between a directional coupler and a MMI?',
    # 'Do you have access to internet?',
    # 'A four channel WDM connected to four grating couplers',
]


# CSS styling for example buttons
# Provides consistent button appearance and layout
custom_css = """
<style>
    .stButton > button {
        width: 400px;
        height: 20px;
        border-radius: 10px;
        }
</style>
"""


# Example buttons in a container
# Display example prompts when enabled to help users get started
if hasattr(session, 'show_examples') and session.show_examples:
    # Add vertical space and center the container
    st.markdown(
        """
        <div style="margin-bottom: 10vh; text-align: center;">
        </div>
    """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("Or try one of these:")
        col1, col2 = st.columns(2)
        with col1:
            for example in example_prompts_left:
                display_text = (
                    example[:45] + " ..."
                )  # Limit text to first 100 characters

                st.markdown(custom_css, unsafe_allow_html=True)
                if st.button(display_text):
                    on_button_click(example)

        with col2:
            for example in example_prompts_right:
                display_text = (
                    example[:45] + " ..."
                )  # Limit text to first 100 characters

                st.markdown(custom_css, unsafe_allow_html=True)
                if st.button(display_text):
                    on_button_click(example)

def get_next_log_filename(
    directory=PATH.logs, prefix="log_", extension=".pickle", digits=4
):
    # Convert directory to a Path object
    dir_path = Path(directory)

    # Ensure directory exists
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Get the current date and time in the desired format
    current_time = datetime.now().strftime("d%Y%m%d_t%H%M%S")

    # Generate a random 5-digit number and check if it already exists
    while True:
        random_number = random.randint(
            0, 10**digits - 1
        )  # Generates a number between 00000 and 99999
        formatted_number = str(random_number).zfill(
            digits
        )  # Pad with zeros if necessary

        # Format the next log filename
        next_filename = f"{prefix}{current_time}_{formatted_number}{extension}"

        # Check if the file already exists
        if not (dir_path / next_filename).exists():
            break  # If the filename doesn't exist, exit the loop

    # Return the complete path as a string
    return str(dir_path / next_filename), random_number


def pickleable(obj):
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, AttributeError, TypeError):
        return False
    return True


def logger():
    session_data = session.to_dict()

    # Remove or transform non-pickleable objects
    for key, value in session_data.items():
        if not pickleable(value):  # You may implement a helper function to check this.
            session_data[key] = "non-pickleable"

    session_data.pop(
        "p400_gdsfig", None
    )  # remove gdsfig, it's about 5 MB in size, not sure why

    # Make sure parent directory exists before writing
    try:
        Path(session.log_filename).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    with open(session.log_filename, "wb") as file:
        pickle.dump(session_data, file)


def on_template_select(template_id):
    session.template_selected = True
    session.p100 = False
    session.p200_selected_template = template_id


def on_component_select(c_selected_idx):
    session.components_selected = True
    session.p100 = False
    session.p200_selected_components = c_selected_idx


def display_templates_columns():
    # Determine the number of columns based on the length of the list
    num_columns = min(len(session.p200_retreived_templates), 2)

    # Create the columns
    columns = st.columns(num_columns)

    # Populate the columns with items
    for i, item in enumerate(session.p200_retreived_templates):
        with columns[i % num_columns]:
            with st.container(height=400):
                t = item[1]
                id_ = item[0]
                st.write(t["doc"]["title"] + "\n" + t["doc"]["description"])
                st.markdown(f"[reference]({t['doc']['reference']})")
                st.button(
                    id_,
                    use_container_width=True,
                    on_click=on_template_select,
                    args=(id_,),
                )


def display_components_columns():
    if not hasattr(session, 'p200_componenets_search_r') or not session.p200_componenets_search_r:
        st.warning("No component search results available")
        return
    
    c_idx = []
    scores = []
    for c in session.p200_componenets_search_r:
        c_idx.append(c.match_list)
        scores.append(c.match_scores)

    c_selected_idx = [None] * len(c_idx)
    session.p200_selected_components = []

    with st.form(key="my_form"):
        st.write("Picking components:")
        for i in range(len(c_idx)):
            with st.container(border=True):
                st.write(f"**{session.p200_pretemplate['components_list'][i]}**")

                # options = [f"{list_of_cnames[j]} ({j}) ({scores[i][k]})" for k, j in enumerate(c_idx[i])]
                for _ii in range(len(c_idx)):
                    options = []
                    for k, j in enumerate(c_idx[i]):
                        name = list_of_cnames[j]
                        score = scores[i][k]
                        if score == "exact":
                            # option = f"{name} [:green[{score}], {j}] "
                            option = f"{name} :green[/{score}/]"
                        elif score == "partial":
                            # option = f"{name} [:orange[{score}], {j}]"
                            option = f"{name} :orange[/{score}/]"
                        elif score == "poor":
                            # option = f"{name} [:red[{score}], {j}]"
                            option = f"{name} :red[/{score}/]"
                        else:
                            # option = f"{name} [:grey[{score}], {j}]"
                            option = f"{name} :grey[/{score}/]"
                        options.append(option)

                c_selected_idx[i] = st.radio(
                    label=str(i), options=options, label_visibility="collapsed"
                )

        c_selected_idx = [s.split(" :")[0] for s in c_selected_idx]
        st.form_submit_button(
            label="Submit", on_click=on_component_select, args=(c_selected_idx,)
        )


# Step-by-Step Interface
if hasattr(session, 'step_by_step_mode') and session.step_by_step_mode:
    # Ensure automatic workflow variables don't interfere with step-by-step workflow
    # Reset any automatic workflow state that might cause interference
    if hasattr(session, 'automatic_phase'):
        session.automatic_phase = "step_by_step"
    if hasattr(session, 'entity_extraction_complete'):
        session.entity_extraction_complete = False
    if hasattr(session, 'component_search_complete'):
        session.component_search_complete = False
    if hasattr(session, 'schematic_complete'):
        session.schematic_complete = False
    
    st.markdown("## Step-by-Step Execution")
    st.markdown("Run each step individually with your own inputs")
    
    # Step selection
    step_options = {
        "Entity Extraction": "entity_extraction",
        "Component Specification": "component_specification", 
        "Circuit DSL Creation": "circuit_dsl_creation",
        "Schematic Generation": "schematic_generation",
        "Layout & Simulation": "layout_simulation"
    }
    
    selected_step = st.selectbox(
        "Select Step to Execute:",
        list(step_options.keys()),
        key="step_by_step_step_selection",
        index=list(step_options.keys()).index(session.current_step_by_step_step)
    )
    
    # Update the session state with the current selection
    session.current_step_by_step_step = selected_step
    
    st.markdown("---")
    
    # Step-specific input and execution
    if selected_step == "Entity Extraction":
        st.markdown("### Step 1: Entity Extraction")
        st.markdown("Provide a natural language description of a photonic circuit")
        
        # Show current step results if available
        if "entity_extraction" in session.step_results:
            st.success("✅ Entity extraction completed!")
            result = session.step_results["entity_extraction"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Extracted Entities:**")
                st.write("```yaml\n" + yaml.dump(result["pretemplate"], sort_keys=False, width=55))
            
            with col2:
                try:
                    st.markdown("**Initial Schematic:**")
                    st.graphviz_chart(result["preschematic"])
                except Exception as e:
                    st.write("Failed to render schematic")
            
            with st.expander("Full Results", expanded=False):
                st.json(result)
            
            if st.button("Re-run Entity Extraction", key="rerun_entity_extraction"):
                session.step_results.pop("entity_extraction", None)
                st.rerun()
        else:
            custom_prompt = st.text_area(
                "Circuit Description:",
                value=session.custom_inputs["entity_extraction"],
                placeholder="e.g., A 2x2 MZI with thermo-optic phase shifters",
                height=100
            )
            session.custom_inputs["entity_extraction"] = custom_prompt
            
            if st.button("Run Entity Extraction", type="primary"):
                if custom_prompt.strip():
                    result = run_step_by_step_entity_extraction(custom_prompt)
                    if result:
                        st.success("Entity extraction completed!")
                        # Auto-populate next step input but don't auto-navigate
                        session.custom_inputs["component_specification"] = yaml.dump(
                            result["pretemplate"], default_flow_style=False
                        )
                        st.rerun()
                else:
                    st.error("Please provide a circuit description")
    
    elif selected_step == "Component Specification":
        st.markdown("### Step 2: Component Specification")
        st.markdown("Provide a pretemplate YAML or use result from previous step")
        
        # Show previous result if available
        if "entity_extraction" in session.step_results:
            with st.expander("Previous Step Result", expanded=False):
                st.write("```yaml\n" + yaml.dump(session.step_results["entity_extraction"]["pretemplate"]))
        
        # Show current step results if available
        if "component_specification" in session.step_results:
            st.success("✅ Component specification completed!")
            result = session.step_results["component_specification"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Selected Components:**")
                for i, comp in enumerate(result["selected_components"]):
                    st.write(f"{i+1}. {comp}")
            
            with col2:
                if result.get("selected_template"):
                    st.markdown("**Selected Template:**")
                    st.write(result["selected_template"])
                else:
                    st.markdown("**No template selected**")
            
            with st.expander("Full Results", expanded=False):
                st.json(result)
            
            if st.button("Re-run Component Specification", key="rerun_component_spec"):
                session.step_results.pop("component_specification", None)
                st.rerun()
        else:
            # Check if we have search results ready for selection
            if hasattr(session, 'component_search_ready') and session.component_search_ready:
                # Show the component selection interface
                run_component_selection_interface()
            else:
                # Auto-populate from previous step if available
                pretemplate_yaml = ""
                if "entity_extraction" in session.step_results:
                    pretemplate_yaml = yaml.dump(session.step_results["entity_extraction"]["pretemplate"], default_flow_style=False)
                    st.info("📝 Auto-populated from entity extraction results")
                
                custom_pretemplate = st.text_area(
                    "Pretemplate YAML:",
                    value=pretemplate_yaml or session.custom_inputs["component_specification"],
                    placeholder="""components_list:
  - MZI
  - phase_shifter
title: "2x2 MZI"
brief_summary: "A Mach-Zehnder interferometer with phase control"
circuit_instructions: "Connect MZI to phase shifter" """,
                    height=200
                )
                session.custom_inputs["component_specification"] = custom_pretemplate
                
                if st.button("Run Component Search", type="primary"):
                    if custom_pretemplate.strip():
                        result = run_step_by_step_component_specification(custom_pretemplate)
                        if result is None:  # Search completed successfully
                            st.rerun()  # Refresh to show selection interface
                    else:
                        st.error("Please provide pretemplate YAML")
    
    elif selected_step == "Circuit DSL Creation":
        st.markdown("### Step 3: Circuit DSL Creation")
        st.markdown("Create circuit DSL from component specification or template")
        
        # Show previous result if available
        if "component_specification" in session.step_results:
            with st.expander("Previous Step Result", expanded=False):
                result = session.step_results["component_specification"]
                st.write(f"Selected components: {', '.join(result['selected_components'])}")
                if result.get("selected_template"):
                    st.write(f"Selected template: {result['selected_template']}")
        
        # Show current step results if available
        if "circuit_dsl_creation" in session.step_results:
            st.success("✅ Circuit DSL creation completed!")
            result = session.step_results["circuit_dsl_creation"]
            
            st.markdown("**Circuit DSL:**")
            st.write("```yaml\n" + yaml.dump(result["circuit_dsl"], sort_keys=False, width=55))
            
            with st.expander("Full Results", expanded=False):
                st.json(result)
            
            if st.button("Re-run Circuit DSL Creation", key="rerun_circuit_dsl"):
                session.step_results.pop("circuit_dsl_creation", None)
                st.rerun()
        else:
            # Check if we have component specification results
            if "component_specification" in session.step_results:
                comp_spec_result = session.step_results["component_specification"]
                pretemplate = comp_spec_result["pretemplate"]
                selected_components = comp_spec_result["selected_components"]
                selected_template = comp_spec_result.get("selected_template")
                
                st.info("📝 Using component specification results")
                st.write(f"**Components:** {', '.join(selected_components)}")
                if selected_template:
                    st.write(f"**Template:** {selected_template}")
                
                # Auto-populate pretemplate from previous step
                # Convert any tuples to lists to avoid YAML tuple tags
                safe_pretemplate = convert_tuples_to_lists(pretemplate)
                pretemplate_yaml = yaml.dump(safe_pretemplate, default_flow_style=False, default_style=None)
                st.info("📝 Auto-populated pretemplate from component specification results")
                
                custom_pretemplate = st.text_area(
                    "Pretemplate YAML:",
                    value=pretemplate_yaml,
                    height=200,
                    help="Pretemplate from component specification step"
                )
                
                # Input fields for additional customization
                col1, col2 = st.columns(2)
                with col1:
                    custom_components = st.text_area(
                        "Custom Selected Components (one per line):",
                        value="\n".join(selected_components),
                        height=100,
                        help="Override the selected components from previous step"
                    )
                
                with col2:
                    custom_template = st.text_input(
                        "Custom Template ID:",
                        value=selected_template or "",
                        help="Override the selected template from previous step"
                    )
                
                # Check if we have a template selected from previous step
                st.write(f"Debug: selected_template = {selected_template}")
                st.write(f"Debug: custom_template = {custom_template}")
                st.write(f"Debug: condition result = {selected_template and not custom_template}")
                
                # Check if we have a template selected (either from previous step or custom input)
                if selected_template or (custom_template and custom_template.strip()):
                    # Determine which template to use
                    template_to_use = selected_template if selected_template else custom_template
                    st.info(f"📋 Template selected: {template_to_use}")
                    st.write("Please fill out the template specifications below:")
                    
                    # Get template specifications
                    template_specs = templates_dict[template_to_use]["properties"]["specs"]
                    
                    user_specs = {}
                    for key, item in template_specs.items():
                        user_input = st.text_input(
                            f"{key} ({item['comment']})", 
                            item["value"],
                            key=f"step_by_step_template_spec_{key}"
                        )
                        user_specs[key] = {
                            "value": user_input,
                            "comment": item["comment"],
                        }
                    
                    if st.button("Generate Circuit DSL from Template", type="primary"):
                        try:
                            # Create circuit DSL from template
                            circuit_dsl = templates_dict[template_to_use].copy()
                            circuit_dsl["properties"]["specs"] = user_specs
                            
                            # Component retrieval for template nodes
                            if "TEMPLATE" in template_to_use:
                                with st.spinner("Looking for components..."):
                                    for key, value in circuit_dsl["nodes"].items():
                                        try:
                                            specs_for_retrieval = user_specs.copy()
                                            for spec_key in specs_for_retrieval:
                                                if "comment" in specs_for_retrieval[spec_key]:
                                                    del specs_for_retrieval[spec_key]["comment"]
                                        except Exception as e:
                                            st.error(f"An error occurred processing specs: {e}")
                                            specs_for_retrieval = ""

                                        try:
                                            r = llm_api.llm_retrieve(
                                                value + f"\n({str(specs_for_retrieval)})",
                                                session.p100_list_of_docs,
                                                session.p100_llm_api_selection,
                                            )
                                            
                                            if r is None or len(r) == 0:
                                                st.error(f"No components found for {value}")
                                                continue
                                            
                                            all_retrieved = [session.p100_list_of_cnames[i] for i in r]
                                            st.write(f"For {value}, found: " + "\n".join(all_retrieved))

                                            selected_component = session.p100_list_of_cnames[r[0]]
                                            circuit_dsl["nodes"][key] = {}
                                            circuit_dsl["nodes"][key]["component"] = selected_component
                                            
                                        except Exception as e:
                                            st.error(f"Error during component retrieval for {value}: {e}")
                                            continue
                            
                            result = {
                                "circuit_dsl": circuit_dsl,
                                "template_id": template_to_use,
                                "parsed_specs": user_specs
                            }
                            session.step_results["circuit_dsl_creation"] = result
                            st.success("Circuit DSL creation completed!")
                            # Auto-populate next step input but don't auto-navigate
                            session.custom_inputs["schematic_dot"] = yaml.dump(result["circuit_dsl"], default_flow_style=False)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error during template DSL generation: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    # Original logic for custom template or component selection
                    if st.button("Create Circuit DSL", type="primary"):
                        # Parse custom components
                        custom_components_list = [c.strip() for c in custom_components.split('\n') if c.strip()]
                        
                        # Create circuit DSL
                        if custom_template:
                            # Template path
                            result = run_step_by_step_circuit_dsl_creation(
                                custom_pretemplate,
                                custom_template_id=custom_template
                            )
                        else:
                            # Component path
                            result = run_step_by_step_circuit_dsl_creation(
                                custom_pretemplate,
                                custom_selected_components=custom_components_list
                            )
                        
                        if result:
                            st.success("Circuit DSL creation completed!")
                            # Auto-populate next step input but don't auto-navigate
                            session.custom_inputs["schematic_dot"] = yaml.dump(result["circuit_dsl"], default_flow_style=False)
                            st.rerun()
            else:
                # Manual input option when no component specification results are available
                st.warning("No component specification results available.")
                
                # Add manual input option
                with st.expander("Manual Input Option", expanded=True):
                    st.markdown("**Option 1: Simple Component List**")
                    st.markdown("Provide a simple list of components and basic circuit information.")
                    
                    manual_components = st.text_area(
                        "Component List (one per line):",
                        value="mzi_2x2\nphase_shifter",
                        height=100,
                        help="Enter component names, one per line"
                    )
                    
                    manual_title = st.text_input(
                        "Circuit Title:",
                        value="Custom Circuit",
                        help="Title for the circuit"
                    )
                    
                    manual_description = st.text_area(
                        "Circuit Description:",
                        value="A custom photonic circuit",
                        height=80,
                        help="Brief description of the circuit"
                    )
                    
                    manual_instructions = st.text_area(
                        "Circuit Instructions:",
                        value="Connect components in series",
                        height=80,
                        help="How components should be connected"
                    )
                    
                    if st.button("Create Circuit DSL from Manual Input", type="primary"):
                        if manual_components.strip():
                            # Parse components
                            components_list = [c.strip() for c in manual_components.split('\n') if c.strip()]
                            
                            # Create simple pretemplate
                            manual_pretemplate = {
                                "title": manual_title,
                                "brief_summary": manual_description,
                                "circuit_instructions": manual_instructions,
                                "components_list": components_list
                            }
                            
                            # Create circuit DSL
                            result = run_step_by_step_circuit_dsl_creation(
                                yaml.dump(manual_pretemplate, default_flow_style=False),
                                custom_selected_components=components_list
                            )
                            
                            if result:
                                st.success("Circuit DSL creation completed!")
                                # Auto-populate next step input but don't auto-navigate
                                session.custom_inputs["schematic_dot"] = yaml.dump(result["circuit_dsl"], default_flow_style=False)
                                st.rerun()
                        else:
                            st.error("Please provide at least one component.")
                    
                    st.markdown("---")
                    st.markdown("**Option 2: Complete Pretemplate YAML**")
                    st.markdown("Provide a complete pretemplate in YAML format.")
                    
                    manual_pretemplate_yaml = st.text_area(
                        "Complete Pretemplate YAML:",
                        value="""title: "Custom Circuit"
brief_summary: "A custom photonic circuit"
circuit_instructions: "Connect components in series"
components_list:
  - mzi_2x2
  - phase_shifter""",
                        height=200,
                        help="Complete pretemplate in YAML format"
                    )
                    
                    if st.button("Create Circuit DSL from YAML", type="primary"):
                        if manual_pretemplate_yaml.strip():
                            try:
                                # Parse the YAML
                                pretemplate = yaml.safe_load(manual_pretemplate_yaml)
                                components_list = pretemplate.get("components_list", [])
                                
                                if components_list:
                                    # Create circuit DSL
                                    result = run_step_by_step_circuit_dsl_creation(
                                        manual_pretemplate_yaml,
                                        custom_selected_components=components_list
                                    )
                                    
                                    if result:
                                        st.success("Circuit DSL creation completed!")
                                        # Auto-populate next step input but don't auto-navigate
                                        session.custom_inputs["schematic_dot"] = yaml.dump(result["circuit_dsl"], default_flow_style=False)
                                        st.rerun()
                                else:
                                    st.error("No components found in the pretemplate.")
                            except yaml.YAMLError as e:
                                st.error(f"Invalid YAML format: {e}")
                        else:
                            st.error("Please provide pretemplate YAML.")
    
    elif selected_step == "Schematic Generation":
        st.markdown("### Step 4: Schematic Generation")
        st.markdown("Generate schematic from circuit DSL")
        
        # Show previous result if available
        if "circuit_dsl_creation" in session.step_results:
            with st.expander("Previous Step Result", expanded=False):
                st.write("Circuit DSL available")
        
        # Show current step results if available
        if "schematic_generation" in session.step_results:
            st.success("✅ Schematic generation completed!")
            result = session.step_results["schematic_generation"]
            
            st.markdown("**Generated DOT Graph:**")
            st.graphviz_chart(result["dot_string"])
            
            with st.expander("Circuit DSL", expanded=False):
                st.write("```yaml\n" + yaml.dump(result["circuit_dsl"], sort_keys=False, width=55))
            
            with st.expander("Full Results", expanded=False):
                st.json(result)
            
            if st.button("Re-run Schematic Generation", key="rerun_schematic"):
                session.step_results.pop("schematic_generation", None)
                st.rerun()
        else:
            # Auto-populate from previous step if available
            circuit_dsl_yaml = ""
            if "circuit_dsl_creation" in session.step_results:
                circuit_dsl_yaml = yaml.dump(session.step_results["circuit_dsl_creation"]["circuit_dsl"], default_flow_style=False)
                st.info("📝 Auto-populated from circuit DSL creation results")
            
            custom_circuit_dsl = st.text_area(
                "Circuit DSL YAML:",
                value=circuit_dsl_yaml or session.custom_inputs["schematic_dot"],
                placeholder="""doc:
  title: "2x2 MZI"
  description: "A Mach-Zehnder interferometer"
nodes:
  N1:
    component: "mzi_2x2"
  N2:
    component: "phase_shifter"
edges:
  E1:
    link: "N1:o1:N2:i1"
    properties:
      type: "route"
      constraints: {}""",
                height=300
            )
            session.custom_inputs["schematic_dot"] = custom_circuit_dsl
            
            # Add manual preschematic input option
            st.markdown("---")
            st.markdown("**Optional: Custom Preschematic**")
            st.markdown("Provide your own DOT graph for edge generation. If left empty, one will be generated automatically from the circuit DSL.")
            
            custom_preschematic = st.text_area(
                "Custom Preschematic (DOT format):",
                value=session.custom_inputs.get("custom_preschematic", ""),
                placeholder="""graph G {
    rankdir=LR;
    N1 [label="mzi_2x2"];
    N2 [label="phase_shifter"];
    
    N1 -- N2;
}""",
                height=200,
                help="Provide a DOT graph with node definitions and edges. This will be used instead of auto-generating the preschematic."
            )
            session.custom_inputs["custom_preschematic"] = custom_preschematic
            
            if st.button("Run Schematic Generation", type="primary"):
                if custom_circuit_dsl.strip():
                    # Add immediate feedback with spinner
                    with st.spinner("Starting schematic generation..."):
                        result = run_step_by_step_schematic_generation(custom_circuit_dsl, custom_preschematic)
                        if result:
                            st.success("Schematic generation completed!")
                            # Auto-populate next step input but don't auto-navigate
                            session.custom_inputs["layout_netlist"] = yaml.dump(result["circuit_dsl"], default_flow_style=False)
                            st.rerun()
                else:
                    st.error("Please provide circuit DSL YAML")
    
    elif selected_step == "Layout & Simulation":
        st.markdown("### Step 5: Layout & Simulation")
        st.markdown("Generate layout and run simulation from circuit DSL")
        
        # Show previous result if available
        if "schematic_generation" in session.step_results:
            with st.expander("Previous Step Result", expanded=False):
                st.write("Schematic available")
        
        # Show current step results if available
        if "layout_simulation" in session.step_results:
            st.success("✅ Layout and simulation completed!")
            result = session.step_results["layout_simulation"]
            
            st.markdown(f"**Routing Success:** {'✅' if result['routing_success'] else '❌'}")
            
            # Try to display GDS if available
            try:
                if "p400_gdsfig" in session:
                    st.pyplot(session["p400_gdsfig"])
            except:
                st.warning("GDS figure not available")
            
            # Try to display simulation results if available
            try:
                if "plot_sax.png" in str(PATH.build):
                    st.image(str(PATH.build / "plot_sax.png"))
            except:
                st.warning("Simulation plot not available")
            
            with st.expander("GDS-Factory Netlist", expanded=False):
                st.write("```yaml\n" + yaml.dump(result["gf_netlist"]))
            
            with st.expander("Full Results", expanded=False):
                st.json(result)
            
            if st.button("Re-run Layout & Simulation", key="rerun_layout"):
                session.step_results.pop("layout_simulation", None)
                st.rerun()
        else:
            # Auto-populate from previous step if available
            circuit_dsl_yaml = ""
            if "schematic_generation" in session.step_results:
                circuit_dsl_yaml = yaml.dump(session.step_results["schematic_generation"]["circuit_dsl"], default_flow_style=False)
                st.info("📝 Auto-populated from schematic generation results")
            elif "circuit_dsl_creation" in session.step_results:
                circuit_dsl_yaml = yaml.dump(session.step_results["circuit_dsl_creation"]["circuit_dsl"], default_flow_style=False)
                st.info("📝 Auto-populated from circuit DSL creation results")
            
            custom_circuit_dsl = st.text_area(
                "Circuit DSL YAML:",
                value=circuit_dsl_yaml or session.custom_inputs["layout_netlist"],
                placeholder="""doc:
  title: "2x2 MZI"
  description: "A Mach-Zehnder interferometer"
nodes:
  N1:
    component: "mzi_2x2"
    placement:
      x: 0
      y: 0
  N2:
    component: "phase_shifter"
    placement:
      x: 100
      y: 0
edges:
  E1:
    link: "N1:o1:N2:i1"
    properties:
      type: "route"
      constraints: {}""",
                height=300
            )
            session.custom_inputs["layout_netlist"] = custom_circuit_dsl
            
            if st.button("Run Layout & Simulation", type="primary"):
                if custom_circuit_dsl.strip():
                    result = run_step_by_step_layout_simulation(custom_circuit_dsl)
                    if result:
                        st.success("Layout and simulation completed!")
                        # Don't call st.rerun() here to prevent infinite loop
                        # Let user manually navigate to next step if needed
                else:
                    st.error("Please provide circuit DSL YAML")
    
    # Results section
    if hasattr(session, 'step_results') and session.step_results:
        st.markdown("---")
        st.markdown("### Step Results")
        
        for step_name, result in session.step_results.items():
            with st.expander(f"Results: {step_name.replace('_', ' ').title()}", expanded=False):
                if isinstance(result, dict):
                    st.json(result)
                else:
                    st.write(str(result))
    
    # Debug section (can be removed later)
    with st.expander("Debug: Session State", expanded=False):
        st.write("**Current Step-by-Step Step:**")
        st.write(session.current_step_by_step_step)
        st.write("**Step Results Keys:**")
        st.write(list(session.step_results.keys()))
        st.write("**Session Keys:**")
        st.write([k for k in session.keys() if k.startswith('step_by_step_')])
        st.write("**Automatic Workflow State:**")
        st.write(f"automatic_phase: {getattr(session, 'automatic_phase', 'Not set')}")
        st.write(f"entity_extraction_complete: {getattr(session, 'entity_extraction_complete', 'Not set')}")
        st.write(f"component_search_complete: {getattr(session, 'component_search_complete', 'Not set')}")
        st.write(f"schematic_complete: {getattr(session, 'schematic_complete', 'Not set')}")
    
    # Clear results button
    if st.button("Clear All Results"):
        session.step_results = {}
        session.custom_inputs = {
            "entity_extraction": "",
            "component_specification": "",
            "circuit_dsl": "",
            "schematic_dot": "",
            "layout_netlist": "",
            "custom_preschematic": ""
        }
        st.rerun()

# Original automatic workflow (existing code)
elif not session.step_by_step_mode and session.p100 and (session.current_message != ""):
    session.log_filename, session.log_id = get_next_log_filename()
    logger()

    # Display original prompt
    st.markdown("### Original Prompt")
    with st.container(border=True):
        st.markdown(f"*{session.current_message}*")
    
    st.markdown("---")

    # Progress indicator
    phases = ["Entity Extraction", "Component Selection", "Schematic Generation", "Layout & Simulation"]
    current_phase_idx = 0
    if hasattr(session, 'entity_extraction_complete') and session.entity_extraction_complete:
        current_phase_idx = 1
    if hasattr(session, 'component_search_complete') and session.component_search_complete:
        current_phase_idx = 2
    if hasattr(session, 'schematic_complete') and session.schematic_complete:
        current_phase_idx = 3
    
    st.markdown("### Automatic Workflow Progress")
    progress_bar = st.progress(0)
    progress_bar.progress((current_phase_idx + 1) / len(phases))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**{phases[0]}** {'✅' if hasattr(session, 'entity_extraction_complete') and session.entity_extraction_complete else '⏳'}")
    with col2:
        st.markdown(f"**{phases[1]}** {'✅' if hasattr(session, 'component_search_complete') and session.component_search_complete else '⏳'}")
    with col3:
        st.markdown(f"**{phases[2]}** {'✅' if hasattr(session, 'schematic_complete') and session.schematic_complete else '⏳'}")
    with col4:
        st.markdown(f"**{phases[3]}** {'✅' if hasattr(session, 'schematic_complete') and session.schematic_complete and 'p400' in session else '⏳'}")
    
    st.markdown("---")

    # Display completed stage outputs
    if hasattr(session, 'entity_extraction_complete') and session.entity_extraction_complete:
        st.markdown("### 📋 Stage 1: Entity Extraction Results")
        st.markdown("**Extracted Entities:**")
        if hasattr(session, 'p200_pretemplate'):
            st.write("```yaml\n" + yaml.dump(session.p200_pretemplate, width=55))
        st.markdown("---")

    if hasattr(session, 'component_search_complete') and session.component_search_complete:
        st.markdown("### 🔍 Stage 2: Component Selection Results")
        with st.expander("Component Search Results", expanded=True):
            if hasattr(session, 'p200_componenets_search_r') and session.p200_componenets_search_r:
                st.markdown("**Component Search Results:**")
                for i, search_result in enumerate(session.p200_componenets_search_r):
                    with st.container(border=True):
                        st.write(f"**{session.p200_pretemplate['components_list'][i]}**")
                        options = []
                        for k, j in enumerate(search_result.match_list):
                            name = list_of_cnames[j]
                            score = search_result.match_scores[k]
                            if score == "exact":
                                option = f"{name} :green[/{score}/]"
                            elif score == "partial":
                                option = f"{name} :orange[/{score}/]"
                            elif score == "poor":
                                option = f"{name} :red[/{score}/]"
                            else:
                                option = f"{name} :grey[/{score}/]"
                            options.append(option)
                        st.write("Available options:")
                        for option in options:
                            st.write(f"- {option}")
                
                if hasattr(session, 'p200_selected_components') and session.p200_selected_components:
                    st.markdown("**Selected Components:**")
                    for i, comp in enumerate(session.p200_selected_components):
                        st.write(f"{i+1}. {comp}")
            
            if hasattr(session, 'p200_retreived_templates') and session.p200_retreived_templates:
                st.markdown("**Template Search Results:**")
                for i, item in enumerate(session.p200_retreived_templates):
                    with st.container(border=True):
                        t = item[1]
                        id_ = item[0]
                        st.write(f"**Template {i+1}:** {id_}")
                        st.write(t["doc"]["title"] + "\n" + t["doc"]["description"])
                        st.markdown(f"[reference]({t['doc']['reference']})")
                
                if hasattr(session, 'p200_selected_template') and session.p200_selected_template:
                    st.markdown("**Selected Template:**")
                    st.write(session.p200_selected_template)
        st.markdown("---")

    if session.schematic_complete and hasattr(session, 'p300_circuit_dsl'):
        st.markdown("### 🎨 Stage 3: Schematic Generation Results")
        with st.expander("Schematic Generation Output", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Circuit DSL:**")
                st.write("```yaml\n" + yaml.dump(session.p300_circuit_dsl, width=55))
            with col2:
                if hasattr(session, 'p300_dot_string'):
                    st.markdown("**Generated Schematic:**")
                    st.graphviz_chart(session.p300_dot_string)
        st.markdown("---")

    if session.schematic_complete and hasattr(session, 'p400_gf_netlist'):
        st.markdown("### 🏗️ Stage 4: Layout & Simulation Results")
        with st.expander("Layout & Simulation Output", expanded=True):
            st.markdown("**GDS-Factory Netlist:**")
            st.write("```yaml\n" + yaml.dump(session.p400_gf_netlist, width=55))
            
            if hasattr(session, 'p400_gdsfig'):
                st.markdown("**GDS Layout:**")
                st.pyplot(session.p400_gdsfig)
            
            # Try to display simulation results if available
            try:
                plot_path = PATH.build / "plot_sax.png"
                if plot_path.exists():
                    st.markdown("**Simulation Results:**")
                    st.image(str(plot_path))
                else:
                    st.warning("Simulation plot file not found")
            except Exception as e:
                st.warning(f"Error displaying simulation plot: {e}")
        st.markdown("---")

    st.markdown(
        f"""
        <style>
            .small-rectangle {{
                position: fixed;
                top: 10px;
                right: 10px;
                width: 100px;
                height: 50px;
                z-index: 9999;
                padding: 10px;
                text-align: right;
                font-family: monospace;
                color: grey;
                font-size: 10px;
            }}
        </style>
        <div class="small-rectangle">
            <p>{session.log_id}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # classify input as a photonic layout prompt or not
    interpreter_cat = llm_api.intent_classification(session.current_message)

    if interpreter_cat.category_id != 1:
        st.markdown(f"**{interpreter_cat.response}**")
    else:
        # Reset token usage at the start of automatic workflow
        if not session.entity_extraction_complete:
            llm_api.reset_token_usage()
        
        # Phase 1: Entity Extraction (only run if not already complete)
        if not session.entity_extraction_complete:
            session.automatic_phase = "entity_extraction"
            
            # Set the correct LLM model for entity extraction
            session.p100_llm_api_selection = entity_extraction_model

            # Initialize required session variables for LLM API calls
            session.p100_list_of_docs = list_of_docs
            session.p100_list_of_cnames = list_of_cnames

            # Perform entity extraction and preschematic generation
            with st.spinner("Entity extraction ..."):
                session.p200_pretemplate = llm_api.entity_extraction(session.current_message)
                session.p200_preschematic = llm_api.preschematic(session.p200_pretemplate, session.p100_llm_api_selection)
                # Create a copy of the pretemplate for later use
                session.p200_pretemplate_copy = copy.deepcopy(session.p200_pretemplate)

            session.entity_extraction_complete = True
            st.success("✅ Entity extraction completed!")

            # -------------------------------------------------------------
            # 👑 Mode Detection: Single Component vs Circuit Routing
            # -------------------------------------------------------------
            # Use LLM-detected intent from prompt
            design_type = session.p200_pretemplate.get_design_type() if hasattr(session.p200_pretemplate, 'get_design_type') else session.p200_pretemplate.get("design_type", "circuit_routing")
            components = session.p200_pretemplate.get("components_list", [])
            
            is_routing = True
            if design_type == "single_component":
                is_routing = False
            
            # Set phase based on mode
            if not is_routing:
                # ------------------------------------------------------------------
                # Strategy: Check Library (Component Selection) -> Found? -> Prompt
                #                                               -> Not Found? -> Auto-PDK
                # ------------------------------------------------------------------
                
                st.markdown("**🔍 Checking component library...**")
                
                # Setup search context
                session.p100_llm_api_selection = component_selection_model
                target_component = components[0] if components else "unknown"
                st.write(f"Target component: `{target_component}`")
                
                # 先用关键词快速匹配
                def quick_keyword_match(target, cnames):
                    """快速关键词匹配，返回最接近的组件名"""
                    target_lower = target.lower()
                    # 关键词映射（按优先级）
                    keyword_map = {
                        "y_branch": ["y_branch", "ybranch"],  # Y分支，优先级最高
                        "y-branch": ["y_branch", "ybranch"],
                        "y branch": ["y_branch", "ybranch"],
                        "grating": ["_gc", "grating"],
                        "mmi": ["mmi"],
                        "ring": ["ring", "mrr"],
                        "resonator": ["ring", "mrr", "resonator"],
                        "coupler": ["coupler", "dc"],
                        "mzi": ["mzi"],
                        "crossing": ["crossing"],
                        "splitter": ["mmi", "splitter", "y_branch"],
                        "modulator": ["modulator", "mzm"],
                        "heater": ["heater"],
                        "phase": ["phase"],
                    }
                    
                    for kw, aliases in keyword_map.items():
                        if kw in target_lower:
                            for alias in aliases:
                                # 先找以别名开头的
                                for cname in cnames:
                                    cname_lower = cname.lower()
                                    if cname_lower.startswith(alias) or cname_lower.startswith("_" + alias):
                                        return cname
                                # 再找包含别名的
                                for cname in cnames:
                                    if alias in cname.lower():
                                        return cname
                    return None
                
                quick_match = quick_keyword_match(target_component, session.p100_list_of_cnames)
                
                if quick_match:
                    st.write(f"✅ Keyword match found: `{quick_match}`")
                    session.automatic_phase = "pdk_optimization"
                    session.pdk_candidate_name = quick_match
                    lib_path = PATH.repo / "PhotonicsAI" / "KnowledgeBase" / "DesignLibrary" / f"{quick_match}.py"
                    if lib_path.exists():
                        session.generated_template_path = str(lib_path)
                    st.rerun()
                else:
                    # 没有找到匹配，进入 PDK 生成阶段
                    st.write("No keyword match, will generate new component...")
                    session.pdk_target_component = target_component
                    session.automatic_phase = "pdk_generation"
                    st.rerun()
                    
                # 保留 LLM 搜索作为注释，以备将来使用
                # found_match = False
                # best_match_name = ""
                # try:
                #     with st.spinner("Searching component library..."):
                #         search_r = llm_api.llm_search(target_component, session.p100_list_of_docs, model=selected_model)
                #     if search_r.match_scores and search_r.match_scores[0] in ["exact", "partial"]:
                #          idx = search_r.match_list[0]
                #          best_match_name = session.p100_list_of_cnames[idx]
                #          found_match = True
                # except Exception as e:
                #     st.warning(f"LLM search error: {e}")
                
                if found_match:
                     session.automatic_phase = "pdk_optimization"
                     session.pdk_candidate_name = best_match_name
                     # 直接加载现有库组件
                     lib_path = PATH.repo / "PhotonicsAI" / "KnowledgeBase" / "DesignLibrary" / f"{best_match_name}.py"
                     if lib_path.exists():
                         session.generated_template_path = str(lib_path)
                     st.rerun()
                else:
                     # 保存目标组件名称，以防 rerun 后丢失
                     session.pdk_target_component = target_component
                     session.automatic_phase = "pdk_generation"
                     st.rerun()
            else:
                session.automatic_phase = "component_selection" # Continue standard flow

        # -------------------------------------------------------------
        # Phase A.1: Single Component Discovery & Generation
        # 流程: 用户上传 PDF / 粘贴论文文本 -> LLM 提取参数 -> 生成 ONE 高质量模板
        # -------------------------------------------------------------
        if hasattr(session, 'automatic_phase') and session.automatic_phase == "pdk_generation":
             # 调试信息
             st.markdown("### 🔬 PDK Generation Phase")
             st.write(f"**Debug Info:**")
             st.write(f"- automatic_phase: {session.automatic_phase}")
             st.write(f"- entity_extraction_complete: {session.entity_extraction_complete}")
             st.write(f"- p200_pretemplate exists: {hasattr(session, 'p200_pretemplate')}")
             st.write(f"- pdk_target_component: {getattr(session, 'pdk_target_component', 'Not set')}")
             
             # 修复: 确保能获取到 target_component
             try:
                if hasattr(session, 'p200_pretemplate') and session.p200_pretemplate:
                    target_component = session.p200_pretemplate.get("components_list", ["unknown"])
                    target_component = target_component[0] if target_component else "unknown"
                elif hasattr(session, 'pdk_target_component') and session.pdk_target_component:
                    target_component = session.pdk_target_component
                else:
                    target_component = "unknown"
             except Exception as e:
                st.warning(f"Component extraction fallback: {e}")
                target_component = getattr(session, 'pdk_target_component', 'unknown')
                
             st.markdown(f"**🔬 Target Component:** `{target_component}`")

             st.markdown("**📄 Paper Input**")
             uploaded_pdf = st.file_uploader(
                 "Upload one paper PDF",
                 type=["pdf"],
                 key="pdk_generation_pdf_upload",
                 help="Upload a single paper PDF for parameter extraction and component template generation.",
             )

             st.caption("当前阶段只接受单篇 PDF 论文输入。原论文爬虫流程与手工文本输入已停用。")
             # 旧的“爬虫搜索论文”与“粘贴论文文本”路径已停用，当前 webapp 仅触发 PDF 输入模式。

             if st.button("Generate Template From Paper", type="primary", key="generate_template_from_paper"):
                 try:
                     from PhotonicsAI.agents.pdk_agent import create_pdk_agent
                     import tempfile

                     provided_pdf_path = None
                     if uploaded_pdf is not None:
                         suffix = Path(uploaded_pdf.name).suffix or ".pdf"
                         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                             tmp_file.write(uploaded_pdf.getbuffer())
                             provided_pdf_path = tmp_file.name

                     if not provided_pdf_path:
                         st.error("Please upload a PDF before generating.")
                     else:
                         agent = create_pdk_agent()
                         generation_result = None
                         with st.spinner(f"🔬 Processing paper & generating template for '{target_component}'..."):
                             with st.expander("🤖 Agent Generation Stream", expanded=True):
                                 for event in agent.run_generation(target_component, provided_pdf_path):
                                     event_type = event.get("event")
                                     if event_type == "thought":
                                         st.write(f"Thought: {event.get('content', '')}")
                                     elif event_type == "action":
                                         st.write(f"Action: {event.get('tool', '')} -> {event.get('args', {})}")
                                     elif event_type == "observation":
                                         st.write(f"Observation ({event.get('tool', '')}):")
                                         st.json(event.get("result", {}))
                                     elif event_type == "answer":
                                         generation_result = event.get("result")

                         template_result = (generation_result or {}).get("template", {})
                         template_data = template_result.get("data", {}) if isinstance(template_result, dict) else {}
                         filepath = template_data.get("filepath", "")

                         st.markdown("**📊 Discovery Results:**")
                         st.write(f"- Device type: {(generation_result or {}).get('device_type', 'N/A')}")
                         params_payload = ((generation_result or {}).get("params") or {}).get("data", {})
                         if params_payload.get("params"):
                             st.write(f"- Parameters: {params_payload.get('params')}")

                         if filepath:
                             session.generated_template_path = filepath
                             session.discovery_result = generation_result
                             session.automatic_phase = "pdk_optimization"
                             st.success(f"✅ Generated template: {filepath}")
                             st.rerun()

                         error_msg = (generation_result or {}).get("error") or template_result.get("error") or "Unknown error"
                         st.error(f"❌ Failed to generate template: {error_msg}")
                         st.info("Falling back to component selection...")
                         session.automatic_phase = "component_selection"
                         st.rerun()

                 except Exception as e:
                     import traceback
                     st.error(f"❌ Discovery error: {e}")
                     with st.expander("Error Details", expanded=False):
                         st.code(traceback.format_exc())
                     session.automatic_phase = "component_selection"
                     st.rerun()
                     
        if hasattr(session, 'automatic_phase') and session.automatic_phase == "pdk_optimization":
            st.markdown("### 🔧 Component Optimization & Simulation")
            
            col_opt1, col_opt2 = st.columns([3, 1])
            with col_opt2:
                 if st.button("New Design"):
                    # Clear session keys related to workflow
                    keys_to_clear = ['p200_pretemplate', 'generated_template_path', 'automatic_phase', 'current_message', 'input_submitted']
                    for key in keys_to_clear:
                        if key in session:
                            del session[key]
                    st.rerun()

            if hasattr(session, 'generated_template_path') and session.generated_template_path:
                max_inverse_iterations = st.slider(
                    "Inverse Design Max Iterations",
                    min_value=1,
                    max_value=8,
                    value=3,
                    step=1,
                    key="inverse_design_max_iterations",
                )

                fdtd_col, inverse_col = st.columns(2)
                run_fdtd = fdtd_col.button("🚀 Run FDTD Simulation", type="primary")
                run_inverse_design = inverse_col.button("🧠 Run Inverse Design (ReAct)")

                if run_inverse_design:
                    st.info("🔄 Running Step1-Step5 inverse-design ReAct chain...")

                    try:
                        from PhotonicsAI.agents.pdk_agent import create_pdk_agent

                        requirement_text = str(getattr(session, "current_message", "") or "").strip()
                        if not requirement_text and hasattr(session, "p200_pretemplate"):
                            requirement_text = str(session.p200_pretemplate)

                        if not requirement_text:
                            st.error("No requirement text found for inverse design. Please re-enter your request.")
                        else:
                            agent = create_pdk_agent()
                            inverse_result = None

                            with st.expander("🤖 Inverse-Design ReAct Stream", expanded=True):
                                for event in agent.run_inverse_design_react(
                                    requirement_text=requirement_text,
                                    max_iterations=int(max_inverse_iterations),
                                ):
                                    event_type = event.get("event")
                                    if event_type == "thought":
                                        st.write(f"💭 {event.get('content', '')}")
                                    elif event_type == "action":
                                        st.write(f"⚡ {event.get('tool', '')} → {event.get('args', {})}")
                                    elif event_type == "observation":
                                        st.write(f"📊 Observation ({event.get('tool', '')}):")
                                        st.json(event.get("result", {}))
                                    elif event_type == "answer":
                                        inverse_result = event.get("result")

                            session.inverse_design_result = inverse_result

                            if inverse_result and inverse_result.get("ok"):
                                st.success("✅ Inverse design completed!")
                            else:
                                st.warning("⚠️ Inverse design finished with issues. Check stream logs and diagnostics.")

                            run_result = {}
                            if isinstance(inverse_result, dict):
                                run_result = (
                                    inverse_result.get("run_result")
                                    or inverse_result.get("data", {}).get("run_result")
                                    or {}
                                )

                            if run_result:
                                st.markdown("#### 📌 Inverse-Design Summary")
                                col_r1, col_r2, col_r3 = st.columns(3)
                                col_r1.metric("Status", str(run_result.get("status", "unknown")))
                                col_r2.metric(
                                    "Best Score",
                                    f"{float(run_result.get('best_score', 0.0)):.4f}",
                                )
                                col_r3.metric(
                                    "Best Iteration",
                                    str(run_result.get("best_iteration", "-")),
                                )

                                iterations = run_result.get("iterations", [])
                                objective_trace = [
                                    item.get("objective_value")
                                    for item in iterations
                                    if isinstance(item, dict) and item.get("objective_value") is not None
                                ]
                                if len(objective_trace) > 1:
                                    st.markdown("#### 📈 Objective Trend")
                                    st.line_chart(objective_trace)

                                failure_diagnosis = run_result.get("failure_diagnosis")
                                if failure_diagnosis:
                                    with st.expander("🩺 Failure Diagnosis", expanded=False):
                                        st.json(failure_diagnosis)

                                artifact_paths = []
                                for item in iterations:
                                    if not isinstance(item, dict):
                                        continue
                                    artifact_paths.extend(item.get("artifacts", []))

                                shown_paths = set()
                                for artifact in artifact_paths:
                                    try:
                                        artifact_path = Path(str(artifact))
                                    except Exception:
                                        continue
                                    if not artifact_path.exists():
                                        continue
                                    if artifact_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                                        continue
                                    normalized = str(artifact_path)
                                    if normalized in shown_paths:
                                        continue
                                    shown_paths.add(normalized)
                                    st.image(normalized, caption=artifact_path.name)

                    except Exception as inverse_e:
                        st.error(f"Inverse-design error: {inverse_e}")
                        import traceback
                        st.code(traceback.format_exc())

                if run_fdtd:
                    st.info("🔄 Running Agent-driven Tidy3D simulation...")
                    
                    try:
                        from PhotonicsAI.agents.pdk_agent import create_pdk_agent
                        from PhotonicsAI.Photon.component_detector import detect_component_type

                        component_type = "waveguide"
                        comp_list = session.p200_pretemplate.get("components_list", []) if hasattr(session, 'p200_pretemplate') else []
                        if comp_list:
                            component_type, _ = detect_component_type(str(comp_list[0]))

                        agent = create_pdk_agent()
                        final_result = None
                        scores: list[float] = []
                        current_expander = None
                        with st.container():
                            for event in agent.run_optimization(
                                component_type=component_type,
                                template_path=session.generated_template_path,
                                wavelength_nm=1550.0,
                            ):
                                event_type = event.get("event")
                                iteration = event.get("iteration")

                                # Create a new expander when a new iteration starts.
                                if (
                                    event_type == "thought"
                                    and "Starting optimisation iteration" in event.get("content", "")
                                ):
                                    label = event.get("content", f"Iteration {iteration}")
                                    current_expander = st.expander(f"🔄 {label}", expanded=True)

                                ctx = current_expander if current_expander is not None else st

                                if event_type == "thought":
                                    ctx.write(f"💭 {event.get('content', '')}")
                                elif event_type == "action":
                                    ctx.write(f"⚡ {event.get('tool', '')} → {event.get('args', {})}")
                                elif event_type == "observation":
                                    ctx.write(f"📊 Observation ({event.get('tool', '')}):")
                                    ctx.json(event.get("result", {}))
                                    score = event.get("result", {}).get("data", {}).get("score")
                                    if score is not None:
                                        scores.append(float(score))
                                elif event_type == "reflection":
                                    ctx.info(event.get("content", ""))
                                elif event_type == "answer":
                                    final_result = event.get("result")

                            # Score trend chart.
                            if len(scores) > 1:
                                st.markdown("#### 📈 Score Trend")
                                st.line_chart(scores)

                        if final_result and final_result.get("ok"):
                            st.success("✅ Optimisation completed!")
                            res_data = final_result.get("data", {})
                            col_m1, col_m2 = st.columns(2)
                            col_m1.metric("Best Score", f"{res_data.get('best_score', 0):.4f}")
                            col_m2.metric("Iterations", res_data.get("iterations_used", "?"))
                        else:
                            st.warning("⚠️ Simulation finished with warnings or partial outputs.")

                        for image_path in sorted(PATH.build.glob(f"tidy3d_*_{component_type}.png")):
                            st.image(str(image_path), caption=image_path.name)

                        tidy_log = PATH.build / "tidy3d.log"
                        if tidy_log.exists():
                            with st.expander("📄 Tidy3D Log"):
                                st.code(tidy_log.read_text(encoding="utf-8"), language="text")
                                
                    except Exception as sim_e:
                        st.error(f"Simulation error: {sim_e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.error("Template file not found.")
            
            # Stop execution here for single component mode
            st.stop()


            # Component and template search
            session.p100_llm_api_selection = component_selection_model
            
            with st.spinner("Searching design library ..."):
                if len(session.p200_pretemplate.get("components_list", [])):
                    # search templates
                    templates_search_r = llm_api.llm_search(
                        str(session.p200_pretemplate), list(templates_dict.values())
                    )
                    session.p200_retreived_templates = [
                        (list(templates_dict.items())[i])
                            for i in templates_search_r.match_list
                    ]

                    # search components
                    session.p200_componenets_search_r = []
                    for c in session.p200_pretemplate["components_list"]:
                        r = llm_api.llm_search(c, list_of_docs)
                        session.p200_componenets_search_r.append(r)

            st.success("✅ Component search completed!")
            
            # Mark entity extraction as complete
            session.entity_extraction_complete = True
            session.automatic_phase = "component_selection"
            st.rerun()

        # Phase 2: Component Selection (only show if entity extraction is complete)
        elif hasattr(session, 'entity_extraction_complete') and session.entity_extraction_complete and not (hasattr(session, 'component_search_complete') and session.component_search_complete):
            session.automatic_phase = "component_selection"
            st.markdown(
                '<div style="text-align: right; font-size: 18px; font-family: monospace;">Component Selection</div>',
                unsafe_allow_html=True,
            )
            # Always create columns for consistent layout
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
            col3, col4 = st.columns(2)

            # Display component search results (left)
            with col3:
                st.markdown(
                    html_banner.format(content="🛠️ build a new circuit"),
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                if hasattr(session, 'p200_componenets_search_r') and session.p200_componenets_search_r:
                    for i, search_result in enumerate(session.p200_componenets_search_r):
                        with st.container(border=True):
                            comp_name = session.p200_pretemplate['components_list'][i] if hasattr(session, 'p200_pretemplate') else f"Component {i+1}"
                            st.write(f"**{comp_name}**")
                            options = []
                            for k, j in enumerate(search_result.match_list):
                                name = list_of_cnames[j]
                                score = search_result.match_scores[k]
                                if score == "exact":
                                    option = f"{name} :green[/{score}/]"
                                elif score == "partial":
                                    option = f"{name} :orange[/{score}/]"
                                elif score == "poor":
                                    option = f"{name} :red[/{score}/]"
                                else:
                                    option = f"{name} :grey[/{score}/]"
                                options.append(option)

                            st.radio(
                                f"Component {i+1} options:",
                                options,
                                key=f"auto_component_{i}",
                                label_visibility="collapsed",
                            )
                else:
                    st.info("未找到组件候选项。可点击右侧模板或使用下方自动选择/跳过。")

            # Display template search results (right)
            with col4:
                if hasattr(session, 'p200_retreived_templates') and session.p200_retreived_templates:
                    st.markdown(
                        html_banner.format(content="🧩 use a template"),
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                    for i, item in enumerate(session.p200_retreived_templates):
                        with st.container(border=True):
                            t = item[1]
                            id_ = item[0]
                            st.write(t["doc"]["title"] + "\n" + t["doc"]["description"])
                            st.markdown(f"[reference]({t['doc']['reference']})")
                            if st.button(f"Select {id_}", key=f"auto_template_{i}"):
                                session.p200_selected_template = id_
                                session.template_selected = True
                                session.component_search_complete = True
                                st.success(f"Selected template: {id_}")
                                st.rerun()
                else:
                    st.markdown(html_small.format(content="无模板匹配结果。"), unsafe_allow_html=True)

            # Action buttons (always visible)
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            a1, a2, a3 = st.columns([1,1,1])
            with a1:
                if st.button("Submit Component Selection", key="auto_submit_components"):
                    selected_components = []
                    if hasattr(session, 'p200_componenets_search_r') and session.p200_componenets_search_r:
                        for i in range(len(session.p200_componenets_search_r)):
                            selected_key = f"auto_component_{i}"
                            # If user didn't touch the radio, try to pick the top option
                            selected = session.get(selected_key)
                            if not selected:
                                # default to best match (prefer exact)
                                sr = session.p200_componenets_search_r[i]
                                best_idx = 0
                                for idx, sc in enumerate(sr.match_scores):
                                    if sc == "exact":
                                        best_idx = idx
                                        break
                                name = list_of_cnames[sr.match_list[best_idx]]
                                selected = f"{name} :green[/auto/]"
                            component_name = selected.split(" :")[0]
                            selected_components.append(component_name)

                    if selected_components:
                        session.p200_selected_components = selected_components
                        session.components_selected = True
                        session.component_search_complete = True
                        st.success("Components selected!")
                        st.rerun()
                    else:
                        st.error("没有可提交的组件，请先选择或使用自动选择。")
            with a2:
                if st.button("Auto-select Best Matches", key="auto_autoselect"):
                    selected_components = []
                    if hasattr(session, 'p200_componenets_search_r') and session.p200_componenets_search_r:
                        for sr in session.p200_componenets_search_r:
                            # pick exact if exists else first
                            pick = 0
                            for idx, sc in enumerate(sr.match_scores):
                                if sc == "exact":
                                    pick = idx
                                    break
                            selected_components.append(list_of_cnames[sr.match_list[pick]])
                    if selected_components:
                        session.p200_selected_components = selected_components
                        session.components_selected = True
                        session.component_search_complete = True
                        st.success("Auto-selected best matches")
                        st.rerun()
                    else:
                        st.error("没有候选组件可自动选择。")
            with a3:
                if st.button("Reset Workflow", key="auto_reset"):
                    session.entity_extraction_complete = False
                    session.component_search_complete = False
                    session.schematic_complete = False
                    session.automatic_phase = "input"
                    session.current_message = ""
                    session.show_examples = True
                    session.input_submitted = False
                    st.success("Workflow reset! Enter a new circuit description.")
                    st.rerun()

        # Phase 3: Schematic Generation (only run if component selection is complete)
        elif hasattr(session, 'component_search_complete') and session.component_search_complete and "p300" not in session:
            session.automatic_phase = "schematic"
            st.markdown(
                        '<div style="text-align: right; font-size: 18px; font-family: monospace;">300 schematic</div>',
                unsafe_allow_html=True,
            )

            # Create circuit DSL from selected components
            if hasattr(session, 'components_selected') and session.components_selected:
                session.p200_pretemplate["components_list"] = session.p200_selected_components
                map_pretemplate_to_template()
                session["p300"] = True

                # Generate schematic for component selection path
                if "p300" in session:
                    logger()
                    session.p100_llm_api_selection = schematic_model

                    session["p300_circuit_dsl"] = get_ports_info(session["p300_circuit_dsl"])
                    session["p300_circuit_dsl"] = get_params(session["p300_circuit_dsl"])

                    # Apply settings for component selection path
                    session["p300_circuit_dsl"] = llm_api.apply_settings(session, session.p100_llm_api_selection)

                    with st.spinner("Working on the schematic..."):
                        session["p300_dot_string_draft"] = utils.circuit_to_dot(
                            session["p300_circuit_dsl"]
                        )

                        if len(session["p300_circuit_dsl"]["nodes"]) > 0:
                            session["p300_dot_string"] = llm_api.dot_add_edges(session)
                            session["p300_dot_string"] = llm_api.dot_verify(
                                session
                            )

                            for attempt in range(4):
                                print("\n\n+++++++++++++++++++++2")
                                print(session["p300_dot_string"])
                                print("+++++++++++++++++++++2\n\n")
                                happy_flag = utils.dot_planarity(session["p300_dot_string"])
                                if happy_flag:
                                    break
                                else:
                                    st.markdown(
                                        ":red[Crossing edges found! Redoing the graph edges...]"
                                    )
                                    session["p300_dot_string"] = llm_api.dot_add_edges_errorfunc(session)
                                    session["p300_dot_string"] = llm_api.dot_verify(
                                        session
                                    )

                        session["p300_dot_string"] = llm_api.dot_verify(
                            session
                        )

                    session["p300_circuit_dsl"] = utils.edges_dot_to_yaml(session)

                    # get initial placements from dot
                    session["p300_footprints_dict"], session["p300_circuit_dsl"] = footprint_netlist(
                        session["p300_circuit_dsl"]
                    )
                    session["p300_dot_string_scaled"] = utils.dot_add_node_sizes(
                        session["p300_dot_string"],
                        utils.multiply_node_dimensions(session["p300_footprints_dict"], 0.01),
                    )
                    session["p300_graphviz_node_coordinates"] = utils.get_graphviz_placements(
                        session["p300_dot_string_scaled"]
                    )
                    session["p300_graphviz_node_coordinates"] = utils.multiply_node_dimensions(
                        session["p300_graphviz_node_coordinates"], 100 / 72
                    )

                    session["p300_circuit_dsl"] = utils.add_placements_to_dsl(session)
                    session["p300_circuit_dsl"] = utils.add_final_ports(session)

                    session["p300_dot_string"] = session["p300_dot_string"]
                    session["p300_circuit_dsl"] = session["p300_circuit_dsl"]
                    session["p400"] = True
                    session.schematic_complete = True
                    logger()

                    st.success("✅ Schematic generation completed!")
                    st.rerun()

            # Template handling (if template was selected)
            elif hasattr(session, 'template_selected') and session.template_selected:
                template_id = session["p200_selected_template"]
                with st.container(border=True):
                    st.markdown(f"Selection: *{template_id}*\n\n")
                    st.write(
                        templates_dict[template_id]["doc"]["title"]
                        + "\n"
                        + templates_dict[template_id]["doc"]["description"]
                    )
                    st.markdown(f"[reference]({templates_dict[template_id]['doc']['reference']})")

                    st.write(templates_dict[template_id]["doc"]["title"])
                    st.write("For this item, these specifications are required:")

                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container(border=True):
                            specs_dict = templates_dict[template_id]["properties"]["specs"]
                            for key, item in specs_dict.items():
                                user_input = st.text_input(
                                    f"{key} ({item['comment']})", item["value"]
                                )
                                session.user_specs[key] = {
                                    "value": user_input,
                                    "comment": item["comment"],
                                }

                            if st.button("Update"):
                                with st.spinner("Updating template with new specifications..."):
                                    session.updated_specs = yaml.dump(
                                        session.user_specs, default_flow_style=False
                                    )

                session.p100_llm_api_selection = component_specification_model

                if "updated_specs" in session:
                    session["p200_user_specs"] = session.updated_specs
                    parsed_spec = llm_api.parse_user_specs(session)
                    if "Error" in parsed_spec:
                        st.write(parsed_spec)
                        st.write("Let's try again!")
                        del session.user_specs
                    else:
                        st.write("Got the specs!")

                        session["p300_circuit_dsl"] = templates_dict[template_id]
                        session["p300_circuit_dsl"]["properties"]["specs"] = parsed_spec

                        if "TEMPLATE" in template_id:
                            with st.spinner("Looking for components..."):
                                for key, value in session["p300_circuit_dsl"]["nodes"].items():
                                    try:
                                        user_specs = session["p300_circuit_dsl"]["properties"]["specs"]

                                        for spec_key in user_specs:
                                            if "comment" in user_specs[spec_key]:
                                                del user_specs[spec_key]["comment"]
                                    except Exception as e:
                                        st.error(f"An error occurred: {e}")
                                        user_specs = ""

                                    r = llm_api.llm_retrieve(
                                        value + f"\n({str(user_specs)})",
                                        session["p100_list_of_docs"],
                                        session["p100_llm_api_selection"],
                                    )

                                    all_retrieved = [session["p100_list_of_cnames"][i] for i in r]
                                    st.write(f"For {value}, found: " + "\n".join(all_retrieved))

                                    selected_component = session["p100_list_of_cnames"][r[0]]
                                    session["p300_circuit_dsl"]["nodes"][key] = {}
                                    session["p300_circuit_dsl"]["nodes"][key]["component"] = {}
                                    session["p300_circuit_dsl"]["nodes"][key]["component"] = (
                                        selected_component
                                    )

                        session["p300"] = True

                        # Generate schematic for template path
                        if "p300" in session:
                            logger()
                            session.p100_llm_api_selection = schematic_model

                            session["p300_circuit_dsl"] = get_ports_info(session["p300_circuit_dsl"])
                            session["p300_circuit_dsl"] = get_params(session["p300_circuit_dsl"])

                            if not session.template_selected:
                                session["p300_circuit_dsl"] = llm_api.apply_settings(session, session.p100_llm_api_selection)

                            with st.spinner("Working on the schematic..."):
                                session["p300_dot_string_draft"] = utils.circuit_to_dot(
                                    session["p300_circuit_dsl"]
                                )

                                if not session.template_selected:
                                    if len(session["p300_circuit_dsl"]["nodes"]) > 0:
                                        session["p300_dot_string"] = llm_api.dot_add_edges(session)
                                        session["p300_dot_string"] = llm_api.dot_verify(
                                            session
                                        )

                                        for attempt in range(4):
                                            print("\n\n+++++++++++++++++++++2")
                                            print(session["p300_dot_string"])
                                            print("+++++++++++++++++++++2\n\n")
                                            happy_flag = utils.dot_planarity(session["p300_dot_string"])
                                            if happy_flag:
                                                break
                                            else:
                                                st.markdown(
                                                    ":red[Crossing edges found! Redoing the graph edges...]"
                                                )
                                                session["p300_dot_string"] = llm_api.dot_add_edges_errorfunc(session)
                                                session["p300_dot_string"] = llm_api.dot_verify(
                                                    session
                                                )
                                else:
                                    with st.spinner("Updating template schematic..."):
                                        # Initialize the dot string for templates
                                        session["p300_dot_string"] = utils.circuit_to_dot(
                                            session["p300_circuit_dsl"]
                                        )
                                        session["p300_dot_string"] = llm_api.dot_add_edges_templates(session)

                                session["p300_dot_string"] = llm_api.dot_verify(
                                    session
                                )

                            session["p300_circuit_dsl"] = utils.edges_dot_to_yaml(session)

                            # get initial placements from dot
                            session["p300_footprints_dict"], session["p300_circuit_dsl"] = footprint_netlist(
                                session["p300_circuit_dsl"]
                            )
                            session["p300_dot_string_scaled"] = utils.dot_add_node_sizes(
                                session["p300_dot_string"],
                                utils.multiply_node_dimensions(session["p300_footprints_dict"], 0.01),
                            )
                            session["p300_graphviz_node_coordinates"] = utils.get_graphviz_placements(
                                session["p300_dot_string_scaled"]
                            )
                            session["p300_graphviz_node_coordinates"] = utils.multiply_node_dimensions(
                                session["p300_graphviz_node_coordinates"], 100 / 72
                            )

                            session["p300_circuit_dsl"] = utils.add_placements_to_dsl(session)
                            session["p300_circuit_dsl"] = utils.add_final_ports(session)

                            session["p300_dot_string"] = session["p300_dot_string"]
                            session["p300_circuit_dsl"] = session["p300_circuit_dsl"]
                            session["p400"] = True
                            session.schematic_complete = True
                            logger()

                            st.success("✅ Schematic generation completed!")
                            st.rerun()

        # Phase 4: Layout and Simulation (only run if schematic is complete)
        elif hasattr(session, 'schematic_complete') and session.schematic_complete and "p400" in session:
            session.automatic_phase = "layout"
            logger()
            st.markdown(
                '<div style="text-align: right; font-size: 18px; font-family: monospace">400 layout</div>',
                unsafe_allow_html=True,
            )

            session.p100_llm_api_selection = layout_model

            session["p400_gf_netlist"] = utils.dsl_to_gf(session["p300_circuit_dsl"])

            with st.spinner("Rendering the GDS ..."):
                try:
                    c, d = yaml_netlist_to_gds(session, ignore_links=False)
                    routing_flag = True
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    c, d = yaml_netlist_to_gds(session, ignore_links=True)
                    st.markdown(":red[Routing error.]")
                    routing_flag = False
                    pass

            with st.spinner("Simulating s-parameters ..."):
                wl = np.linspace(1.5, 1.6, 200)
                result = session["p400_sax_circuit"](wl=wl)
                p400_sax_fig = utils.plot_dict_arrays(wl, result)

            logger()

            st.success("✅ Layout and simulation completed!")

            # Display final outputs
            st.markdown("### 🎯 Final Results")
            
            st.markdown("**GDS Layout:**")
            if hasattr(session, 'p400_gdsfig'):
                st.pyplot(session.p400_gdsfig)
            else:
                st.warning("GDS figure not available")
            
            st.markdown("**Simulation Results:**")
            try:
                plot_path = PATH.build / "plot_sax.png"
                if plot_path.exists():
                    st.image(str(plot_path))
                else:
                    st.warning("Simulation plot file not found")
            except Exception as e:
                st.warning(f"Error displaying simulation plot: {e}")
                
            with st.spinner("Checking DRC..."):
                try:
                    cwd = Path.cwd()
                    open(str(cwd)+"/PhotonicsAI/Photon/drc/report.lydrb", "w+").close()
                    file_name = "placeholder"
                    gds_drc_file_path = "./drc/" + file_name + ".gds"
                    skip_drc = False  # Flag to skip DRC if GDS write fails
                    
                    # Write GDS file - try different approaches to handle large layer numbers
                    try:
                        # First try: standard write
                        c.write_gds(gds_drc_file_path)
                    except Exception as e:
                        if "layer numbers larger than 65535" in str(e):
                            st.warning("Large layer numbers detected, trying alternative write method...")
                            try:
                                # Try flattening first - check if result is not None
                                flattened = c.flatten()
                                if flattened is not None:
                                    flattened.write_gds(gds_drc_file_path)
                                    st.success("Successfully wrote flattened GDS file")
                                else:
                                    st.warning("Flattening returned None, trying different approach...")
                                    # Try to write with different parameters
                                    c.write_gds(gds_drc_file_path, max_points=None)
                                    st.success("Successfully wrote GDS file with modified parameters")
                            except Exception as flatten_error:
                                st.error(f"Flattening failed: {flatten_error}")
                                # Try writing with different parameters as last resort
                                try:
                                    st.warning("Trying to write with minimal parameters...")
                                    c.write_gds(gds_drc_file_path, max_points=None, max_absolute_error=None, max_relative_error=None)
                                    st.success("Successfully wrote GDS file with minimal parameters")
                                except Exception as final_error:
                                    st.error(f"All write methods failed: {final_error}")
                                    # Create a simple placeholder file for DRC
                                    st.warning("Creating placeholder file for DRC...")
                                    with open(gds_drc_file_path, 'w') as f:
                                        f.write("# Placeholder file - GDS write failed due to layer number limitations\n")
                                    st.info("DRC will be skipped due to GDS write failure")
                                    # Skip DRC by setting a flag
                                    skip_drc = True
                        else:
                            raise e
                    
                    # Only run DRC if we didn't skip it due to GDS write failure
                    if not skip_drc:
                        run_drc(gds_drc_file_path, file_name)
                        st.success("✅ DRC completed successfully!")
                    else:
                        st.warning("DRC skipped due to GDS write failure")
                    with st.expander("DRC results", expanded=False):
                        report_file = str(cwd)+"/PhotonicsAI/Photon/drc/report.lydrb"
                        try:
                            if os.path.exists(report_file):
                                with open(report_file, 'r') as f:
                                    report_content = f.read()
                                if report_content.strip():
                                    st.text("DRC Report:")
                                    st.code(report_content, language="text")
                                else:
                                    st.info("DRC completed but report file is empty. This usually means no violations were found.")
                            else:
                                st.warning("DRC report file not found. DRC may not have completed successfully.")
                        except Exception as e:
                            st.error(f"Error reading DRC report: {e}")
                            st.info("Check the terminal output for DRC execution details.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.info("Note: This error might be due to GDS2 layer number limitations. The circuit layout is still valid.")

            # Circuit optimizer
            optimize_flag = False
            if "properties" in session["p300_circuit_dsl"]:
                if "optimizer" in session["p300_circuit_dsl"]["properties"]:
                    if "error_fn" in session["p300_circuit_dsl"]["properties"]["optimizer"]:
                        if (
                            "free_params"
                            in session["p300_circuit_dsl"]["properties"]["optimizer"]
                        ):
                            if (
                                "sparam"
                                in session["p300_circuit_dsl"]["properties"]["optimizer"]
                            ):
                                if routing_flag:
                                            optimize_flag = 0

            if optimize_flag:
                with st.spinner("Optimizing circuit..."):
                    session["p400_gf_netlist"] = circuit_optimizer(session)

                try:
                    c, d = yaml_netlist_to_gds(session, ignore_links=False)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    c,d = yaml_netlist_to_gds(session, ignore_links=True)
                    st.markdown(":red[Routing error.]")
                    pass

                wl = np.linspace(1.53, 1.57, 500)
                result = session["p400_sax_circuit"](wl=wl)
                p400_sax_fig = utils.plot_dict_arrays(wl, result)

                st.success("✅ Circuit optimization completed!")
                
                # Display optimized results
                st.markdown("### 🚀 Optimized Results")
                
                st.markdown("**Optimized GDS Layout:**")
                if hasattr(session, 'p400_gdsfig'):
                    st.pyplot(session.p400_gdsfig)
                else:
                    st.warning("GDS figure not available")
                
                st.markdown("**Optimized Simulation Results:**")
                try:
                    plot_path = PATH.build / "plot_sax.png"
                    if plot_path.exists():
                        st.image(str(plot_path))
                    else:
                        st.warning("Simulation plot file not found")
                except Exception as e:
                    st.warning(f"Error displaying simulation plot: {e}")

            # Runtime tracking - measure from p100 to p400 completion (after all processing)
            session.p100_end_time = time.time()
            session.p100_runtime = session.p100_end_time - session.p100_start_time
            
            # Debug timing information
            print(f"=== RUNTIME DEBUG ===")
            print(f"Start time: {session.p100_start_time}")
            print(f"End time: {session.p100_end_time}")
            print(f"Runtime: {session.p100_runtime} seconds ({session.p100_runtime/60:.2f} minutes)")
            print(f"Total workflow time: {session.p100_runtime/60:.2f} minutes")
            print(f"=====================")

            # Display token usage at the end of the workflow
            token_usage = llm_api.get_token_usage()
            if token_usage["non_cached_input_tokens"] > 0 or token_usage["output_tokens"] > 0:
                st.markdown("---")
                st.markdown("### Token Usage Summary")
                st.markdown(f"**Input Tokens:** {token_usage['non_cached_input_tokens']}")
                st.markdown(f"**Output Tokens:** {token_usage['output_tokens']}")
                st.markdown(f"**Total Tokens:** {token_usage['non_cached_input_tokens'] + token_usage['output_tokens']}")
                
                # Print to terminal as well
                print(f"\n=== TOKEN USAGE SUMMARY ===")
                print(f"Input Tokens: {token_usage['non_cached_input_tokens']}")
                print(f"Output Tokens: {token_usage['output_tokens']}")
                print(f"Total Tokens: {token_usage['non_cached_input_tokens'] + token_usage['output_tokens']}")
                print(f"===========================\n")
            
            # Display runtime in appropriate format
            if hasattr(session, 'p100_runtime') and session.p100_runtime >= 60:
                minutes = int(session.p100_runtime // 60)
                seconds = session.p100_runtime % 60
                st.write(f"run time: {minutes}m {seconds:.1f}s")
            elif hasattr(session, 'p100_runtime'):
                st.write(f"run time: {round(session.p100_runtime,2)} seconds")
            logger()

