"""Utility functions for the PhotonicsAI package."""

import ast
import glob
import os
import pathlib
import re
from functools import lru_cache

import gdsfactory as gf
import importlib
import inspect
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygraphviz as pgv
import yaml
from sax.saxtypes import Float, Model

from PhotonicsAI.config import PATH


def extract_docstring(file_path):
    """Extracts the module-level docstring of a Python file."""
    with open(file_path, encoding="utf-8") as file:
        module_content = file.read()

    parsed_module = ast.parse(module_content)

    # Extract the module-level docstring
    docstring = ast.get_docstring(parsed_module)

    # Extract the module name from the file path
    module_name = os.path.basename(file_path).replace(".py", "")

    return {"module_name": module_name, "docstring": docstring, "file_path": file_path}


def search_directory_for_docstrings(directory=PATH.pdk):
    """Uses glob to find Python files and extract their docstrings."""
    all_docs = []
    # Recursive glob pattern to find all Python files
    file_paths = glob.glob(f"{directory}/*.py", recursive=True)
    filtered_paths = [
        path for path in file_paths if not os.path.basename(path) == "__init__.py"
    ]

    for file_path in filtered_paths:
        doc = extract_docstring(file_path)
        all_docs.append(doc)

    sorted_all_docs = sorted(all_docs, key=lambda x: x["module_name"])
    return sorted_all_docs


def circuit_to_dot(circuit_dsl):
    """Converts a circuit DSL to a DOT graph string."""
    # data = yaml.safe_load(yaml_str)

    if "name" in circuit_dsl["doc"]:
        graph_name = circuit_dsl["doc"]["name"]
    else:
        graph_name = "graph_name_placeholder"
    nodes = circuit_dsl.get("nodes", {})

    dot_lines = [f"graph {graph_name} {{", "  rankdir=LR;", "  node [shape=record];"]

    for node_name, node_info in nodes.items():
        # Handle both string and dictionary node_info
        if isinstance(node_info, str):
            component_name = node_info
            ports_info = ""
        else:
            component_name = node_info.get("component", "")
            ports_info = node_info.get("properties", {}).get("ports", "")

        if "x" in ports_info:
            input_ports, output_ports = map(int, ports_info.split("x"))

            # Generate input and output port labels
            input_labels = (
                "|".join([f"<o{i}> o{i}" for i in range(input_ports, 0, -1)])
                if input_ports > 0
                else ""
            )
            output_labels = (
                "|".join(
                    [
                        f"<o{i}> o{i}"
                        for i in range(input_ports + 1, input_ports + output_ports + 1)
                    ]
                )
                if output_ports > 0
                else ""
            )

            # Format the label with conditional parts for input and output labels
            if input_labels and output_labels:
                label = f"{{{{{input_labels}}} | {node_name}: {component_name} | {{{output_labels}}}}}"
            elif input_labels:
                label = f"{{{{{input_labels}}} | {node_name}: {component_name} }}"
            elif output_labels:
                label = f"{{ {node_name}: {component_name} | {{{output_labels}}}}}"
            else:
                label = f"{node_name}: {component_name}"
        else:
            # handle cases without ports info
            label = f"{node_name}: {component_name}"

        dot_lines.append(f'  {node_name} [label="{label}"];')

    dot_lines.append("}")

    return "\n".join(dot_lines)


def edges_dot_to_yaml(session):
    """Converts a DOT graph string to a YAML dictionary."""
    # Regular expression to find the edges
    edge_pattern = re.compile(r"(\w+):(\w+) -- (\w+):(\w+);")

    # Find all matches in the DOT graph string
    edges = edge_pattern.findall(session["p300_dot_string"])

    # Format edges as required
    formatted_edges = [f"{edge[0]},{edge[1]}: {edge[2]},{edge[3]}" for edge in edges]

    circuit = session["p300_circuit_dsl"]
    circuit["edges"] = {}
    for i, edge in enumerate(formatted_edges):
        circuit["edges"][f"E{i+1}"] = {}
        circuit["edges"][f"E{i+1}"]["link"] = edge

    return circuit


def dsl_to_gf(circuit_dsl):
    """Converts a circuit DSL to a GDSFactory netlist.

    Also sanitizes instance settings to only include parameters accepted by the
    target component (drop unknown kwargs to avoid runtime TypeError).
    """

    @lru_cache(maxsize=256)
    def _allowed_settings_for(component_name: str) -> set:
        """Return the set of valid setting keys for a given component name.

        Uses gdsfactory's from_yaml + get_netlist to introspect default settings.
        Caches results for performance.
        """
        try:
            # minimal netlist to introspect settings
            gf_netlist = {"instances": {"X": {"component": component_name}}}
            c_tmp = gf.read.from_yaml(yaml.dump(gf_netlist))
            d_tmp = c_tmp.get_netlist(recursive=False)
            settings_keys = set(d_tmp["instances"]["X"].get("settings", {}).keys())
            if settings_keys:
                return settings_keys
        except Exception:
            pass

        # Fallback: introspect the function signature in our DesignLibrary directly
        try:
            mod = importlib.import_module(
                f"PhotonicsAI.KnowledgeBase.DesignLibrary.{component_name}"
            )
            func = getattr(mod, component_name)
            sig = inspect.signature(func)
            params = []
            for name, p in sig.parameters.items():
                if name in {"self"}:
                    continue
                # Only include named params (positional-or-keyword or keyword-only)
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    params.append(name)
            return set(params)
        except Exception:
            return set()

    # Create a new nodes dictionary with renamed keys and sanitized settings
    new_nodes = {}
    for node_id, node_info in circuit_dsl.get("nodes", {}).items():
        component_name = node_info.get("component", "")
        # Prefer params; some flows may already use settings
        raw_settings = node_info.get("params", {}) or node_info.get("settings", {}) or {}

        # Filter out unknown parameters to prevent unexpected keyword errors
        allowed = _allowed_settings_for(component_name)
        if allowed:
            clean_settings = {k: v for k, v in raw_settings.items() if k in allowed}
        else:
            # If we couldn't introspect, pass through as-is (best effort)
            clean_settings = raw_settings

        new_nodes[node_id] = {
            "component": component_name,
            "info": node_info.get("properties", {}),  # Rename properties to info
            "settings": clean_settings,  # Sanitized settings
        }

    new_routes = {}
    for _edge_id, edge_info in circuit_dsl["edges"].items():
        link = edge_info["link"]
        source, target = link.split(": ")
        new_routes[source] = target

    placements = {}
    for i, (node_id, node_info) in enumerate(circuit_dsl.get("nodes", {}).items()):
        pl = node_info.get("placement", {}) if isinstance(node_info, dict) else {}
        x = pl.get("x")
        y = pl.get("y")
        rot = pl.get("rotation")
        if x is None or y is None or rot is None:
            # Fallback to a simple linear placement to avoid KeyError and keep pipeline running
            x = float(i * 100.0)
            y = 0.0
            rot = 0
        placements[node_id] = {"x": x, "y": y, "rotation": rot}

    gf_netlist = {
        "instances": new_nodes,
        "routes": {
            "optical": {
                "links": new_routes,  # Move the list of links under routes: optical: links
            }
        },
        "placements": placements,
        "ports": circuit_dsl.get("ports", {}),
    }

    return gf_netlist


def get_graphviz_placements(dot_string):
    """Get the node positions from a DOT graph string."""
    # Sanitize DOT similar to dot_planarity
    def sanitize_dot_string(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = s.replace("\r", "")
        s = re.sub(r"```(?:dot)?", "", s)
        s = re.sub(r"/\*[\s\S]*?\*/", "", s)
        s = re.sub(r"//.*", "", s)
        m = re.search(r"\b(digraph|graph)\b", s)
        s2 = s[m.start():] if m else s
        start = s2.find("{")
        if start != -1:
            depth = 0
            end_index = None
            for i, ch in enumerate(s2[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_index = i
                        break
            if end_index is not None:
                header = s2[:start]
                block = s2[start:end_index + 1]
                if not re.search(r"\b(digraph|graph)\b", header):
                    s_clean = "graph G " + block
                else:
                    s_clean = header + block
            else:
                if re.search(r"\b(digraph|graph)\b", s2[:start]):
                    s_clean = s2 + "\n}"
                else:
                    s_clean = "graph G " + s2[start:] + "\n}"
        else:
            s_clean = f"graph G {{\n{s2}\n}}"
        s_clean = s_clean.replace("```", "")
        return s_clean.strip()

    dot_string = sanitize_dot_string(dot_string)

    # Create a graph from the DOT string
    try:
        graph = pgv.AGraph(string=dot_string)
    except Exception:
        # Fallback: dump and try naive placement
        try:
            PATH.build.mkdir(parents=True, exist_ok=True)
            with open(str(PATH.build / "invalid_dot_positions.dot"), "w", encoding="utf-8") as fh:
                fh.write(dot_string)
        except Exception:
            pass
        # naive positions: parse node ids and place on a line
        node_ids = []
        for line in dot_string.splitlines():
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*\[", line)
            if m:
                node_id = m.group(1)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        positions = {nid: (i * 100.0, 0.0) for i, nid in enumerate(node_ids)}
        return positions

    # Set node separation (horizontal) (inches)
    graph.graph_attr["nodesep"] = ".05"

    # Set rank separation (vertical) (inches)
    graph.graph_attr["ranksep"] = ".05"

    # Layout the graph using the dot layout
    graph.layout(prog="dot")

    # Render the graph to a file
    graph.draw(PATH.build / "graph.svg")

    # Get node positions ---> THIS IS THE CENTER OF THE NODES. BUT GDSFACTORY USES THE BOTTOM LEFT CORNER.
    positions = {}
    for node in graph.nodes():
        pos = node.attr.get("pos")
        if not pos:
            continue
        try:
            x, y = map(float, pos.split(","))
        except Exception:
            continue
        # Primary key: actual DOT node id
        positions[str(node)] = (x, y)
        # Alias key: if label contains "<dsl_id>: ..."，也记录为坐标键，便于 DSL 对齐
        label = node.attr.get("label")
        if label:
            m = re.search(r"([A-Za-z0-9_]+)\s*:", label)
            if m:
                alias = m.group(1)
                positions.setdefault(alias, (x, y))
    # If Graphviz produced no positions, fallback to naive
    if not positions:
        node_ids = [str(n) for n in graph.nodes()]
        positions = {nid: (i * 100.0, 0.0) for i, nid in enumerate(node_ids)}
    return positions


def dot_add_node_sizes(dot_string, node_dimensions):
    """Add node sizes to a DOT graph string."""
    PADDING_FACTOR = 1.2  # in percent
    PADDING_MARGIN = 1  # ? in microns or inches?
    lines = dot_string.split("\n")
    output_lines = []
    nodes_added = set()

    for line in lines:
        stripped_line = line.strip()
        # if stripped_line.startswith("C") and "[" in stripped_line and stripped_line.endswith("];"):
        if ("label=" in stripped_line) and stripped_line.endswith("];"):
            node_name = stripped_line.split("[")[0].strip()
            if node_name in node_dimensions and node_name not in nodes_added:
                width, height = node_dimensions[node_name]
                size_line = f"  {node_name} [width={width*PADDING_FACTOR+PADDING_MARGIN}, height={height*PADDING_FACTOR+PADDING_MARGIN}, shape=record, fixedsize=true];"
                output_lines.append(size_line)
                nodes_added.add(node_name)
        output_lines.append(line)

    return "\n".join(output_lines)


def multiply_node_dimensions(node_dimensions, factor=0.01):
    """Multiply the dimensions of nodes by a factor."""
    return {
        node: (round(width * factor, 3), round(height * factor, 3))
        for node, (width, height) in node_dimensions.items()
    }


def add_placements_to_dsl(session):
    """Add node placements to the circuit DSL.

    Robust to node-name drift between DOT and DSL. Uses positions for matching DSL node ids;
    if a DOT node label contains "<dsl_id>: ..." 该 id 也会在 positions 中作为别名键。
    未找到坐标的节点将被跳过（保留默认/缺省），避免 KeyError。
    """
    placements = session["p300_graphviz_node_coordinates"]
    circuit = session["p300_circuit_dsl"]

    # Add placements to the data
    # GRAPHVIZ returns CENTER OF THE NODES. BUT GDSFACTORY USES THE BOTTOM LEFT CORNER
    for node_id in list(circuit["nodes"].keys()):
        if node_id not in placements:
            # 尝试用大小写/别名匹配（常见 LLM 将 N1 改为 n1 或 A/B）
            alt = None
            if node_id.lower() in placements:
                alt = node_id.lower()
            elif node_id.upper() in placements:
                alt = node_id.upper()
            else:
                # 最后尝试：寻找完全相同 id 的别名键（get_graphviz_placements 会把 label 中的 dsl id 也塞到键里）
                # 如果没有就跳过该节点
                pass
            if alt and alt in placements:
                pos = placements[alt]
            else:
                # 没有坐标可用，跳过，避免 KeyError
                continue
        else:
            pos = placements[node_id]

        x, y = pos
        node = circuit["nodes"][node_id]
        props = node.get("properties", {})
        dx = props.get("dx", 0)
        dy = props.get("dy", 0)

        node.setdefault("placement", {})
        node["placement"]["rotation"] = 0  # TODO
        node["placement"]["x"] = x - dx / 2
        node["placement"]["y"] = y - 0 * dy / 2

    # circuit['']['placements'] = {key: {'x': value[0]-circuit['nodes'][key]['info']['dx']/2,
    #                             'y': value[1]-0*circuit['nodes'][key]['info']['dy']/2} for key, value in placements.items()}

    return circuit


def add_final_ports(session):
    """Add final ports to the circuit DSL."""

    def find_open_ports(dot_source):
        nodes_ports = {}

        # Parse the graph source to identify edges and node labels
        connected_ports = set()
        for raw_line in dot_source.splitlines():
            line = raw_line.strip()
            # Collect node label-defined ports
            if '[label="' in line:
                try:
                    node = line.split()[0]
                    m = re.search(r'\[label="(.*)"\]', line)
                    if not m:
                        continue
                    label = m.group(1)
                    ports = re.findall(r"<(o\d+)>", label)
                    if ports:
                        nodes_ports[node] = ports
                except Exception:
                    continue
            # Collect connected endpoints (robust to extra colons or attributes)
            if "--" in line:
                # Match occurrences like Node:o1 possibly followed by compass or attrs, we only take first two groups
                for n, p in re.findall(r"(\w+):(o\d+)", line):
                    connected_ports.add(f"{n}:{p}")

        # Determine open ports
        open_ports = []
        for node, ports in nodes_ports.items():
            for port in ports:
                if f"{node}:{port}" not in connected_ports:
                    open_ports.append(f"{node}:{port}")

        return open_ports

    def build_dot_to_dsl_map(dot_source):
        """Build a mapping from DOT node ids to DSL node ids using labels.

        If a DOT node has a label like "N1: component ...", we map DOT_id -> "N1".
        """
        mapping = {}
        for raw_line in dot_source.splitlines():
            line = raw_line.strip()
            if not line or "[label=" not in line:
                continue
            try:
                node = line.split()[0]
                m = re.search(r'\[label="(.*)"\]', line)
                if not m:
                    continue
                label = m.group(1)
                m2 = re.search(r"([A-Za-z0-9_]+)\s*:", label)
                if m2:
                    dsl_id = m2.group(1)
                    mapping[node] = dsl_id
            except Exception:
                continue
        return mapping

    def create_port_dict(open_ports):
        port_dict = {}
        for i, open_port in enumerate(open_ports):
            port_dict[f"o{i+1}"] = open_port.replace(":", ",")
        return port_dict

    dot_src = session["p300_dot_string"]
    open_ports_list = find_open_ports(dot_src)

    # Map DOT node ids in ports to DSL node ids to avoid mismatches like 'C1' vs 'N1'
    dot_to_dsl = build_dot_to_dsl_map(dot_src)
    dsl_nodes = set(session["p300_circuit_dsl"].get("nodes", {}).keys())

    def map_node_to_dsl(node: str) -> str | None:
        if node in dsl_nodes:
            return node
        if node in dot_to_dsl and dot_to_dsl[node] in dsl_nodes:
            return dot_to_dsl[node]
        if node.lower() in dsl_nodes:
            return node.lower()
        if node.upper() in dsl_nodes:
            return node.upper()
        return None

    mapped_open_ports = []
    for npair in open_ports_list:
        try:
            node, port = npair.split(":", 1)
        except ValueError:
            continue
        target_node = map_node_to_dsl(node)
        if target_node:
            mapped_open_ports.append(f"{target_node}:{port}")
    open_ports_list = mapped_open_ports

    # Fallback: ensure at least 2 external ports for SAX circuit validation
    if len(open_ports_list) < 2:
        try:
            nodes = list(session["p300_circuit_dsl"].get("nodes", {}).keys())
            if nodes:
                first = nodes[0]
                last = nodes[-1] if len(nodes) > 1 else nodes[0]
                # Prefer expose o1 of first and last nodes
                # Avoid duplicates if only one node
                open_ports_list = [f"{first}:o1", f"{last}:o2" if last == first else f"{last}:o1"]
        except Exception:
            # As a last resort, create generic o1/o2 on a synthetic node name; 
            # but keep best effort to avoid crashing
            open_ports_list = ["N1:o1", "N2:o1"]

    circuit_ports_dict = create_port_dict(open_ports_list)

    session["p300_circuit_dsl"]["ports"] = circuit_ports_dict
    # print(yaml.dump(data, default_flow_style=False))
    return session["p300_circuit_dsl"]


def get_file_path(dir_string):
    """Get the absolute file path from a relative directory string."""
    current_dir = os.path.dirname(__file__)
    current_dir = os.path.join(
        current_dir, "..", "KnowledgeBase"
    )  # Use os.path.join for path construction
    file_path = os.path.join(
        current_dir, *dir_string.split(os.sep)
    )  # Use os.sep for splitting
    file_path = os.path.abspath(file_path)
    return file_path


matplotlib.use("Agg")

# Set the plot style to dark background
plt.style.use("dark_background")

# Set font type globally
font = {
    "family": "monospace",  # You can change this to any available font family
    "size": 11,
}
matplotlib.rc("font", **font)


def plot_dict_arrays(wl, data_dict):
    """Plot a dictionary of arrays."""
    # only keeping the s-params from port 1 to others:
    data_dict = {k: v for k, v in data_dict.items() if k[0] == "o1"}

    cols = 6  # Number of columns in the subplot grid

    num_plots = len(data_dict)
    rows = (num_plots // cols) + (
        num_plots % cols > 0
    )  # Calculate the number of rows needed

    # Calculate the figure size to maintain a 4:3 aspect ratio for each subplot
    subplot_width = 3
    subplot_height = 2
    fig_width = subplot_width * cols
    fig_height = subplot_height * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easy iteration if rows and cols are more than 1
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for _idx, (ax, (key, array)) in enumerate(zip(axes, data_dict.items())):
        ax.plot(
            wl,
            10 * np.log10(np.abs(array) ** 2),
            label=f"{key[0]}-{key[1]}",
            linewidth=2,
            color="darksalmon",
        )  # Make plot lines thicker
        ax.text(
            0.95,
            0.95,
            f"{key[0]}-{key[1]}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    # Remove any unused subplots
    for ax in axes[len(data_dict) :]:
        fig.delaxes(ax)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    plt.savefig(PATH.build / "plot_sax.png")
    plt.close()

    return fig


wl_cband = np.linspace(1.500, 1.600, 128)
PathType = str | pathlib.Path


def model_from_npz(
    filepath: PathType | np.ndarray,
    xkey: str = "wavelengths",
    xunits: float = 1,
) -> Model:
    """This is a modified version of the original function in gplugins/sax/read.py
    Returns a SAX Sparameters Model from a npz file.

    The SAX Model is a function that returns a SAX SDict interpolated over wavelength.

    Args:
        filepath: CSV Sparameters path or pandas DataFrame.
        xkey: key for wavelengths in file.
        xunits: x units in um from the loaded file (um). 1 means 1um.
    """
    sp = np.load(filepath) if isinstance(filepath, pathlib.Path | str) else filepath
    keys = list(sp.keys())

    if xkey not in keys:
        raise ValueError(f"{xkey!r} not in {keys}")

    x = jnp.asarray(sp[xkey] * xunits)
    wl = jnp.asarray(wl_cband)

    # make sure x is sorted from low to high
    idxs = jnp.argsort(x)
    x = x[idxs]
    sp = {k: v[idxs] for k, v in sp.items()}

    @jax.jit
    def model(wl: Float = wl):
        S = {}
        zero = jnp.zeros_like(x)

        for key in sp:
            if not key.startswith("wav"):
                port_mode0, port_mode1 = key.split(",")
                port0, _ = port_mode0.split("@")
                port1, _ = port_mode1.split("@")

                m = jnp.interp(wl, x, np.abs(sp.get(key, zero)))
                a = jnp.interp(wl, x, np.unwrap(np.angle(sp.get(key, zero))))
                S[(port0, port1)] = m * jnp.exp(1j * a)

        return S

    return model


def dot_crossing_edges(session):
    """Check if a Graphviz dot string has any crossing edges.

    Args:
        session: The Streamlit session object containing the dot string to check.
    """
    dot_string = session.p300_dot_string

    happy_flag = dot_planarity(dot_string)
    if happy_flag:
        return "No crossing edges found."
    else:
        return "Eroor: crossing edges found!"
    # , prompts["dot_verify"], session.p100_llm_api_selection


def dot_planarity(dot_string):
    """Check if a Graphviz dot string has any crossing edges.
    This function takes a dot string as input, applies a layout algorithm to position
    the nodes and edges, and then checks for any crossing edges in the graph.

    Args:
        session: The Streamlit session object containing the dot string to check.
    """
    # Sanitize DOT text to avoid pygraphviz parsing errors from LLM artifacts
    def sanitize_dot_string(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        # Normalize newlines and strip code fences
        s = s.replace("\r", "")
        s = re.sub(r"```(?:dot)?", "", s)
        # Remove C/C++ style comments and // line comments
        s = re.sub(r"/\*[\s\S]*?\*/", "", s)
        s = re.sub(r"//.*", "", s)
        # Trim leading noise before 'graph' or 'digraph'
        m = re.search(r"\b(digraph|graph)\b", s)
        s2 = s[m.start():] if m else s
        # Extract the first balanced brace block
        start = s2.find("{")
        if start != -1:
            depth = 0
            end_index = None
            for i, ch in enumerate(s2[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_index = i
                        break
            if end_index is not None:
                header = s2[:start]
                block = s2[start:end_index + 1]
                if not re.search(r"\b(digraph|graph)\b", header):
                    s_clean = "graph G " + block
                else:
                    s_clean = header + block
            else:
                # No matching closing brace; try to close it
                if re.search(r"\b(digraph|graph)\b", s2[:start]):
                    s_clean = s2 + "\n}"
                else:
                    s_clean = "graph G " + s2[start:] + "\n}"
        else:
            # No braces found; wrap contents in a minimal graph
            s_clean = f"graph G {{\n{s2}\n}}"
        # Final cleanup of stray backticks
        s_clean = s_clean.replace("```", "")
        return s_clean.strip()

    dot_string = sanitize_dot_string(dot_string)
        
    try:
        # Load the dot string
        graph = pgv.AGraph(string=dot_string)

        # Apply a layout to the graph
        graph.layout(prog="dot")

        # Get edge coordinates
        edges = []
        for edge in graph.edges():
            pos = edge.attr.get("pos")
            if not pos:
                continue
            points = pos.split()
            if not points:
                continue
            # Fallback parsing for edge pos strings
            try:
                start = tuple(map(float, points[0].split(",")))
                end = tuple(map(float, points[-1].split(",")))
                if len(start) >= 2 and len(end) >= 2:
                    edges.append(((start[0], start[1]), (end[0], end[1])))
            except Exception:
                continue
    except Exception as e:
        # If parsing/layout fails, dump the DOT for debugging and return False to trigger edge-fix retry
        try:
            PATH.build.mkdir(parents=True, exist_ok=True)
            with open(str(PATH.build / "invalid_dot_last.dot"), "w", encoding="utf-8") as fh:
                fh.write(dot_string)
        except Exception:
            pass
        return False

    # Function to check if two line segments (p1, q1) and (p2, q2) intersect
    def do_intersect(p1, q1, p2, q2):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
                1
            ] <= max(p[1], r[1]):
                return True
            return False

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    # Check each pair of edges for intersection
    crossings = []
    for i, (p1, q1) in enumerate(edges):
        for j, (p2, q2) in enumerate(edges):
            if i != j and do_intersect(p1, q1, p2, q2):
                crossings.append(((p1, q1), (p2, q2)))

    if len(crossings) == 0:
        return True
    else:
        return False
