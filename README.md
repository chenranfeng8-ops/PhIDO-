# PhIDO

Photonics Intelligent Design and Optimization

PhIDO is an AI-assisted design and inverse-optimization system for photonic integrated circuits (PICs). It converts natural-language design intent into validated, executable, and recoverable Tidy3D simulation workflows, then delivers traceable optimization records, diagnostic evidence, layout-related artifacts, and reports.

This repository is not a single-device script and is not limited to the 1x5 mode mux case. Devices such as the 1x5 mode mux and 1x2 splitter are validation scenarios; the system goal is a reusable and extensible PIC design workflow.

## Current Positioning

PhIDO is organized around an end-to-end inverse-design loop:

- Parse user requirements into structured design intent.
- Retrieve and bind relevant Tidy3D/Flexcompute documentation context.
- Generate simulation, geometry, source, monitor, objective, and optimizer configuration.
- Validate geometry, monitors, objectives, and cloud-cost risks before execution.
- Run Tidy3D forward simulation and adjoint inverse design.
- Diagnose failed or weak optimization runs from checkpoints, FOM traces, and structured records.
- Recover through rollback target selection, local replanning, and checkpoint resume.
- Export inspectable traces, summaries, generated scripts, simulation records, GDS/DRC-related artifacts, and reports.

Tidy3D is the current primary simulation and optimization backend. Historical or exploratory Meep/SAX paths are not treated as the main architecture.

## Workflow

The inverse-design pipeline is organized as Step1 through Step7.

1. **Step1 Requirement Parsing**  
   Convert natural-language requests into device type, ports, wavelength range, target metrics, constraints, and optimization intent.

2. **Step2 Documentation Context**  
   Retrieve relevant Tidy3D/Flexcompute documentation and bind API rules, examples, and implementation constraints to the current task.

3. **Step3 Configuration Generation**  
   Produce structured simulation configuration, including geometry, domain, sources, monitors, objectives, optimizer settings, and runtime parameters.

4. **Step4 Validation**  
   Run hard gates for schema validity, geometry sanity, source/monitor observability, objective completeness, and cloud-cost risk.

5. **Step5 Execution and Recording**  
   Use the component-aware Tidy3D builder to run forward simulation or adjoint optimization, while persisting iteration records, checkpoints, and execution artifacts.

6. **Step6 Diagnosis and Recovery**  
   Analyze failure signatures and FOM curves, score rollback candidates, select a local replan target, and resume execution from checkpoint. Deterministic fallback is reserved for exhausted recovery budgets or low-confidence candidates.

7. **Step7 Artifact Delivery**  
   Collect optimization results, traces, summaries, generated scripts, simulation records, layout-related files, and reports.

## Architecture

The codebase is organized around explicit workflow boundaries.

- `PhotonicsAI/agents/`  
  ReAct orchestration, inverse-design agents, recovery policy, rollback target selection, and UI event streaming.

- `PhotonicsAI/tools/inverse_design_steps/`  
  Tool boundaries for Step1 through Step6, including requirement parsing, documentation context, configuration generation, validation, execution support, and recovery.

- `PhotonicsAI/tools/inverse_design_execution.py`  
  Step5 execution contract, backend dispatch, checkpoint handling, resume support, artifact path management, and cross-source consistency checks.

- `PhotonicsAI/Photon/tidy3d_runner.py`  
  Component-aware Tidy3D simulation construction, topology replay, source/monitor handling, and simulation artifact generation.

- `PhotonicsAI/Photon/webapp.py`  
  Streamlit interface and the current primary user entry point.

- `build/`  
  Runtime outputs, experiment results, temporary validation scripts, and local artifacts. This directory is not the source of system truth.

## Project Structure

```text
PhIDO/
|-- PhotonicsAI/                         # Main application package
|   |-- agents/                          # ReAct orchestration and recovery control
|   |   |-- react_loop.py                # Generic event-yielding ReAct loop
|   |   `-- pdk_agent.py                 # Inverse-design agent assembly and recovery workflow
|   |-- Photon/                          # Core application and photonics runtime modules
|   |   |-- webapp.py                    # Streamlit web application, primary UI entry point
|   |   |-- tidy3d_runner.py             # Component-aware Tidy3D simulation builder and artifacts
|   |   |-- llm_api.py                   # LLM provider integrations and model configuration
|   |   |-- utils.py                     # Shared utilities for circuit and file processing
|   |   |-- prompts.yaml                 # Prompt templates and system instructions
|   |   |-- templates.yaml               # Circuit templates and reusable configurations
|   |   |-- DemoPDK.py                   # Demo process design kit helpers
|   |   `-- drc/                         # Design rule checking support
|   |       |-- drc.py                   # DRC execution wrapper
|   |       `-- drc_script.drc           # KLayout DRC script
|   |-- KnowledgeBase/                   # Local design knowledge and component examples
|   |   |-- DesignLibrary/               # Photonic component library
|   |   `-- FDTD/                        # Reference FDTD data and examples
|   |-- tools/                           # Tool surfaces used by agents and workflows
|   |   |-- inverse_design_steps/        # Step1-Step6 callable inverse-design tools
|   |   |   |-- step1_requirements.py    # Natural-language requirement parsing
|   |   |   |-- step2_doc_context.py     # Tidy3D/Flexcompute documentation context
|   |   |   |-- step3_config_generation.py # Simulation and optimization config generation
|   |   |   |-- step4_validation.py      # Hard validation gates and constraint packets
|   |   |   |-- step5_execution.py       # Step5 execution wrapper
|   |   |   `-- mcp_surface.py           # MCP-facing helper surface
|   |   |-- inverse_design_execution.py  # Step5 execution contract, checkpoints, resume, artifacts
|   |   |-- inverse_design_config.py     # Pydantic config models and runtime configuration
|   |   |-- inverse_design_config_generation.py # Deterministic/LLM-assisted config builder
|   |   |-- inverse_design_config_validation.py # Config validation implementation
|   |   |-- inverse_design_failure_diagnosis.py # Failure signatures and repair candidates
|   |   |-- inverse_design_optimizer_attribution.py # Optional Step6 optimizer attribution helper
|   |   |-- inverse_design_replan.py     # Structured patch and local replan support
|   |   `-- inverse_design_working_memory.py # Recovery memory and scenario records
|   `-- config.py                        # Application-level configuration
|-- scripts/                             # Utility scripts and legacy-compatible helpers
|-- requirements.txt                     # Runtime dependency set
|-- pyproject.toml                       # Python project metadata and tool configuration
|-- Makefile                            # Convenience commands where available
|-- README.md                           # Project overview and setup guide
`-- build/                              # Generated runtime artifacts, ignored by default
```

Local planning documents, staged experiments, test fixtures, and large simulation outputs may exist in a developer workspace, but they are not part of the public source layout unless intentionally curated.

## Core Principles

- **Tidy3D-first execution**: production simulation and optimization should use the unified component-aware Tidy3D builder.
- **Single source of truth**: requirements, configuration, optimizer parameters, working directories, and checkpoints should be explicit rather than hardcoded.
- **Recoverable optimization**: Step6 follows diagnosis, rollback candidate scoring, local replanning, and checkpoint resume.
- **Traceable data**: iteration records should identify whether values come from optimizer simulation, diagnostic simulation, or derived metrics.
- **System-level fixes**: improvements should generalize beyond a single device, experiment, or case.
- **Diagnosis first**: statistical helpers such as scikit-learn are advisory and only used when enough CSV samples exist. Failure diagnosis remains the primary decision path.

## Installation

Use Python 3.11 or later in an isolated environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If local photonics or cloud-simulation dependencies require additional platform setup, install them into the same environment used to run the application.

## Configuration

Create a local `.env` file for credentials and runtime settings. Do not commit secrets.

Common variables:

```text
ZHIPU_API_KEY=...
OPENAI_API_KEY=...
TIDY3D_API_KEY=...
```

Real Tidy3D cloud execution requires `TIDY3D_API_KEY` and available account credit. Full optimization runs may consume FlexCredit, so local validation should be completed before submitting cloud jobs.

## Run

From the repository root:

```powershell
$env:PYTHONPATH='C:\Users\PC\Desktop\OPTI-AI'
streamlit run PhotonicsAI/Photon/webapp.py
```

The Streamlit Web App is the primary interactive entry point. Notebooks may be used for assembly, visualization, or one-off inspection, but they should not become the long-term source of configuration truth.

## Validation

Documentation-only edits usually do not require runtime tests. For code changes, run the narrowest relevant checks first, then broaden validation based on the affected workflow surface.

Useful local checks:

```powershell
python -m py_compile PhotonicsAI\Photon\webapp.py
python -m pytest -q
```

Real Step5/Step6 cloud validation should only be run when credentials, account state, and expected cost are clear.

## Outputs and Repository Hygiene

Runtime artifacts include simulation traces, JSONL/CSV summaries, plots, generated scripts, GDS files, and diagnostic reports. These are usually local outputs and should not enter version control unless deliberately curated as public examples.

Local or ignored content typically includes:

- `.env` and other secret-bearing configuration.
- `build/` experiment outputs and generated artifacts.
- Large HDF5, GDS, PNG, Tidy3D viewer, trace, and runtime files.
- Local planning, memory-bank, staged experiment records, and temporary validation documents.

## Current Boundaries

PhIDO is under active development. The most important engineering boundary is to keep Step1 through Step7 coherent: use explicit configuration, share the same component-aware simulation builder, and record enough evidence for failure diagnosis and recovery.

New capabilities should strengthen reusable system mechanisms rather than add single-case patches.
