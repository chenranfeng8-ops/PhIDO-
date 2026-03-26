## 2026-03-25

### Missing Component Discovery Refactor

- Refactored `scripts/auto_pdk_generator.py` discovery pipeline into 5 explicit step functions while keeping `discover_and_generate()` as the stable facade used by web workflow integration.
- New step APIs:
	- `resolve_device_type_and_keywords(component_name)`
	- `retrieve_papers_multi_source(device_type, keywords, max_papers)`
	- `rank_and_dedup_papers(papers, device_type, top_n)`
	- `extract_or_aggregate_params(papers, device_type)`
	- `generate_template_file(device_type, params, papers, confidence_note="")`
- Architectural impact:
	- Preserved existing UI call contract (no webapp orchestrator rewrite required in this step).
	- Isolated external I/O (crawler), ranking logic, parameter strategy, and file rendering into independently callable units.
	- Added standardized step return envelope (`ok/data/error`) to improve future agent orchestration compatibility.

