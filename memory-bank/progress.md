## 2026-03-24

- Added `wsl_venv/` to `.gitignore` to stop local virtual-environment files from inflating Git/VS Code change counts.
- Added missing-component workflow entries to `.vscode/bookmarks.json` using existing workspace bookmark schema.
- Removed temporary inline `🔖 BOOKMARK` comments from source files; bookmarks are now managed only via plugin JSON.

## 2026-03-25

- Refactored `scripts/auto_pdk_generator.py` by splitting monolithic `discover_and_generate()` workflow into 5 independent step functions:
	- `resolve_device_type_and_keywords`
	- `retrieve_papers_multi_source`
	- `rank_and_dedup_papers`
	- `extract_or_aggregate_params`
	- `generate_template_file`
- Kept `discover_and_generate()` as compatibility facade so existing web integration remains unchanged.
- Standardized step-level outputs to `ok/data/error` envelopes for future agent-style orchestration.
- Verified no new file-level errors in `scripts/auto_pdk_generator.py`.
