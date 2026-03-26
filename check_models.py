#!/usr/bin/env python
"""Check all get_model functions return correct format."""
import importlib
from pathlib import Path

pdk_path = "/mnt/c/Users/PC/Desktop/PhIDO-/PhotonicsAI/KnowledgeBase/DesignLibrary"
module_names = [f.stem for f in Path(pdk_path).glob("*.py") if f.name != "__init__.py"]

failed = []
for module_name in module_names:
    full_module_name = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
        if hasattr(module, "get_model"):
            func = module.get_model
            result = func()
            if isinstance(result, dict):
                print(f"✓ {module_name}: dict with keys {list(result.keys())[:3]}...")
            else:
                print(f"✗ {module_name}: returns {type(result).__name__}, not dict")
                failed.append(module_name)
        else:
            print(f"- {module_name}: no get_model")
    except Exception as e:
        print(f"✗ {module_name}: {type(e).__name__}: {e}")
        failed.append(module_name)

print()
if failed:
    print(f"FAILED: {len(failed)} modules")
    for f in failed:
        print(f"  - {f}")
else:
    print("ALL OK")