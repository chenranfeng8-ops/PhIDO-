#!/usr/bin/env python
"""Check all DesignLibrary modules for missing function aliases."""
import importlib
import sys
from pathlib import Path

pdk_path = "/mnt/c/Users/PC/Desktop/PhIDO-/PhotonicsAI/KnowledgeBase/DesignLibrary"
module_names = [f.stem for f in Path(pdk_path).glob("*.py") if f.name != "__init__.py"]

failed = []
for module_name in module_names:
    full_module_name = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
        func = getattr(module, module_name)
        print(f"✓ {module_name}")
    except AttributeError as e:
        print(f"✗ {module_name}: {e}")
        failed.append(module_name)
    except Exception as e:
        print(f"? {module_name}: {type(e).__name__}: {e}")
        failed.append(module_name)

print()
if failed:
    print(f"FAILED: {len(failed)} modules")
    for f in failed:
        print(f"  - {f}")
else:
    print("ALL OK")