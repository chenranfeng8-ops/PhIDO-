#!/usr/bin/env python
import sys

modules = [
    ("zhipuai", "from zhipuai import ZhipuAI"),
    ("anthropic", "import anthropic"),
    ("google.generativeai", "import google.generativeai"),
    ("tiktoken", "import tiktoken"),
    ("dotenv", "from dotenv import load_dotenv"),
    ("deepseek_tokenizer", "from deepseek_tokenizer import ds_token"),
    ("PhotonicsAI.config", "from PhotonicsAI.config import PATH"),
    ("PhotonicsAI.Photon.llm_api", "from PhotonicsAI.Photon import llm_api"),
    ("PhotonicsAI.Photon.webapp", "from PhotonicsAI.Photon import utils"),
]

failed = []
for name, stmt in modules:
    try:
        exec(stmt)
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        failed.append((name, str(e)))

print()
if failed:
    print(f"FAILED: {len(failed)} modules")
    for n, e in failed:
        print(f"  - {n}: {e}")
else:
    print("ALL OK")