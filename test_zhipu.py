#!/usr/bin/env python3
import os
from dotenv import load_dotenv
load_dotenv()

print("Testing ZhipuAI API...")
print(f"ZHIPUAI_API_KEY: {os.environ.get('ZHIPUAI_API_KEY', 'NOT SET')[:20]}...")

try:
    from zhipuai import ZhipuAI
    client = ZhipuAI()
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
    )
    print(f"SUCCESS! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"FAILED: {e}")