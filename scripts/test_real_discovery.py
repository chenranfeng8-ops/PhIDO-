# -*- coding: utf-8 -*-
"""
测试真实的动态组件生成功能（完整流程：网络爬虫 + 论文搜索 + 参数提取）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

print("=" * 70)
print("Test: Real Dynamic Component Generation (Full Pipeline)")
print("=" * 70)

# Step 1: 检查依赖
print("\n[Step 1] Checking Dependencies...")

try:
    import selenium
    print(f"  [OK] selenium version: {selenium.__version__}")
except ImportError:
    print("  [ERROR] selenium not installed. Run: pip install selenium")
    sys.exit(1)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    print("  [OK] webdriver modules available")
except ImportError as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

try:
    import gdsfactory as gf
    print(f"  [OK] gdsfactory version: {gf.__version__}")
except ImportError:
    print("  [ERROR] gdsfactory not installed")
    sys.exit(1)

try:
    from zhipuai import ZhipuAI
    print("  [OK] zhipuai available (for LLM)")
except ImportError:
    print("  [WARN] zhipuai not installed, LLM extraction may fail")

# Step 2: 检查 Chrome 浏览器
print("\n[Step 2] Checking Chrome Browser...")

try:
    from selenium.webdriver.chrome.options import Options
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # 尝试初始化 Chrome driver
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.quit()
        print("  [OK] Chrome browser and ChromeDriver ready")
    except Exception as e:
        print(f"  [WARN] ChromeDriver auto-install failed: {e}")
        print("  Trying direct Chrome launch...")
        driver = webdriver.Chrome(options=options)
        driver.quit()
        print("  [OK] Chrome browser ready (using system driver)")
        
except Exception as e:
    print(f"  [ERROR] Chrome browser check failed: {e}")
    print("  Please install Chrome browser and ChromeDriver")

# Step 3: 导入 auto_pdk_generator
print("\n[Step 3] Importing auto_pdk_generator...")

try:
    import auto_pdk_generator as apg
    print("  [OK] auto_pdk_generator imported")
    print(f"  Available templates: {list(apg.TEMPLATES.keys())}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# Step 4: 测试组件类型识别
print("\n[Step 4] Testing Component Type Recognition...")

test_components = ["y_branch_splitter", "mmi_1x2", "ring_resonator"]
for comp in test_components:
    device_type = apg._resolve_device_type(comp)
    print(f"  '{comp}' -> device_type: {device_type}")

# Step 5: 运行真实的动态生成
print("\n" + "=" * 70)
print("[Step 5] Running Real Discovery and Generation")
print("=" * 70)

# 选择一个简单的组件进行测试
component_name = "y_branch_splitter"
print(f"\nComponent: {component_name}")
print("-" * 70)

try:
    result = apg.discover_and_generate(
        component_name=component_name,
        max_papers=5  # 减少论文数量加快测试
    )
    
    print("\n[Result]")
    print(f"  Device type: {result.get('device_type', 'N/A')}")
    print(f"  Papers found: {result.get('papers_found', 0)}")
    print(f"  Parameters extracted: {result.get('params', {})}")
    print(f"  Confidence: {result.get('confidence_note', 'N/A')}")
    
    if result.get('filepath'):
        print(f"  Generated file: {result['filepath']}")
        print("  [SUCCESS] Component template generated!")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
        print("  [FAILED] Could not generate component template")
        
except Exception as e:
    import traceback
    print(f"\n[ERROR] Exception occurred: {e}")
    print(traceback.format_exc())

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)