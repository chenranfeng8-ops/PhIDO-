# 缺失组件识别与入库工作流 - 实现检查清单

> 本文档标记当前已实现的工作流各阶段代码位置，用于快速定位与开发验收。

---

## 📋 工作流全景

```
缺失组件识别 → 关键词提取 → 论文搜索 → 论文排序 
    ↓
参数提取 → LLM聚合 → 组件模板生成 → Web UI集成
```

---

## ✅ 已实现模块详情

### 第一阶段：关键词提取与设备类型映射

| 功能 | 文件 | 位置 | 状态 | 说明 |
|------|------|------|------|------|
| 设备类型映射 | `scripts/auto_pdk_generator.py` | 全局 `DEVICE_keywords` 字典 | ✅ | 预定义设备类型和关键词关 |
| LLM关键词生成 | `scripts/auto_pdk_generator.py#Line~300-350` | `_generate_search_keywords_with_llm()` | ✅ | 动态生成搜索关键词 |
| 关键词降级策略 | `scripts/auto_pdk_generator.py#Line~1020-1025` | `discover_and_generate()` 中Step 2 | ✅ | LLM失败→硬编码→组件名 |

### 第二阶段：多源论文爬虫搜索

| 功能 | 文件 | 位置 | 状态 | 说明 |
|------|------|------|------|------|
| ArXiv爬虫 | `scripts/auto_pdk_generator.py` | `search_arxiv()` | ✅ | Selenium爬取ArXiv论文 |
| Google Scholar爬虫 | `scripts/auto_pdk_generator.py` | `search_google_scholar()` | ✅ | Selenium爬取Google Scholar谷歌学术 |
| Optica/OSA爬虫 | `scripts/auto_pdk_generator.py` | `search_optica()` | ✅ | Selenium爬取Optica官方库 |
| 多源检索编排 | `scripts/auto_pdk_generator.py#Line~1026-1065` | `discover_and_generate()` 中Step 3 | ✅ | 按优先级调用ArXiv→Scholar→Optica |
| 去重逻辑 | `scripts/auto_pdk_generator.py#Line~1066-1075` | 按标题去重 | ✅ | 避免重复论文 |

### 第三阶段：论文质量评判与排序

| 功能 | 文件 | 位置 | 状态 | 说明 |
|------|------|------|------|------|
| LLM论文排序 | `scripts/auto_pdk_generator.py` | `_rank_papers_with_llm()` | ✅ | 请优质论文，返回评分与原因 |
| 质量评分 | `scripts/auto_pdk_generator.py#Line~1076-1085` | `discover_and_generate()` 中Step 3.5 | ✅ | 限制处理Top N论文 |
| 排序详情保存 | `scripts/auto_pdk_generator.py#Line~1090-1100` | `result["paper_rankings"]` | ✅ | 记录论文标题、来源、评分、理由 |

### 第四阶段：参数提取与聚合

| 功能 | 文件 | 位置 | 状态 | 说明 |
|------|------|------|------|------|
| Zhipu AI参数提取 | `scripts/auto_pdk_generator.py#Line~377-450` | `extract_params_with_zhipuai()` | ✅ | 需`ZHIPUAI_API_KEY`环境变量 |
| 启发式参数提取 | `scripts/auto_pdk_generator.py` | `extract_params_heuristic()` | ✅ | 无API时的降级方案，220nm SOI参数 |
| LLM参数聚合 | `scripts/auto_pdk_generator.py` | `_aggregate_params_with_llm()` | ✅ | 综合多篇论文生成共识参数 |
| 参数验证 | `scripts/auto_pdk_generator.py#Line~410-425` | 类型约束检查 | ✅ | 使用`type_validator`检查范围 |
| 单论文提取 | `scripts/auto_pdk_generator.py#Line~1101-1103` | `discover_and_generate()` 中Step 4 | ✅ | 仅一篇论文时的处理 |
| 默认参数降级 | `scripts/auto_pdk_generator.py#Line~1104-1106` | 无论文时的降级 | ✅ | 使用启发式默认参数 |

### 第五阶段：组件模板生成

| 功能 | 文件 | 位置 | 状态 | 说明 |
|------|------|------|------|------|
| 主函数入口 | `scripts/auto_pdk_generator.py#Line~961-1150` | `discover_and_generate(component_name, max_papers=8)` | ✅ | 完整的发现→生成流程 |
| 流程编排 | `scripts/auto_pdk_generator.py#Line~975-1145` | 5步走完整流程 | ✅ | 映射→关键词→爬虫→排序→生成 |
| 代码模板库 | `scripts/auto_pdk_generator.py` | 全局 `TEMPLATES` 字典 | ✅ | 按device_type存储代码框架 |
| 文件生成 | `scripts/auto_pdk_generator.py#Line~1114-1140` | `discover_and_generate()` 中Step 5 | ✅ | 保存为`auto_{device_type}_consensus.py` |
| 来源标注 | `scripts/auto_pdk_generator.py#Line~1120-1124` | 代码注释中包含论文来源 | ✅ | 可追溯参数来源 |

### 第六阶段：Web UI集成

| 功能 | 文件 | 位置 | 状态 | 说明 |
|------|------|------|------|------|
| 自动PDK生成Phase | `PhotonicsAI/Photon/webapp.py#Line~2385-2405` | `Phase A.0: Auto-PDK Mode Detection` | ✅ | 检测缺失组件并切换到自动生成 |
| 目标组件识别 | `PhotonicsAI/Photon/webapp.py#Line~2308-2350` | 快速关键词匹配+库检索 | ✅ | `quick_keyword_match()` 函数 |
| 组件库检索 | `PhotonicsAI/Photon/webapp.py#Line~2326-2365` | 组件搜索逻辑 | ✅ | 遍历DesignLibrary检查匹配度 |
| Discovery执行 | `PhotonicsAI/Photon/webapp.py#Line~2420-2445` | `auto_pdk_generator.discover_and_generate()` 调用 | ✅ | 触发完整发现与生成流程 |
| 结果展示 | `PhotonicsAI/Photon/webapp.py#Line~2437-2460` | 论文数量、设备类型、参数、置信度  | ✅ | Streamlit结果展示 |
| 下一阶段转换 | `PhotonicsAI/Photon/webapp.py#Line~2447-2456` | 成功→优化Phase，失败→回退选择 | ✅ | `session.automatic_phase` 状态机 |

### 第七阶段：参数提取指南与文档

| 功能 | 文件 | 位置 | 状态 | 说明 |
|------|------|------|------|------|
| 论文提取规范 | `docs/paper_extraction_template.md#Line~251-300` | "提取完成后" 部分 | ✅ | 参数范围、默认值、端口定义 |
| 组件代码模板 | `docs/paper_extraction_template.md#Line~260-290` | Python代码框架 | ✅ | gdsfactory集成示例 |
| 元数据定义 | `docs/paper_extraction_template.md` | 文件头部分 | ✅ | Name、Description、Args、Reference |

---

## 📍 关键代码位置速查

### 快速导航

```plaintext
├─ 第1阶段：关键词生成
│  └─ scripts/auto_pdk_generator.py ~300-350    (_generate_search_keywords_with_llm)
│  └─ scripts/auto_pdk_generator.py ~1020-1025  (discover_and_generate中Step 2)
│
├─ 第2阶段：论文爬虫
│  └─ scripts/auto_pdk_generator.py             (search_arxiv/scholar/optica)
│  └─ scripts/auto_pdk_generator.py ~1026-1065  (discover_and_generate中Step 3)
│
├─ 第3阶段：论文排序
│  └─ scripts/auto_pdk_generator.py             (_rank_papers_with_llm)
│  └─ scripts/auto_pdk_generator.py ~1076-1085  (discover_and_generate中Step 3.5)
│
├─ 第4阶段：参数提取
│  └─ scripts/auto_pdk_generator.py ~377-450    (extract_params_with_zhipuai)
│  └─ scripts/auto_pdk_generator.py             (extract_params_heuristic)
│  └─ scripts/auto_pdk_generator.py             (_aggregate_params_with_llm)
│  └─ scripts/auto_pdk_generator.py ~1101-1106  (discover_and_generate中Step 4)
│
├─ 第5阶段：模板生成
│  └─ scripts/auto_pdk_generator.py ~961-1150   (discover_and_generate主函数)
│  └─ scripts/auto_pdk_generator.py ~1114-1140  (discover_and_generate中Step 5)
│
├─ 第6阶段：Web UI集成
│  └─ PhotonicsAI/Photon/webapp.py ~2308-2350   (组件检索策略)
│  └─ PhotonicsAI/Photon/webapp.py ~2402-2480   (Phase A.1: Discovery执行)
│
└─ 第7阶段：文档与指南
   └─ docs/paper_extraction_template.md ~251-300
```

---

## 🔄 完整工作流演示

用户输入 ("Ge photodetector") 
    ↓
[webapp.py#2308] 快速关键词匹配 → 组件库检索
    ↓
[webapp.py#2385] 检测缺失 → 自动PDK模式
    ↓
[auto_pdk_generator.py#961] discover_and_generate() 入口
    ↓
[auto_pdk_generator.py#1020-1025] Step 1: 设备类型映射 + 关键词生成
    ↓
[auto_pdk_generator.py#1026-1065] Step 2-3: 多源论文爬虫 (ArXiv→Scholar→Optica)
    ↓
[auto_pdk_generator.py#1076-1085] Step 3.5: LLM论文排序 (Top 8)
    ↓
[auto_pdk_generator.py#1101-1106] Step 4: 参数提取与聚合
    ↓
[auto_pdk_generator.py#1114-1140] Step 5: ONE模板代码生成
    ↓
[webapp.py#2437-2460] 结果展示与下一阶段转换 (成功→优化/失败→回退)

```

---

## 🚀 后续扩展空间

- [ ] 组件入库验证脚本 (`component_ingest_validator.py`)
- [ ] 缺失组件对比分析报告生成
- [ ] 论文来源优先级配置化
- [ ] 参数约束更细粒度的配置
- [ ] 生成的模板自动集成测试

---

**更新时间**: 2026-03-24  
**版本**: 1.0  
**检查者**: GitHub Copilot  
