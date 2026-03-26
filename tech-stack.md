# PhIDO 技术栈建议（基于 PRD）

## 1. 目标

在满足 PRD（LLM 自动化设计 → GDS 生成 → Meep 仿真 → KLayout DRC）的前提下，给出**最简单但健壮**且仅保留自动主线路的技术栈，并与当前已使用技术栈对照。

---

## 2. 当前已有技术栈（As-Is）

基于 `pyproject.toml` 与现有代码实际使用情况整理。

### 2.1 语言与运行时

- Python `3.11+`

### 2.2 Web / 服务层

- `streamlit`（主 Web 应用）
- `fastapi` + `uvicorn`（已在依赖中，当前主流程以 Streamlit 为主）
- `sse-starlette`、`sseclient`

### 2.3 光子设计与仿真

- `gdsfactory~=8.18.1`（版图生成核心）
- `gplugins[sax,tidy3d]~=1.1.2`（历史依赖，待收敛）
- `sax==0.13.3`（历史仿真线路，待清理）
- `meep`（代码已有 runner，环境侧使用）
- `tidy3d`（历史云仿真线路，待清理）
- `KLayout`（系统级依赖，DRC 执行）

### 2.4 LLM 与 AI 相关

- `openai==1.43.0`
- `google-generativeai`
- `anthropic`（代码中使用）
- `zhipuai`（代码中使用）
- `transformers[torch]` + `torch/torchvision/torchaudio`
- `sentence_transformers`
- 其他 AI 生态依赖：`groq`、`fireworks-ai`、`replicate`

### 2.5 检索与数据处理

- `rank_bm25`
- `nltk`
- `haystack-ai`
- `numpy`、`pandas`
- `pyyaml`

### 2.6 可视化与工程质量

- `mpld3`、`pygraphviz`
- `pytest`、`pytest-cov`、`ruff`

---

## 3. 推荐：最简单但健壮技术栈（To-Be）

> 设计原则：**单体优先、依赖收敛、关键路径稳定、可渐进扩展**。

### 3.1 核心推荐栈（MVP-Production）

#### 应用层

- **Python 3.11**
- **Streamlit**（唯一前后端承载，先不拆前后端）

#### 工作流与编排

- 原生 Python 模块化（不引入额外工作流框架）
- `pydantic`（结构化输出校验）
- `backoff`（外部 API 重试）

#### PIC 设计与验证

- **gdsfactory**（版图与路由核心）
- **Meep**（唯一仿真主线路）
- **KLayout**（DRC，批处理调用）

#### 检索与知识库

- **BM25 + sentence_transformers**（保留当前有效混合检索）
- 文件系统知识库（不引入数据库，降低复杂度）

#### 新元件闭环能力（新增）

- **意图解析层**：Pydantic + 结构化 schema（统一需求字段）
- **缓存层**：SQLite（元数据）+ 文件缓存（生成结果/仿真产物）
- **多源搜索层**：本地 KB + 文献/网页检索接口（至少双源）
- **交叉验证层**：来源打分与一致性规则（参数范围、单位一致性、来源可信度）
- **约束融合层**：Foundry DRC 规则 + 物理公式 + 用户目标的统一约束求解
- **敏感性分析层**：参数扰动 + 指标变化排序
- **反馈学习层**：用户评分驱动模板权重更新

#### LLM 策略（简化且稳）

- **主模型：智谱 GLM（`zhipuai`）**
- **备份模型：OpenAI（`openai`）**
- 其他供应商（Anthropic/Gemini/Qwen/Groq/Fireworks/Replicate）改为可选插件，不作为核心运行必需

#### 工程质量

- `ruff` + `pytest`
- `.env` 管理密钥

---

## 4. 当前栈 vs 推荐栈（对照）

| 维度 | 当前状态 | 推荐方案（最简健壮） | 结论 |
|:---|:---|:---|:---|
| Web 架构 | Streamlit + FastAPI 混合依赖 | 仅 Streamlit 承载主流程 | 简化 |
| LLM 提供商 | 多供应商并行 | GLM 主用 + OpenAI 兜底 | 收敛 |
| 仿真后端 | SAX + Meep + Tidy3D | 仅 Meep 自动主线路（SAX/Tidy3D 清理） | 收敛 |
| 数据层 | 文件系统为主 | 继续文件系统（MVP 不上 DB） | 保持 |
| 检索 | BM25 + 语义检索 + haystack | BM25 + sentence_transformers，haystack 可选 | 适度收敛 |
| 组件闭环 | 静态模板为主 | 缓存→多源→约束→生成→分析→反馈 | 补强 |
| 约束来源 | 混合经验值 | DRC 文件 + 物理公式 + 用户目标 | 规范化 |
| 路由校验 | 几何校验偏多 | 几何 + 拓扑双校验 | 补强 |
| 可视化 | mpld3 + pygraphviz | 保留（满足拓扑/报告可视化） | 保持 |
| 质量保障 | ruff + pytest | 保持并强化门禁 | 保持 |

---

## 5. 最小依赖基线（建议）

> 下列为“核心运行必需”集合（建议维护 `requirements-minimal.txt`）。

- `streamlit`
- `gdsfactory~=8.18.1`
- `gplugins~=1.1.2`
- `openai==1.43.0`
- `zhipuai`
- `pydantic`
- `backoff`
- `pyyaml`
- `numpy`
- `pandas`
- `rank_bm25`
- `sentence_transformers`
- `tiktoken`
- `mpld3`
- `pygraphviz`
- `python-dotenv`
- `sqlalchemy`（或标准库 `sqlite3`，用于缓存与反馈元数据）
- `scipy`（敏感性分析与数值计算）
- `networkx`（拓扑一致性检查）

系统侧必需：
- `KLayout`
- `Meep`
- `Graphviz`

---

## 5.1 新增能力与组件映射

| 工作流环节 | 最简实现组件 | 说明 |
|:---|:---|:---|
| 解析意图 | `pydantic` | 统一请求字段，减少后续分支 |
| 检查缓存 | `sqlite3` + 本地文件目录 | 低成本、易落地、可追溯 |
| 多源搜索 | 本地 KB + 外部检索 API | 至少双源，降低单源偏差 |
| 交叉验证 | 规则打分器（Python 原生） | 比较参数一致性与来源可靠度 |
| 应用约束 | DRC 规则解析 + 物理公式模块 | 约束必须有来源 |
| 生成组件 | gdsfactory API + 模板参数化 | 保持与现有系统兼容 |
| 敏感性分析 | `scipy` / `numpy` 扰动分析 | 输出参数影响排序 |
| 仿真验证 | Meep（默认） | 自动线路单一化，降低维护复杂度 |
| 保存缓存 | SQLite + 文件落盘 | 便于命中复用 |
| 用户反馈 | SQLite 反馈表 + 权重更新策略 | 持续改进候选排序 |

---

## 5.2 必须/可选分层

### 必须（第一阶段）

- Streamlit + gdsfactory + Meep + KLayout
- `pydantic` + `backoff`
- SQLite 缓存与反馈记录
- 拓扑校验（`networkx`）

### 可选（第二阶段）

- Meep 深度仿真优化（速度/精度权衡）
- 更复杂的多源检索编排框架

---

## 6. 为什么这套“最简单但健壮”

1. **简单**：单体 Streamlit + 文件系统，不引入数据库与微服务拆分。
2. **健壮**：保留你当前最关键、已验证的链路（gdsfactory / Meep / KLayout）。
3. **可控**：LLM 供应商收敛到“主 + 备”，减少 API 差异导致的不稳定。
4. **可扩展**：后续若并发或团队协作增长，可再平滑拆分 FastAPI 后端。

---

## 7. 分阶段落地建议

### Phase 1（立即）

- 固定核心依赖版本（尤其 gdsfactory/gplugins/openai/meep）
- 建立最小可运行环境（`requirements-minimal.txt`）
- 明确主模型/备份模型策略（GLM + OpenAI）

### Phase 2（稳定化）

- 为 5 步工作流补齐回归测试（最少 1 条 happy path + 1 条异常 path）
- 统一异常与日志格式（LLM 调用、仿真、DRC）

### Phase 3（再扩展）

- 需要 API 化时再启用 FastAPI
- 需要高并发时再引入任务队列和缓存

---

## 8. 一句话结论

对于当前 PRD，**最简且健壮**的选择是：

**Python 3.11 + Streamlit + gdsfactory + Meep + KLayout +（GLM 主模型 / OpenAI 备份）+ BM25/语义混合检索 + ruff/pytest**。
