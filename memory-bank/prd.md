# PhIDO 产品需求文档（PRD）

> **Photonics Intelligent Design & Optimization**
> 利用大型语言模型自动化光子集成电路设计的智能网络应用

| 字段 | 值 |
|:---|:---|
| 版本 | v1.0.0 |
| 状态 | Draft |
| 作者 | PhotonicsAI Team |
| 创建日期 | 2026-03-22 |
| 最后更新 | 2026-03-22 |

---

## 1. 产品概述

### 1.1 愿景

PhIDO 是一款基于 Streamlit 的智能网络应用，通过大型语言模型（LLMs）⾃动化光子集成电路（PIC）的完整设计流程——从自然语言规格描述到版图生成和设计规则检查（DRC），将设计周期从数周压缩到数小时。

### 1.2 目标用户

| 用户角色 | 典型场景 | 核心诉求 |
|:---|:---|:---|
| 光子芯片设计工程师 | 快速原型验证、参数扫描 | 快速获得可制造版图与仿真结果 |
| 科研人员 | 论文复现、新器件探索 | 低门槛生成 GDS，对接仿真工具 |
| 教学/学生 | 学习光子电路设计流程 | 引导式工作流、可视化反馈 |
| PDK 维护者 | 扩展元件库、验证设计规则 | 结构化元件管理与 DRC 脚本集成 |

### 1.3 核心价值主张

1. **自然语言驱动**：用户用自然语言描述电路需求，LLM 自动提取规格
2. **全流程覆盖**：规格 → 元件选型 → 电路原理图 → 版图 → 仿真 → DRC
3. **多 LLM 灵活切换**：支持智谱 GLM / 阿里通义 / OpenAI / Anthropic / Google Gemini
4. **工业级输出**：直接输出 GDSII 制造文件，对接 KLayout DRC

### 1.4 当前系统根本问题（需优先修复）

#### P0-1 电路路由与拓扑稳定性不足

- LLM 生成的 DOT 不稳定，后处理清洗成本高
- 节点 ID、端口 ID 命名不一致（大小写/格式混用）
- 边提取依赖脆弱正则，存在连接信息丢失风险
- 当前偏重几何检查，缺少电路级拓扑正确性验证

#### P0-2 新组件生成能力不足

- 现状高度依赖静态模板库，库外需求难以落地
- `auto_*` 组件多数为参数提取，不等同于可验证的新结构生成
- 部分“逆设计”命名组件缺少实际优化闭环

#### P0-3 约束来源不统一

- 约束值混合人工经验与默认值，来源不可追溯
- 目标应统一为三类来源：代工厂 DRC 规则、物理公式、用户性能需求

---

## 2. 功能需求

### 2.1 工作流模式

系统仅保留**自动引导主线路**，移除分步/分布式残留线路：

#### 2.1.1 自动引导工作流（Automatic Workflow）

全自动五步流水线，适合标准设计场景：

```
用户自然语言输入
    ↓
[Step 1] 实体提取（Entity Extraction）
    ↓
[Step 2] 元件选型（Component Selection）
    ↓
[Step 3] 原理图/DSL 生成（Schematic Generation）
    ↓
[Step 4] 版图生成 & 仿真（Layout & Simulation）
    ↓
[Step 5] 设计规则检查（DRC）
    ↓
输出：GDS 文件 + 仿真报告 + DRC 报告
```

**关键行为：**
- 各步骤自动衔接，失败时提供重试机制
- 每步输出实时展示在 Web UI 上
- 全流程计时与 Token 消耗追踪


---

### 2.2 电路设计自动化

#### FR-2.2.1 实体提取（Entity Extraction）

| 项目 | 描述 |
|:---|:---|
| 输入 | 用户自然语言描述（中/英文） |
| 处理 | LLM 解析输入，提取器件类型、数量、连接关系、性能指标 |
| 输出 | 结构化 JSON/YAML：器件列表、参数约束、拓扑关系 |
| 验证 | 使用 Pydantic schema 验证输出格式完整性 |

**示例输入：**
> "设计一个 2×2 马赫-曾德尔干涉仪，带 TiN 加热器，工作波长 1550nm"

**示例输出：**
```yaml
components:
  - type: mzi_2x2
    heater: TiN
    wavelength: 1550nm
    ports: [input_1, input_2, output_1, output_2]
```

#### FR-2.2.2 智能元件匹配（Component Selection）

| 项目 | 描述 |
|:---|:---|
| 知识库 | 180+ 光子元件（涵盖定向耦合器、光栅耦合器、MMI、Y 分支、MZI、环形谐振器、相移器、PBS/PBR、波导交叉等） |
| 匹配策略 | BM25 + 语义嵌入（sentence_transformers）混合检索 |
| 候选排序 | LLM 根据提取参数二次评分与筛选 |
| 兜底机制 | 若知识库无精确匹配，LLM 基于 gdsfactory API 动态生成新元件代码 |

#### FR-2.2.5 新元件生成与入库闭环（重点新增）

当知识库低命中或未命中时，系统必须执行以下闭环流程：

```
用户请求
    ↓
解析意图
    ↓
检查缓存（命中直接返回）
    ↓（未命中）
多源搜索 + 交叉验证
    ↓
应用约束（DRC + 物理 + 用户目标）
    ↓
生成组件 + 敏感性分析
    ↓
仿真验证（可选）
    ↓
返回结果 + 写入缓存
    ↓
采集用户反馈并调整模板权重
```

**约束数据来源要求（必须可追溯）：**

| 数据类型 | 来源 |
|:---|:---|
| 组件模板 | gdsfactory 内置 + LLM 多源检索结果 |
| 制造约束 | Foundry DRC 规则文件/工艺手册 |
| 物理约束 | 光学公式（如 MMI 自成像、耦合长度关系等） |
| 性能目标 | 用户输入规格（插损、带宽、ER、尺寸等） |
| 参数参考 | 论文历史数据 + 本地缓存积累 |

**输出要求：**

- 输出可实例化组件定义（可进入布局编译）
- 输出约束应用报告（参数如何被限制/调整）
- 输出敏感性分析摘要（关键参数影响排序）
- 若生成失败，必须返回明确回退路径与原因

**元件库来源分类：**

| 来源 | 命名前缀 | 数量（约） | 说明 |
|:---|:---|:---|:---|
| 论文自动提取 | `auto_*` | 130+ | 从学术论文中提取参数并自动生成 |
| 手工设计 | 无固定前缀 | 50+ | 人工编写的高质量参考元件 |
| gdsfactory 原生 | `_*` | 10+ | gdsfactory 内置元件的薄封装 |

#### FR-2.2.3 自动路由与电路连接

| 项目 | 描述 |
|:---|:---|
| DSL 生成 | LLM 输出电路 DSL（YAML 格式），定义元件实例与端口连接 |
| 拓扑验证 | 检查端口匹配、悬空端口、连接回环 |
| 波导路由 | gdsfactory 内置路由器生成低损耗波导连接 |
| 可视化 | 生成 DOT 图展示电路拓扑 |

#### FR-2.2.4 布局生成

| 项目 | 描述 |
|:---|:---|
| 输出格式 | GDSII (`.gds`) |
| 后端引擎 | `gdsfactory` >= 8.18 |
| 层系统 | 兼容标准 PDK 层定义；自动处理层号越界问题 |
| 导出选项 | GDS 文件下载 + PNG 预览 |

---

### 2.3 仿真集成（自动主线路）

#### FR-2.3.1 FDTD 全波仿真（Meep）

| 项目 | 描述 |
|:---|:---|
| 主引擎 | Meep（本地 FDTD） |
| 备选引擎 | 无（当前版本关闭分布式/云仿真线路） |
| 适用场景 | 单元件验证、耦合效率分析、模态求解 |
| 集成方式 | `meep_runner.py` 封装，接受 gdsfactory 几何输入 |
| 输出 | 场分布图、S 参数、仿真日志 |

---

### 2.4 设计规则检查（DRC）

#### FR-2.4.1 KLayout DRC 集成

| 项目 | 描述 |
|:---|:---|
| 引擎 | KLayout（批处理模式 `-b`） |
| 规则脚本 | `drc_script.drc`（可扩展） |
| 检查范围 | 最小线宽、最小间距、层级合规、几何自洽 |

#### FR-2.4.2 DRC 报告

| 项目 | 描述 |
|:---|:---|
| 报告格式 | KLayout `.lydrb` 报告 + Web UI 可视化摘要 |
| 违规分类 | 按严重等级（Error / Warning）与层级分组 |
| 反馈方式 | 违规列表 + 位置坐标 + 建议修复方向 |

#### FR-2.4.3 层兼容性

- 自动检测 GDS 层号是否在目标 PDK 合法范围内
- 超出范围时提供层号映射回退策略
- 记录层号冲突日志供用户复核

---

### 2.5 模板工作流

| 项目 | 描述 |
|:---|:---|
| 模板来源 | `templates.yaml` 与 `prompts.yaml` |
| 可用模板 | MZI + 加热器、环形谐振器、定向耦合器等常见拓扑 |
| 参数化 | 用户可修改长度、宽度、耦合间距、加热器规格等参数 |
| 扩展方式 | 新增模板：在 `templates.yaml` 添加条目 + 在 `prompts.yaml` 补充对应提示词 |

---

### 2.6 LLM 集成

#### 支持模型矩阵

| 供应商 | 模型 | 用途建议 |
|:---|:---|:---|
| 阿里云百炼 | glm-4.7, qwen-plus/turbo/max/long | 默认推荐，中文优化 |
| 智谱 AI | glm-4, glm-4-flash, chatglm_turbo/pro/std | 实体提取格式化（Pydantic） |
| OpenAI | gpt-4o, o1, o3-mini | 英文场景、复杂推理 |
| Anthropic | claude-3-7-sonnet, claude-opus-4 | 长上下文、代码生成 |
| Google | gemini-2.5-pro, gemini-1.5-pro/flash, gemini-2.0-flash | 多模态、高速 |

#### 关键设计

- 所有步骤共享统一模型选择器，也支持逐步配置不同模型
- Token 用量按 session 追踪（输入/输出/缓存分别计数）
- 调用失败时指数退避重试（`backoff` 库）
- API Key 通过 `.env` 文件注入，不硬编码

---

## 3. 非功能需求

### 3.1 性能

| 指标 | 目标 |
|:---|:---|
| 实体提取延迟 | < 10s（取决于 LLM 响应） |
| 版图生成 | < 30s（中等复杂度电路） |
| DRC 执行 | < 60s（超时中断） |
| 页面首次加载 | < 5s（Streamlit 冷启动后） |

### 3.2 可靠性

- LLM 调用失败自动重试（最多 3 次）
- 仿真超时保护（可配置）
- 每步输出持久化到 `log/` 目录，支持断点恢复
- 缓存命中优先返回，避免重复搜索与重复生成
- 多源搜索结果需交叉验证，降低单源幻觉风险
- 约束应用失败时返回可读诊断，不允许静默降级

### 3.3 安全

- API Key 仅存储在 `.env`，不进入版本控制
- 无用户认证（当前为本地/内网部署场景）
- GDS 文件写入限定在 `build/` 目录

### 3.4 可维护性

- 代码质量：`ruff` lint + `pytest` 测试
- Python 3.11+ 类型提示
- 模块边界清晰：`llm_api` / `utils` / `drc` / 仿真 runner 各自独立

### 3.5 兼容性

| 环境 | 要求 |
|:---|:---|
| Python | >= 3.11 |
| OS | Ubuntu/Debian（推荐）、Windows（WSL 或原生） |
| 浏览器 | Chrome / Firefox / Edge（Streamlit 兼容） |
| KLayout | 需系统安装（DRC 功能） |

---

## 4. 系统架构

### 4.1 技术栈

```
┌─────────────────────────────────────────────────┐
│                 Web UI (Streamlit)               │
├─────────────────────────────────────────────────┤
│            Workflow Engine (webapp.py)            │
├──────────┬──────────┬──────────┬────────────────┤
│ LLM API  │ Component│ Layout   │ Simulation     │
│ Layer    │ KB       │ Engine   │ Engine         │
│          │          │          │                │
│ ·智谱    │ ·Design  │·gdsfact- │ ·Meep (FDTD)   │
│ ·OpenAI  │  Library │ ory 8.18 │ ·Meep (FDTD)   │
│ ·Claude  │ ·BM25    │·GDS 导出 │ ·(自动单线路)  │
│ ·Gemini  │ ·Embed   │          │                │
│ ·Qwen    │  Search  │          │                │
├──────────┴──────────┴──────────┴────────────────┤
│              DRC Engine (KLayout -b)             │
├─────────────────────────────────────────────────┤
│        Config / .env / prompts.yaml              │
└─────────────────────────────────────────────────┘
```

### 4.2 目录结构

```
PhotonicsAI/
├── Photon/
│   ├── webapp.py              # Streamlit 主入口 & 工作流引擎
│   ├── llm_api.py             # 多 LLM 统一调用层
│   ├── utils.py               # 工具函数（文档提取、搜索、格式转换）
│   ├── meep_runner.py         # Meep FDTD 仿真封装
│   ├── fdtd_runner.py         # 通用 FDTD 接口
│   ├── DemoPDK.py             # 演示 PDK 定义
│   ├── prompts.yaml           # LLM 提示词模板
│   ├── templates.yaml         # 电路模板配置
│   └── drc/
│       ├── drc.py             # DRC 执行引擎
│       └── drc_script.drc     # KLayout DRC 规则脚本
├── KnowledgeBase/
│   ├── DesignLibrary/         # 180+ 光子元件 Python 文件
│   └── FDTD/                  # 预计算仿真数据
├── config.py                  # 路径 & 全局配置
└── __init__.py
```

### 4.3 数据流

```
用户输入 (自然语言)
  │
  ├─[LLM] 实体提取 ──→ 结构化 JSON/YAML
  │
  ├─[LLM+KB] 元件匹配 ──→ gdsfactory 元件实例列表
  │    └─ 未命中时 → LLM 动态生成元件代码
  │
  ├─[LLM] DSL 生成 ──→ 电路拓扑 YAML + DOT 图
  │
    ├─[gdsfactory] 版图编译 ──→ .gds 文件
    │    └─ [Meep] 仿真 ──→ S 参数 / 场图
  │
  └─[KLayout] DRC ──→ 报告 (.lydrb) + UI 摘要
```

---

## 5. 外部依赖

### 5.1 核心 Python 依赖

| 包 | 版本约束 | 用途 |
|:---|:---|:---|
| gdsfactory | ~=8.18.1 | 版图引擎 |
| gplugins | ~=1.1.2 | gdsfactory 生态插件（非 SAX 主链路） |
| streamlit | latest | Web UI |
| fastapi + uvicorn | latest | API 服务（备选部署） |
| openai | ==1.43.0 | OpenAI API 客户端 |
| anthropic | latest | Claude API 客户端 |
| google-generativeai | latest | Gemini API 客户端 |
| zhipuai | latest | 智谱 AI SDK |
| sentence_transformers | latest | 语义嵌入检索 |
| rank_bm25 | latest | BM25 关键词检索 |
| torch + transformers | latest | 本地模型推理（可选） |
| pydantic | latest | 输出 schema 验证 |
| backoff | latest | API 重试 |

### 5.2 系统级依赖

| 工具 | 用途 | 安装方式 |
|:---|:---|:---|
| KLayout | DRC 执行引擎 | `apt install klayout` |
| Graphviz | DOT 图渲染 | `apt install graphviz libgraphviz-dev` |
| Meep | 本地 FDTD | conda / pip |

---

## 6. API Key 与环境变量

| 变量名 | 必需 | 说明 |
|:---|:---|:---|
| `ZHIPU_API_KEY` | **是** | 智谱 AI（实体提取格式化依赖） |
| `DASHSCOPE_API_KEY` | 否 | 阿里云百炼（Qwen 系列） |
| `OPENAI_API_KEY` | 否 | OpenAI 模型 |
| `ANTHROPIC_API_KEY` | 否 | Claude 模型 |
| `GOOGLE_API_KEY` | 否 | Gemini 模型 |

> 至少配置 `ZHIPU_API_KEY` + 一个其他供应商 Key 即可启动完整流程。

---

## 7. 用户故事与验收标准

### US-1：自然语言驱动的全自动电路设计

**作为**光子设计工程师，**我想要**输入一段电路描述后自动获得 GDS 版图，**以便**快速验证设计思路。

**验收标准：**
- [ ] 输入 "Design a 1×2 MZI with TiN heaters at 1550nm"，5 步自动完成
- [ ] 输出 `.gds` 文件可在 KLayout 中正确打开
- [ ] 每步耗时与 token 用量在页面上实时展示
- [ ] 全流程 < 3 分钟（网络正常时）

### US-2：智能元件匹配与缺失元件生成

**作为**科研人员，**我想要**系统在知识库中找不到精确元件时自动生成新元件，**以便**我不被现有库限制。

**验收标准：**
- [ ] 对库中已有元件（如 `mzi_2x2_heater_tin_cband`），返回匹配结果并展示参数
- [ ] 对库中不存在的元件（如 "异质集成 InP 激光器"），LLM 生成 gdsfactory 兼容代码
- [ ] 生成的代码可通过 `gf.Component` 实例化并输出合法 GDS
- [ ] 新元件生成流程满足“缓存→多源→约束→生成→分析→（可选）仿真→回写缓存”闭环
- [ ] 结果中包含约束来源与参数调整依据
- [ ] 用户反馈可记录并影响后续模板权重

### US-7：新元件闭环生成可追溯

**作为**器件研发工程师，**我想要**系统在库外需求下给出可追溯的新元件生成过程，**以便**确认结果可信且可复用。

**验收标准：**
- [ ] 生成结果附带多源证据摘要（至少 2 个来源）
- [ ] 约束应用前后参数差异可查看
- [ ] 缓存命中时直接返回并标注命中版本
- [ ] 用户评分后，下一次同类请求候选排序发生可解释变化

### US-3：Meep FDTD 仿真

**作为**器件设计者，**我想要**对关键元件执行 FDTD 仿真，**以便**验证耦合效率和传输特性。

**验收标准：**
- [ ] 选择 Meep 作为仿真后端时，自动从 gdsfactory 几何生成 Meep 仿真脚本
- [ ] 仿真完成后输出场分布图和 S 参数
- [ ] 仿真超时（默认 60s）时优雅中断并给出提示

### US-4：DRC 自动检查

**作为**流片准备工程师，**我想要**版图生成后自动运行 DRC，**以便**提前发现制造违规。

**验收标准：**
- [ ] 版图生成后自动调用 KLayout DRC
- [ ] 违规报告在 Web UI 上以表格形式展示（位置、类型、严重等级）
- [ ] 无违规时明确提示 "DRC 通过"
- [ ] KLayout 未安装时给出清晰错误提示而非崩溃

### US-5：模板快速启动

**作为**新用户，**我想要**从预定义模板开始设计，**以便**快速理解系统能力。

**验收标准：**
- [ ] 模板列表可在 UI 上浏览
- [ ] 选择模板后参数可编辑（长度、宽度、耦合间距等）
- [ ] 一键启动后进入正常工作流

---

## 8. 版本路线图

### v1.0（当前）—— 基础全流程

- [x] 五步自动引导工作流
- [x] 多 LLM 供应商支持（智谱 / OpenAI / Claude / Gemini / Qwen）
- [x] 180+ 元件知识库
- [x] gdsfactory 版图生成
- [x] Meep 电路仿真主线路
- [x] KLayout DRC 集成
- [x] Token 用量追踪

### v1.1 —— 仿真增强

- [ ] Meep FDTD 一键仿真流程打通
- [ ] 仿真结果嵌入式可视化（mpld3 交互图）
- [ ] 参数扫描（波长 / 几何扫描）支持

### v1.2 —— 智能增强

- [ ] 知识库自动扩充：从论文 PDF 自动提取器件参数并生成元件代码
- [ ] 多轮对话式设计：用户可在任意步骤通过对话修正设计
- [ ] 设计历史管理：保存/加载/对比历史设计
- [ ] 新元件生成闭环（缓存、交叉验证、约束融合、反馈学习）

### v2.0 —— 生产就绪

- [ ] 多用户支持与认证
- [ ] FastAPI 后端 + Streamlit 前端分离部署
- [ ] PDK 管理系统（多 PDK 切换）
- [ ] CI/CD 自动化测试（pytest + GDS 回归）
- [ ] 云端部署方案（Docker compose）

---

## 9. 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|:---|:---|:---|:---|
| LLM 输出不稳定（幻觉） | 生成错误元件参数或无效代码 | 高 | Pydantic schema 验证 + 重试机制 + 人工复核步骤 |
| LLM API 不可用/限流 | 工作流中断 | 中 | 多供应商兜底 + 指数退避重试 + 本地模型降级 |
| DOT/端口命名不一致 | 路由或拓扑解析失败 | 高 | 统一命名规范 + 结构化解析替代正则 |
| 单源检索偏差 | 新元件参数不可靠 | 高 | 多源交叉验证 + 证据评分 |
| KLayout 系统未安装 | DRC 功能不可用 | 中 | 启动时检测并提示；DRC 为可选步骤 |
| GDS 层号越界 | 版图导出失败 | 低 | 自动层号映射回退 + 日志记录 |
| 知识库元件参数过时 | 匹配质量下降 | 中 | 建立元件版本管理 + 定期校验 |
| 仿真耗时过长 | 用户体验差 | 中 | 超时保护 + 进度条 + 异步执行 |

---

## 10. 成功度量

| 指标 | 目标（v1.0） | 测量方式 |
|:---|:---|:---|
| 全流程成功率 | >= 80%（标准模板场景） | 自动化测试 + 日志统计 |
| 实体提取准确率 | >= 90%（常见器件） | 与人工标注对比 |
| DRC 首次通过率 | >= 70% | DRC 报告分析 |
| 平均端到端耗时 | < 3 分钟 | 日志计时 |
| 用户使用模板数 | 跟踪 top-5 热门模板 | 应用日志 |

---

## 11. 术语表

| 术语 | 定义 |
|:---|:---|
| PIC | Photonic Integrated Circuit，光子集成电路 |
| GDS / GDSII | 版图交换标准格式 |
| DRC | Design Rule Check，设计规则检查 |
| PDK | Process Design Kit，工艺设计包 |
| MZI | Mach-Zehnder Interferometer，马赫-曾德尔干涉仪 |
| MMI | Multimode Interference，多模干涉 |
| PBS / PBR | Polarization Beam Splitter / Rotator，偏振分束/旋转器 |
| FDTD | Finite-Difference Time-Domain，时域有限差分 |
| DSL | Domain Specific Language，领域特定语言 |
| LLM | Large Language Model，大型语言模型 |
| BM25 | Best Matching 25，经典关键词检索算法 |

---

## 附录 A：快速启动命令

```bash
# 安装依赖
pip install -r requirements.txt
pip install kfactory==0.21.1

# 配置环境变量
cp .env.example .env   # 编辑并填入 API Key

# 创建日志目录
mkdir -p PhotonicsAI/log

# 设置 PYTHONPATH
export PYTHONPATH='.'

# 启动应用
streamlit run PhotonicsAI/Photon/webapp.py
```

## 附录 B：相关文档

| 文档 | 位置 | 说明 |
|:---|:---|:---|
| 用户教程 | `GETTING_STARTED.md` | 逐步使用指南与示例 |
| Agent 约束 | `AGENTS.md` | AI Agent 行为规范 |
| 变更日志 | `CHANGELOG.md` | 版本变更记录 |
| 工作流详情 | `PhIDO_Complete_Workflow_Details.md` | 五步工作流技术细节 |
