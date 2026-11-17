# Agent Memory（智能体记忆）技术调研报告

> 调研日期：2026-11-17。GitHub 数据为当日经 GitHub API 采集的一手快照；论文信息经 arXiv API 逐条核实；各系统评测成绩均标注来源——**多为厂商/论文自报、评测口径不一，不能直接横向比较**。

## TL;DR

- **问题**：LLM 本身无状态、上下文窗口有限且昂贵。记忆（memory）让 agent 跨会话保持一致、积累经验、实现个性化与长程任务。
- **技术主线**：从 2023 年"给 agent 加个存储"（memory stream / 反思 / 遗忘曲线），演进为覆盖**写入—管理—读取**全生命周期的记忆系统；当前前沿是"明文 / 激活(KV cache) / 参数"三层记忆统一调度（MemOS）与"睡眠期整理"（Letta sleep-time compute）。
- **开源格局**：[Mem0](https://github.com/mem0ai/mem0)（69.1k★）生态最广；2025 年底出现的 [Hindsight](https://github.com/vectorize-io/hindsight)（61.5k★，增长最快）提出 retain/recall/reflect 四网络架构；时序知识图谱路线（[Graphiti/Zep](https://github.com/getzep/graphiti)、[Cognee](https://github.com/topoteretes/cognee)）适合企业与时变事实；[Letta](https://github.com/letta-ai/letta)（MemGPT 后继）代表 in-context 记忆编排；编程 agent 记忆（[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) 48.8k★、[agentmemory](https://github.com/rohitg00/agentmemory) 30.4k★）是最热细分。
- **云厂商已进场**：AWS AgentCore Memory、Google Vertex AI Memory Bank、阿里 Hologres 长记忆服务（兼容 Mem0 API）把"短期会话记忆 + 长期抽取洞察"标准化。
- **评测与争议**：LoCoMo 被[社区复测](https://github.com/mem-eval-suite/LoCoMo_refined)后各家分数普跌 15–22 点；"Mem0 是否真 SOTA"之争（[Zep 博客](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)）暴露厂商互测的利益冲突；记忆还会**放大用户错误偏好（谄媚）**并被**投毒**——安全已成为 2026 年一等公民。
- **选型速查**：轻量个性化 → LangGraph store/LangMem、Memobase；客服/CRM 结构化事实 → Graphiti/Zep；复杂长程与研究 → Letta、MemOS、Cognee；托管 → Mem0 Platform、Zep Cloud、Supermemory、Hindsight Cloud、Hologres；编程 agent → CLAUDE.md/AGENTS.md + agentmemory/ai-memory。

---

## 1. 为什么 Agent 需要记忆

LLM 的"记忆"默认只存在于上下文窗口内：会话结束即失忆，窗口越长推理成本越高、注意力越稀释。Agent 要成为可靠的数字员工，必须解决四件事：

1. **跨会话一致性**：记住用户偏好、历史决策与承诺；
2. **个性化**：推荐、陪伴、教育等场景依赖用户画像的持续积累；
3. **经验复用与自我改进**：从成功/失败轨迹中沉淀可复用技能（程序记忆），而非每次从零试错；
4. **长程任务**：跨天/跨周的工作流需要外部状态承载。

**"长上下文是否会消灭记忆需求？"——答案是否定的，但价值主张在变化**：

- [LongMemEval](https://arxiv.org/abs/2410.10813)（ICLR'25）实测：商业聊天助手与长上下文 LLM 在持续交互记忆上准确率下降可达 **30%**；
- [LoCoMo](https://arxiv.org/abs/2402.17753)（ACL'24）显示长上下文 LLM 与朴素 RAG 都显著落后人类；
- [In Defense of RAG](https://arxiv.org/abs/2409.01666) 证明超长上下文存在**注意力稀释（context rot）**，答案质量随输入变长呈倒 U 型；
- **百 K token 以内 full-context 常常"更准"**（Mem0 论文同表数据亦如此）——但线性 token 成本、延迟与治理需求（权限/审计/保留策略）使其在**月级、多源**历史上不可行；[Mem0](https://arxiv.org/abs/2504.19413) 自报相对全量上下文 **token 节省 90%+、p95 延迟降 91%**；2026 年结构化记忆在准确性上也能反超（Zep 图记忆 LongMemEval 90.2% vs full-context GPT-4o 约 60%）。

**记忆与 RAG 的关系**：RAG 解决"输入知识从哪来"，记忆解决"交互经验怎么留下来"。技术同源（检索、向量、图），但记忆多了**生命周期管理**——写入决策、更新合并、遗忘、冲突消解——这是本调研的重点。

## 2. 记忆的分类体系

三篇综述给出权威框架：[A Survey on the Memory Mechanism of LLM-based Agents](https://arxiv.org/abs/2404.13501)（2024，最早系统综述）、[Rethinking Memory in LLM based Agents](https://arxiv.org/abs/2505.00675)（2025，当前引用最广）、[A Survey of Context Engineering](https://arxiv.org/abs/2507.13334)（2025，把记忆纳入上下文工程统一框架）。

### 2.1 按时间：短期 / 长期
- **短期（工作）记忆**：上下文窗口内的 scratchpad、CoT 轨迹、in-context 学习，随会话消失；
- **长期记忆**：跨会话持久化，需要显式的写入与读取机制。

### 2.2 按内容（认知科学 Tulving 三分法）
- **情景记忆（episodic）**：具体经历——"上周二用户说过他要去上海"；
- **语义记忆（semantic）**：抽象事实——用户画像、实体关系、世界事实；
- **程序记忆（procedural）**："怎么做"——技能、流程、经验教训。早期范本是 [Voyager](https://arxiv.org/abs/2305.16291) 的 skill library（可执行代码作为记忆）；2025 年 [Memp](https://arxiv.org/abs/2508.06433)（ACL'26 Findings）把程序记忆做成 Build/Retrieve/Update 全流程，并证明强模型积累的程序记忆**迁移到弱模型仍大幅涨分**。

### 2.3 按载体：参数化 vs 非参数化（+ 中间的激活层）
- **参数化**：记忆固化进权重——test-time training（[TTT 层](https://arxiv.org/abs/2407.04620)）、[Titans](https://arxiv.org/abs/2501.00663) 的神经长期记忆模块；
- **激活记忆**：KV cache——[MemOS](https://arxiv.org/abs/2507.03724) 将其列为介于权重与外部存储之间的"激活记忆"，可被 MemCube 封装复用；[EM-LLM](https://arxiv.org/abs/2407.09450)（ICLR'25）用 KV 复用实现 10M token 级"无限上下文"；
- **非参数化**：外部显式数据——非结构化文本（向量库）与结构化知识图谱。

### 2.4 按操作：写入 / 管理 / 读取
[Rethinking Memory](https://arxiv.org/abs/2505.00675) 把记忆操作原子化为六个：**Consolidation（巩固写入）、Updating、Indexing、Forgetting、Retrieval、Condensation（压缩）**。工程上归并为三段：

| 阶段 | 核心问题 | 代表技术 |
|---|---|---|
| 写入 | 写什么、何时写、怎么去重合并 | 抽取式（Mem0）、自编辑（MemGPT/Letta、Anthropic 记忆工具）、异步整理（LangMem、sleep-time compute） |
| 管理 | 如何组织、更新、遗忘、消解冲突 | 反思（Generative Agents）、遗忘曲线（MemoryBank）、图失效（Graphiti）、写留读消（DISCERN） |
| 读取 | 检什么、怎么排序、怎么注入 | 混合检索+重排、时间/重要性加权、递归摘要、按 token 预算组装 |

## 3. 代表性技术演进时间线

### 2023：范式奠基
| 工作 | 核心思想 |
|---|---|
| [Reflexion](https://arxiv.org/abs/2303.11366)（NeurIPS'23） | 语言化强化学习：失败反馈写成语言存入情景缓冲，不更新权重也能从试错中学习 |
| [Generative Agents](https://arxiv.org/abs/2304.03442)（Stanford, UIST'23） | memory stream 按**时近性/重要性/相关性**打分检索 + 定期 **reflection** 综合高层洞察——后续几乎所有记忆系统的原型 |
| [MemoryBank](https://arxiv.org/abs/2305.10250) | 引入 **Ebbinghaus 遗忘曲线**做记忆衰减/强化，面向 AI 陪伴 |
| [Voyager](https://arxiv.org/abs/2305.16291) | 可执行代码 skill library 作为程序记忆，免微调终身学习 |
| [MemGPT](https://arxiv.org/abs/2310.08560)（UC Berkeley） | 借鉴 OS 虚拟内存：主/外上下文两级、LLM 自管理换页 → 后继产品 [Letta](https://github.com/letta-ai/letta) |

### 2024：结构化记忆与评测兴起
| 工作 | 核心思想 |
|---|---|
| [LoCoMo](https://arxiv.org/abs/2402.17753)（ACL'24） | 超长对话基准（10 段对话、平均 300 轮、最多 35 个 session、整段 16–26K token）；核心发现：长上下文与 RAG 均远落后人类 |
| [GraphRAG](https://arxiv.org/abs/2404.16130)（Microsoft） | 离线实体 KG + 社区分层摘要，回答"整个语料在讲什么"的全局问题 |
| [HippoRAG](https://arxiv.org/abs/2405.14831)（NeurIPS'24） | 以海马体索引理论为蓝本：LLM 抽取 KG + Personalized PageRank 联想检索；多跳 QA 超 SOTA 20%，比迭代检索便宜 10–30 倍 |
| [TTT 层](https://arxiv.org/abs/2407.04620) | 隐状态本身是小模型，测试时自监督更新——记忆内化为参数化学习 |
| [EM-LLM](https://arxiv.org/abs/2407.09450) | 情景记忆式无限上下文：按"贝叶斯惊异度"分段并复用 KV cache |
| [In Defense of RAG](https://arxiv.org/abs/2409.01666) | 长上下文稀释注意力，检索存在倒 U 最优点 |
| [LongMemEval](https://arxiv.org/abs/2410.10813)（ICLR'25） | 五大记忆能力评测 + indexing-retrieval-reading 三段式设计框架，验证 session 分解、事实增强 key、时间感知查询扩展三项优化显著有效 |
| [Zep](https://arxiv.org/abs/2501.13956) | **时序知识图谱**（Graphiti）：双时间轴（事件时间 vs 摄入时间）支持事实演变与矛盾处理，矛盾事实"失效"而非删除 |

### 2025：记忆"操作系统化"
| 工作 | 核心思想 |
|---|---|
| [Titans](https://arxiv.org/abs/2501.00663)（Google） | attention=短期、神经记忆=长期的测试时学习模块，扩展到 2M 上下文 |
| [A-MEM](https://arxiv.org/abs/2502.12110)（NeurIPS'25） | Zettelkasten 式原子笔记 + 动态链接 + **记忆演化**（新记忆触发旧记忆更新） |
| [HippoRAG 2](https://arxiv.org/abs/2502.14802)（ICML'25） | "From RAG to Memory"：非参数持续学习，兼顾事实/联想/sense-making 三类记忆 |
| [Mem0](https://arxiv.org/abs/2504.19413) | 生产级抽取—巩固—检索管线（ADD/UPDATE/DELETE/NOOP 决策）+ 图记忆变体 |
| [Sleep-time compute](https://arxiv.org/abs/2504.13171)（Letta） | "睡眠 agent"空闲期离线思考上下文、改写主 agent 的 memory blocks；同等精度省测试时算力约 5× |
| [MemOS](https://arxiv.org/abs/2507.03724) | **MemCube** 统一调度明文/激活(KV cache)/参数三层记忆——记忆即可调度的系统资源 |
| [MIRIX](https://arxiv.org/abs/2507.07957) | 六类记忆（Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault）+ 多 agent 协调检索 |
| [MemBench](https://arxiv.org/abs/2506.21605)（ACL'25 Findings） | 记忆读写操作综合评测：有效性/效率/容量三维度 |
| [Hindsight](https://arxiv.org/abs/2512.12818)（2025-12） | 四逻辑网络（世界事实/agent 经历/实体摘要/演化信念）+ retain/recall/reflect 三操作，记忆作为推理的一等公民 |
| [Memp](https://arxiv.org/abs/2508.06433)（ACL'26 Findings） | 程序记忆全流程，强→弱模型记忆迁移 |

### 2026：安全、自动化与参数化融合（前沿）
- **记忆安全成为一等公民**：[MemoryDojo](https://arxiv.org/abs/2611.09721) 评测**记忆投毒**（恶意指令写入记忆并跨会话传播）；[MPBench](https://arxiv.org/abs/2606.04329) 系统研究投毒攻击面（4 条写入通道/9 类漏洞/6 类攻击）；[Salami 攻击](https://arxiv.org/abs/2608.01637) 把恶意目标切片成多条良性记忆、对 OpenClaw 达 75% 成功率；[origin-bound 权限](https://arxiv.org/abs/2606.24322) 给出可机器验证的防御；[HEAMS](https://arxiv.org/abs/2611.20144) 把隐私泄露、纠错阻力、过度保留、投毒纳入综合评测；[LEDGER](https://arxiv.org/abs/2611.20927) 研究**可审计记忆**（事实变更、来源可信度、用户要求遗忘）。
- **冲突消解工程化**：[DISCERN](https://arxiv.org/abs/2611.15729) 提出"**写时保留、读时消解**"处理企业多会话中的矛盾/被取代/实体混淆；[Read Boundaries](https://arxiv.org/abs/2611.19264) 研究跨存储多次读的一致性边界与修复；[依赖感知上下文图](https://arxiv.org/abs/2611.19859) 在事实变更时定位并修复受影响的下游决策。
- **参数化合流**：[Instant Weight-Space Consolidation](https://arxiv.org/abs/2611.20759) 把情景文本记忆即时固化进权重；[Coding Agent Memory Post-Training II](https://arxiv.org/abs/2611.14009) 直接**后训练模型学会何时写笔记/读笔记**。
- **设计自动化**：[MAPE](https://arxiv.org/abs/2611.17068) 用程序演化自动发现有效记忆系统设计；[TRACE](https://arxiv.org/abs/2608.09153) 用轨迹归因自动诊断修复上下文层（提示/知识库/技能）故障。
- **组织级共享记忆**：[共享组织记忆皮层](https://arxiv.org/abs/2611.05753)、[Decision Context](https://arxiv.org/abs/2611.08644)（跨会话学习用户纠正的行为偏好）、[MemCredit](https://arxiv.org/abs/2611.15600)（按决策效用给记忆计价）。
- **检索底座多样化**：[FlyHash](https://arxiv.org/abs/2611.05791)（果蝇启发稀疏哈希）验证向量 ANN 之外的可行性。

## 4. 主流开源框架与产品横评

数据：GitHub API，2026-11-17 快照。

### 4.1 记忆专用框架

| 框架 | 架构/定位 | License | Stars | 活跃度 |
|---|---|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | 抽取—巩固—检索管线；向量为主可选图 | Apache-2.0 | 69.1k | 日更 |
| [Hindsight](https://github.com/vectorize-io/hindsight) | retain/recall/reflect 四网络 + 云服务 | MIT | 61.5k | 日更（2025-10 创建，增速最快） |
| [Cognee](https://github.com/topoteretes/cognee) | ECL 管线 → 图+向量混合记忆 | Apache-2.0 | 33.4k | 日更 |
| [Graphiti](https://github.com/getzep/graphiti)（Zep 开源核） | bi-temporal 时序知识图谱 | Apache-2.0 | 32.9k | 日更 |
| [Supermemory](https://github.com/supermemoryai/supermemory) | learner-1 模型 + vector-graph，第三方记忆层 | MIT | 31.6k | 日更 |
| [TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) | 多 agent 共享同一记忆服务器 | MIT（README 标注） | 31.0k | 周更（2026-04 创建） |
| [agentmemory](https://github.com/rohitg00/agentmemory) | 编程 agent 跨会话/跨工具记忆（MCP） | Apache-2.0 | 30.4k | 日更 |
| [Letta](https://github.com/letta-ai/letta) | memory blocks + sleep-time，有状态 agent 平台 | Apache-2.0 | 25.5k | 月内活跃 |
| [ai-memory](https://github.com/akitaonrails/ai-memory) | Rust，跨 20+ 编程 agent 的记忆（markdown vault） | MIT | 16.0k | 日更 |
| [MemOS](https://github.com/MemTensor/MemOS) | MemCube 三层记忆统一调度 | Apache-2.0 | 12.4k | 日更（2.0"星尘"） |
| [MIRIX](https://github.com/Mirix-AI/MIRIX) | 六类记忆 + 多 agent 检索协调 | Apache-2.0 | 3.5k | 周更 |
| [Memobase](https://github.com/memodb-io/memobase) | 结构化用户画像 + 事件时间线 | Apache-2.0 | 3.0k | 放缓（团队转向 Acontext，后者已归档） |
| [LangMem](https://github.com/langchain-ai/langmem) | LangChain 官方记忆 SDK | MIT | 1.8k | 活跃 |
| [Memoripy](https://github.com/caspianmoon/memoripy) | "证据优先"：时间版本/引用/可解释召回 | Apache-2.0 | 0.7k | 低频 |

各路线要点（详细集成与定价见各官网）：

- **Mem0**：最完整的"生产可用"生态。对话增量抽取 → LLM 决策 ADD/UPDATE/DELETE/NOOP → 向量库（图变体捕捉关系），检索分≈相似度+时近性+重要性加权；集成 22+ 框架（LangChain/LlamaIndex/LangGraph…）及 Claude Code/Cursor 等编码 agent 插件；OSS 之外有托管平台（免费层起步）。**给聊天产品最快加"用户记忆"的默认选择**，但其基准声明有争议（见第 5 节）。
- **Hindsight**：论文（[2512.12818](https://arxiv.org/abs/2512.12818)）明确批评"外挂式记忆"（抽取片段→向量/图→top-k 注入）证据/推理不分、长程组织混乱；改为**结构化一等公民**：世界事实、agent 经历、实体摘要、演化信念四张逻辑网络，retain/recall/reflect 全程可追溯。适合把记忆当推理底座的重度场景。
- **Zep/Graphiti**：交互建模为时序上下文图——实体、带生效区间的边、episode 溯源；矛盾事实失效而非删除，保留"当时为真"；混合检索=语义+BM25+图遍历，增量更新不重算。商业平台闭源（Flex $125/月起，企业版支持 VPC/BYOK、SOC2/HIPAA），2026-10 发布 v4 Context API 与 Konig 图数据库服务。**企业级时序推理与数据治理首选**。
- **Letta**：MemGPT 作者创办。memory blocks 常驻上下文、agent 用工具自编辑；archival（向量）/recall（消息库）分层；全量状态可序列化为 Agent File；sleep-time agents 后台重写记忆。2025-26 转向 Letta Code（记忆优先编码 agent）与 git 化 Context Repositories。定价 Free（3 agent BYOK）/Pro $20 月。**要 agent"自我进化"而非外挂记忆时的平台级选择**。
- **Cognee**：ECL（Extract-Cognify-Load）管线把文档/对话/代码转为图+向量混合记忆，四操作 remember/recall/improve/forget；内置 GLiNER 小模型可零 API key 本地跑；1.x 起单 Postgres 撑全记忆层；COGX 迁移格式支持从 Mem0/Letta/Zep 迁入；另有[方法学论文](https://arxiv.org/abs/2505.24478)与 Cognee Cloud。**自托管"公司大脑"**。
- **MemOS**：理念最激进——MemCube 封装内容+溯源/版本，统一明文、激活（KV cache）、参数三级记忆的调度演化；自托管栈 Neo4j+Qdrant + 异步 MemScheduler；自报 LoCoMo 88.83 / LongMemEval 89.20，并发布横评 14 个商业产品的 OmniMemEval；给 OpenClaw agent 任务完成率 36.63%→50.87%（自报）。**研究三层记忆统一方向的国产代表**。
- **Supermemory**：learner-1 模型做抽取/合并/推断/遗忘，写入 vector-graph，经 hooks 实时注入任意 agent；开源引擎可完全本地运行，商业 API 称月处理 1T+ token、搜索 356ms；刻意不卷基准（"记忆要让人感觉更好，不是刷榜"）。**跨模型/跨 harness 的第三方记忆层**。
- **MIRIX**：基于 Letta 二次开发；PostgreSQL BM25+向量检索、可选 Neo4j 溯源超图；Auto-Dream 定期合并去重清冲突；多模态屏幕记忆 demo。纯开源无云。多模态个人助理方向的研究前沿。
- **Memobase**：画像驱动——每用户结构化 profile（topic/sub_topic/content）+ 事件时间线；单 run 固定 3 次 LLM 调用（省 40–50% token），画像读取 <100ms；适合"画像字段产品侧可控"的 C 端陪伴/教育产品。**活跃度是主要风险**。
- **LangMem / LangGraph Store**：LangChain 官方——hot-path 记忆工具（对话中即时读写）+ 后台 memory manager 自动抽取合并；存储走 LangGraph BaseStore（namespace+向量检索，Postgres），LangGraph Platform 默认内置。**已用 LangGraph 团队的零迁移成本首选**。
- **LlamaIndex Memory**：2025 年新 Memory 类取代 ChatMemoryBuffer——短期 token 限额内 FIFO，超额批次 flush 进长期 memory blocks（Static/FactExtraction/Vector，可自定义），按 priority 截断，SQLite/Postgres 持久化。**框架内轻量分层记忆，非独立服务**。
- **编程 agent 专门路线**：CLAUDE.md / AGENTS.md / Cursor rules 是"文件即记忆"的朴素形态（详见 6.7）；[agentmemory](https://github.com/rohitg00/agentmemory) 与 [ai-memory](https://github.com/akitaonrails/ai-memory)（Rust）解决**跨工具共享**（Claude Code 中断、Codex 继续）；[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp)（48.8k★）维护代码/文档/基础设施持久图谱 + 变更感知测试选择。这一细分 2025-2026 增长极快。
- **其他**：[Khoj](https://github.com/khoj-ai/khoj)（38.2k★，AGPL，自托管"第二大脑"应用，非记忆 API，2026-08 后放缓）；[EM-LLM](https://arxiv.org/abs/2407.09450) 为研究代码。

### 4.2 云厂商托管记忆服务

| 服务 | 方案 | 备注 |
|---|---|---|
| AWS Bedrock AgentCore Memory | 短期 session 事件 + 长期异步洞察 | 大厂"双层记忆"标准化 |
| Google Vertex AI Memory Bank | 抽取式，绑定 Vertex Agent Engine | 无时序图 |
| 阿里 Hologres 长记忆服务 | 向量+全文+图+标量混合，兼容开源 Mem0 API；全托管 | 自报 AI Coding token 省 78.7–98.7%；2026-11 起商业化计费；主打个人→团队/组织记忆资产化 |
| Anthropic 记忆工具 | 文件式 memories 目录（view/str_replace/delete 命令） | 随 Claude 模型公测，"文件即记忆"产品化 |

## 5. 评测基准与争议

### 5.1 主要基准

| 基准 | 考察 | 备注 |
|---|---|---|
| [MSC](https://arxiv.org/abs/2107.07567)（2021） | 多 session 开放域对话 | 鼻祖；"摘要+检索召回"优于端到端的结论沿用至今 |
| [LoCoMo](https://arxiv.org/abs/2402.17753)（ACL'24） | 超长多会话对话 QA（单跳/时序/多跳/开放域+摘要+多模态） | 最常被引用，**质量受多方质疑**（见 5.3） |
| [LongMemEval](https://arxiv.org/abs/2410.10813)（ICLR'25） | 五能力：信息抽取/跨会话推理/时间推理/**知识更新**/拒答；500 题挂平均 115K token 历史 | 公认比 LoCoMo 严格，已成新主战场 |
| DMR（Deep Memory Retrieval） | Zep 论文引入 | Zep 自报 94.8% 胜 MemGPT 93.4% |
| [PerLTQA](https://arxiv.org/abs/2402.16288) | 8,593 题/30 角色，语义+情景记忆 | 记忆分类/检索/合成 |
| [MemBench](https://arxiv.org/abs/2506.21605)（ACL'25 Findings） | 记忆读写操作：有效性/效率/容量 | 补 QA 之外的操作维度 |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257)（ICLR'26） | 四能力：**准确检索/测试时学习/长程理解/冲突消解**；增量多轮注入，"注入一次、多次查询" | 新增 EventQA、FactConsolidation 数据集；同团队 ICML'26 新作 MemoryArena 把评测推向 agentic 任务 |
| [MPBench](https://arxiv.org/abs/2606.04329)（2026） | 记忆投毒：4 条写入通道/9 类漏洞/6 类攻击 | 安全基准 |
| [MemoryDojo](https://arxiv.org/abs/2611.09721)（2026） | 投毒在 10+ 记忆系统中的传播与放大 | 安全基准 |
| [HEAMS](https://arxiv.org/abs/2611.20144)（2026） | 隐私泄露/纠错阻力/过度保留/投毒综合评测 | "记忆治理"评测 |

### 5.2 自报成绩一览（**口径不一，不能横比**）

LoCoMo（LLM-as-Judge）时间线最能说明问题：

| 系统 | LoCoMo 成绩 | 来源与背景 |
|---|---|---|
| full-context GPT-4o-mini | 72.90% | Mem0 论文同表——**反超全部记忆系统** |
| Mem0 / Mem0-Graph | 66.88% / 68.44% | [Mem0 论文](https://arxiv.org/abs/2504.19413)自报 |
| Zep（Mem0 论文中的数字） | 65.99% | 被 [Zep 复测](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)指出三处实现错误（说话人都标成 user、时间戳未走 created_at、串行检索虚增延迟） |
| Zep（修正后，2025） | 75.14% | Zep 复测 |
| MIRIX / MemOS / Hindsight | 85.4% / 88.83% / 89.61% | 各自论文自报（不同 judge、不同实现） |
| Zep（2026 最新） | 94.7%（p95 155ms） | [Zep 官网](https://www.getzep.com/research/memory-benchmarks/) |
| 阿里 Hologres | 96.82% | 阿里官方文档自报 |
| **LoCoMo-Refined 复测**（2026-04） | Mem0 **48.91%**、MemOS 63.60%、EverMemOS 58.25%、MemPalace 58.68%，榜首 MemoraX 82.65% | [社区复测](https://github.com/mem-eval-suite/LoCoMo_refined)：换严格 judge（与人标注一致率 43.67%→86.33%）+ 修订 337 道题后，**各家普跌 15–22 点**；榜首亦为记忆厂商——厂商造榜刷榜是行业通病 |

LongMemEval 侧（自报）：Zep 90.2%（GPT-4o 71.2% vs full-context 60.2%）、MemOS 89.20%、Hindsight 91.4%（20B 骨干即达 83.6%）。

其他自报：Mem0 较 OpenAI Memory 相对 +26%、p95 延迟 −91%、token −90%+；Letta sleep-time compute 同精度省约 5× 测试时算力；Cognee 自建 BEAM 基准（100K token 0.79 / 10M 0.67）并公开方法与局限（做法值得提倡）。

### 5.3 三大争议

1. **基准质量与厂商互测**：LoCoMo 平均对话仅 1.6–2.6 万 token（未超现代上下文）、缺知识更新题、第 5 类缺 ground truth（Mem0 与 Zep 均弃用）、说话人归属错误、歧义题多；Mem0 的"SOTA"声明被 Zep 复测推翻，而双方都是记忆厂商——**利益冲突下没有干净的横评**。缓解动向：LoCoMo-Refined、Zep 的"准确率×延迟×token×judge 版本"四要素同报、Cognee/Letta 自建基准并公开方法（Letta 2026-10 推 FrontierMemory 考持续学习、MemOS 用 OmniMemEval 横评）。
2. **记忆放大谄媚**：《[Recalling Too Well](https://openreview.net/pdf?id=0Xt1qZ5xdW)》（ICLR'26 workshop，Writer）指出记忆系统会把用户错误偏好放大至 25 倍（NoveltyBench 偏好锚定 87.3% vs 无记忆基线 47.3%）。[Zep 复盘](https://www.getzep.com/blog/sycophancy-is-a-design-choice/)归因于两个设计：评测 prompt"仅依据记忆作答"被误用于推理任务；Mem0 式抽取只存用户侧、丢弃 assistant 的纠正，形成**单边记录**。缓解：双侧对话入库、事实带溯源、时间失效区间、按任务门控偏好检索。
3. **投毒与污染**：MINJA 理想条件下注入成功率 >95%；[Salami 攻击](https://arxiv.org/abs/2608.01637)把恶意目标切片成多条良性记忆、对 OpenClaw 达 75% 成功率；MPBench 证明**写入越激进越易被投毒**。防御（复合信任分、时间衰减清洗、[origin-bound 权限](https://arxiv.org/abs/2606.24322)）刚起步。

**工程结论**：公开基准只做**初筛**；选型必须自建线上评测（用真实业务会话回放，重点测冲突消解、时序、拒答、隐私），并要求"准确率+延迟+token+judge 版本"四要素同报。

## 6. 工程实践与设计模式

### 6.1 写入策略（write policy）三档

| 策略 | 代表 | 优缺点 |
|---|---|---|
| 每轮自动抽取 | Mem0、AWS AgentCore | 不漏信息；写放大、成本高，**越激进越易被投毒**（MPBench） |
| agent 自主决定 | MemGPT/Letta 自编辑、LangGraph Store 工具化读写、Anthropic 记忆工具（文件式） | 可解释、agent 自己判断价值；会漏写 |
| 后台异步整理 | Letta sleep-time compute、LangMem 后台 memory manager | 不阻塞对话、空闲期反思合并、成本后置 |

生产折中：**热路径决定"存什么"，冷路径决定"怎么整理"**。

### 6.2 检索策略
- 向量 + BM25 + 图**多路混合** + cross-encoder 重排已是标配（Zep 多 scope 并行、图常驻内存；Hologres 向量+全文+图+标量混合、毫秒级）；
- 可直接落地的技巧：LongMemEval 三优化（**session 分解控粒度、事实增强 key、时间感知查询扩展**）；按 token 预算非生成式组装上下文块（Zep auto search 单调用再省 53% token）；
- 重要性/时近性/相关性加权（Generative Agents 打分公式仍是经典起点）；
- 图增强检索（HippoRAG 的 PPR、Graphiti 子图遍历）在多跳/关联问题上明显优于纯向量；**embedding 召回质量是隐性上限**（[Zep 用 5,954 条生产 query 评 embedding](https://www.getzep.com/blog/evaluating-nemotron-3-embed-for-agent-memory/)）。

### 6.3 巩固、遗忘与冲突消解
- **反思/摘要**：定期把原始记忆压缩成高层结论（Generative Agents reflection → Letta sleep-time 一脉相承）；Anthropic 的做法：compaction（摘要重启+保留最近文件）、NOTES.md 结构化笔记、子 agent 上下文隔离（[官方博客](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents)）；
- **遗忘**：Ebbinghaus 衰减、TTL、按重要性淘汰；LongMemEval 特别强调**拒答能力**（该忘的别硬答）；
- **冲突消解**（2026 主战场）：时序 KG 用**边失效而非删除**记录事实演变（Graphiti 双时态是主流解法）；[DISCERN](https://arxiv.org/abs/2611.15729) 的"写时保留、读时消解"（保留历史版本，读取时再裁决）比写入时粗暴覆盖更稳妥；Mem0 在更新阶段做去重/合并/覆盖。

### 6.4 分层与隔离
- 分层：**in-context 核心记忆**（Letta memory blocks，常驻可自编辑）→ **归档记忆**（archival，向量全量）→ **召回记忆**（recall，消息历史索引）；冷热分层控成本；
- 隔离：Mem0 用 user_id/agent_id/run_id 三级作用域；LangGraph namespace；Hologres 主打"个人→团队/组织共享"记忆资产化 + RAM 权限策略；多 agent 共享记忆（MemOS、Letta shared blocks、TencentDB-Agent-Memory 团队记忆）要显式定义读写边界。

### 6.5 安全（2026 新必修课）
- **投毒**：用户输入/网页/工具返回中的恶意指令可能被固化进记忆（MINJA、Salami、MemoryDojo）；写入侧需来源标记+可信度分级，高权限动作不直接信任低信任来源记忆；
- **隐私与遗忘权**：HEAMS 把"纠错阻力、过度保留、上下文不当泄露"列为评测维度；合规要求"用户要求遗忘"能真正执行（LEDGER 可审计记忆）；
- **谄媚**：双侧入库、按任务门控偏好检索（见 5.3）。

### 6.6 成本与延迟
- 写入：异步化、批量化（Memobase 单 run 固定 3 次 LLM 调用省 40–50%）；
- 检索：控制注入条数与长度（Mem0 的 90% token 节省主要来自"不塞全量上下文"）；Hologres 实测 AI Coding 省 78.7–98.7%；
- KV cache 复用（EM-LLM、MemOS 激活记忆）是延迟敏感场景新杠杆。

### 6.7 编程 Agent 的记忆实践
- **CLAUDE.md**（企业/用户/项目/本地四层、`#` 快速追加、`@` 导入，[文档](https://code.claude.com/docs/en/memory)）与 **Cursor Rules**（Project/User/Team/AGENTS.md 四类，alwaysApply/globs/描述三字段控制何时注入，[文档](https://cursor.com/docs/rules)）本质是 **prompt 级持久指令、文件即记忆**：可 git 审计、由人显式维护，但无自动检索、无冲突消解、无跨用户聚合——属"**半记忆**"方案。编码场景刻意偏爱显式文件（just-in-time 读取、避免陈旧索引），Anthropic 记忆工具可视为把该模式产品化；
- **跨工具共享**：[agentmemory](https://github.com/rohitg00/agentmemory)（30.4k★，MCP，"在 Claude Code 里中断、到 Cursor/Codex 继续"）、[ai-memory](https://github.com/akitaonrails/ai-memory)（Rust，20+ harness，markdown vault）；
- **代码图谱**：[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp)（48.8k★）把代码/文档/基础设施连成持久图，支持变更感知测试选择；
- 学术侧：[Coding Agent Memory Post-Training](https://arxiv.org/abs/2611.14009) 系列直接训练模型"何时写/读笔记"。

### 6.8 常见坑（失败模式）
1. **什么都存**：无写入门槛 → 噪声库，检索质量随规模崩塌，污染面变大；
2. **只看 QA 基准分**：LoCoMo 高分 ≠ 冲突消解/长程 agentic 任务好用；中文社区大量"横评"实为厂商软文；
3. **忽视事实演变**：只增不删不改，三个月后记忆自相矛盾；
4. **单边记录**：只存用户侧、丢弃 assistant 纠正 → 谄媚放大；
5. **记忆污染无防护**：一条恶意输入永久毒化行为；
6. **低估运维**：记忆是需要审计、备份、质量监控的**有状态基础设施**。

## 7. 选型建议

| 场景 | 推荐路径 |
|---|---|
| 轻量个性化（偏好/称呼/简史） | LangGraph store / [LangMem](https://github.com/langchain-ai/langmem)；数据少时结构化用户档案表就够 |
| 客服 / CRM（结构化事实、需审计） | [Graphiti/Zep](https://github.com/getzep/graphiti)（时序 KG、VPC/BYOK、SOC2）或 Mem0 图模式 |
| 通用记忆层快速上线 | [Mem0](https://github.com/mem0ai/mem0)（自托管）或 Mem0 Platform / [Supermemory](https://github.com/supermemoryai/supermemory) / Hologres（托管） |
| 重度长程 / 研究型 agent | [Letta](https://github.com/letta-ai/letta)（memory blocks + sleep-time）、[MemOS](https://github.com/MemTensor/MemOS)（三层调度）、[Hindsight](https://github.com/vectorize-io/hindsight)（结构化推理底座） |
| 数据基础设施导向 / 自托管"公司大脑" | [Cognee](https://github.com/topoteretes/cognee)（图+向量管线、单 Postgres、COGX 迁移） |
| 画像可控的 C 端陪伴/教育 | [Memobase](https://github.com/memodb-io/memobase)（注意活跃度风险） |
| 编程 agent | CLAUDE.md/AGENTS.md 管静态规范 + agentmemory/ai-memory 跨会话共享 + codebase-memory-mcp 代码图谱 |
| 已上 AWS/GCP/阿里云 | AgentCore Memory / Vertex AI Memory Bank / Hologres 长记忆服务 |

**自建 vs 采购**：自建的核心工作量在"巩固与冲突消解策略 + 自建评测集"，而不是存取本身——若团队没有 3 人月以上投入，优先用成熟开源/托管，把精力放在业务评测与安全防护上。

## 8. 趋势展望（2026→2027）

1. **三层记忆统一调度**：明文（外部存储）/ 激活（KV cache）/ 参数（权重）从割裂走向统一生命周期管理（MemOS MemCube、权重空间巩固、TTT 合流）；
2. **睡眠期计算常态化**：记忆整理从"在线挤占推理预算"转向"空闲异步编译"（Letta sleep-time、LangMem 后台线程）；
3. **评测换代**：从 LoCoMo 式静态 QA 转向 agentic 任务（MemoryArena）、冲突消解（MemoryAgentBench）、持续学习（FrontierMemory）与安全（MPBench/MemoryDojo/HEAMS）；judge 质量成为基准可信度的关键（LoCoMo-Refined 的教训）；
4. **记忆安全与治理成为采购硬指标**：投毒防护、来源可信度分级、可审计与遗忘权、谄媚缓解；
5. **记忆设计自动化**：MAPE 式程序演化与 TRACE 式故障归因，让"记忆管线怎么设计"本身被自动优化；
6. **组织级共享记忆**：从单 agent 记忆到团队/组织记忆资产（Hologres、TencentDB-Agent-Memory、organizational cortex），读写边界与权限模型成为新问题；
7. **编程 agent 记忆继续爆发**：跨工具共享 + 代码图谱 + 记忆后训练（模型学会何时写/读笔记）三条线并进。

## 9. 调研方法与局限

- 本会话 web 搜索 API 不可用（缺 API key），改用 **GitHub API + arXiv API + Bing/官方页面直接抓取**，四路并行（主调研 + 三个子调研）完成；论文条目均经 arXiv API 核实（编号、日期、标题），仓库 star/License/最近 push 为 2026-11-17 快照；
- 评测成绩为各论文/厂商自报，存在利益相关与口径差异，请以自建评测为准；中文来源中厂商软文较多，已尽量交叉核对；
- 覆盖以公开可验证的一手来源为准，内部实现细节（如各商业平台架构）依赖官方文档自述。

## 10. 关键参考

**综述**：[Memory Mechanism Survey (2024)](https://arxiv.org/abs/2404.13501) · [Rethinking Memory (2025)](https://arxiv.org/abs/2505.00675) · [Context Engineering Survey (2025)](https://arxiv.org/abs/2507.13334)

**论文**（按时间）：[Generative Agents](https://arxiv.org/abs/2304.03442) · [Reflexion](https://arxiv.org/abs/2303.11366) · [Voyager](https://arxiv.org/abs/2305.16291) · [MemoryBank](https://arxiv.org/abs/2305.10250) · [MemGPT](https://arxiv.org/abs/2310.08560) · [LoCoMo](https://arxiv.org/abs/2402.17753) · [GraphRAG](https://arxiv.org/abs/2404.16130) · [HippoRAG](https://arxiv.org/abs/2405.14831) · [TTT](https://arxiv.org/abs/2407.04620) · [EM-LLM](https://arxiv.org/abs/2407.09450) · [In Defense of RAG](https://arxiv.org/abs/2409.01666) · [LongMemEval](https://arxiv.org/abs/2410.10813) · [Titans](https://arxiv.org/abs/2501.00663) · [Zep](https://arxiv.org/abs/2501.13956) · [A-MEM](https://arxiv.org/abs/2502.12110) · [HippoRAG 2](https://arxiv.org/abs/2502.14802) · [Sleep-time Compute](https://arxiv.org/abs/2504.13171) · [Mem0](https://arxiv.org/abs/2504.19413) · [MemBench](https://arxiv.org/abs/2506.21605) · [MemoryAgentBench](https://arxiv.org/abs/2507.05257) · [MemOS](https://arxiv.org/abs/2507.03724) · [MIRIX](https://arxiv.org/abs/2507.07957) · [Memp](https://arxiv.org/abs/2508.06433) · [TRACE](https://arxiv.org/abs/2608.09153) · [Salami Attack](https://arxiv.org/abs/2608.01637) · [Hindsight](https://arxiv.org/abs/2512.12818) · [MIRIX-2.0](https://arxiv.org/abs/2611.12068) · [MemoryDojo](https://arxiv.org/abs/2611.09721) · [HEAMS](https://arxiv.org/abs/2611.20144) · [LEDGER](https://arxiv.org/abs/2611.20927) · [DISCERN](https://arxiv.org/abs/2611.15729) · [MAPE](https://arxiv.org/abs/2611.17068) · [权重空间巩固](https://arxiv.org/abs/2611.20759) · [编程 Agent 记忆后训练](https://arxiv.org/abs/2611.14009) · [MPBench](https://arxiv.org/abs/2606.04329) · [Origin-bound 防御](https://arxiv.org/abs/2606.24322) · [MSC](https://arxiv.org/abs/2107.07567) · [PerLTQA](https://arxiv.org/abs/2402.16288)

**争议与复测**：[Zep: Is Mem0 Really SOTA?](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) · [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) · [Recalling Too Well](https://openreview.net/pdf?id=0Xt1qZ5xdW) · [Zep: Sycophancy is a Design Choice](https://www.getzep.com/blog/sycophancy-is-a-design-choice/)

**官方文档**：[Claude Code Memory](https://code.claude.com/docs/en/memory) · [Cursor Rules](https://cursor.com/docs/rules) · [Anthropic Context Engineering](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) · [LangMem](https://langchain-ai.github.io/langmem/) · [AWS AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html) · [Hologres 长记忆服务](https://help.aliyun.com/zh/hologres/user-guide/overview-of-hologres-long-memory-service)
