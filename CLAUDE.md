# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GeoMRF V2** (Geological Risk Management Framework) 是一个基于知识图谱的 TBM 隧道施工不良地质智能处置推荐系统。系统通过整合多源异构数据（设计资料、地质预报、施工记录、专家经验等），构建领域知识图谱，并结合深度学习与大语言模型，为隧道施工提供智能化的风险识别与处置方案推荐。

### 项目状态

- **kg_construction**：✅ 已完成，可正常运行
- **retrieval**：🔄 正在重构中
- **recommendation**：🔄 正在重构中

---

## 开发环境

本项目使用 **Conda 虚拟环境**进行开发，以确保依赖隔离和环境一致性。

### 环境信息

- **环境名称**：NLP
- **环境路径**：D:\envs\NLP
- **Python 版本**：3.11.9
- **Python 解释器**：D:\envs\NLP\python.exe

### 在命令行中使用环境

所有 Python 命令需要在 NLP 环境中运行。使用以下方式：

```bash
# 方法1：使用 conda run（推荐）
conda run -n NLP python -m kg_construction.scripts.chunk_documents

# 方法2：在命令前激活环境（PowerShell/CMD）
conda activate NLP
python -m kg_construction.scripts.chunk_documents

# 方法3：直接使用环境中的 Python
D:\envs\NLP\python.exe -m kg_construction.scripts.chunk_documents
```

**注意**：在本文档的所有命令示例中，默认使用方法1（`conda run -n NLP`），如无特殊说明，请确保在 NLP 环境中运行。

---

## Development Commands

### 知识图谱构建（kg_construction）

```bash
# 1. 文档分块（支持PDF和JSON）
conda run -n NLP python -m kg_construction.scripts.chunk_documents

# 2. 实体关系抽取（regex/lexicon/LLM三种方法）
conda run -n NLP python -m kg_construction.scripts.extract_triples

# 3. 构建Neo4j知识图谱
conda run -n NLP python -m kg_construction.scripts.build_kg

# 4. 生成向量嵌入（可选）
conda run -n NLP python -m kg_construction.scripts.generate_embeddings
# 或指定节点类型
conda run -n NLP python -m kg_construction.scripts.generate_embeddings --node-type 紧急响应措施
```

### Web界面启动

```bash
cd recommendation/core
conda run -n NLP python gradio_demo.py
# 访问 http://localhost:7860
```

### 依赖安装

```bash
conda run -n NLP pip install -r requirements.txt
```

### 预训练模型下载

由于模型权重文件较大，需要自行下载：

```bash
# 使用 Hugging Face CLI
conda run -n NLP pip install huggingface-hub

# 下载 BERT 模型
conda run -n NLP huggingface-cli download bert-base-chinese --local-dir models/bert-base-chinese

# 下载 BGE 模型
conda run -n NLP huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/bge-large-zh-v1.5
```

---

## 开发规范

### 代码组织规范

1. **测试代码位置**
   - 所有测试代码必须放置在 `tests/` 目录下
   - 单元测试：`tests/unit/`
   - 集成测试：`tests/integration/`
   - 测试数据：`tests/data/`
   - **禁止在项目根目录放置测试文件**

2. **测试文件命名**
   - 单元测试：`test_*.py` 或 `*_test.py`
   - 测试函数：`test_*()`
   - 测试类：`Test*`

3. **源代码组织**
   - 源代码按模块组织：`kg_construction/`, `retrieval/`, `recommendation/`
   - 每个模块包含：`core/`（核心逻辑）、`models/`（模型）、`utils/`（工具）

4. **配置文件**
   - 所有配置文件集中在各模块的 `config.py` 或配置目录中
   - 避免硬编码路径和参数

5. **路径规范**
   - 使用 `Path(__file__).parent.parent...` 计算项目根目录
   - 不要假设当前工作目录
   - 所有模块需要支持从不同位置导入

### 测试文件管理规范

1. **测试脚本位置**
   - 所有测试脚本必须放置在 `tests/` 目录下
   - 单元测试：`tests/unit/`
   - 集成测试：`tests/integration/`
   - 诊断测试：`tests/diagnostic/`
   - 测试数据：`tests/data/`
   - 测试输出：`tests/output/`

2. **测试输出文件位置**
   - **所有测试产生的输出文件必须放置在 `tests/output/` 目录**
   - 包括但不限于：临时文件、日志文件、诊断报告、测试结果文件
   - **禁止在项目根目录创建任何测试输出文件**

3. **临时文件处理**
   - 测试脚本应将所有输出写入 `tests/output/` 目录
   - 测试脚本应负责清理自己产生的临时文件
   - 诊断报告应保存为 `tests/output/diagnostic_*.md`

4. **禁止的文件名和位置**
   - 禁止在项目根目录创建：`test_output.txt`、`debug.log`、`DIAGNOSTIC_REPORT_*.md` 等
   - 禁止在模块目录创建临时测试文件
   - 所有测试相关输出必须统一管理在 `tests/output/` 下

5. **输出目录管理**
   - `tests/output/` 目录已在 `.gitignore` 中配置，不会被提交
   - 建议定期清理此目录以节省磁盘空间
   - 测试脚本可在此目录创建子目录以组织输出

---

## 测试

本项目使用 `pytest` 框架进行测试，所有测试代码位于 `tests/` 目录中。

**重要提醒**：
- 所有测试代码必须在 `tests/` 目录下
- 项目根目录不应包含测试文件
- 遵循开发规范中的代码组织要求

### 测试目录结构

```
tests/
├── unit/                   # 单元测试
│   ├── test_ner.py         # NER 模块测试
│   └── test_dynamic_weight.py  # 动态权重模块测试
├── integration/            # 集成测试
│   ├── test_association_retrieval.py  # 关联检索测试
│   ├── test_association_comprehensive.py  # 综合关联检索测试
│   └── test_association_quick.py  # 快速关联检索测试
├── diagnostic/             # 诊断测试
│   ├── test_orphaned_nodes.py      # 孤立节点诊断
│   ├── test_extraction_results.py  # 抽取结果诊断
│   └── test_chainage_check.py      # chainage检查
├── data/                   # 测试数据
└── conftest.py             # pytest 配置和 fixtures
```

### 测试分类说明

1. **单元测试 (unit/)**: 测试单个模块/类的功能
   - 例如：NER模型、动态权重模型

2. **集成测试 (integration/)**: 测试多个模块协作的功能
   - 例如：关联检索流程、推荐流程

3. **诊断测试 (diagnostic/)**: 诊断系统状态和数据的脚本
   - 例如：孤立节点检查、抽取结果验证
   - 可使用pytest运行，也可直接运行

### 运行测试

```bash
# 运行所有测试
conda run -n NLP pytest

# 运行特定类型的测试
conda run -n NLP pytest tests/unit/
conda run -n NLP pytest tests/integration/
conda run -n NLP pytest tests/diagnostic/

# 运行特定测试文件
conda run -n NLP pytest tests/diagnostic/test_orphaned_nodes.py -v

# 显示详细输出
conda run -n NLP pytest -v

# 显示测试覆盖率（需要安装 pytest-cov）
conda run -n NLP pytest --cov=retrieval --cov=kg_construction
```

### 直接运行测试脚本

某些测试脚本也可以直接运行（不依赖 pytest）：

```bash
# 运行诊断测试（不使用pytest）
conda run -n NLP python -m tests.diagnostic.test_orphaned_nodes
conda run -n NLP python -m tests.diagnostic.test_extraction_results
conda run -n NLP python -m tests.diagnostic.test_chainage_check
```

### 重要规范

**❌ 禁止行为**：
- **禁止在项目根目录放置测试文件**
- **禁止在`kg_construction/scripts/`或其他模块目录放置测试/诊断脚本**
- **所有测试代码必须放置在`tests/`目录下**

**✅ 正确做法**：
- 单元测试 → `tests/unit/`
- 集成测试 → `tests/integration/`
- 诊断测试 → `tests/diagnostic/`
- 测试数据 → `tests/data/`

**命名规范**：
- 单元测试：`test_*.py`
- 集成测试：`test_*.py`
- 诊断测试：`test_*.py`
- 测试函数：`test_*()`
- 测试类：`Test*`

---

### 编写测试

- 单元测试文件命名：`test_*.py`
- 测试函数命名：`test_*()`
- 使用 `conftest.py` 中定义的 fixtures
- 参考 `tests/unit/test_ner.py` 作为示例

---

## Architecture Overview

### kg_construction/（已完成）

多源数据处理与知识图谱构建模块。

**核心组件**：

- `core/chunking/`：文档分类、PDF解析、分块策略
  - `DocumentClassifier`：根据文件名/内容识别文档类型
  - `PDFParser`：PDF文本提取
  - `TextChunker`：文本分块（支持按页、按标题）
- `core/extraction/`：实体关系抽取
  - **三种抽取方法**：
    - `RegexExtractor`：正则表达式（里程、时间等）
    - `LexiconExtractor`：字典匹配（地质术语词典）
    - `LLMExtractor`：大模型语义理解
  - `EntityExtractor`：统一抽取接口
  - `DocumentAggregator`：文档级聚合（针对变更纪要）
- `core/storage/`：Neo4j图数据库操作
  - `Neo4jClient`：数据库连接
  - `GraphBuilder`：图谱构建（显式关系+隐式推理）
  - `DataLoader`：加载抽取结果
  - `Neo4jSyncTracker`：增量同步追踪
- `core/graph_inference/`：隐式关系推理
  - 基于里程区间匹配（`range_overlap`）
  - 多跳路径推理（如：紧急响应措施→历史处置案例→施工信息→探测方法）
- `core/embedding/`：向量嵌入生成
  - 支持 BGE-large-zh-v1.5（优先）或 BERT-base-chinese

**关键配置**：

- `kg_construction/core/extraction/config.py`：
  - `NODE_TYPES`：节点类型定义（15种节点类型）
  - `RELATION_TYPES`：关系类型定义
  - `DOCUMENT_EXTRACTION_CONFIG`：文档类型配置（8种文档类型）
  - `RELATION_INFERENCE_CONFIG`：关系推理规则
- `kg_construction/core/storage/neo4j_config.json`：Neo4j连接配置

**数据流程**：

```
原始文档(PDF/JSON)
    ↓
文档分块（chunk_documents.py）
    ↓
实体关系抽取（extract_triples.py）
    ├─ regex: 里程、时间等结构化信息
    ├─ lexicon: 地质术语匹配
    └─ LLM: 语义理解
    ↓
图谱构建（build_kg.py）
    ├─ 显式关系：从文档直接抽取
    └─ 隐式关系：里程匹配+多跳推理
    ↓
向量嵌入（generate_embeddings.py，可选）
```

**追踪文件**：

- `kg_construction/core/extraction/extraction_mapping.json`：记录已抽取的文档
- `kg_construction/core/storage/neo4j_sync_status.json`：记录Neo4j同步状态
- `kg_construction/data/processed/embedding_cache.json`：向量嵌入缓存

### retrieval/（重构中）

智能检索模块，混合检索引擎。

**核心组件**：

- `core/search_engine.py`：混合检索
  - 向量检索（BGE/BERT embeddings）
  - 关键词匹配（keywords相似度）
  - 动态权重融合（alpha * similarity + beta * key_score）
- `core/query_pipeline.py`：检索流程编排
  - 关键信息提取（里程、线路、风险类型）
  - NER实体识别
  - 图谱检索（`kg_plan_relevance_retrieval`、`kg_mileage_relevance_retrieval`）
- `models/ner/`：BiLSTM-CRF 命名实体识别
- `models/dynamic_weight/`：动态权重模型（学习alpha/beta参数）
- `utils/kg_utils.py`：图谱查询工具函数

**检索流程**：

```
用户查询
    ↓
关键信息提取（里程、线路、风险类型）
    ↓
NER实体识别（地质实体）
    ↓
混合检索
    ├─ 向量检索：BGE embeddings + cosine相似度
    ├─ 图谱检索：基于风险类型查询方案节点
    └─ 动态权重：学习alpha/beta融合
    ↓
图谱增强
    ├─ kg_plan_relevance_retrieval：方案关联信息
    └─ kg_mileage_relevance_retrieval：里程相关设计/探测信息
```

### recommendation/（重构中）

隧道不良地质处置推荐模块。

**核心组件**：

- `core/recommendation_engine.py`：推荐引擎
  - 意图识别：判断是否为处置相关查询
  - 方案推荐：基于图谱路径检索
  - 状态管理：多轮对话、方案切换
- `core/conversation_manager.py`：对话管理器
  - 多轮对话处理
  - 拒绝反馈处理（最多3次拒绝）
  - LLM增强生成
- `core/feedback_handler.py`：反馈学习
- `core/gradio_demo.py`：Gradio Web界面
- `models/intention/`：BiLSTM二分类意图识别
- `models/llm/`：LLM客户端（支持OpenAI兼容API，如Qwen）

**推荐流程**：

```
用户查询
    ↓
意图识别（BiLSTM分类）
    ↓
检索引擎（retrieval模块）
    ↓
推荐引擎
    ├─ 获取Top-K方案
    ├─ 状态管理（当前方案、备选方案）
    └─ 拒绝处理（切换到下一个方案）
    ↓
LLM增强生成
    ├─ 使用prompt模板构建上下文
    └─ 生成专业处置建议
    ↓
用户反馈
    └─ 拒绝记录 → 反馈学习
```

---

## Configuration Files

### Neo4j配置

`kg_construction/core/storage/neo4j_config.json`：

```json
{
  "uri": "bolt://localhost:7687",
  "username": "neo4j",
  "password": "YOUR_NEO4J_PASSWORD_HERE",
  "database": "neo4j",
  "batch_size": 100
}
```

### LLM配置

`config/llm_config.json`：

```json
{
  "current_model": "qwen",
  "models": {
    "qwen": {
      "api_base": "YOUR_API_URL",
      "api_key": "YOUR_API_KEY_HERE",
      "model_name": "gpt-4o",
      "temperature": 0.1,
      "max_tokens": 512
    }
  }
}
```

### 嵌入模型配置

`kg_construction/core/embedding/embedding_config.json`：

```json
{
  "model_type": "bge",
  "bge_model_path": "models/bge-large-zh-v1.5",
  "bert_model_path": "models/bert-base-chinese",
  "batch_size": 32,
  "vector_field": "embedding_vector"
}
```

### Prompt模板

`recommendation/data/prompt_template.json`：LLM生成回复的模板

---

## Key Design Patterns

### 1. 配置驱动

- 节点/关系schema在`config.py`中集中定义
- 文档类型配置支持自适应处理策略
- 单例节点预创建（围岩等级、风险类型等）

### 2. 增量处理

- `extraction_mapping.json`追踪已处理文档
- `neo4j_sync_status.json`追踪Neo4j同步状态
- 跳过已处理文件，支持断点续传

### 3. 节点合并与去重

- 文档级聚合：变更纪要使用`merged_text`策略
- 节点合并：相同`merge_keys`的节点合并
- Neo4j去重：基于属性值去重节点

### 4. 多策略抽取

- **Regex**：里程、时间等结构化信息
- **Lexicon**：地质术语词典匹配（CSV格式）
- **LLM**：复杂语义理解（支持fallback到分section提取）

### 5. 关系推理

- **里程匹配**：`range_overlap`策略匹配里程区间
- **多跳推理**：通过图谱路径推理隐式关系
  - 例如：紧急响应措施 → 历史处置案例 → 施工信息 → 探测方法 → 探测结论

### 6. 懒加载单例

- BERT/BGE模型懒加载（只在首次使用时加载）
- Neo4j连接懒加载
- 配置文件集中管理

---

## Important Notes

### 模型权重

- BERT-base-chinese 和 BGE-large-zh-v1.5 需要单独下载到 `models/` 目录
- 检查点文件（NER、动态权重、意图识别）需要训练后生成

### 数据目录

- 源数据：`kg_construction/data/source/`
- 中间结果：`kg_construction/data/processed/`
  - `chunks/`：分块结果
  - `extraction_results/`：抽取结果
  - `knowledge_graph_data/global_graph.json`：全局图数据

### 日志

- `kg_construction/logs/`：知识图谱构建日志
- 使用`setup_logger`工具函数创建日志器

### Neo4j图数据库

- 需要预先安装并启动Neo4j（>= 4.4）
- 默认端口：7687
- 默认数据库：neo4j

---

## Troubleshooting

### 常见问题

1. **Neo4j连接失败**
   - 检查Neo4j服务是否启动
   - 确认`neo4j_config.json`中的连接信息正确

2. **模型加载失败**
   - 确认模型权重已下载到正确的目录
   - 检查`embedding_config.json`中的模型路径

3. **LLM调用失败**
   - 确认API密钥和URL正确
   - 检查网络连接

4. **抽取结果为空**
   - 检查文档类型是否正确识别
   - 查看日志确认抽取方法是否匹配
   - 对于LLM抽取，确认token限制

### 调试技巧

- 使用`logging.getLogger(__name__).setLevel(logging.DEBUG)`开启详细日志
- 检查`extraction_mapping.json`确认文档处理状态
- 使用Neo4j Browser查看图谱结构
