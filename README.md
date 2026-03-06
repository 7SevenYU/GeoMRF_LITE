# GeoMRF V2 - TBM隧道不良地质处置知识图谱与推荐系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

GeoMRF V2（Geological Risk Management Framework）是一个基于知识图谱的TBM隧道施工不良地质智能处置推荐系统。系统通过整合多源异构数据（设计资料、地质预报、施工记录、专家经验等），构建领域知识图谱，并结合深度学习与大语言模型，为隧道施工提供智能化的风险识别与处置方案推荐。

### 核心功能

- **🔗 知识图谱构建**：自动从多源文档中抽取实体与关系，构建Neo4j知识图谱
- **🔍 智能检索**：基于NER、动态权重模型和向量嵌入的混合检索引擎
- **🎯 处置推荐**：结合意图识别的隧道不良地质处置方案智能推荐
- **💬 Web交互**：基于Gradio的对话式推荐界面
- **🤖 LLM增强**：集成大语言模型生成专业的处置建议

### 应用场景

- TBM隧道施工风险防控
- 隧道不良地质（岩爆、掉块、突涌、塌方等）处置方案推荐
- 施工安全预警与决策支持
- 领域知识管理与经验沉淀

---

## 🏗️ 系统架构

```
GeoMRF V2
│
├── kg_construction/     # 知识图谱构建模块
│   ├── core/
│   │   ├── chunking/        # 文档分块与分类
│   │   ├── extraction/      # 实体关系抽取（规则/LLM）
│   │   ├── storage/         # Neo4j图数据库操作
│   │   ├── graph_inference/ # 隐式关系推理
│   │   └── embedding/       # 向量嵌入生成
│   └── scripts/             # 构建脚本
│
├── retrieval/            # 智能检索模块
│   ├── core/
│   │   ├── search_engine.py    # 混合检索引擎
│   │   └── query_pipeline.py   # 检索流程编排
│   ├── models/
│   │   ├── ner/                # 命名实体识别
│   │   └── dynamic_weight/     # 动态权重模型
│   └── utils/              # 工具函数
│
└── recommendation/       # 隧道不良地质处置推荐模块
    ├── core/
    │   ├── recommendation_engine.py   # 推荐引擎
    │   ├── conversation_manager.py    # 对话管理
    │   ├── feedback_handler.py        # 反馈处理
    │   └── gradio_demo.py             # Web界面
    ├── models/
    │   ├── intention/         # 意图识别（BiLSTM）
    │   └── llm/               # LLM客户端
    └── utils/              # 配置管理
```

### 技术栈

| 类别 | 技术选型 |
|------|---------|
| **图数据库** | Neo4j + py2neo |
| **深度学习** | PyTorch, Transformers, Sentence-Transformers |
| **NLP模型** | BERT-base-chinese, BGE-large-zh-v1.5 |
| **LLM** | OpenAI兼容API（Qwen等） |
| **Web界面** | Gradio |
| **文档处理** | pdfplumber, PyPDF2 |

---

## ✨ 功能特性

### 1. 知识图谱构建

- **多源数据支持**：PDF变更纪要、JSON地质预报、处置记录、设计资料
- **智能分块**：基于文档类型自适应分块策略
- **混合抽取**：
  - 规则抽取（正则表达式）
  - 字典抽取（地质术语词典）
  - LLM抽取（大模型语义理解）
- **关系推理**：基于里程区间、属性匹配的隐式关系推理
- **向量化**：为节点生成BGE向量嵌入，支持语义检索

### 2. 智能检索

- **多阶段检索流程**：
  1. 关键信息提取（里程、线路、风险类型）
  2. NER实体识别
  3. 混合检索（BM25 + 向量 + 图谱）
  4. 动态权重融合
- **实体识别**：基于BiLSTM-CRF的地质实体识别
- **动态权重**：学习不同检索方法的最佳组合

### 3. 隧道不良地质处置推荐

- **意图识别**：BiLSTM二分类模型判断是否为处置相关查询
- **方案推荐**：基于图谱路径检索相关处置方案
- **对话管理**：多轮对话、方案切换、反馈学习
- **LLM增强**：结合检索结果生成专业处置建议

---

## 🚀 安装部署

### 环境要求

- Python >= 3.8
- Neo4j >= 4.4
- CUDA >= 11.0（可选，用于GPU加速）

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/GeoMRF.git
cd GeoMRF/GeoMRFV2
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置Neo4j数据库

编辑 `config/kg_construction/core/storage/neo4j_config.json`：

```json
{
  "uri": "bolt://localhost:7687",
  "username": "neo4j",
  "password": "YOUR_NEO4J_PASSWORD_HERE",
  "database": "neo4j",
  "batch_size": 100
}
```

### 4. 配置LLM（可选）

编辑 `config/llm_config.json`：

```json
{
  "current_model": "qwen",
  "models": {
    "qwen": {
      "api_base": "YOUR_API_URL",
      "api_key": "YOUR_API_KEY_HERE",
      "model_name": "gpt-4o",
      "temperature": 0.7,
      "max_tokens": 2000
    }
  }
}
```

### 5. 下载预训练模型

由于模型权重文件较大，需要自行下载并放置在指定目录：

#### 方式一：使用 Hugging Face CLI（推荐）

```bash
# 安装 huggingface-hub
pip install huggingface-hub

# 下载 BERT 模型到 models/bert-base-chinese/
huggingface-cli download bert-base-chinese --local-dir models/bert-base-chinese

# 下载 BGE 模型到 models/bge-large-zh-v1.5/
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/bge-large-zh-v1.5
```

#### 方式二：手动下载

1. **BERT-base-chinese**
   - 下载地址：https://huggingface.co/bert-base-chinese
   - 放置路径：`models/bert-base-chinese/`

2. **BGE-large-zh-v1.5**
   - 下载地址：https://huggingface.co/BAAI/bge-large-zh-v1.5
   - 放置路径：`models/bge-large-zh-v1.5/`

下载后的目录结构应为：
```
models/
├── bert-base-chinese/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
└── bge-large-zh-v1.5/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

---

## 🏃 快速开始

### 1. 构建知识图谱

```bash
# 准备数据文件，放入 data/ 目录
# 支持格式：PDF（变更纪要）、JSON（地质预报、处置记录等）

# 分块文档
python -m kg_construction.scripts.chunk_documents

# 抽取实体关系
python -m kg_construction.scripts.extract_triples

# 构建图谱
python -m kg_construction.scripts.build_kg

# 生成向量嵌入（可选）
python -m kg_construction.scripts.generate_embeddings
```

### 2. 启动推荐系统Web界面

```bash
cd recommendation/core
python gradio_demo.py
```

访问 `http://localhost:7860` 使用Web界面。

### 3. 使用示例

**查询示例**：
- "IV级深埋硬质岩掉块如何处置？"
- "DK15+200处出现岩爆风险，有什么应对措施？"
- "富水破碎带的应急响应方案"

---

## 📁 项目结构

```
GeoMRFV2/
│
├── config/                   # 配置文件目录
│   ├── llm_config.json       # LLM配置
│   └── kg_construction/
│       └── core/
│           └── storage/
│               └── neo4j_config.json
│
├── kg_construction/          # 知识图谱构建模块
│   ├── core/
│   │   ├── chunking/        # 文档分块
│   │   ├── extraction/      # 实体抽取
│   │   ├── storage/         # 图存储
│   │   ├── graph_inference/ # 关系推理
│   │   └── embedding/       # 向量化
│   └── scripts/             # 构建脚本
│
├── retrieval/               # 智能检索模块
│   ├── core/                # 检索引擎
│   ├── models/              # NER、动态权重模型
│   └── utils/               # 工具函数
│
├── recommendation/          # 隧道不良地质处置推荐模块
│   ├── core/                # 推荐引擎、Web界面
│   ├── models/              # 意图识别、LLM客户端
│   └── utils/               # 配置管理
│
├── models/                  # 预训练模型
│   ├── bert-base-chinese/
│   └── bge-large-zh-v1.5/
│
├── data/                    # 数据目录
├── requirements.txt         # 依赖列表
└── README.md               # 项目文档
```

---

## ⚙️ 配置说明

### 知识图谱构建配置

- `kg_construction/core/extraction/config.py`：节点类型、关系类型、抽取策略配置
- `kg_construction/core/graph_inference/inference_config.py`：隐式关系推理规则

### 检索配置

- `retrieval/utils/config.py`：模型路径、检索阈值、Top-K配置

### 推荐配置

- `recommendation/utils/config.py`：意图识别阈值、路径配置
- `recommendation/data/prompt_template.json`：LLM提示词模板

---

## 📊 数据流程

```
原始文档 (PDF/JSON)
    ↓
文档分块
    ↓
实体关系抽取 → 规则/字典/LLM
    ↓
Neo4j图谱构建
    ↓
向量嵌入生成 (BGE)
    ↓
智能检索 (BM25 + 向量 + 图谱)
    ↓
隧道不良地质处置推荐 (意图识别 + 方案检索)
    ↓
LLM增强生成
    ↓
用户界面 (Gradio)
```

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/GeoMRF/issues)
- 邮箱：your.email@example.com

---

## 🙏 致谢

- BERT模型来自 [Hugging Face](https://huggingface.co/)
- BGE模型来自 [BAAI](https://github.com/FlagOpen/FlagEmbedding)
- 感谢所有贡献者的支持
