# StoryRAG

StoryRAG 是一个**本地运行的中文小说 / 长文本知识库问答桌面应用**。它从 `.txt` 文件构建本地 Chroma 向量库,结合元数据与可选的 LLM 预处理增强检索,并以**受检索证据约束**的方式作答、给出引用与可信度护栏。

面向个人阅读 / 研究场景:导入文本 → 建立索引 → 提问 → 查看证据 → 固定上下文继续追问 → 知识库打包导出 / 导入。

> 隐私:所有文本、向量库与 API Key 都留在本地,仓库不包含任何小说原文或密钥。

## 功能特性

- **桌面界面**(PySide6):左侧建库 / 选库,中间对话,右侧证据与详情。
- **本地预处理**:章节切分、分块、实体抽取、关系线索、合成「角色总表」块。
- **混合检索**:稠密向量 + 词面关键词 + 元数据 + 关系信号 + 分卷过滤 + 相邻上下文扩展,按问题类型动态调权。
- **可选 LLM 增强**(DeepSeek 兼容接口):预处理、查询改写、分解规划、答案生成;简单事实问题会**自动跳过分解规划**,仅「只有 / 分别 / 哪些 / 有没有去过」等枚举类问题才触发多路核对。
- **Embedding 客户端**(SiliconFlow / OpenAI 兼容):分批、**并发、超时、失败退避重试**。
- **知识库管理**:多语料导入、增量更新、别名管理、建库包导出 / 导入、调试证据视图。

## 目录结构

| 目录 | 职责 |
|------|------|
| `desktop_app/` | PySide6 桌面界面 |
| `knowledge_base/` | 语料导入、预处理编排、Chroma 写入 |
| `preprocessing/` | 文本加载、分块、元数据抽取、产物持久化 |
| `retrieval/` | 查询分析、混合排序、向量库访问、上下文扩展、格式化 |
| `qa/` | 问答主流程、可选分解、追问证据 |
| `llm/` | OpenAI 兼容客户端、提示词、响应 schema、证据约束校验 |
| `core/` | 配置收口与日志 |
| `tests/` | 策略点与纯函数回归测试 |

## 环境要求

- Python **3.11 / 3.12 / 3.13**
- **DeepSeek 兼容**的对话 API Key(答案生成 + 可选 LLM 预处理)
- **SiliconFlow 兼容**的 Embedding API Key(建库与查询向量化)

安装运行依赖:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

开发 / 测试依赖:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 配置

从 `.env.example` 复制一份本地 `.env`,填入你自己的 Key:

```powershell
Copy-Item .env.example .env
```

必填:

```dotenv
DEEPSEEK_API_KEY=
SILICONFLOW_API_KEY=
```

常用可选项:

```dotenv
DEEPSEEK_MODEL=dsv4pro
DEEPSEEK_API_BASE=https://api.deepseek.com
RAG_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
RAG_EMBEDDING_CONCURRENCY=8       # Embedding 建库并发数,长建库不建议 >16
RAG_DOC_DIR=docs
RAG_PROCESSED_DIR=processed
RAG_CHROMA_DB_DIR=chroma_db
```

更多旋钮见 `.env.example`。应用内的「设置」对话框也能把这些值写回本地 `.env`。

## 使用

启动桌面应用:

```powershell
.\.venv\Scripts\python.exe run_desktop.py
```

直接对 `docs/` 下已有的 `.txt` 建库(命令行):

```powershell
.\.venv\Scripts\python.exe build_db.py
```

打包成 Windows 桌面可执行文件:

```powershell
.\.venv\Scripts\python.exe build_desktop_exe.py
```

典型桌面流程:

1. 启动应用,在「设置」里配置 API Key 与模型。
2. 导入本地 `.txt` 到新建或已有知识库,等待建库完成。
3. 在左侧勾选搜索范围(可选到具体卷),提问。
4. 在右侧查看检索证据,固定有用的上下文用于追问。
5. 需要跨机器迁移时,导出 / 导入建库包。

## 开发

跑测试:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

静态检查:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall app.py app_services.py build_db.py build_desktop_exe.py core desktop_app knowledge_base llm preprocessing qa retrieval run_desktop.py
```

界面冒烟自检(不弹窗,仅验证可加载):

```powershell
.\.venv\Scripts\python.exe run_desktop.py --smoke-test
```

CI(GitHub Actions)会在干净环境里跑 ruff / compileall / pytest。

## 数据与版权

本仓库**刻意不包含**小说原文、受版权文本、生成的 Chroma 向量库、预处理产物、缓存、本地虚拟环境、IDE 状态以及 API Key。以下本地数据已在 `.gitignore` 中忽略:

```
.env  docs/  processed/  chroma_db/  cache/  .venv/  .claude/
```

请**只导入你拥有、已获授权或法律允许处理的文本**。应用导出的建库包可能包含原文与向量,除非你拥有相应权利,否则请按私有内容对待、不要公开分享。

## 说明

- 简单事实问题会跳过分解规划 LLM,降低单次问答的串行调用与延迟;枚举 / 排他类问题仍走多路核对。
- 相邻上下文扩展默认开启——叙事类问答常依赖紧邻的前后文。
- 内部仍使用了部分 Chroma 私有集合 API:在锁定的依赖版本下可用,后续加固建议用仓储适配层收口。

## 许可证

[MIT](LICENSE)
