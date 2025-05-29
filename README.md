# 企业知识库构建系统 - MVP原型

本项目是企业知识库构建方案的原型（MVP）实现，用于验证原设计方案的可行性。系统演示了从文档导入、清洗、语义分段到向量化与存储的完整流程。

## 项目结构

- `document_processor.py`: 处理PDF/DOCX/TXT等文档格式
- `structured_data_processor.py`: 处理CSV/JSON/XML等结构化数据格式
- `segmenter.py`: 实现文本的语义分段
- `vectorizer.py`: 负责文本向量化和向量存储
- `knowledge_base.py`: 主程序，整合所有组件并提供示例
- `api_server.py`: REST API服务接口
- `streamlit_app.py`: 基于Streamlit的Web界面
- `setup_venv.bat`: 自动设置虚拟环境的批处理脚本
- `requirements.txt`: 依赖包列表
- `VENV_GUIDE.md`: 使用虚拟环境的详细指南

## 解决循环依赖问题的两种方法

运行该项目时，您可能遇到了llama_index库的循环依赖问题。我们提供了两种解决方案：

### 方案1：使用专用虚拟环境（推荐）

虚拟环境可以隔离项目依赖，避免与系统环境冲突，是解决依赖问题的最佳实践。

1. 使用提供的批处理脚本创建虚拟环境：
   ```
   setup_venv.bat
   ```

2. 脚本会自动创建环境并安装所需依赖，完成后可以运行：
   ```
   python knowledge_base.py
   ```

3. 详细指南请参考 [VENV_GUIDE.md](VENV_GUIDE.md)

### 方案2：使用我们的代码修改

如果您不方便使用虚拟环境，我们也提供了代码级别的修复方案，通过替换problematic imports为自定义实现来解决循环引用问题。

1. 我们已替换了`segmenter.py`中的llama_index依赖，改为自定义实现
2. 修改后的代码保持原有API不变，但避免了循环导入问题

## 运行示例

### 基本演示
```bash
python knowledge_base.py
```
基本演示将创建样例文档，将它们添加到知识库中并执行一系列预定义的查询，展示系统的基本功能。

### 交互式命令行界面
```bash
python interactive_demo.py
```
交互式演示提供了一个命令行菜单界面，让您可以：
- 导入自己的文档（单个文件或整个目录）
- 对知识库进行自定义查询
- 查看知识库统计信息
- 清除和重置知识库

### 网页图形界面

原版Web应用（如果遇到ChromaDB API兼容性问题）：
```bash
streamlit run web_app.py
```

简化版Web应用（推荐使用，内存存储无兼容问题）：
```bash
streamlit run simple_web_demo.py
```

网页应用提供了友好的图形界面，包含：
- 文件上传与处理（支持TXT、PDF、Office格式等）
- 交互式搜索与结果展示
- 知识库统计与可视化
- 搜索历史记录

要运行网页界面，您需要先安装streamlit和相关依赖：
```bash
pip install streamlit pandas python-docx openpyxl python-pptx
```

#### Office文件支持

简化版Web应用支持以下Office格式：
- Word文档 (.docx)：提取文本内容和段落
- Excel表格 (.xlsx, .xls)：处理多个工作表的表格数据
- PowerPoint演示文稿 (.pptx)：提取幻灯片中的文本内容

## 当前实现的功能

- ✅ 多格式文档处理 (PDF, DOCX, TXT)
- ✅ 结构化数据支持 (JSON, CSV, XML)
- ✅ 文本清洗与规范化
- ✅ 基于语义的文本分段
- ✅ 文本向量化与存储
- ✅ 语义搜索功能
- ✅ API服务接口
- ✅ 用户友好的Web界面

## 性能说明

当前MVP实现使用sentence-transformers作为向量化模型，ChromaDB作为向量存储。这些组件在没有GPU的环境中也能运行，但处理大量文档时性能会受限。

对于生产环境，建议：
- 使用GPU加速模型推理
- 考虑使用高性能向量数据库（如Milvus, FAISS）
- 实现增量更新机制减少处理开销

## 扩展与改进方向

- 开发更精细的语义分段算法
- 实现领域专用embedding模型的微调
- 增加文档变更检测和增量更新能力
- 添加反馈驱动的优化循环
- 增强多模态内容支持
- 改进向量存储的分片和可扩展性

## 许可证

[许可证信息]

---

此项目验证了在企业知识库构建方案设计中描述的核心功能的可行性，为后续完整实施提供了基础。
