# 使用虚拟环境解决依赖冲突

为了解决 `llama_index` 及其他库的循环依赖问题，使用专用的Python虚拟环境是最佳实践。这种方法的优点是不需要修改原始代码，同时可以确保所有组件在隔离的环境中正常工作。

## 什么是虚拟环境？

虚拟环境是一个独立的Python解释器环境，它有自己的包安装目录，与其他虚拟环境和全局（系统级）Python环境隔离。这意味着您可以在一个虚拟环境中安装特定版本的包，而不会影响其他项目或系统环境。

## 为什么使用虚拟环境？

使用虚拟环境可以解决以下问题：

1. **依赖冲突**：不同项目可能需要同一个库的不同版本
2. **循环依赖**：某些库之间可能存在循环引用问题
3. **环境污染**：避免在系统Python环境中安装过多包
4. **版本隔离**：确保项目在特定版本的依赖上正常工作
5. **可复制性**：方便在其他机器上重现相同的环境

## 设置虚拟环境 (Windows)

我们提供了一个批处理脚本 `setup_venv.bat` 来自动化虚拟环境的创建过程。

### 使用自动化脚本

1. 在命令提示符或PowerShell中导航到项目目录：
   ```
   cd d:\mark重要资料\G-SQL\workdata\mvp\knowledge_base_poc
   ```

2. 运行设置脚本：
   ```
   setup_venv.bat
   ```

3. 脚本将自动:
   - 创建名为 `kb_venv` 的虚拟环境
   - 激活该环境
   - 安装所有必要的依赖（指定版本以避免冲突）
   - 保持命令窗口打开

### 手动设置步骤

如果您想手动创建虚拟环境，请按照以下步骤操作：

1. 创建虚拟环境：
   ```
   python -m venv kb_venv
   ```

2. 激活虚拟环境：
   ```
   kb_venv\Scripts\activate
   ```

3. 升级pip：
   ```
   python -m pip install --upgrade pip
   ```

4. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用虚拟环境

激活虚拟环境后，您的命令提示符前会显示 `(kb_venv)`，表示您正在虚拟环境中工作。

现在您可以运行知识库示例：

```
python knowledge_base.py
```

所有命令都将在虚拟环境的上下文中执行，使用虚拟环境中安装的包。

## 停用虚拟环境

完成工作后，您可以停用虚拟环境：

```
deactivate
```

## 依赖版本说明

为了确保所有组件能够协同工作，我们在脚本中指定了以下依赖版本：

- onnxruntime==1.16.3：用于ChromaDB的嵌入函数计算
- sentence-transformers==2.3.1：用于文本向量化
- chromadb==0.4.22：向量数据库
- llama-index==0.9.26：文本索引和检索框架
- pymupdf==1.23.7：PDF处理
- python-docx==1.0.1：Word文档处理
- unidecode==1.3.7：文本规范化
- 其他支持库（numpy, pandas, tqdm等）

这些版本已经过测试，可以一起正常工作，避免循环依赖问题。

## 关于onnxruntime和ChromaDB的特别说明

### onnxruntime依赖
ChromaDB依赖onnxruntime来处理其默认嵌入功能。如果不安装onnxruntime，您可能会遇到以下错误：

```
ImportError: DLL load failed while importing onnxruntime_pybind11_state: 动态链接库(DLL)初始化例程失败
```

```
ValueError: The onnxruntime python package is not installed. Please install it with `pip install onnxruntime`
```

我们已在设置脚本中添加了onnxruntime的安装。如果您使用的是NVIDIA GPU并希望加速，可以改为安装onnxruntime-gpu版本：

```
pip install onnxruntime-gpu==1.16.3
```

### ChromaDB接口变更
从ChromaDB 0.4.16版本开始，嵌入函数接口发生了变化。如果您使用自定义嵌入函数，需要确保：

1. `__call__`方法的参数名为`input`而不是`texts`：
```python
def __call__(self, input: List[str]) -> List[List[float]]:
    # 而不是 def __call__(self, texts: List[str])
    ...
```

2. 如果您遇到以下错误：
```
Expected EmbeddingFunction.__call__ to have the following signature: odict_keys(['self', 'input']), got odict_keys(['self', 'texts'])
```

说明您的嵌入函数接口与ChromaDB的期望不匹配。请参考：https://docs.trychroma.com/embeddings

### ChromaDB租户与数据库参数
在最新版本的ChromaDB中（0.4.22+），初始化客户端时需要额外的tenant和database参数：

1. 如果您遇到以下错误：
```
ValueError: Could not connect to tenant default_tenant. Are you sure it exists?
```

需要在初始化ChromaDB客户端时指定tenant和database参数：
```python
client = chromadb.PersistentClient(
    path="./chroma_db",
    tenant="default",  # 添加租户参数
    database="default"  # 添加数据库参数
)
```

我们的实现已经更新，适配了ChromaDB的最新API要求。

## 常见问题解决

1. **ImportError 或 ModuleNotFoundError**
   - 确保虚拟环境已正确激活（提示符前应显示 `(kb_venv)`）
   - 尝试重新安装特定的依赖：`pip install <package_name>==<version>`

2. **循环导入错误**
   - 如果仍然出现循环导入错误，可能需要调整导入顺序或结构
   - 考虑使用我们提供的代码修改方案（使用自定义实现替代问题组件）

3. **虚拟环境激活失败**
   - 确保您有足够的权限创建和运行虚拟环境
   - 尝试使用管理员权限运行命令提示符
   - 检查Python环境变量是否正确设置

## 进一步阅读

- [Python官方虚拟环境文档](https://docs.python.org/zh-cn/3/library/venv.html)
- [pip用户指南](https://pip.pypa.io/en/stable/user_guide/)
- [Python依赖管理最佳实践](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
