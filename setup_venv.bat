@echo off
REM 设置虚拟环境的脚本

echo 为知识库项目创建虚拟环境...

REM 创建虚拟环境
python -m venv kb_venv

REM 激活虚拟环境
call kb_venv\Scripts\activate.bat

REM 升级pip
python -m pip install --upgrade pip

REM 安装依赖包 (指定版本避免冲突)

echo 安装核心依赖...
pip install onnxruntime==1.16.3
pip install sentence-transformers==2.3.1
pip install chromadb==0.4.22
pip install llama-index==0.9.26

echo 安装文档处理依赖...
pip install unidecode==1.3.7
pip install pymupdf==1.23.7
pip install python-docx==1.0.1
pip install regex==2023.10.3
pip install openpyxl==3.1.2
pip install python-pptx==1.0.1

echo 安装数据处理依赖...
pip install numpy==1.26.2
pip install pandas==2.1.4
pip install tqdm==4.66.1

echo 安装API服务依赖...
pip install fastapi==0.109.0
pip install uvicorn==0.25.0
pip install python-multipart==0.0.6

echo 安装Web界面依赖...
pip install streamlit==1.30.0

echo 安装系统工具...
pip install redis==5.0.1
pip install watchdog==3.0.0

echo.
echo 虚拟环境设置完成!
echo 使用 'kb_venv\Scripts\activate.bat' 激活环境
echo 使用 'python knowledge_base.py' 运行知识库演示
echo.

REM 保持命令窗口打开
cmd /k
