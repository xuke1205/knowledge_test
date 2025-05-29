"""
简化版知识库Web应用 - 使用内存向量存储

这个脚本是一个简化版的Web应用，使用内存向量存储，避免ChromaDB持久化客户端的API问题。
"""

import os
import streamlit as st
import tempfile
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import time
from knowledge_base import KnowledgeBaseConstructor

# 初始化设置
st.set_page_config(page_title="企业知识库系统", layout="wide")
st.title("企业知识库系统")

# 初始化session state
if 'kb' not in st.session_state:
    # 使用内存中的知识库
    st.session_state.kb = KnowledgeBaseConstructor(
        embedding_model_name="all-MiniLM-L6-v2",
        collection_name="knowledge_base",
        db_directory=None,  # 使用内存存储
        tenant="default_tenant",
        database="default_database"
    )
    
if 'doc_count' not in st.session_state:
    st.session_state.doc_count = 0
    
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# 侧边栏 - 文件上传
with st.sidebar:
    st.header("文档上传")
    
    uploaded_files = st.file_uploader(
        "选择文档上传", 
        type=["txt", "pdf", "docx", "xlsx", "pptx", "csv", "json"],
        accept_multiple_files=True
    )
    
    # 默认分段参数 (隐藏UI控件)
    do_chunk = True
    chunk_size = 500
    
    if uploaded_files and st.button("处理文档"):
        progress_bar = st.progress(0)
        processed_files = 0
        
        for i, file in enumerate(uploaded_files):
            try:
                # 保存临时文件
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.getbuffer())
                    temp_path = tmp.name
                
                # 读取文件内容 (简化实现)
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 分段处理
                paragraphs = [p for p in content.split('\n') if p.strip()]
                chunks = []
                for para in paragraphs:
                    if len(para) > chunk_size:
                        for i in range(0, len(para), chunk_size):
                            chunks.append(para[i:i+chunk_size])
                    else:
                        chunks.append(para)
                
                # 添加到知识库
                try:
                    doc_ids = st.session_state.kb.batch_add_documents([{
                        "text": "\n".join(chunks),
                        "metadata": {"source": file.name, "file_type": file.type}
                    }], segment=False)
                except Exception as e:
                    st.error(f"处理文件失败: {file.name} - 文件格式可能不受支持 ({str(e)})")
                    continue
                st.session_state.doc_count += len(chunks)
                processed_files += 1
                
            except Exception as e:
                st.error(f"处理文件失败: {file.name} - {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
            progress_bar.progress(min((i+1)/len(uploaded_files), 1.0))
        
        st.success(f"成功处理 {processed_files}/{len(uploaded_files)} 个文件")

# 主界面标签
tab1, tab2, tab3 = st.tabs(["搜索", "文档管理", "智能问答"])

# 搜索标签
with tab1:
    st.header("语义搜索")
    
    search_query = st.text_input("输入搜索查询")
    if search_query and st.button("搜索"):
        try:
            # 使用session中的知识库
            results = st.session_state.kb.search_kb(search_query, n_results=5)
            
            # 显示结果
            st.subheader("搜索结果")
            for i, (doc, meta, score) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                with st.expander(f"结果 {i+1} (相关性: {1-score:.2f})"):
                    st.write(f"来源: {meta.get('source','未知')}")
                    st.write(doc)
        except Exception as e:
            st.error(f"搜索出错: {str(e)}")

# 文档管理标签
with tab2:
    st.header("文档管理")
    
    if st.session_state.doc_count == 0:
        st.info("知识库为空")
    else:
        # 获取文档统计
        results = st.session_state.kb.search_kb("", n_results=st.session_state.doc_count)
        file_stats = {}
        
        for meta in results['metadatas'][0]:
            file_name = meta.get("source", "未知")
            if file_name not in file_stats:
                file_stats[file_name] = {
                    "count": 0,
                    "upload_time": "2023-01-01"  # 实际应记录真实时间
                }
            file_stats[file_name]["count"] += 1
        
        # 显示文件列表
        for file_name, stats in file_stats.items():
            with st.expander(f"{file_name} ({stats['count']}段落)"):
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"上传时间: {stats['upload_time']}")
                    st.write(f"段落数: {stats['count']}")
                with col2:
                    st.download_button(
                        "下载", 
                        data="", 
                        file_name=file_name,
                        disabled=True,
                        help="内存存储不支持下载"
                    )
                    st.button(
                        "删除",
                        key=f"del_{file_name}",
                        disabled=True,
                        help="内存存储不支持删除"
                    )
        
        # 统计信息
        st.subheader("统计")
        col1, col2 = st.columns(2)
        col1.metric("文件数", len(file_stats))
        col2.metric("总段落数", st.session_state.doc_count)

# 智能问答标签
with tab3:
    st.header("智能问答")
    
    question = st.text_input("输入您的问题")
    if question and st.button("获取答案"):
        try:
            # 使用session中的知识库
            results = st.session_state.kb.search_kb(question, n_results=3)
            
            # 简单问答实现 (实际应该集成LLM)
            context = "\n\n".join(results['documents'][0])
            answer = f"根据知识库内容，相关信息如下:\n\n{context}"
            
            st.success(answer)
            st.info("注: 这是基于检索的简单回答，完整LLM集成将在后续版本实现")
        except Exception as e:
            st.error(f"问答出错: {str(e)}")

st.caption("当前使用内存存储 - 数据在页面刷新后将丢失")
