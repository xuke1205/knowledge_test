"""
企业知识库Web应用

使用Streamlit构建的简单Web界面，用于演示企业知识库系统。
启动方法: streamlit run web_app.py
"""

import os
import time
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import shutil

from knowledge_base import KnowledgeBaseConstructor

# 页面配置
st.set_page_config(
    page_title="企业知识库系统",
    page_icon="📚",
    layout="wide"
)

# 定义常量
DEFAULT_KB_DIR = "./web_kb_db"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# 会话状态初始化
if 'kb' not in st.session_state:
    st.session_state.kb = None

if 'stats' not in st.session_state:
    st.session_state.stats = None

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# 样式
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E90FF;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .source-info {
        color: #666;
        font-size: 0.8rem;
    }
    .relevance-score {
        color: #1E90FF;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #1E90FF;
    }
    .stat-card {
        background-color: #f0f8ff;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    .highlight {
        background-color: #FFFF00;
    }
</style>
""", unsafe_allow_html=True)

def initialize_kb(kb_dir: str = DEFAULT_KB_DIR, model: str = DEFAULT_MODEL) -> KnowledgeBaseConstructor:
    """
    初始化或获取知识库构造器
    """
    if st.session_state.kb is None:
        with st.spinner('正在初始化知识库系统...'):
            try:
                st.session_state.kb = KnowledgeBaseConstructor(
                    embedding_model_name=model,
                    collection_name="web_kb",
                    db_directory=kb_dir
                )
            except Exception as e:
                st.error(f"初始化知识库失败: {str(e)}")
                st.stop()
            update_stats()
    
    return st.session_state.kb

def update_stats():
    """
    更新知识库统计信息
    """
    if st.session_state.kb is not None:
        st.session_state.stats = st.session_state.kb.get_stats()

def search_kb(query: str, n_results: int = 5):
    """
    执行知识库搜索
    """
    kb = initialize_kb()
    
    with st.spinner('正在搜索...'):
        start_time = time.time()
        results = kb.search_kb(query, n_results=n_results)
        search_time = time.time() - start_time
    
    # 添加到搜索历史
    search_record = {
        "query": query,
        "n_results": len(results['documents'][0]) if results and 'documents' in results and results['documents'] else 0,
        "time": search_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.search_history.append(search_record)
    
    return results, search_time

def clear_kb():
    """
    清除知识库
    """
    kb_dir = DEFAULT_KB_DIR
    if os.path.exists(kb_dir):
        shutil.rmtree(kb_dir, ignore_errors=True)
    
    # 重新初始化
    st.session_state.kb = None
    initialize_kb()
    update_stats()
    
    # 清除搜索历史
    st.session_state.search_history = []

def process_uploaded_file(uploaded_file, segment: bool = True):
    """
    处理上传的文件
    """
    kb = initialize_kb()
    
    # 创建临时文件
    temp_file_path = os.path.join("./temp", uploaded_file.name)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        # 处理文件
        with st.spinner(f'正在处理文件: {uploaded_file.name}...'):
            start_time = time.time()
            doc_ids = kb.add_document_to_kb(temp_file_path, segment=segment)
            processing_time = time.time() - start_time
        
        # 更新统计信息
        update_stats()
        
        return len(doc_ids), processing_time
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def display_stats():
    """
    显示知识库统计信息
    """
    stats = st.session_state.stats
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <h3>文档数</h3>
                <h2>{stats['documents_processed']}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <h3>段落数</h3>
                <h2>{stats['segments_created']}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <h3>向量数</h3>
                <h2>{stats['vectors_stored']}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    if stats['document_types']:
        st.markdown("### 文档类型分布")
        
        # 转换为DataFrame
        df = pd.DataFrame({
            '文件类型': list(stats['document_types'].keys()),
            '数量': list(stats['document_types'].values())
        })
        
        st.bar_chart(df.set_index('文件类型'))

def display_search_history():
    """
    显示搜索历史
    """
    if not st.session_state.search_history:
        st.info("暂无搜索历史")
        return
    
    st.markdown("### 最近的搜索")
    
    # 转换为DataFrame
    df = pd.DataFrame(st.session_state.search_history)
    df.columns = ['查询', '结果数', '用时(秒)', '时间戳']
    df['用时(毫秒)'] = df['用时(秒)'].apply(lambda x: round(x * 1000, 2))
    df = df.drop(columns=['用时(秒)'])
    
    st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

def main():
    """主函数"""
    
    st.markdown('<h1 class="main-title">企业知识库系统</h1>', unsafe_allow_html=True)
    
    # 初始化知识库
    kb = initialize_kb()
    
    # 侧边栏
    with st.sidebar:
        st.header("控制面板")
        
        # 文件上传区域
        st.markdown("## 文档导入")
        
        uploaded_files = st.file_uploader(
            "选择文件上传到知识库", 
            accept_multiple_files=True,
            type=["txt", "pdf", "docx", "csv", "json", "xml"]
        )
        
        segment_docs = st.checkbox("对文档进行语义分段", value=True)
        
        if uploaded_files and st.button("处理所选文件"):
            with st.spinner('处理上传的文件...'):
                total_segments = 0
                successful = 0
                
                # 创建进度条
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    try:
                        n_segments, _ = process_uploaded_file(file, segment=segment_docs)
                        total_segments += n_segments
                        successful += 1
                        # 更新进度
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    except Exception as e:
                        st.error(f"处理文件 '{file.name}' 时发生错误: {str(e)}")
                
                st.success(f"成功处理 {successful}/{len(uploaded_files)} 个文件，共添加 {total_segments} 个段落。")
        
        # 知识库管理
        st.markdown("## 知识库管理")
        
        if st.button("清除知识库", key="clear_kb"):
            if st.session_state.stats and st.session_state.stats['segments_created'] > 0:
                if st.warning("确定要清除所有知识库数据吗?", icon="⚠️"):
                    clear_kb()
                    st.success("知识库已清除")
            else:
                st.info("知识库为空，无需清除")
        
        # 关于部分
        st.markdown("---")
        st.markdown("""
        ### 关于
        
        这是企业知识库系统的Web演示界面，目前支持:
        
        - 文档导入与处理
        - 语义搜索
        - 知识库统计
        
        使用了Streamlit构建界面，结合了先前开发的知识库后端。
        """)
    
    # 主页面标签
    tab1, tab2, tab3 = st.tabs(["搜索", "统计信息", "使用指南"])
    
    # 搜索标签
    with tab1:
        st.markdown('<h2 class="section-title">知识库搜索</h2>', unsafe_allow_html=True)
        
        # 检查知识库是否为空
        if st.session_state.stats and st.session_state.stats['segments_created'] == 0:
            st.info("知识库为空! 请先上传一些文档。")
        
        # 搜索框
        query = st.text_input("输入您的搜索查询:", placeholder="例如: 知识库系统的关键组件有哪些?")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            num_results = st.slider("返回结果数量", min_value=1, max_value=10, value=5)
        with col2:
            search_button = st.button("搜索", key="search_button")
        
        # 执行搜索
        if search_button and query:
            results, search_time = search_kb(query, num_results)
            
            # 显示结果
            st.markdown(f"### 搜索结果 (用时: {search_time*1000:.2f}毫秒)")
            
            if not results or not results['documents'] or not results['documents'][0]:
                st.info("没有找到匹配的结果")
            else:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 转换距离为相关度分数 (0-1范围)
                    # ChromaDB的距离是余弦距离: 0=完全相同 2=完全相反
                    relevance_score = max(0, 1.0 - (distance / 2)) if distance is not None else 0
                    # 设置阈值，只有高于0.3相关度的结果才显示
                    if relevance_score < 0.3:
                        continue
                    
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <h4>结果 {i+1}</h4>
                            <div class="source-info">来源: {metadata.get('file_name', '未知')} | 
                            <span class="relevance-score">相关度: {relevance_score:.2%}</span></div>
                            <div style="margin-top:10px;white-space: pre-wrap;">{doc if isinstance(doc, str) else str(doc)}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        
        # 显示搜索历史
        st.markdown('<h3 class="section-title">搜索历史</h3>', unsafe_allow_html=True)
        display_search_history()
    
    # 统计信息标签
    with tab2:
        st.markdown('<h2 class="section-title">知识库统计信息</h2>', unsafe_allow_html=True)
        
        # 显示统计信息
        if st.session_state.stats:
            display_stats()
            
            # 知识库位置
            st.markdown("### 知识库位置")
            st.code(os.path.abspath(DEFAULT_KB_DIR))
        else:
            st.info("无法获取统计信息，知识库可能未初始化")
    
    # 使用指南标签
    with tab3:
        st.markdown('<h2 class="section-title">使用指南</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 快速入门
        
        1. **导入文档**
           - 从左侧栏上传文件（支持TXT、PDF、DOCX、CSV、JSON和XML）
           - 选择是否对文档进行语义分段
           - 点击处理按钮导入文档
        
        2. **搜索知识库**
           - 在搜索标签页输入问题或关键词
           - 调整返回结果数量
           - 查看搜索结果和相关度评分
        
        3. **查看统计信息**
           - 在统计信息标签页查看知识库状态
           - 了解文档和段落数量分布
        
        ### 注意事项
        
        - 文档导入后会进行自动分段和向量化
        - 搜索使用语义匹配而非关键词匹配
        - 相关度分数越高表示结果与查询越相关
        - 可以随时清除知识库重新开始
        
        ### 高级提示
        
        - 使用完整自然语言问题通常比关键词搜索效果更好
        - 对于特定领域文档，使用该领域的专业术语搜索效果更佳
        - 分段选项影响检索粒度，可根据文档结构选择是否启用
        """)

if __name__ == "__main__":
    main()
