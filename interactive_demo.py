"""
交互式知识库演示应用

这个脚本提供了一个简单的交互式界面，用于:
1. 导入自定义文档到知识库
2. 对知识库进行语义搜索查询
3. 查看知识库统计信息
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any, Optional

from knowledge_base import KnowledgeBaseConstructor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveKnowledgeBase:
    """交互式知识库应用"""
    
    def __init__(self, kb_directory: str = "./custom_kb_db"):
        """
        初始化交互式应用
        
        Args:
            kb_directory: 知识库数据目录
        """
        # 初始化知识库系统
        self.kb = KnowledgeBaseConstructor(
            embedding_model_name="all-MiniLM-L6-v2", 
            collection_name="custom_kb",
            db_directory=kb_directory
        )
        
        self.kb_directory = kb_directory
        self.running = True
        print(f"\n{'='*20} 企业知识库系统 {'='*20}\n")
    
    def show_menu(self):
        """显示主菜单"""
        print("\n请选择操作:")
        print("1. 导入文档到知识库")
        print("2. 搜索知识库")
        print("3. 显示知识库统计信息")
        print("4. 清除知识库")
        print("0. 退出")
        
        choice = input("\n请输入选项 [0-4]: ").strip()
        
        if choice == "1":
            self.import_documents()
        elif choice == "2":
            self.search_knowledge_base()
        elif choice == "3":
            self.show_stats()
        elif choice == "4":
            self.clear_database()
        elif choice == "0":
            self.running = False
            print("退出应用. 再见!")
        else:
            print("无效选项，请重试")
    
    def import_documents(self):
        """导入文档到知识库"""
        print("\n==== 文档导入 ====")
        print("支持的文件类型: .txt, .pdf, .docx, .csv, .json, .xml")
        
        # 获取文档路径
        doc_path = input("\n请输入文档路径或目录 (输入'c'取消): ").strip()
        
        if doc_path.lower() == 'c':
            print("导入已取消")
            return
        
        # 处理输入
        if os.path.isfile(doc_path):
            # 单个文件
            self._process_file(doc_path)
        elif os.path.isdir(doc_path):
            # 处理目录
            self._process_directory(doc_path)
        else:
            print(f"错误: 找不到路径 '{doc_path}'")
    
    def _process_file(self, file_path: str) -> bool:
        """处理单个文件"""
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 '{file_path}'")
            return False
        
        # 检查文件类型
        _, ext = os.path.splitext(file_path)
        supported_exts = ['.txt', '.pdf', '.docx', '.csv', '.json', '.xml']
        if ext.lower() not in supported_exts:
            print(f"警告: 不支持的文件类型 '{ext}'. 支持的类型: {', '.join(supported_exts)}")
            confirm = input("是否仍然尝试处理这个文件? (y/n): ").lower() == 'y'
            if not confirm:
                return False
        
        # 处理文件
        try:
            print(f"正在处理文件: {file_path}")
            
            # 询问是否分段
            segment = input("是否对文档进行语义分段? (y/n, 默认y): ").lower() != 'n'
            
            # 添加到知识库
            start_time = time.time()
            doc_ids = self.kb.add_document_to_kb(file_path, segment=segment)
            processing_time = time.time() - start_time
            
            print(f"处理完成! 用时: {processing_time:.2f}秒")
            print(f"添加了 {len(doc_ids)} 个文档段落")
            
            return True
        except Exception as e:
            print(f"处理文件出错: {e}")
            return False
    
    def _process_directory(self, dir_path: str):
        """处理目录中的所有支持的文件"""
        if not os.path.exists(dir_path):
            print(f"错误: 目录不存在 '{dir_path}'")
            return
        
        # 查找支持的文件
        supported_exts = ['.txt', '.pdf', '.docx', '.csv', '.json', '.xml']
        files_to_process = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in supported_exts:
                    files_to_process.append(os.path.join(root, file))
        
        if not files_to_process:
            print(f"在目录 '{dir_path}' 中没有找到支持的文件")
            return
        
        print(f"找到 {len(files_to_process)} 个支持的文件:")
        for i, file_path in enumerate(files_to_process):
            print(f"{i+1}. {file_path}")
        
        # 询问用户确认
        confirm = input(f"\n确认处理这些文件? (y/n): ").lower() == 'y'
        if not confirm:
            print("处理已取消")
            return
        
        # 询问是否分段
        segment = input("是否对文档进行语义分段? (y/n, 默认y): ").lower() != 'n'
        
        # 处理文件
        successful = 0
        start_time = time.time()
        
        try:
            results = self.kb.batch_add_documents(files_to_process, segment=segment)
            total_segments = sum(len(doc_ids) for doc_ids in results.values() if doc_ids)
            
            # 计算成功数量
            for file_path, doc_ids in results.items():
                if doc_ids:  # 非空列表表示成功
                    successful += 1
            
            processing_time = time.time() - start_time
            print(f"\n处理完成! 用时: {processing_time:.2f}秒")
            print(f"成功: {successful}/{len(files_to_process)} 个文件")
            print(f"总计添加: {total_segments} 个文档段落")
            
        except Exception as e:
            print(f"批量处理出错: {e}")
    
    def search_knowledge_base(self):
        """搜索知识库"""
        print("\n==== 知识库搜索 ====")
        
        # 检查知识库状态
        stats = self.kb.get_stats()
        if stats["segments_created"] == 0:
            print("知识库为空! 请先导入一些文档。")
            return
        
        # 获取搜索查询
        print("输入您的搜索查询 (输入'c'取消):")
        while self.running:
            query = input("\n> ").strip()
            
            if query.lower() == 'c':
                break
                
            if not query:
                continue
                
            # 设置结果数量
            try:
                num_results = int(input("返回结果数量 [默认5]: ").strip() or "5")
            except ValueError:
                num_results = 5
                print("使用默认值: 5")
            
            # 执行搜索
            try:
                print("\n搜索中...")
                start_time = time.time()
                results = self.kb.search_kb(query, n_results=num_results)
                search_time = time.time() - start_time
                
                # 显示结果
                self._display_search_results(query, results, search_time)
                
                # 询问是否继续
                if input("\n继续搜索? (y/n, 默认y): ").lower() == 'n':
                    break
                    
            except Exception as e:
                print(f"搜索出错: {e}")
    
    def _display_search_results(self, query: str, results: Dict[str, Any], search_time: float):
        """格式化显示搜索结果"""
        print(f"\n查询: '{query}'")
        print(f"搜索用时: {search_time*1000:.2f}毫秒")
        print("="*60)
        
        if not results or not results['documents'] or not results['documents'][0]:
            print("没有找到匹配的结果")
            return
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            relevance_score = 1.0 - distance  # 转换为相关度分数
            
            print(f"\n结果 {i+1}:")
            print(f"- 来源: {metadata.get('file_name', '未知')}")
            print(f"- 相关度: {relevance_score:.2%}")
            print("-"*40)
            
            # 内容太长时截取
            max_display_len = 400
            if len(doc) > max_display_len:
                truncated_doc = doc[:max_display_len] + "..."
            else:
                truncated_doc = doc
                
            print(truncated_doc)
            print("-"*60)
    
    def show_stats(self):
        """显示知识库统计信息"""
        print("\n==== 知识库统计 ====")
        
        stats = self.kb.get_stats()
        
        print(f"文档处理数: {stats['documents_processed']}")
        print(f"段落生成数: {stats['segments_created']}")
        print(f"向量存储数: {stats['vectors_stored']}")
        
        if stats['document_types']:
            print("\n文档类型统计:")
            for ext, count in stats['document_types'].items():
                print(f"- {ext}: {count}个")
        
        print(f"\n知识库位置: {self.kb_directory}")
    
    def clear_database(self):
        """清除知识库"""
        print("\n==== 清除知识库 ====")
        
        # 确认
        confirm = input("警告: 这将删除所有知识库数据! 确定吗? (yes/no): ").lower()
        
        if confirm != 'yes':
            print("操作已取消")
            return
        
        try:
            # 删除数据库目录
            if os.path.exists(self.kb_directory):
                import shutil
                shutil.rmtree(self.kb_directory, ignore_errors=True)
                print("正在重新初始化知识库...")
                
                # 重新初始化知识库
                self.kb = KnowledgeBaseConstructor(
                    embedding_model_name="all-MiniLM-L6-v2", 
                    collection_name="custom_kb",
                    db_directory=self.kb_directory
                )
                
                print("知识库已清除并重新初始化")
            else:
                print("知识库目录不存在，无需清除")
        except Exception as e:
            print(f"清除知识库出错: {e}")
    
    def run(self):
        """运行交互式循环"""
        while self.running:
            self.show_menu()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="交互式企业知识库应用")
    parser.add_argument("--db-dir", type=str, default="./custom_kb_db", 
                        help="知识库数据目录")
    args = parser.parse_args()
    
    try:
        app = InteractiveKnowledgeBase(kb_directory=args.db_dir)
        app.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
