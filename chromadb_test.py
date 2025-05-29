"""
ChromaDB连接测试脚本

这个简单脚本用于测试ChromaDB连接并创建一个基本集合，确保我们能正确处理tenant和database参数。
"""

import os
import chromadb
from chromadb.config import Settings

def test_chromadb_connection():
    """测试ChromaDB连接并创建基本集合"""
    print("开始测试ChromaDB连接...")
    
    # 确保目录存在
    db_path = "./test_chromadb"
    os.makedirs(db_path, exist_ok=True)
    
    # 尝试不同的参数组合
    try:
        # 方法1: 使用最新API (0.4.22+)
        print("\n尝试方法1: 使用tenant和database参数")
        client = chromadb.PersistentClient(
            path=db_path + "/method1",
            tenant="default",
            database="default"
        )
        collection = client.create_collection(name="test_collection_1")
        print(f"成功创建集合: {collection.name}")
    except Exception as e:
        print(f"方法1失败: {e}")
    
    try:
        # 方法2: 使用旧版API
        print("\n尝试方法2: 不使用tenant和database参数")
        client = chromadb.PersistentClient(path=db_path + "/method2")
        collection = client.create_collection(name="test_collection_2")
        print(f"成功创建集合: {collection.name}")
    except Exception as e:
        print(f"方法2失败: {e}")
    
    try:
        # 方法3: 内存客户端
        print("\n尝试方法3: 使用内存客户端")
        client = chromadb.EphemeralClient()
        collection = client.create_collection(name="test_collection_3")
        print(f"成功创建集合: {collection.name}")
    except Exception as e:
        print(f"方法3失败: {e}")
        
    # 打印ChromaDB版本信息
    print(f"\nChromaDB版本: {chromadb.__version__}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_chromadb_connection()
