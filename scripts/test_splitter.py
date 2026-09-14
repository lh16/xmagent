"""测试文档加载与分块"""

from src.rag.loader import DocumentLoader
from src.rag.splitter import DocumentSplitter


def main():
    # 1. 加载文档
    print("=" * 60)
    print("步骤1：加载文档")
    print("=" * 60)
    
    documents = DocumentLoader.load_directory("./data/knowledge")
    print(f"加载了 {len(documents)} 个文档")
    
    # 查看第一个文档
    if documents:
        print(f"\n第一个文档预览：")
        print(documents[0].page_content[:200])
        print(f"\n元数据：{documents[0].metadata}")
    
    # 2. 分块
    print("\n" + "=" * 60)
    print("步骤2：文档分块")
    print("=" * 60)
    
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split(documents)
    print(f"切分为 {len(chunks)} 个文本块")
    
    # 查看第一个块
    if chunks:
        print(f"\n第一个块预览：")
        print(chunks[0].page_content[:300])
        print(f"\n块元数据：{chunks[0].metadata}")


if __name__ == "__main__":
    main()