import os
import pymupdf
import chromadb

def load_documents(data_dir="data"):
    docs = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            doc = pymupdf.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            docs.append({"filename": filename, "content": text})
            print(f"已加载PDF: {filename} ({len(text)} 字符)")
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"filename": filename, "content": text})
            print(f"已加载TXT: {filename} ({len(text)} 字符)")
    return docs

def chunk_documents(docs, chunk_size=800, overlap=100):
    chunks = []
    for doc in docs:
        content = doc["content"]
        content = " ".join(content.split())
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "source": doc["filename"]
                })
            start = end - overlap
    print(f"共切割成 {len(chunks)} 个chunk")
    return chunks

def build_index(chunks):
    client = chromadb.PersistentClient(path="index")
    try:
        client.delete_collection("medical_docs")
    except:
        pass
    collection = client.create_collection("medical_docs")
    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"]} for c in chunks]
    print("正在建立索引...")
    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    print(f"索引建立完成，共 {collection.count()} 个chunk")

if __name__ == "__main__":
    docs = load_documents("data")
    if not docs:
        print("没有找到文件")
    else:
        chunks = chunk_documents(docs)
        build_index(chunks)
        print("完成！")
