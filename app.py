import os
import pymupdf
import chromadb
import anthropic
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"filename": filename, "content": text})
    return docs

def chunk_documents(docs, chunk_size=800, overlap=100):
    chunks = []
    for doc in docs:
        content = " ".join(doc["content"].split())
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]
            if chunk_text.strip():
                chunks.append({"text": chunk_text.strip(), "source": doc["filename"]})
            start = end - overlap
    return chunks

@st.cache_resource
def init():
    client = chromadb.PersistentClient(path="index")
    
    # 如果索引不存在就建
    try:
        collection = client.get_collection("medical_docs")
        if collection.count() == 0:
            raise Exception("empty")
    except:
        docs = load_documents("data")
        chunks = chunk_documents(docs)
        try:
            client.delete_collection("medical_docs")
        except:
            pass
        collection = client.create_collection("medical_docs")
        texts = [c["text"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": c["source"]} for c in chunks]
        collection.add(documents=texts, ids=ids, metadatas=metadatas)
    
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return collection, anthropic_client

def search(collection, query, n=5):
    results = collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return docs, sources

def answer(anthropic_client, query, docs, sources):
    context = "\n\n---\n\n".join(docs)
    unique_sources = list(set(sources))
    system = """You are a helpful assistant that helps people understand medical bills and financial assistance options in Atlanta, Georgia.

Answer questions based ONLY on the provided documents. If the answer is not in the documents, say so clearly.
Be specific, practical, and compassionate. The user may be stressed about medical bills.
Always mention relevant phone numbers, deadlines, or next steps when available.
Keep answers concise but complete."""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"}]
    )
    return message.content[0].text, unique_sources

# UI
st.title("🏥 Medical Bill Helper")
st.caption("Ask questions about medical bills and financial assistance in Atlanta")

with st.sidebar:
    st.header("About")
    st.write("This tool helps you understand medical bills and find financial assistance at Atlanta hospitals.")
    st.write("**Data sources:**")
    st.write("- Grady Memorial Hospital")
    st.write("- Emory Healthcare")
    st.divider()
    st.write("**Example questions:**")
    if st.button("I can't afford my bill"):
        st.session_state.example = "I can't afford my medical bill. What can I do?"
    if st.button("How do I apply for help?"):
        st.session_state.example = "How do I apply for financial assistance?"
    if st.button("Grady vs Emory?"):
        st.session_state.example = "What's the difference between Grady and Emory financial assistance?"
    st.divider()
    st.caption("⚠️ This tool provides general information only and does not constitute legal or medical advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

default_input = st.session_state.pop("example", "") if "example" in st.session_state else ""

if query := st.chat_input("Ask about your medical bill..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Looking up information..."):
            try:
                collection, anthropic_client = init()
                docs, sources = search(collection, query)
                response, unique_sources = answer(anthropic_client, query, docs, sources)
                st.write(response)
                st.caption(f"Sources: {', '.join(unique_sources)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": unique_sources
                })
            except Exception as e:
                import traceback
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
