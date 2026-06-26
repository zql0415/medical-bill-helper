import os
import requests
import pymupdf
import chromadb
import anthropic
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 启动时自动下载PDF
PDF_SOURCES = {
    "grady_policy.pdf": "https://www.gradyhealth.org/wp-content/uploads/Financial-Assistance-Program-Policy-1.pdf",
    "emory_summary.pdf": "https://www.emoryhealthcare.org/-/media/Project/EH/Emory/ui/pdfs/insurance/financial-assistance/plain-language-summary-ehc-financial-assistance-policy-2026-final-04-01-2026-aod.pdf"
}

def download_pdfs():
    os.makedirs("data", exist_ok=True)
    for filename, url in PDF_SOURCES.items():
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(filepath, "wb") as f:
                        f.write(r.content)
            except:
                pass

def load_documents(data_dir="data"):
    docs = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            try:
                doc = pymupdf.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                if text.strip():
                    docs.append({"filename": filename, "content": text})
            except:
                pass
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
    download_pdfs()
    client = chromadb.PersistentClient(path="index")
    try:
        collection = client.get_collection("medical_docs")
        if collection.count() < 110:
            raise Exception("rebuild")
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
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    return collection, anthropic_client

def search(collection, query, n=8):
    results = collection.query(query_texts=[query], n_results=n)
    return results["documents"][0], [m["source"] for m in results["metadatas"][0]]

def answer(anthropic_client, query, docs, sources):
    context = "\n\n---\n\n".join(docs)
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
    return message.content[0].text, list(set(sources))

# UI
st.title("🏥 Medical Bill Helper")
st.caption("Ask questions about medical bills and financial assistance in Atlanta")

with st.sidebar:
    st.header("About")
    st.write("This tool helps you understand medical bills and find financial assistance at Atlanta hospitals.")
    st.write("**Data sources:**")
    st.write("- Grady Memorial Hospital")
    st.write("- Emory Healthcare")
    st.write("- Piedmont Healthcare")
    st.divider()
    st.write("**Example questions:**")
    if st.button("I can't afford my bill"):
        st.session_state.example = "I can't afford my medical bill. What can I do?"
    if st.button("How do I apply for help?"):
        st.session_state.example = "How do I apply for financial assistance?"
    if st.button("Compare all 3 hospitals"):
        st.session_state.example = "What are the income limits for Grady, Emory, and Piedmont financial assistance?"
    st.divider()
    st.caption("⚠️ This tool provides general information only and does not constitute legal or medical advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

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
