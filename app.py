import os
import base64
import chromadb
import anthropic
import streamlit as st


def detect_media_type(file_bytes):
    if file_bytes[:4] == b'\x89PNG':
        return "image/png"
    elif file_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif file_bytes[:4] == b'%PDF':
        return "application/pdf"
    else:
        return "image/png"


@st.cache_resource
def init():
    client = chromadb.PersistentClient(path="index")
    collection = client.get_collection("medical_docs")
    ac = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
    )
    return collection, ac


def search(collection, query, n=5):
    results = collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return docs, sources


def extract_bill_items(ac, file_bytes, media_type):
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
    if media_type == "application/pdf":
        content_block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}}
    else:
        content_block = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}
    msg = ac.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": [
            content_block,
            {"type": "text", "text": """You are a medical bill analyst. Extract all line items from this medical bill.

Return a structured list in this exact format (one item per line):
- [Service/Charge Name]: $[Amount] | [Date if visible] | [Code if visible]

After the list, add:
TOTAL BILLED: $[amount]
INSURANCE PAID: $[amount or "Not shown"]
PATIENT RESPONSIBILITY: $[amount or "Not shown"]

Include every charge, fee, and adjustment shown."""}
        ]}]
    )
    return msg.content[0].text


def analyze_bill(ac, collection, extracted_text):
    query = "financial assistance patient cannot afford bill uninsured"
    docs, sources = search(collection, query)
    context = "\n\n---\n\n".join(docs)
    msg = ac.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system="""You are a compassionate medical billing advocate helping patients in Atlanta, Georgia.
You have access to financial assistance policies from Grady Memorial Hospital and Emory Healthcare.
Be specific, practical, and empathetic.""",
        messages=[{"role": "user", "content": f"""Here is a patient's medical bill:

{extracted_text}

Here are the financial assistance policies from local hospitals:
{context}

Please provide:
1. **Bill Summary** - What services were billed and whether amounts seem typical
2. **Red Flags** - Any charges that seem unusual or worth questioning
3. **Financial Assistance Options** - What programs might help based on hospital policies
4. **Next Steps** - Specific actions: who to call, what to ask for

Be practical and actionable."""}]
    )
    return msg.content[0].text, list(set(sources))


def answer_question(ac, collection, query):
    docs, sources = search(collection, query)
    context = "\n\n---\n\n".join(docs)
    msg = ac.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="""You are a helpful assistant helping people understand medical bills and financial assistance in Atlanta, Georgia.
Answer based ONLY on the provided documents. Be specific, practical, and compassionate.""",
        messages=[{"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"}]
    )
    return msg.content[0].text, list(set(sources))


st.set_page_config(page_title="Medical Bill Helper", page_icon="🏥", layout="wide")
st.title("🏥 Medical Bill Helper")
st.caption("Upload your medical bill for analysis, or ask questions about financial assistance in Atlanta")

with st.sidebar:
    st.header("About")
    st.write("This tool helps Atlanta residents understand medical bills and find financial assistance.")
    st.write("**Knowledge base:**")
    st.write("• Grady Memorial Hospital")
    st.write("• Emory Healthcare")
    st.write("• Piedmont Healthcare")
    st.divider()
    st.write("**Example questions:**")
    if st.button("I can't afford my bill"):
        st.session_state.pending_q = "I can't afford my medical bill. What can I do?"
    if st.button("How do I apply for help?"):
        st.session_state.pending_q = "How do I apply for financial assistance?"
    if st.button("Income limits for all 3?"):
        st.session_state.pending_q = "What are the income limits for financial assistance at Grady, Emory, and Piedmont?"
    st.divider()
    st.caption("⚠️ General information only. Not legal or medical advice.")

tab1, tab2 = st.tabs(["📄 Upload & Analyze My Bill", "💬 Ask a Question"])

with tab1:
    st.subheader("Upload Your Medical Bill")
    st.write("Upload a photo or PDF of your bill — we'll extract all charges and explain your options.")
    uploaded = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"], help="Supported: JPG, PNG, PDF")

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            if uploaded.type.startswith("image"):
                st.image(uploaded, caption="Uploaded bill", width='stretch')
            else:
                st.info(f"📄 PDF uploaded: **{uploaded.name}** ({uploaded.size // 1024} KB)")
        with col2:
            st.write("**File details:**")
            st.write(f"• Name: {uploaded.name}")
            st.write(f"• Type: {uploaded.type}")
            st.write(f"• Size: {uploaded.size:,} bytes")
        st.divider()

        if st.button("🔍 Analyze This Bill", type="primary", use_container_width=True):
            file_bytes = uploaded.read()
            media_type = detect_media_type(file_bytes)
            with st.spinner("Step 1/2: Reading your bill..."):
                try:
                    collection, ac = init()
                    extracted = extract_bill_items(ac, file_bytes, media_type)
                except Exception as e:
                    st.error(f"Error reading bill: {e}")
                    st.stop()
            st.success("✅ Bill items extracted!")
            with st.expander("📋 Extracted Line Items", expanded=True):
                st.text(extracted)
            with st.spinner("Step 2/2: Analyzing charges and finding assistance options..."):
                try:
                    analysis, sources = analyze_bill(ac, collection, extracted)
                except Exception as e:
                    st.error(f"Error analyzing bill: {e}")
                    st.stop()
            st.subheader("💡 Analysis & Recommendations")
            st.markdown(analysis)
            st.caption(f"Based on policies from: {', '.join(sources)}")

with tab2:
    st.subheader("Ask About Financial Assistance")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources: {', '.join(msg['sources'])}")
    pending = st.session_state.pop("pending_q", None)
    if query := (st.chat_input("Ask about your medical bill...") or pending):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Looking up information..."):
                try:
                    collection, ac = init()
                    response, sources = answer_question(ac, collection, query)
                    st.markdown(response)
                    st.caption(f"Sources: {', '.join(sources)}")
                    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())
