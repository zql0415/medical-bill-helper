# 🏥 Medical Bill Helper

A RAG-powered chatbot that helps uninsured and underinsured patients in Atlanta understand their medical bills and find financial assistance.

**Live demo:** https://medical-bill-apper-4whz2suchfqnuy7mfh5cmh.streamlit.app/

---

## The Problem

Medical bills in the US are notoriously hard to read. Most patients don't know:
- Whether they qualify for financial assistance
- How to apply, or what documents to bring
- The difference between programs at different hospitals
- Whether their bill contains errors

This tool lets users describe their situation in plain language and get specific, actionable guidance — without needing to navigate complex hospital websites or wait on hold.

---

## What It Does

- Answers questions about medical bills and financial assistance in Atlanta
- Pulls answers from real hospital policy documents (not Claude's general knowledge)
- Covers Grady Memorial Hospital and Emory Healthcare
- Shows the source documents behind every answer
- Available 24/7 via public URL — no login required

---

## How It Works

This is a RAG (Retrieval-Augmented Generation) system:

1. **Data ingestion** — Hospital financial assistance policy PDFs are parsed and split into chunks
2. **Embedding** — Each chunk is embedded using ChromaDB's built-in model
3. **Retrieval** — When a user asks a question, the most relevant chunks are retrieved
4. **Generation** — Claude reads the retrieved chunks and generates a grounded answer
5. **UI** — Streamlit provides the chat interface

The key difference from just asking Claude directly: answers are grounded in specific, up-to-date local documents. Claude cannot make up information that isn't in the source files.

---

## Tech Stack

- **Python 3.10**
- **ChromaDB** — vector database and embedding
- **PyMuPDF** — PDF parsing
- **Anthropic Claude API** (claude-sonnet-4-6) — answer generation
- **Streamlit** — chat UI and deployment

---

## Data Sources

- Grady Memorial Hospital Financial Assistance Program Policy
- Emory Healthcare Financial Assistance Plain Language Summary

---

## Known Limitations

- Only covers Atlanta hospitals currently in the knowledge base
- Cannot read or interpret uploaded bill images (planned feature)
- Information accuracy depends on source document recency

---

## Local Setup

```bash
git clone https://github.com/zql0415/medical-bill-helper
cd medical-bill-helper
conda create -n medical python=3.10
conda activate medical
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your_key" > .env
python build_index.py
streamlit run app.py
```
