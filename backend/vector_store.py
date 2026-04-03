import os
import json
import api_config  # still allowed if you want other configs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# -------------------- 1. ENV SETUP --------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
chat_history = []

# -------------------- 2. LOAD SCRAPED JSON --------------------

with open("website_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print("Website data found!")

texts = []
metadatas = []

# -------- HTML PAGES --------
for page in data.get("pages", []):
    page_title = page.get("title", "Untitled Page")
    page_url = page.get("url", "unknown")

    for section in page.get("sections", []):
        heading = section.get("heading", "General")
        content = section.get("content", "")

        # Combine intelligently for better embeddings
        full_text = f"Page Title: {page_title}\nHeading: {heading}\nContent: {content}"

        texts.append(full_text)
        metadatas.append({
            "source": page_url,
            "type": "html",
            "title": page_title,
            "heading": heading
        })

# -------- PDFs (unchanged logic but slightly improved) --------
for pdf in data.get("pdfs", []):
    pdf_text = pdf.get("text", "")
    pdf_url = pdf.get("url", "unknown")

    texts.append(pdf_text)
    metadatas.append({
        "source": pdf_url,
        "type": "pdf",
        "title": "PDF Document",
        "heading": "Full Document"
    })

# -------------------- 3. VECTOR STORE SETUP --------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

split_texts = []
split_metadatas = []

print(f"Splitting {len(texts)} documents into chunks...")
for i, (text, meta) in enumerate(zip(texts, metadatas)):
    chunks = text_splitter.split_text(text)
    split_texts.extend(chunks)
    split_metadatas.extend([meta] * len(chunks))
    if (i + 1) % 500 == 0:
        print(f"Progress: Splitted {i+1}/{len(texts)} documents...")

print(f"Total chunks created: {len(split_texts)}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(f"Starting vector building for {len(split_texts)} chunks...")

# Process in batches to track progress
batch_size = 100
vector_store = None

for i in range(0, len(split_texts), batch_size):
    batch_texts = split_texts[i:i + batch_size]
    batch_metadatas = split_metadatas[i:i + batch_size]
    
    if vector_store is None:
        vector_store = FAISS.from_texts(
            texts=batch_texts,
            embedding=embeddings,
            metadatas=batch_metadatas
        )
    else:
        vector_store.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas
        )
    
    current_progress = min(i + batch_size, len(split_texts))
    print(f"Building vectors: {current_progress}/{len(split_texts)} chunks completed...")

print("Vector building complete! Saving store...")
vector_store.save_local("faiss_store")
print("All done!")
