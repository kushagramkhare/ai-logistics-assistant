import os
import time
import config  # Assumes config.text and config.GOOGLE_API_KEY exist
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. ENVIRONMENT SETUP
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Stay on CPU to avoid DLL errors


# 2. THE CHUNKER
def split_text(text, chunk_size=1000, chunk_overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

chunks = split_text(config.text)

# 3. INITIALIZE EMBEDDINGS
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 4. INITIALIZE CHROMA (Empty)
# We initialize the object first so we can add to it batch-by-batch
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 5. THE BATCH LOADER (Scalability Logic)
def load_in_batches(all_chunks, batch_size=20):
    print(f"🚀 Starting upload of {len(all_chunks)} chunks...")
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        
        # add_texts is better for scale than from_texts
        vector_store.add_texts(texts=batch)
        
        print(f"✅ Success: Added chunks {i} to {i + len(batch)}")
        
        # This sleep prevents the "429 Resource Exhausted" error
        # 10-15 seconds is the sweet spot for the Free Tier
        time.sleep(15) 

# Run the batch loader
load_in_batches(chunks)

# 6. RETRIEVAL
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
user_query = input("\nWhat do you wanna know about Zendaya? ")
results = retriever.invoke(user_query)

for i, doc in enumerate(results):
    print(f"\n--- Snippet {i+1} ---")
    print(doc.page_content)



# Use .content for the most reliable streaming output across providers
