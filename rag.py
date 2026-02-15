import os
import json
import config  # still allowed if you want other configs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# -------------------- 1. ENV SETUP --------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
chatHistory = []

# -------------------- 2. LOAD SCRAPED JSON --------------------

with open("website_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print("Website data found!")

texts = []
metadatas = []

for page in data["pages"]:
    texts.append(page["text"])
    metadatas.append({
        "source": page["url"],
        "type": "html"
    })

for pdf in data["pdfs"]:
    texts.append(pdf["text"])
    metadatas.append({
        "source": pdf["url"],
        "type": "pdf"
    })

# -------------------- 3. VECTOR STORE SETUP --------------------

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

split_texts = []
split_metadatas = []

for text, meta in zip(texts, metadatas):
    chunks = textSplitter.split_text(text)
    split_texts.extend(chunks)
    split_metadatas.extend([meta] * len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_texts(
    texts=split_texts,
    embedding=embeddings,
    metadatas=split_metadatas
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------- 4. MODEL --------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

def rewrite_query(chatHistory, latest_query):
    history_text = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage)
         else f"Assistant: {m.content}"
         for m in chatHistory[-6:]]
    )

    prompt = f"""
You are a query rewriting assistant.

Conversation History:
{history_text}

Latest User Question:
{latest_query}

Rewrite the latest question into a clear, standalone search query.
Do NOT answer it. Only rewrite.
"""
    rewrittenQuery = model.invoke(prompt).content.strip()
    print("Query rewritten accorinding to chat history for better context:")
    print(rewrittenQuery)
    return rewrittenQuery

# -------------------- 5. TOOL --------------------

@tool
def logistics_search(query: str) -> str:
    """Search the local knowledge base and return relevant information with sources using soft relevance scoring."""
    
    # Step A: Retrieve
    query = rewrite_query(chatHistory, query)
    docs = retriever.invoke(query)

    if not docs:
        return "No information found in the local database."

    # Collect context and sources
    context_chunks = []
    sources = set()

    for doc in docs:
        context_chunks.append(doc.page_content)
        sources.add(doc.metadata.get("source", "unknown"))

    context_text = "\n".join(context_chunks)
    # Step D: Return context + sources
    sources_text = "\n".join(f"- {s}" for s in sources)

    return f"""
{context_text}

Sources:
{sources_text}
"""


# -------------------- 6. AGENT --------------------

agent = create_agent(
    model=model,
    tools=[logistics_search],
    system_prompt=(
        "You are a helpful assistant. "
        "Always use the logistics_search tool to find facts. "
        "Cite the provided sources in your answer. "
        "If no information found,say so"
    ),
)

# -------------------- 7. RUN --------------------

if __name__ == "__main__":
    try:
        print("System Online...")
        print("Hello, how can I help you today?\n")
        while True:
            user_query = input()
            if user_query.lower() in ["exit", "quit"]:
                break

            chatHistory.append(HumanMessage(content=user_query))
            res = agent.invoke({"messages": chatHistory})

            final_answer = res["messages"][-1]
            print("\n✨ ANSWER:\n", final_answer.text)

            chatHistory.append(AIMessage(content=final_answer.text))

    except Exception as e:
        print(f"❌ Error: {e}")
