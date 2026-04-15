"""
The main rag pipeline
Builds a retriever tool for retrieval of relevant chunks
A query rewriter for rewriting current query accordint to chat history for better context
Builds a logitics tool which returns the appropriate response and sources to the LLM 
"""
import os
import json
import api_config  # still allowed if you want other configs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disabling gpu usage 
#chat_history = [] #list for storing the user queries and responses as well. Only the ones during current runtime.


embeddings = HuggingFaceEmbeddings(  #vector embeddings 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(   #storing the vector embeddings locally
    "faiss_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,           # still retrieve wide (since DB is big)
        "fetch_k": 30,     # candidates before diversity selection
        "lambda_mult": 0.7 # balance relevance vs diversity (0.5–0.8 is good)
    }
) #building a retriever tool which returns top 20 relevant chunks
#as_retriever is a inbuilt langchain function

model = ChatGoogleGenerativeAI( #LLM model used for query rewriting and giving final responses
    model="gemini-2.5-flash-lite",
    temperature=0 #so that the model is deterministic and doesnt hallucinate
)
"""
Adding a preload fxn
it is observed that it takes 25secs for the system to start taking user input
for reducing that we will be using this preload fxn
this fucntion will load everything in the server beforehand, so that when a user uses it there is no need for doing it again and again for every user separately.
"""
def preload():
    print("🔄 Preloading system...")

    # force FAISS + embeddings to load
    _ = retriever

    # warm up LLM (optional but useful)
    try:
        model.invoke("hi")
    except:
        pass

    print("✅ System ready.")

"""
Function for rewriting a query according to the chat history
appends the user queries and LLM responses and sends it to an LLM
LLM tries to make the current query sound more contextual and relevant to the previous queries for better retrieval
"""
def f_rewrite_query(chat_history, latest_query): 
    history_text = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage)
         else f"Assistant: {m.content}"
         for m in chat_history[-6:]]
    )

    prompt = f"""
You are a query rewriting assistant.

Conversation History:
{history_text}

Latest User Question:
{latest_query}

Rewrite the latest question into a clear, standalone question.

IMPORTANT:
- If the question depends on previous conversation, include relevant context.
- Preserve important details like names, entities, or references.
- If the user mentioned their name earlier, include it explicitly.
- If no context is needed, keep it simple.

Do NOT answer the question. Only rewrite it.
"""
    rewritten_query = model.invoke(prompt).content.strip()
    print("Query rewritten accorinding to chat history for better context:")
    print(rewritten_query)
    return rewritten_query


"""
The tool below is the main tool of the entire pipeline.
This tool uses the retriever tool for receiving the chunks of the content that is relevant to the query.
These chunks are saved in the 'docs' list
If the list is empty then the model will respond saying no relevant info available
Then the docs list is separated into actual information and source links(metadata)
"""
@tool
def f_logistics_search(query: str) -> str:
    """Search the local knowledge base and return relevant information with sources using soft relevance scoring."""
    
    # Step A: Retrieve
    #query = f_rewrite_query([], query)
    docs = retriever.invoke(query)

    if not docs:
        return "No information found in the local database."

    context_chunks = []
    sources = set()

    for doc in docs:
        context_chunks.append(doc.page_content)
        sources.add(doc.metadata.get("source", "unknown"))

    context_text = "\n".join(context_chunks)
    
    sources_text = "\n".join(f"- {s}" for s in sources)

    return f"""
{context_text}

Sources:
{sources_text}
"""


agent = create_agent( #creating the agent using the model and tool which were defined above and a system prompt for a more systematic reponse.
    model=model,
    tools=[f_logistics_search],
    system_prompt=(
        "You are a helpful assistant.\n\n"
        "Always use the logistics_search tool to find facts.\n\n"
        "Answer using this structure:\n"
        "1. Title (based on the query)\n"
        "2. Key Information (bullet points)\n"
        "3. Additional Details (if any)\n"
        "4. Sources (bullet list)\n\n"
        "Use clear spacing between sections.\n"
        "Do not write large paragraphs.\n"
    ),
)

"""
Taking input and generating responses!
"""
preload()

"""
if __name__ == "__main__":
    try:
        print("System Online...")
        print("Hello, how can I help you today?\n")
        while True:
            user_query = input()
            if user_query.lower() in ["exit", "quit"]:
                break

            chat_history.append(HumanMessage(content=user_query))
            res = agent.invoke({"messages": chat_history})

            final_answer = res["messages"][-1].text
            print("\nANSWER:\n", final_answer)

            chat_history.append(AIMessage(content=final_answer))

    except Exception as e:
        print(f"❌ Error: {e}")
"""

