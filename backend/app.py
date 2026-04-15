from fastapi import FastAPI
from pydantic import BaseModel
from main_rag import agent
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (keep as is)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Query(BaseModel):
    question: str
    history: list = []   # coming from frontend


@app.get("/")
def home():
    return {"message": "AI Logistics Assistant API is running"}


@app.post("/chat")
def chat(q: Query):

    # 🔥 Convert chat history into plain text
    chat_history_text = ""
    for msg in q.history:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_history_text += f"{role}: {msg['content']}\n"

    # 🔥 Inject history into query
    final_query = f"""
You are a helpful assistant.

Conversation so far:
{chat_history_text}

Now answer the latest question:
User: {q.question}
"""

    # 🤖 Call agent with ONLY this query
    res = agent.invoke({
        "messages": [HumanMessage(content=final_query)]
    })

    # 📤 Extract answer safely
    if isinstance(res, dict) and "messages" in res:
        answer = res["messages"][-1].text
    else:
        answer = str(res)

    return {"answer": answer}