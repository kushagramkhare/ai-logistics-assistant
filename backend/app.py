from fastapi import FastAPI
from pydantic import BaseModel
from main_rag import agent, f_rewrite_query
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
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
    history: list = []


@app.get("/")
def home():
    return {"message": "AI Logistics Assistant API is running"}


@app.post("/chat")
def chat(q: Query):

    # 🔁 Convert history → LangChain messages
    messages = []
    for msg in q.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 🔥 Use FULL history for rewriting
    rewritten_query = f_rewrite_query(messages, q.question)

    # 🤖 Send ONLY rewritten query to agent
    res = agent.invoke({
        "messages": [HumanMessage(content=rewritten_query)]
    })

    # 📤 Extract answer safely
    if isinstance(res, dict) and "messages" in res:
        answer = res["messages"][-1].content
    else:
        answer = str(res)

    return {"answer": answer}