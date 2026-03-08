from fastapi import FastAPI
from pydantic import BaseModel
from main_rag import agent
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    history: list = []

@app.get("/")
def home():
    return {"message": "AI Logistics Assistant API is running"}

@app.post("/chat")
def chat(q: Query):

    messages = []

    for msg in q.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=q.question))

    res = agent.invoke({"messages": messages})

    answer = res["messages"][-1].text

    return {"answer": answer}