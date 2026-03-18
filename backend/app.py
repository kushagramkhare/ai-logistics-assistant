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

chat_sessions = {}

class Query(BaseModel):
    question: str
    history: list = []
    session_id: str

@app.get("/")
def home():
    return {"message": "AI Logistics Assistant API is running"}

@app.post("/chat")
def chat(q: Query):

    if q.session_id not in chat_sessions:
        chat_sessions[q.session_id] = []

    messages = chat_sessions[q.session_id]

    messages.append(HumanMessage(content=q.question))

    res = agent.invoke({"messages": messages})

    answer = res["messages"][-1].text

    messages.append(AIMessage(content=answer))

    return {"answer": answer}