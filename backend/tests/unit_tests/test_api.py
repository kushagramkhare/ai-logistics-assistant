import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app

client = TestClient(app)

# ==========================================
# API Route Units
# ==========================================

def test_UT_api_1_home_endpoint():
    """UT_api_1: Verify the API is online and the root route works"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Logistics Assistant API is running"}

@patch("app.agent.invoke")
def test_UT_api_2_chat_endpoint(mock_invoke):
    """UT_api_2: Verify the /chat endpoint returns a successful response"""
    # Mock the LangChain agent response
    mock_res = MagicMock()
    mock_res.__getitem__.side_effect = lambda key: [MagicMock(text="Test Answer")] if key == "messages" else None
    mock_invoke.return_value = mock_res

    payload = {
        "question": "What are the hostel fees?",
        "history": []
    }
    
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["answer"] == "Test Answer"

@patch("app.agent.invoke")
def test_UT_api_3_chat_history_processing(mock_invoke):
    """UT_api_3: Ensure history is correctly formatted before reaching the agent"""
    mock_res = MagicMock()
    mock_res.__getitem__.side_effect = lambda key: [MagicMock(text="History Received")] if key == "messages" else None
    mock_invoke.return_value = mock_res

    payload = {
        "question": "Tell me more.",
        "history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"}
        ]
    }
    
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    # Capture the arguments passed to agent.invoke
    args, _ = mock_invoke.call_args
    messages = args[0]["messages"]
    
    # Assert messages length (2 history + 1 new query = 3)
    assert len(messages) == 3
    assert messages[0].content == "Hi"
    assert messages[1].content == "Hello! How can I help?"
    assert messages[2].content == "Tell me more."
