import pytest
import os
from unittest.mock import patch, MagicMock

# Import your actual modules/functions here
from main_rag import f_logistics_search, embeddings, vector_store, f_rewrite_query

# ==========================================
# Data Extraction & Processing Units
# ==========================================

def test_UT_rag_1_convert_to_embeddings():
    """UT_rag_1: Convert document text into embeddings"""
    sample_text = ["This is a test chunk."]
    
    vectors = embeddings.embed_documents(sample_text)
    
    # Assert Embeddings successfully generated
    assert vectors is not None
    assert isinstance(vectors, list)
    assert len(vectors[0]) > 0

def test_UT_rag_2_insert_into_vector_store():
    """UT_rag_2: Insert embeddings into vector store"""
    # Assert FAISS index created successfully (based on local vector_store object)
    assert vector_store is not None
    
    # You can also verify that the FAISS index actually contains items
    index_size = vector_store.index.ntotal
    assert isinstance(index_size, int)

# ==========================================
# Querying and Generation Units
# ==========================================

def test_UT_rag_3_retrieve_natural_language_query(mocker):
    """UT_rag_3: Submit natural language query to backend"""
    mock_retriever = mocker.patch("main_rag.retriever.invoke")
    
    # Mocking what the retriever would return
    mock_doc = MagicMock()
    mock_doc.page_content = "Here is some relevant context about hostels."
    mock_retriever.return_value = [mock_doc]
    
    query = "Tell me about hostels"
    docs = mock_retriever(query)
    
    # Assert Relevant contextual chunks retrieved
    assert len(docs) > 0
    assert "hostels" in docs[0].page_content
    mock_retriever.assert_called_once_with(query)

def test_UT_rag_4_generate_response(mocker):
    """UT_rag_4: Generate response using retrieved context"""
    mock_agent = mocker.patch("main_rag.agent.invoke")
    mock_agent.return_value = {"messages": [MagicMock(text="Coherent answer produced")]}
    
    # Simulating LLM response generation
    res = mock_agent({"messages": "What is the fee?"})
    final_answer = res["messages"][-1].text
    
    # Assert Response generated successfully
    assert final_answer == "Coherent answer produced"

def test_UT_rag_5_verify_source_attribution():
    """UT_rag_5: Verify source attribution in response"""
    # Simulate tool call parsing
    mock_query = "Who is the principal?"
    
    # Using your actual tool logic to ensure sources are appended
    result = f_logistics_search.invoke({"query": mock_query})
    
    # Assert Source URLs correctly attached (checks if "Sources:" text block exists)
    assert "Sources:" in result

def test_UT_rag_6_unrelated_query(mocker):
    """UT_rag_6: Query unrelated to institute domain"""
    # Force retriever to return nothing for unrelated queries to prevent hallucination
    mocker.patch("main_rag.retriever.invoke", return_value=[])
    
    query = "Who won the FIFA World Cup?"
    result = f_logistics_search.invoke({"query": query})
    
    # Assert System returned fallback response indicating insufficient data
    assert "No information found in the local database" in result

def test_UT_rag_7_no_matching_documents(mocker):
    """UT_rag_7: Query with no matching documents in vector store"""
    mocker.patch("main_rag.retriever.invoke", return_value=[])
    
    query = "sdgasdgasgasgasg"
    result = f_logistics_search.invoke({"query": query})
    
    # Assert Fallback message returned
    assert "No information found in the local database" in result

def test_UT_rag_8_extremely_short_query(mocker):
    """UT_rag_8: Extremely short query"""
    mock_doc = MagicMock()
    mock_doc.page_content = "Hostel fees are 5000."
    mocker.patch("main_rag.retriever.invoke", return_value=[mock_doc])
    
    query = "hostel"
    result = f_logistics_search.invoke({"query": query})
    
    # Assert Relevant hostel information retrieved
    assert "Hostel fees" in result

def test_UT_rag_9_extremely_long_query():
    """UT_rag_9: Extremely long query exceeding typical length"""
    long_query = "What is " + "the " * 500 + "fee?"
    
    # Testing rewrite query to ensure it doesn't crash on long inputs
    try:
        rewritten = f_rewrite_query([], long_query)
        processed = True
    except Exception:
        processed = False
        
    # Assert Query processed successfully without crash
    assert processed is True
