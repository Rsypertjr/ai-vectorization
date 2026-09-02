import os 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, Annotated, List

# 1. Environment Setup 
# Ensure your .env file contains:  GEMINI_API_KEY=your_api_key_here
load_dotenv() 

# 2. Define the Agent State 
class AgentState(TypedDict):
    messages: List[dict]
    query: str 
    vector_context: str 
    response: str 
    
    
# 3. Dummy Vector Store Search (Simulating Vectorization retrieval)
def search_vector_db(query: str) -> str:
    """
    In a full production RAG pipeline, you would use GoogleGenerativeAIEmbeddings
    to vectorize the incoming query and search a vector database like Pinecone or pgvector.
    
    Example under the hood:
    embeddings_model = GoogleGenerativeAIEmbeddings(model="text-embeddings-004")
    query_vector = embeddings_model.embed_query(query)    
    """
    # Mocking a vector search hit based on semantic relevance 
    if "password" in query.lower() or "login" in query.lower():
        return "Internal Policy Doc #402: Users must reset password via security dashboard profile settings."
    return "No highly relevant vector context found."

# 4. Define Graph Nodes
def retrieve_node(state: AgentState) -> dict:
    """Node that extracts vector context based on query vectorization."""
    print("--- [Vector DB] Querying vector space via embeddings... ---")
    context = search_vector_db(state["query"])
    return {"vector_context": context}

def llm_node(state: AgentState) -> dict: 
    """Node that uses Gemini model to generate a response using the vector context."""
    print("--- [Gemini AI] Generating response with context...---")
  
    # Initialize the modern Gemini model setup 
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2,
        api_key=os.getenv("GEMINI_API_KEY")
    )  
    
    system_prompt = (
        f"You are a helpful assistant.  Use this vectorized internal context to answer the user query.\n"
        f"Context: {state['vector_context']}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]}
    ]
    
    ai_msg = llm.invoke(messages)
    return {"response": ai_msg.content}

# 5. Build the LangGraph Workflow 
builder = StateGraph(AgentState) 

# Add processing units (nodes)
builder.add_node("retrieve_context", retrieve_node)
builder.add_node("call_model", llm_node)

# Context the structural execution flow (edges)
builder.add_edge(START, "retrieve_context")  # Start execution here 
builder.add_edge("retrieve_context","call_model") # Pass data to the model 
builder.add_edge("call_model", END)

# Compile the workflow graph 
graph = builder.compile()

# 6. Execute the Graph 
if __name__ == "__main__":
    initial_input = {"query": "How can I fix my password problem?"}
    
    final_output = graph.invoke(initial_input)
    
    print("\n--- FINAL SYSTEM RESPONSE ---")
    print(final_output["response"])

    
    