import os 
from dotenv import load_dotenv 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

load_dotenv() 

# Initialize the vectorizer model 
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 1. Vectorize a user query 
user_query = "How can I fix my password problem?"
query_vector = embeddings_model.embed_query(user_query)

# 2. Vectorize target document chunks to compare against
doc_1 = "To change your account security credentials, visit your profile dashboard settings."
doc_2 = "The company office kitchen has fresh coffee and tea daily."

doc_vector_1 = embeddings_model.embed_query(doc_1)
doc_vector_2 = embeddings_model.embed_query(doc_2)

# Inspecting the data profile
print(f"Vector Length: {len(query_vector)} dimensions")
print(f"Sample values: {query_vector[:3]}...")