import os 
import numpy as np 
from dotenv import load_dotenv 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

load_dotenv()

# 1. Initialize the embedding generator 
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 2. Define the user query and data corpus 
user_query = "How can I fix my password problem?"

documents = [
    "To change your account security credentials, visit your profile dashboard settings.", # Match 
    "The company office kitchen has fresh coffee and tea daily.", # No Match
    "System administrators can trigger a secure login credential reset upon request.", # Match
    "Quarterly financial projections indicate strong infrastructure growth over Q3." # No Match
]

# 3. Generate numerical vector embeddings 
print("--- Vectorizing text into numerical arrays... ---")
query_vector = np.array(embeddings_model.embed_query(user_query))
doc_vectors = np.array(embeddings_model.embed_documents(documents))

# 4. Define the Cosine Similarity Function 
def calculate_cosine_similarity(v1: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity between a single vector (v1)
    and a matrix of multiple document vectors.
    Formula: (A * B) / (||A|| * ||B||)
    """
    
    # Compute the dot product (numerator) for all items simultaneously
    dot_product = np.dot(matrix, v1)
    
    # Compute the geometric magnitude/lengths (denominator) 
    v1_norm = np.linalg.norm(v1) 
    matrix_norms = np.linalg.norm(matrix, axis=1)
    
    # Complete the division 
    return dot_product / (v1_norm * matrix_norms)

# 5. Calculate scores and match indices 
scores = calculate_cosine_similarity(query_vector, doc_vectors)

# Get indices sorted from highest score to lowest score 
ranked_indices = np.argsort(scores)[::-1]

# 6. Display Ranked Results
print("\n--- SEMANTIC SEARCH RANKING RESULTS ---")
for rank, idx in enumerate(ranked_indices, start=1):
    print(f"Rank {rank}: [Score: {scores[idx]:.4f}]")
    print(f"Text: \{documents[idx]}\"\n")