"""
Enterprise Python Implementation Code
To connect securely, install the database adapter: pip install psycopg2-binary 
(or psycopg for async workflows).

This script connects to your Docker container, provisions a standard relational 
architecture with a native vector(768) data type, applies a high-performance HNSW 
(Hierarchical Navigable Small World) vector index for sub-millisecond querying, and 
provides a multi-user safe workflow.

"""

import os 
import psycopg2 
from psycopg2.extras import execute_values
from dotenv import load_dotenv 
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Establish Secure Database Connection 
try: 
    conn = psycopg2.connect(
        host="localhost",
        database="vector_enterprise",
        user="admin_user",
        password="secure_password_2026",
        port="5432"
    )
    cursor = conn.cursor() 
except Exception as e: 
    print(f"Database connection failed: {e}")
    exit(1)
    
# 2. Initialize pgvector Extension & Unified Schema
# gemini-embedding-001 utilizes 3072 dimensions
DIMENSIONS = 1536

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Unified Table Strategy using native 'vector' type 
cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS enterprise_documents (
        id SERIAL PRIMARY KEY, 
        content TEXT NOT NULL,
        embedding vector({DIMENSIONS}),
        tenant_id VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

# 3. Apply HNSW Vector Index for Enterprise Scaling
# This enables sub-millisecond Approximate Nearest Neighbor (ANN) search over millions of rows.
# 'vector_cosine_ops' optimizes the index for cosine distance calculation (<=>)
cursor.execute("""
    CREATE INDEX IF NOT EXISTS enterprise_docs_hnsw_idx 
    ON enterprise_documents
    USING hnsw (embedding vector_cosine_ops);
""")
conn.commit()

# 4. Ingest and Vectorize Documents
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536, # LangChain handles the MRL truncation parameters under the hood
    google_api_key=os.getenv("GEMINI_API_KEY")
) 

documents_to_store = [
    "To change your account security credentials, visit your profile dashboard settings.",
    "The company office kitchen has fresh coffee and tea daily.",
    "System adminstrators can trigger a secure login credential reset upon request.",
    "Quarterly financial projections indicate strong infrastructure growth over Q3."
]

print("--- Mass vectorizing data arrays... ---")
embeddings = embeddings_model.embed_documents(documents_to_store)

#  Prepare row payload data (Including a multi-user tenant partition tag)
data_to_insert = [
    (content, emb, "tenant_company_abc")
    for content, emb in zip(documents_to_store, embeddings)
]

# High-speed bulk insertion statement 
insert_query = """ 
    INSERT INTO enterprise_documents (content, embedding, tenant_id)
    VALUES %s
"""
execute_values(cursor, insert_query, data_to_insert)
conn.commit()
print(f"Successfully ingested {len(documents_to_store)} documents.")

# 5. Multi-User Isolated Query Execution
user_query = "How can I fix my password problem?"
active_tenant = "tenant_company_abc"

query_vector = embeddings_model.embed_query(user_query)

print(f":\n--- Running Multi-User Isolated Search for Tenant: '{active_tenant}' ---")

# PostgresSQL pgvector operator '<=>' calculates Cosine Distance natively on hardware. 
# Standard relational conditions (WHERE tenant_id) seemlessly filter vector spaces.
search_query = """
    SELECT 
        id,
        content,
        1 - (embedding <=> %s::vector) AS similarity
    FROM enterprise_documents 
    WHERE tenant_id = %s
    ORDER BY embedding <=> %s::vector ASC 
    LIMIT 2;
"""

cursor.execute(search_query, (query_vector, active_tenant, query_vector))
results = cursor.fetchall() 

for row_id, text, similarity in results:
    print(f"Row ID: {row_id} | Cosine Similarity: {similarity:.4f}")
    print(f"Matched Text: \"{text}\"\n")

cursor.close()
conn.close()
