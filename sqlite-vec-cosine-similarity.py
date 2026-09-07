"""
Why This Scaled Solution Beats Manual Matrix Operations

Memory Safety: 
When scaling to hundreds of thousands of chunks, keeping a colossal raw array array in your system RAM 
will cause application crashes. SQLite safely pages data from the local solid-state storage disk as needed.

Metadata Cohesion (JOINs): 
You can easily attach complex production filtering inside the same single string transaction block 
(e.g., WHERE docs.user_id = 42 AND docs.created_at > '2026-01-01').

C-Level Performance Acceleration: 
The database handles multi-dimensional arithmetic inside native compiled operations using raw pointer byte 
streams, meaning it runs faster than raw Python array processing loops.
"""


import os 
import sqlite3 
import struct 
from typing import List 
import sqlite_vec 
from dotenv import load_dotenv 
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Connect to local SQLite file and safely register the vector math engine 
DB_FILE = "local_vector_store.db"
conn = sqlite3.connect(DB_FILE)
conn.enable_load_extension(True)
sqlite_vec.load(conn) # Registers vector functions natively inside the SQL environment
conn.enable_load_extension(False) # Best practice security boundary locking

# Verify extensions initialized properly
sqlite_v, vec_v = conn.execute("SELECT sqlite_version(), vec_version()").fetchone()
print(f"Loaded: SQLite v{sqlite_v} | sqlite-vec v{vec_v}")

# 2. Define schema using the specialized vec0 engine (e.g.,  has 3072 dimensions)
DIMENSIONS = 3072
cursor = conn.cursor()

# Create a traditional metadata table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY,  -- Removed AUTO_INCREMENT
        content TEXT NOT NULL
    )               
""")

# Create the corresponding virtual vector indexing table
cursor.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_document_chunks USING vec0(
        id INTEGER PRIMARY KEY,
        embedding float[{DIMENSIONS}]
    )
""")
conn.commit()

# Helper function to serialize float arrays into raw binary blobs for fast DB storage 
def serialize_f32(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)

# 3. Vectorize and Ingest context chunks
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

documents_to_store = [
    "To change your account security credentials, visit your profile dashboard settings.",
    "The company office kitchen has fresh coffee and tea daily.",
    "System administrators can trigger a secure login credential reset upon request.",
    "Quarterly financial projections indicate strong infrastructure growth over Q3."
]

print("--- Vectorizing text and writing to SQLite storage... ---")
embeddings = embeddings_model.embed_documents(documents_to_store)

for content, emb in zip(documents_to_store, embeddings):
    # Insert metadata row first to grab the tracking ID
    cursor.execute("INSERT INTO document_chunks (content) VALUES (?)", (content,))
    assigned_id = cursor.lastrowid 
    
    # Store raw bytes blob in the parallel virtual indexing table 
    cursor.execute(
        "INSERT INTO vec_document_chunks(id, embedding) VALUES (?,?)",
        (assigned_id, serialize_f32(emb))
    )
conn.commit()

# 4. Querying the Database Using Native SQL Vector Operators 
user_query = "How can I fix my password problem?"
query_vector = embeddings_model.embed_query(user_query)

print(f"\n--- Running Native Semantic SQL Search for: '{user_query}' ---")

# vec_distance_cosine computes similarity on the fly inside the database engine
search_query = """
    SELECT 
        docs.id,
        docs.content,
        vec_distance_cosine(vec.embedding, ?) AS distance
    FROM vec_document_chunks vec
    JOIN document_chunks docs ON vec.id = docs.id
    ORDER BY distance ASC
    LIMIT 2
"""

cursor.execute(search_query, (serialize_f32(query_vector),))
results = cursor.fetchall()

# Display matched results natively computed by SQLite
for row_id, text, distance in results:
    # Cosine distance ranges 0 to 2 ( 0 = identical direction ). Convert to a 0-1 similarity metric:
    similarity = 1.0 - (distance / 2.0)
    print(f"Row ID: {row_id} | Similiarity: {similarity:.4f}")
    print(f"Matches Text: \"{text}\"\n")
    
conn.close()
    


