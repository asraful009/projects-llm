import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# tokenizer for chunking
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def split_into_chunks(text, max_tokens=800, overlap=100):
    tokens = tokenizer.encode(text, truncation=False)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))

        start += max_tokens - overlap

    return chunks

# 1. Load embedding model (FIXED: no token param needed here)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load files safely
folder = ".data"

files = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, f))
]

# 3. Read + chunk documents
chunks = []
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks.extend(split_into_chunks(text))

print(f"Total chunks: {len(chunks)}")

# 4. Convert text → embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# IMPORTANT: ensure float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

# 5. Create FAISS index (use cosine-like similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. Ask question
query = input("Ask: ")

# 7. Query embedding
query_vec = model.encode([query]).astype("float32")

# 8. Search
k = 3
distances, indices = index.search(query_vec, k)

# 9. FIX: retrieve from correct source (chunks, not "lines")
context = "\n\n".join([chunks[i] for i in indices[0]])

print("\n[Retrieved Context]")
print(context)

# 10. Prompt for LLM
prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

# 11. Call Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:latest",
        "prompt": prompt,
        "stream": False
    }
)

answer = response.json()["response"]

print("\n[Answer]")
print(answer)