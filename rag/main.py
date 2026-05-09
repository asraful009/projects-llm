import os
import pickle

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from rab_db import RagDB

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# =========================
# Lazy Loaded Globals
# =========================

tokenizer = None
model = None


def get_tokenizer():
  global tokenizer
  if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2",
      token=HF_TOKEN
    )
  return tokenizer


def get_model():
  global model

  if model is None:
    model = SentenceTransformer(
      "all-MiniLM-L6-v2"
    )

  return model


# =========================
# File Helpers
# =========================

def list_files_in_directory(directory):
  return [
    f for f in os.listdir(directory)
    if os.path.isfile(os.path.join(directory, f))
    and f.endswith(".txt")
  ]


def read_file(directory, file_path) -> str | None:
  full_path = os.path.join(directory, file_path)

  with open(full_path, "r", encoding="utf-8") as f:
    text = f.read().strip()

  return text if text else None


# =========================
# Chunking
# =========================

def split_into_chunks(text, max_tokens=300, overlap=50):
  tokenizer = get_tokenizer()
  tokens = tokenizer.encode(text, add_special_tokens=False)

  chunks = []
  start = 0

  while start < len(tokens):
    end = start + max_tokens
    chunk_tokens = tokens[start:end]
    chunk = tokenizer.decode(chunk_tokens)
    chunks.append(chunk)
    start += max_tokens - overlap

  return chunks


def large_chunk_splitter(text: str | None, max_tokens=300, overlap=50):
  if text is None:
    return []

  paragraphs = [
    p.strip()
    for p in text.split("\n\n")
    if p.strip()
  ]

  print(f"Text split into {len(paragraphs)} paragraphs.")

  all_chunks = []

  for paragraph in paragraphs:
    paragraph_chunks = split_into_chunks(
      paragraph,
      max_tokens,
      overlap
    )
    all_chunks.extend(paragraph_chunks)

  return all_chunks


# =========================
# Debug Save
# =========================

def save_chunks_to_files(chunks, output_dir=".chunks"):
  os.makedirs(output_dir, exist_ok=True)
  for i, chunk in enumerate(chunks):
    with open(os.path.join(output_dir, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
      f.write(chunk)


# =========================
# Build FAISS Index
# =========================

def row_data_to_vector(root_path=".data", rag_db: RagDB = None):
  os.makedirs(".ai_data", exist_ok=True)
  faiss_path = ".ai_data/faiss.index"
  if os.path.exists(faiss_path):
    print("Loading FAISS index from file...")
    return faiss.read_index(faiss_path)



  if not os.path.exists(root_path):
    print(f"Directory {root_path} does not exist.")
    return

  if rag_db is None:
    rag_db = RagDB()
  files = list_files_in_directory(root_path)
  print(f"Read {len(files)} files.")
  rag_db.create_table()
  model = get_model()
  tokenizer = get_tokenizer()
  for i, file in enumerate(files):
    if rag_db.get_documents_by_file_name(file):
      print(f"{file} already exists. Skipping.")
      continue
    text = read_file(root_path, file)
    if not text:
      continue

    chunks = large_chunk_splitter(text)
    print(f"File {i} split into {len(chunks)} chunks.")

    document_id = rag_db.insert_document(file_name=file, total_chunks=len(chunks))

    for j, chunk in enumerate(chunks):
      chunk = f"{file} [line {j}]: {chunk}"
      token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
      embedding = model.encode(chunk, normalize_embeddings=True)
      embedding_blob = pickle.dumps(embedding)

      rag_db.insert_chunk(
        document_id=document_id,
        chunk_text=chunk,
        chunk_index=j,
        token_count=token_count,
        embedding=embedding_blob,
      )

  rows = rag_db.get_all_chunks()

  embeddings = [
      pickle.loads(row["embedding"])
      for row in rows
  ]

  rag_db.close()

  embeddings = np.array(embeddings).astype("float32")

    # print("Embeddings shape:", embeddings.shape)

  dimension = embeddings.shape[1]
  # Cosine Similarity
  faiss_index = faiss.IndexFlatIP(dimension)
  faiss_index.add(embeddings)

  faiss.write_index(faiss_index, faiss_path)

  return faiss_index


# =========================
# Search
# =========================

def search(query, index, rag_db=None, k=5):
    model = get_model()

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)

    if rag_db is None:
      rag_db = RagDB()

    rows = rag_db.get_all_chunks()

    results = []

    for idx, score in zip(indices[0], scores[0]):
      if idx < 0:
        continue

      results.append({"text": rows[idx]["chunk_text"], "score": float(score)})

    return results


# =========================
# Ollama
# =========================

def ask_ollama(prompt, ollama_model="llama3.1:latest"):
  url = "http://localhost:11434/api/generate"

  payload = {
    "model": ollama_model,
    "prompt": prompt,
    "stream": False
  }

  response = requests.post(url, json=payload)

  return response.json()["response"]


def generate_answer(query, context):
  prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not in the context,
say:
"I could not find the answer in the documents."

Context:
{context}

Question:
{query}

Answer:
"""

  return ask_ollama(prompt)

def ask_ai_model(query, results):
  contexts = []
  for i, item in enumerate(results):
    contexts.append(item["text"])

  context = "\n\n".join(contexts)

  answer = generate_answer(query, context)

  print("\n========== ANSWER ==========\n")
  print(answer)
  print("\n========== ANSWER ==========\n")

# =========================
# Main
# =========================

def main():

  faiss_index = row_data_to_vector()
  query = (
    "in `THE ADVENTURE OF THE THREE GABLES` "
    "story what is main story. "
    "explain in one paragraph."
  )
  results = search(query=query, index=faiss_index, k=1024)
  for i, item in enumerate(results):
    print(
      f"[{i + 1}] "
      f"(score: {item['score']:.4f}) : "
      f" {item["text"]}"
    )
  ask_ai_model(query, results)



if __name__ == "__main__":
  main()
