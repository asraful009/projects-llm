import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from rab_db import RagDB
import requests

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(
  "sentence-transformers/all-MiniLM-L6-v2", token=HF_TOKEN
)
model = SentenceTransformer("all-MiniLM-L6-v2")

def list_files_in_directory(directory):
  return [
    f for f in os.listdir(directory) 
    if os.path.isfile(os.path.join(directory, f))
    and f.endswith(".txt")
  ]

def read_file(directory, file_path) -> str | None:
  text = None
  full_path = os.path.join(directory, file_path)
  with open(full_path, "r", encoding="utf-8") as f:
    text = f.read().strip()
  return text if text else None

def split_into_chunks(text, max_tokens=300, overlap=50):
  tokens = tokenizer.encode(text, add_special_tokens=False)
  chunks = []
  start = 0

  while start < len(tokens):
    end = start + max_tokens
    chunk_tokens = tokens[start:end]
    print(f"Chunk tokens: {len(chunk_tokens)}")
    chunk = tokenizer.decode(chunk_tokens)
    chunks.append(chunk)
    start += max_tokens - overlap

  return chunks


def large_chunk_splitter(text: str| None, max_tokens=300, overlap=50):
  if text is None:
    return []
  paragraphs = text.split("\n\n")
  print(f"Text split into {len(paragraphs)} paragraphs.")
  chunks = []
  for paragraph in paragraphs:
    paragraph_chunks = split_into_chunks(paragraph, max_tokens, overlap)
    chunks.extend(paragraph_chunks)
  return chunks

def save_chunks_to_files(chunks, output_dir=".chunks"):
  os.makedirs(output_dir, exist_ok=True)
  for i, chunk in enumerate(chunks):
    with open(os.path.join(output_dir, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
      f.write(chunk)

def row_data_to_vector(root_path = ".data", rag_db: RagDB = None):
  if os.path.exists(".ai_data/faiss.index"):
    print("Loading FAISS index from file...")
    faiss_index = faiss.read_index(".ai_data/faiss.index")
    return faiss_index

  if rag_db is None:
    rag_db = RagDB()
  if not os.path.exists(root_path):
    print(f"Directory {root_path} does not exist. Please create it and add .txt files to process.")
    return
  files = list_files_in_directory(root_path)
  print(f"Read {len(files)} files.")
  chunks_all = []
  rag_db.create_table()
  for i, file in enumerate(files):
    if rag_db.get_documents_by_file_name(file) is not None:
      print(f"Document with file name {file} already exists in the database. Skipping.")
      continue
    meta : dict[str, str| None] = {"file_name": file, "data": read_file(root_path, file)}
    if not meta["data"]:
      continue
    
    chunks = large_chunk_splitter(meta["data"])
    print(f"File {i} split into {len(chunks)} chunks.")
    chunks_all.extend(chunks)
    document_id = rag_db.insert_document(file_name=f"{meta['file_name']}", total_chunks=len(chunks_all))
    for j, chunk in enumerate(chunks):
      token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
      embedding = model.encode(chunk)
      embedding_blob = pickle.dumps(embedding)
      rag_db.insert_chunk(
        document_id=document_id,
        chunk_text=chunk,
        chunk_index=j,
        token_count=token_count,
        embedding=embedding_blob,
      )
  print(f"Total chunks: {len(chunks_all)}")
  embeddings = [ pickle.loads(x["embedding"]) for x in rag_db.get_all_chunks() ]
  rag_db.close()
  embeddings = np.array(embeddings).astype("float32")
  print(embeddings.shape)
  dimension = embeddings.shape[1]
  faiss_index = faiss.IndexFlatL2(dimension)
  faiss_index.add(embeddings)
  faiss.write_index(faiss_index, ".ai_data/faiss.index")
  return faiss_index

def search(query, index, model, rag_db, k=5):
  # 1. query → vector
  query_vec = model.encode([query]).astype("float32")
  # 2. FAISS search
  distances, indices = index.search(query_vec, k)
  if rag_db is None:
    rag_db = RagDB()
  rows = rag_db.get_all_chunks()
  results = []

  for idx, dist in zip(indices[0], distances[0]):
    results.append({
      "text": rows[idx]["chunk_text"],
      "distance": float(dist)
    })
  results = sorted(results, key=lambda x: x["distance"])
  return results

def generate_answer(query, context):
  prompt = f"""You are an AI assistant. Use this context to answer:\n
    Context: {context}\n
    Question: {query}\n
    Answer:
  """

  return ask_ollama(prompt)

def ask_ollama(prompt, ollama_model="llama3.1:latest"):
  url = "http://localhost:11434/api/generate"

  payload = {
    "model": ollama_model,
    "prompt": prompt,
    "stream": False
  }

  response = requests.post(url, json=payload)
  return response.json()["response"]

def main():
  faiss_index = row_data_to_vector()
  # query = input("Query: ")
  query = "in `THE ADVENTURE OF THE THREE GABLES` story what is main story. explain in one line."
  results = search(query, faiss_index, model, None, 250)
  context = ""
  for i, item in enumerate(results):
    print(f"[ {i+1} ] (score: {item['distance']}):  {item['text']} ")
    context = "\n".join([c for c in item['text']])
  answers = generate_answer(query, context)
  print(answers)

if __name__ == "__main__":
  main()
  