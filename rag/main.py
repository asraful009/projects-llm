import os

from dotenv import load_dotenv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from rab_db import RagDB
import pickle


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

def main():
  root_path = ".data"
  if not os.path.exists(root_path):
    print(f"Directory {root_path} does not exist. Please create it and add .txt files to process.")
    return
  files = list_files_in_directory(root_path)
  print(f"Read {len(files)} files.")
  chunks_all = []
  rag_db = RagDB()
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
      # print(embedding)
      rag_db.insert_chunk(
        document_id=document_id,
        chunk_text=chunk,
        chunk_index=j,
        token_count=token_count,
        embedding=embedding_blob,
      )
  print(f"Total chunks: {len(chunks_all)}")
  rag_db.close()

if __name__ == "__main__":
  main()