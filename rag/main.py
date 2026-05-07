

import os
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(
  "sentence-transformers/all-MiniLM-L6-v2", token=HF_TOKEN
)

def list_files_in_directory(directory):
  return [
    f for f in os.listdir(directory) 
    if os.path.isfile(os.path.join(directory, f))
    and f.endswith(".txt")
  ]

def read_file(file_path):
  text = None
  with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().strip()
  return text if text else None

def read_all_files_in_directory(directory):
  data = []
  files = list_files_in_directory(directory)
  for file in files:
    data.append(read_file(os.path.join(directory, file)))
  return data

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


def large_chunk_splitter(text, max_tokens=300, overlap=50):
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
  print("Hello, RAG!")
  data = read_all_files_in_directory(".data")
  print(f"Read {len(data)} files.")
  chunks_all = []
  for i, text in enumerate(data):
    if not text:
      continue
    chunks = large_chunk_splitter(text)
    print(f"File {i} split into {len(chunks)} chunks.")
    chunks_all.extend(chunks)

  print(f"Total chunks: {len(chunks_all)}")
  save_chunks_to_files(chunks_all)

if __name__ == "__main__":
  main()