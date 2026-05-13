
from rag_2.rag_2 import Rag_2
import json

def save_to_file(data, filename="chunks.json"):
  with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

def main():
  rag = Rag_2()
  documents = rag.read_files()
  print(len(documents))


  paragraphs = rag.chunk_document(documents[0])
  print(len(paragraphs))
  chunks = rag.chunk_paragraphs(paragraphs)
  print(len(chunks))

  save_to_file(chunks, "chunks.json")

if __name__ == "__main__":
  main()