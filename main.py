import os

from rag_2.rag_2 import Rag_2
import json

def save_to_file(data, filename="chunks.json"):
  with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

def main():
  rag = Rag_2()
  chuck_for_token = []
  if not os.path.isfile(".ai_data/chunks_pargraph.json"):

    documents = rag.read_files()
    print(len(documents))


    paragraphs = rag.chunk_document(documents[0])
    print(len(paragraphs))
    paragraphs_chunks = rag.chunk_paragraphs(paragraphs)
    print(len(paragraphs_chunks))

    save_to_file(paragraphs_chunks, ".ai_data/chunks_pargraph.json")

    for paragraph in paragraphs_chunks:
      chucks = rag.chunk_paragraph(paragraph)
      chucks = rag.chunk_for_token(chucks)
      chuck_for_token.extend(chucks)

    save_to_file(chuck_for_token, ".ai_data/chuck_for_token.json")

  else:
    with open('.ai_data/chuck_for_token.json', 'r') as file:
      chuck_for_token = json.load(file)

  print(len(chuck_for_token))

if __name__ == "__main__":
  main()