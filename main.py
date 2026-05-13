
from rag_2.rag_2 import Rag_2

def main():
  rag = Rag_2()
  documents = rag.read_files()
  print(len(documents))

  paragraphs = rag.chunk_document(documents[0])
  print(len(paragraphs))
  print(paragraphs[0])
  print(paragraphs[1])
  print(paragraphs[2])

if __name__ == "__main__":
  main()