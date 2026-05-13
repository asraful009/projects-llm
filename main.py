
from rag_2.rag_2 import Rag_2

def main():
  rag = Rag_2()
  documents = rag.read_files()
  print(len(documents))

if __name__ == "__main__":
  main()