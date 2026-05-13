import os


from readers.file_reader import FileReader
from chunking.recursive_chunking import RecursiveChunking

class Rag_2:
  def __init__(self):
    self.__recursiveChunking = RecursiveChunking()

  def read_files(self):
    reader = FileReader()
    documents = []
    for f in reader.get_all_files(os.path.join(os.getcwd(), ".data")):
      print(f"file: {f}")
      documents.append(reader.read_one_file(f))
    # print(len(documents))
    return documents

  def chunk_document(self, document):
    paragraphs =  self.__recursiveChunking.split_document(document)
    return paragraphs
