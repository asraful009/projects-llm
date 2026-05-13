
import os



from readers.file_reader import FileReader
from chunking.recursive_chunking import RecursiveChunking
from chunking.semantic_refinement import SemanticRefinement


class Rag_2:
  def __init__(self):
    self.__recursiveChunking = RecursiveChunking()
    self.__semanticRefinement = SemanticRefinement()

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

  def chunk_paragraphs(self, paragraphs: list[str]) -> list[str]:
    return self.__semanticRefinement.refine_chunks(paragraphs)