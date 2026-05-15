
import os

from chunking.local_llm_chuck import LocalLLMChunking
from readers.file_reader import FileReader
from chunking.recursive_chunking import RecursiveChunking
from chunking.semantic_refinement import SemanticRefinement
from chunking.token_chunker import TokenChunker


class Rag_2:
  def __init__(self):
    self.__recursiveChunking = RecursiveChunking()
    self.__semanticRefinement = SemanticRefinement()
    self.__tokenChunker = TokenChunker()
    self.__localLLMChunking = LocalLLMChunking()

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

  def chunk_paragraph(self, paragraph: str) -> list[str]:
    return self.__recursiveChunking.split_paragraph(paragraph)

  def chunk_for_token(self, chunks: list[str]) -> list[str]:
    refine_chuck = []
    for chunk in chunks:
      c = self.__tokenChunker.chunk_text(chunk)
      refine_chuck.extend(c)
    return refine_chuck

  def extract_meta_chunks(self, chunks: list[str]):
    return self.__localLLMChunking.extract_add_more_info(chunks)