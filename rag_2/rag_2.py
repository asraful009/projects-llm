
import os

from chunking.local_llm_chuck import LocalLLMChunking
from readers.file_reader import FileReader
from chunking.recursive_chunking import RecursiveChunking
from chunking.semantic_refinement import SemanticRefinement
from chunking.token_chunker import TokenChunker


class Rag_2:
  def __init__(self):
    self.__recursiveChunking = None
    self.__semanticRefinement = None
    self.__tokenChunker = None
    self.__localLLMChunking = None

  def read_files(self):
    reader = FileReader()
    documents = []
    for f in reader.get_all_files(os.path.join(os.getcwd(), ".data")):
      print(f"file: {f}")
      documents.append(reader.read_one_file(f))
    # print(len(documents))
    return documents

  def chunk_document(self, document):
    if not self.__recursiveChunking:
      self.__recursiveChunking = RecursiveChunking()
    paragraphs =  self.__recursiveChunking.split_document(document)
    return paragraphs

  def chunk_paragraphs(self, paragraphs: list[str]) -> list[str]:
    if not self.__semanticRefinement:
      self.__semanticRefinement = SemanticRefinement()
    return self.__semanticRefinement.refine_chunks(paragraphs)

  def chunk_paragraph(self, paragraph: str) -> list[str]:
    if not self.__recursiveChunking:
      self.__recursiveChunking = RecursiveChunking()
    return self.__recursiveChunking.split_paragraph(paragraph)

  def chunk_for_token(self, chunks: list[str]) -> list[str]:
    if not self.__tokenChunker:
      self.__tokenChunker = TokenChunker()
    refine_chuck = []
    for chunk in chunks:
      c = self.__tokenChunker.chunk_text(chunk)
      refine_chuck.extend(c)
    return refine_chuck

  def extract_meta_chunks(self, chunks: list[str]):
    if not self.__localLLMChunking:
      self.__localLLMChunking = LocalLLMChunking()
    return self.__localLLMChunking.extract_add_more_info(chunks)