
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

  def extract_faqs(self, meta_info: dict):
    if not self.__localLLMChunking:
      self.__localLLMChunking = LocalLLMChunking()
    return self.__localLLMChunking.extract_faqs_info(meta_info)

  def generate_meta_for_embeddings(self, meta_info: dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)  # go up to projects-llm/
    full_path = os.path.join(root_dir, "./prompt_templates/ai_prompt_template_for_embed.txt")
    meta_info_template = ""
    with open(full_path, "r", encoding="utf-8") as f:
      meta_info_template = f.read()
    meta_info_embedded_arr = []

    if not meta_info["chunks"]:
      return meta_info_embedded_arr

    for i, chunk in enumerate(meta_info["chunks"]):
      embedding_text = meta_info_template.format(
        chunk_type=chunk.get("chunk_type", ""),
        title=chunk.get("title", ""),
        summary=chunk.get("summary", ""),
        chunk=chunk.get("chunk", ""),
        keywords=", ".join(chunk.get("keywords", [])),
        topics=", ".join(chunk.get("topics", [])),
        people=", ".join(chunk.get("entities", {}).get("people", [])),
        locations=", ".join(chunk.get("entities", {}).get("locations", [])),
        orgs=", ".join(chunk.get("entities", {}).get("organizations", [])),
        questions=", ".join(chunk.get("questions_raised", []))
      )
      meta_info_embedded_arr.append(embedding_text)
    return meta_info_embedded_arr


  def generate_faq_for_embeddings(self, faqs: dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)  # go up to projects-llm/
    full_path = os.path.join(root_dir, "./prompt_templates/ai_prompt_template_for_faq_embed.txt")
    meta_info_template = ""
    with open(full_path, "r", encoding="utf-8") as f:
      meta_info_template = f.read()
    meta_info_embedded_arr = []
    if not faqs["FAQs"]:
      return meta_info_embedded_arr
    for i, chunk in enumerate(faqs["FAQs"]):
      embedding_text = meta_info_template.format(
        questions=chunk.get("faq", ""),
        answer=chunk.get("ans", "")
      )
      meta_info_embedded_arr.append(embedding_text)
    return meta_info_embedded_arr

