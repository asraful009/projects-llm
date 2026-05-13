
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

class RecursiveChunking:
  def __init__(self):
    pass

  def split_document(self, document):
    splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=50,
      separators=[
        "\n\n"
      ]
    )
    document = self.__clean_chunk(document)
    chunks = splitter.split_text(document)

    chunks = self.__clean_chunks(chunks)
    return chunks

  def __clean_chunks(self, chunks: list[str]) -> list[str]:
    cleaned_chunks = []
    for chunk in chunks:
      chunk = re.sub(r'[ \t]+', ' ', chunk) # remove extra spaces/tabs
      chunk = re.sub(r'\n+', '\n', chunk) # remove too many newlines
      chunk = chunk.strip() # remove start/end whitespace

      cleaned_chunks.append(chunk)

    return cleaned_chunks

  def __clean_chunk(self, chunk: str) -> str:
    chunk = re.sub(r'[ \t]+', ' ', chunk)
    chunk = chunk.strip()
    return chunk