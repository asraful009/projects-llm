
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    chunks = splitter.split_text(document)
    return chunks