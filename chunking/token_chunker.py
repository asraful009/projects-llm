
from transformers import AutoTokenizer
import re

class TokenChunker:
  def __init__(self,
               chunk_size=512,
               chunk_overlap=64):
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    self.tokenizer = AutoTokenizer.from_pretrained(
      "BAAI/bge-m3"
    )


  def token_count(self, text):
    return len(
      self.tokenizer.encode(
        text,
        add_special_tokens=False
      )
    )


  def split_sentences(self, text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


  def chunk_text(self, text):
    sentences = self.split_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
      sentence_tokens = self.token_count(sentence)
      # sentence too large
      if sentence_tokens > self.chunk_size:
        continue

      # chunk full
      if current_tokens + sentence_tokens > self.chunk_size:
        chunk = " ".join(current_chunk)
        chunks.append(chunk)

        # overlap
        overlap_chunk = []
        overlap_tokens = 0

        for s in reversed(current_chunk):
          s_tokens = self.token_count(s)
          if overlap_tokens + s_tokens > self.chunk_overlap:
            break
          overlap_chunk.insert(0, s)
          overlap_tokens += s_tokens
        current_chunk = overlap_chunk
        current_tokens = overlap_tokens

      current_chunk.append(sentence)
      current_tokens += sentence_tokens

    # final chunk
    if current_chunk:
      chunks.append(" ".join(current_chunk))

    return chunks
