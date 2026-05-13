
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
import os

class SemanticRefinement:
  def __init__(self) -> None:
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
      login(hf_token)

    self.__model = SentenceTransformer("BAAI/bge-m3")

  def refine_chunks(self, chunks: list[str], threshold: float = 0.75) -> list[str]:
    if not chunks:
      return []
    print(f"row chuck len :  {len(chunks)}")
    embeddings = self.__embed(chunks)
    print(f"emb len : {len(embeddings)}")
    merged = []
    current_chunk = chunks[0]
    current_emb = embeddings[0]

    for i in range(1, len(chunks)):
      sim = np.dot(current_emb, embeddings[i])

      if len(current_chunk) > 2000:
        merged.append(current_chunk)
        current_chunk = chunks[i]
        current_emb = embeddings[i]
        continue

      if sim >= threshold:
        current_chunk += "\n" + chunks[i]
        current_emb = 0.7 * current_emb + 0.3 * embeddings[i]
      else:
        merged.append(current_chunk)
        current_chunk = chunks[i]
        current_emb = embeddings[i]

    merged.append(current_chunk)

    return merged

  def __embed(self, row_chuck: list[str]):
    return self.__model.encode(row_chuck, normalize_embeddings=True)