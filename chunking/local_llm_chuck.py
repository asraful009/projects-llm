
import requests
import json
import re
import os

class LocalLLMChunking:
  def __init__(self):
    self.__model = "llama3.1:latest" #  "qwen2.5:14b-instruct"
    self.__url = "http://localhost:11434/api/generate"

  def extract_add_more_info(self, chucks: list[str]):
    chucks_with_meta = self.__enrich_chunks(chucks)
    return chucks_with_meta

  def __load_prompt(self, prompt_path: str, chunks_text: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)  # go up to projects-llm/
    full_path = os.path.join(root_dir, prompt_path)
    with open(full_path, "r", encoding="utf-8") as f:
      template = f.read()
    return template.replace("{chunks}", chunks_text)

  def __enrich_chunks(self, chunks: list[str]):
    chunks_text = json.dumps(chunks, indent=2, ensure_ascii=False)
    prompt = self.__load_prompt("./prompt_templates/ai_prompt_template.txt", chunks_text)

    res = requests.post(
      self.__url ,
      json={
        "model": self.__model,
        "prompt": prompt,
        "stream": False,
        "options": {
          "temperature": 0
        }
      }
    )
    output = res.json()["response"]

    return output

  def __safe_json_parse(self, text: str):
    """
    Extract JSON even if model adds extra text
    """

    # find JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
      return {
        "title": "",
        "summary": "",
        "keywords": []
      }

    try:
      return json.loads(match.group())
    except json.JSONDecodeError:
      return {
        "title": "",
        "summary": "",
        "keywords": []
      }