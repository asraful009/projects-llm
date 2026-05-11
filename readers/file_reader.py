
import os
from pathlib import Path

class FileReader():
  ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
  }


  def __init__(self):
    pass

  def get_all_files(self, full_path) -> list[str]:
    path = Path(full_path)
    return [
      str(file)
      for file in path.rglob("*")
      if file.is_file()
         and file.suffix.lower() in self.ALLOWED_EXTENSIONS
    ]

  def read_one_file(self, full_path) -> str:
    path = Path(full_path)
    with path.open() as f:
      return f.read()
