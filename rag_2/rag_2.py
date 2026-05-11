import os

from readers.file_reader import FileReader


class Rag_2:
  def __init__(self):
    pass

  def read_files(self):
    reader = FileReader()
    document = []
    for f in reader.get_all_files(os.path.join(os.getcwd(), ".data")):
      print(f)
      document.append(reader.read_one_file(f))
    print(len(document))