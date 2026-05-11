
import os
from readers.file_reader import FileReader

def main():
  reader = FileReader()
  document = []
  for f in reader.get_all_files(os.path.join(os.getcwd(), ".data")):
    print(f)
    document.append(reader.read_one_file(f))
  print(document)

if __name__ == "__main__":
  main()