import sqlite3

class RagDB:
  def __init__(self, db_path="db.sqlite3"):
    self.db_path = db_path
    self.conn = sqlite3.connect(self.db_path)
    self.cursor = self.conn.cursor()
    
  def create_table(self):
    self.cursor.execute('''
      CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        file_name TEXT NOT NULL,
        file_path TEXT,
        total_chunks INTEGER
      )
    ''')
    self.conn.commit()
    self.cursor.execute('''
      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER,
        token_count INTEGER,        
        embedding BLOB,
        
        FOREIGN KEY(document_id)
        REFERENCES documents(id)
      )
    ''')
    self.conn.commit()
    
  def insert_document(self, file_name, file_path=None, total_chunks=0):
    self.cursor.execute("""
      INSERT INTO documents (
        file_name,
        file_path,
        total_chunks
      )
      VALUES (?, ?, ?)
    """, (file_name, file_path, total_chunks))

    self.conn.commit()

    # return inserted document id
    return self.cursor.lastrowid

  # insert chunk using document_id
  def insert_chunk(
      self,
      document_id,
      chunk_text,
      chunk_index=None,
      token_count=None,
      embedding=None,
  ):
    self.cursor.execute("""
      INSERT INTO chunks (
        document_id,
        chunk_text,
        chunk_index,
        token_count,
        embedding
      )
      VALUES (?, ?, ?, ?, ?)
      """, (
      document_id,
      chunk_text,
      chunk_index,
      token_count,
      embedding
    ))

    self.conn.commit()

  def get_all_documents(self):
    self.cursor.execute("SELECT * FROM documents")
    return self.cursor.fetchall()
  
  def get_documents_by_file_name(self, file_name):
    self.cursor.execute("SELECT * FROM documents WHERE file_name = ?", (file_name,))
    return self.cursor.fetchone()

  def get_all_chunks(self) -> list[dict] :
    self.cursor.execute("""
      SELECT id, document_id, chunk_text, chunk_index, token_count, embedding
      FROM chunks
      """
    )
    rows = self.cursor.fetchall()
    results:list[dict] = []
    for row in rows:
      results.append({
        "id": row[0],
        "document_id": row[1],
        "chunk_text": row[2],
        "chunk_index": row[3],
        "token_count": row[4],
        "embedding": row[5]
      })
    return results

  def get_chunks_by_document(self, document_id):
    self.cursor.execute("""
      SELECT * FROM chunks
      WHERE document_id = ?
      ORDER BY chunk_index
    """, (document_id,))

    return self.cursor.fetchall()

  def close(self):
    self.conn.close()

