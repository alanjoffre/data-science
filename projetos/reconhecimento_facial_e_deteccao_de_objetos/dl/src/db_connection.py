import sqlite3
import pickle
import os

# Caminho para o banco de dados SQLite
DATABASE_PATH = "encodings.db"

def create_database():
    """Cria o banco de dados e a tabela de encodings se não existirem."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_encoding(name, encoding):
    """Salva um novo encoding no banco de dados."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    encoding_blob = pickle.dumps(encoding)  # Serializa o encoding
    cursor.execute("INSERT INTO encodings (name, encoding) VALUES (?, ?)", (name, encoding_blob))
    conn.commit()
    conn.close()

def load_encodings():
    """Carrega todos os encodings do banco de dados."""
    if not os.path.exists(DATABASE_PATH):
        create_database()

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM encodings")
    rows = cursor.fetchall()
    conn.close()

    # Desserializa os encodings
    encodings = [(row[0], pickle.loads(row[1])) for row in rows]
    return encodings

def delete_user(name):
    """Exclui um usuário do banco de dados pelo nome."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM encodings WHERE name = ?", (name,))
    conn.commit()
    conn.close()
