import sqlite3

DB_PATH = "fyp.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chats(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        role TEXT,
        message TEXT
    )
    """)
    return conn

def add_user(username, password):
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO users VALUES(?,?)", (username, password))
    conn.commit()

def check_user(username, password):
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM users WHERE username=? AND password=?", (username,password)).fetchone()
    return row is not None

def save_chat(username, role, message):
    conn = get_conn()
    conn.execute("INSERT INTO chats(username, role, message) VALUES(?,?,?)", (username, role, message))
    conn.commit()