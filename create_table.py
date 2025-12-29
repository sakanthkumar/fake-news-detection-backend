import sqlite3

conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    text TEXT,
    label TEXT,
    confidence REAL
)
""")

conn.commit()
conn.close()
print("✅ Table created successfully.")
